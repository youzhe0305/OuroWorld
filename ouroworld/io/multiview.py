"""The ``MultiviewVideos`` artifact: generated multi-view videos plus their cameras.

On-disk layout (paths relative to the artifact root)::

    cameras.json        {"schema_version": 1, "views": [{"view_id", width, height, K, c2w}]}
    observations.json   {"schema_version": 1, "reference_view_id": int,
                         "observations": [{"view_id", "time", "path", "role"}]}
    frames/view_XX/*    the images referenced by ``path``

``time`` is the normalised loop phase in ``[0, 1)``. ``role`` states how an
observation was produced (paper §4.2):

* ``reference`` -- a frame of the looping reference video (the reference view);
* ``t0`` -- a non-reference view at ``t = 0``, i.e. a render of the static scene;
* ``generated`` -- a non-reference view at ``t > 0``, synthesised by the video model.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ouroworld.geometry.camera import Camera
from ouroworld.io.cameras import camera_from_record, camera_to_record
from ouroworld.io.errors import ArtifactError

SCHEMA_VERSION = 1
CAMERAS_FILE = "cameras.json"
OBSERVATIONS_FILE = "observations.json"


class Role(enum.StrEnum):
    """How an observation was produced."""

    REFERENCE = "reference"
    T0 = "t0"
    GENERATED = "generated"


@dataclass(frozen=True)
class Observation:
    """One supervision image.

    Attributes:
        view_id: Index of the camera that sees this image.
        time: Normalised loop phase in ``[0, 1)``.
        image_path: Absolute path of the image file.
        role: How the image was produced.
    """

    view_id: int
    time: float
    image_path: Path
    role: Role


@dataclass(frozen=True)
class MultiviewVideos:
    """Cameras and observations of one scene.

    Attributes:
        root: Directory the artifact was loaded from.
        cameras: Camera of every view, keyed by ``view_id``.
        observations: Every observation, sorted by ``(view_id, time)``.
        reference_view_id: The view holding the reference video.
    """

    root: Path
    cameras: dict[int, Camera]
    observations: tuple[Observation, ...]
    reference_view_id: int

    @property
    def view_ids(self) -> list[int]:
        """Sorted ids of all views."""
        return sorted(self.cameras)

    def with_role(self, *roles: Role) -> list[Observation]:
        """Return the observations whose role is one of ``roles``."""
        return [observation for observation in self.observations if observation.role in roles]


def expected_role(view_id: int, time: float, reference_view_id: int) -> Role:
    """Return the role an observation must have given where and when it lies."""
    if view_id == reference_view_id:
        return Role.REFERENCE
    return Role.T0 if time == 0.0 else Role.GENERATED


def load_multiview(root: Path) -> MultiviewVideos:
    """Read and validate a ``MultiviewVideos`` directory."""
    root = Path(root).resolve()
    cameras_doc = _read_json(root / CAMERAS_FILE)
    observations_doc = _read_json(root / OBSERVATIONS_FILE)
    cameras = {
        int(record["view_id"]): camera_from_record(record, f"{CAMERAS_FILE} view")
        for record in cameras_doc["views"]
    }
    reference_view_id = int(observations_doc["reference_view_id"])
    if reference_view_id not in cameras:
        raise ArtifactError(f"reference view {reference_view_id} has no camera")
    observations = tuple(
        sorted(
            (
                _observation_from_record(record, root, cameras, reference_view_id)
                for record in observations_doc["observations"]
            ),
            key=lambda item: (item.view_id, item.time),
        )
    )
    if not observations:
        raise ArtifactError(f"{root / OBSERVATIONS_FILE} lists no observations")
    return MultiviewVideos(root, cameras, observations, reference_view_id)


def save_multiview_index(
    root: Path,
    cameras: dict[int, Camera],
    observations: list[dict[str, Any]],
    reference_view_id: int,
) -> None:
    """Write ``cameras.json`` and ``observations.json``; images are written by the caller.

    Args:
        root: Artifact directory.
        cameras: Camera of every view.
        observations: Records with ``view_id``, ``time`` and a root-relative ``path``;
            ``role`` is derived from the other fields.
        reference_view_id: The view holding the reference video.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    views = [
        {"view_id": view_id, **camera_to_record(cameras[view_id])} for view_id in sorted(cameras)
    ]
    records = [
        {
            "view_id": int(record["view_id"]),
            "time": float(record["time"]),
            "path": str(record["path"]),
            "role": expected_role(
                int(record["view_id"]), float(record["time"]), reference_view_id
            ).value,
        }
        for record in observations
    ]
    _write_json(root / CAMERAS_FILE, {"schema_version": SCHEMA_VERSION, "views": views})
    _write_json(
        root / OBSERVATIONS_FILE,
        {
            "schema_version": SCHEMA_VERSION,
            "reference_view_id": int(reference_view_id),
            "observations": records,
        },
    )


def _observation_from_record(
    record: dict[str, Any], root: Path, cameras: dict[int, Camera], reference_view_id: int
) -> Observation:
    view_id = int(record["view_id"])
    time = float(record["time"])
    if view_id not in cameras:
        raise ArtifactError(f"observation {record['path']} refers to unknown view {view_id}")
    if not 0.0 <= time < 1.0:
        raise ArtifactError(f"observation {record['path']} has time {time} outside [0, 1)")
    role = Role(record["role"])
    if role is not expected_role(view_id, time, reference_view_id):
        raise ArtifactError(
            f"observation {record['path']} is marked {role.value} but lies at view {view_id}, "
            f"t={time} with reference view {reference_view_id}"
        )
    image_path = root / record["path"]
    if not image_path.is_file():
        raise ArtifactError(f"missing image {image_path}")
    return Observation(view_id, time, image_path, role)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactError(f"missing {path}")
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"{path}: expected schema_version {SCHEMA_VERSION}")
    return document


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
