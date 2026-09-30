"""The ``ReferenceVideo`` artifact: the looping video seen from the reference camera.

On-disk layout (paths relative to the artifact root)::

    reference_video.json  {"schema_version": 1, "cycle_seconds": float,
                           "frames": [{"path", "time"}]}
    frames/*              the frames, at the reference camera's resolution

``time`` is the loop phase in ``[0, 1]``. The first frame lies at ``t = 0`` and
the last at ``t = 1``: the video starts and ends on the same state (it is
generated with the reference image as both first and last frame), so the last
frame closes the loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ouroworld.io.errors import ArtifactError

SCHEMA_VERSION = 1
INDEX_FILE = "reference_video.json"


@dataclass(frozen=True)
class ReferenceVideo:
    """Frames of one loop of the reference video.

    Attributes:
        root: Directory the artifact was loaded from.
        frame_paths: Absolute path of every frame, in time order.
        times: ``(F,)`` loop phase of every frame, from 0 to 1 inclusive.
        cycle_seconds: Duration of one loop.
    """

    root: Path
    frame_paths: tuple[Path, ...]
    times: np.ndarray
    cycle_seconds: float

    def __len__(self) -> int:
        return len(self.frame_paths)

    def evenly_spaced(self, count: int) -> list[int]:
        """Return the indices of ``count`` frames at uniform phase steps, both ends included.

        Raises:
            ArtifactError: If the frames cannot be subsampled at exact steps.
        """
        if count < 2 or (len(self) - 1) % (count - 1):
            raise ArtifactError(
                f"{len(self)} reference frames cannot be split into {count - 1} equal steps"
            )
        stride = (len(self) - 1) // (count - 1)
        return list(range(0, len(self), stride))


def load_reference_video(root: Path) -> ReferenceVideo:
    """Read and validate a ``ReferenceVideo`` directory."""
    root = Path(root).resolve()
    index_path = root / INDEX_FILE
    if not index_path.is_file():
        raise ArtifactError(f"missing {index_path}")
    document = json.loads(index_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"{index_path}: expected schema_version {SCHEMA_VERSION}")
    records = document["frames"]
    times = np.asarray([float(record["time"]) for record in records])
    if len(records) < 2 or times[0] != 0.0 or times[-1] != 1.0 or np.any(np.diff(times) <= 0):
        raise ArtifactError(f"{index_path}: frame times must increase from 0 to 1")
    paths = tuple(root / record["path"] for record in records)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ArtifactError(f"{index_path}: missing frames {missing[:3]}")
    return ReferenceVideo(root, paths, times, float(document["cycle_seconds"]))


def save_reference_video_index(
    root: Path, records: list[dict[str, Any]], cycle_seconds: float
) -> None:
    """Write ``reference_video.json``; the frames are written by the caller.

    Args:
        root: Artifact directory.
        records: ``{"path", "time"}`` per frame, ``path`` relative to ``root``.
        cycle_seconds: Duration of one loop.
    """
    document = {
        "schema_version": SCHEMA_VERSION,
        "cycle_seconds": float(cycle_seconds),
        "frames": [{"path": str(item["path"]), "time": float(item["time"])} for item in records],
    }
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / INDEX_FILE).write_text(json.dumps(document, indent=2), encoding="utf-8")
