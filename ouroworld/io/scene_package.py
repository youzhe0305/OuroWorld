"""The scene package: the input static 3DGS, its cameras and its pivot.

On-disk layout::

    gaussians.ply   the input 3DGS (INRIA layout)
    scene.json      {"schema_version": 1,
                     "background": [r, g, b],
                     "clip": {"near", "far"},
                     "world_up": [x, y, z],
                     "import_camera": {"width", "height", "K", "c2w"},
                     "pivot": null | {"world", "pixel", "depth", "source"},
                     "reference": null | {"camera": {...}, "image": "reference.png",
                                          "selected": "reference" | "candidate_XXX"}}
    reference.png   the input 3DGS rendered from the reference camera

A package is built in three steps (paper App. A, §4.1): an importer writes the
PLY and the camera the scene came with (``import_camera``); the pivot is
picked on the import camera's render; the reference camera is chosen around
the pivot and its render saved. :class:`SceneSource` is the package before the
last step, :class:`ScenePackage` the complete package every later stage reads.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ouroworld.geometry.camera import Camera
from ouroworld.io.cameras import camera_from_record, camera_to_record
from ouroworld.io.errors import ArtifactError
from ouroworld.io.images import save_rgb8

SCHEMA_VERSION = 1
GAUSSIANS_FILE = "gaussians.ply"
METADATA_FILE = "scene.json"
REFERENCE_IMAGE_FILE = "reference.png"
PIVOT_SOURCES = ("click", "center_depth")


@dataclass(frozen=True)
class PivotAnnotation:
    """The world point both camera orbits turn around (paper §4.1, §4.3).

    Attributes:
        world: ``(3,)`` pivot in world space.
        pixel: ``(2,)`` pixel of the import camera the pivot was picked at.
        depth: Camera-z depth of the pivot in the import camera.
        source: ``"click"`` (picked by hand) or ``"center_depth"`` (image centre).
    """

    world: np.ndarray
    pixel: np.ndarray
    depth: float
    source: str


@dataclass(frozen=True)
class SceneSource:
    """A package before its reference camera is chosen.

    Attributes:
        root: Package directory.
        gaussians_path: Path of the input 3DGS PLY.
        background: RGB background colour in ``[0, 1]``.
        world_up: ``(3,)`` unit up direction of the world.
        import_camera: Camera the scene was imported with, with the clip planes.
        pivot: Pivot annotation, or ``None`` if not picked yet.
    """

    root: Path
    gaussians_path: Path
    background: tuple[float, float, float]
    world_up: np.ndarray
    import_camera: Camera
    pivot: PivotAnnotation | None


@dataclass(frozen=True)
class ScenePackage:
    """Input static scene.

    Attributes:
        root: Package directory.
        gaussians_path: Path of the input 3DGS PLY.
        background: RGB background colour in ``[0, 1]``.
        reference_camera: Camera of the reference view, with the scene's clip planes.
        reference_image_path: Render of the input 3DGS from the reference camera.
        pivot: ``(3,)`` orbit pivot in world space, or ``None`` if not annotated.
    """

    root: Path
    gaussians_path: Path
    background: tuple[float, float, float]
    reference_camera: Camera
    reference_image_path: Path
    pivot: np.ndarray | None


def load_scene_source(root: Path) -> SceneSource:
    """Read and validate everything but the reference view."""
    root = Path(root).resolve()
    metadata, path = _read_metadata(root)
    background = tuple(float(value) for value in metadata["background"])
    if len(background) != 3:
        raise ArtifactError(f"{path}: background must have three channels")
    world_up = _finite_vector(metadata["world_up"], 3, f"{path} world_up")
    norm = float(np.linalg.norm(world_up))
    if norm < 1e-12:
        raise ArtifactError(f"{path}: world_up must be non-zero")
    camera = _clipped(
        camera_from_record(metadata["import_camera"], f"{path} import_camera"), metadata, path
    )
    pivot = None if metadata.get("pivot") is None else _pivot_from_record(metadata["pivot"], path)
    return SceneSource(
        root,
        root / GAUSSIANS_FILE,
        background,
        world_up / norm,
        camera,
        pivot,  # type: ignore[arg-type]
    )


def load_scene_package(root: Path) -> ScenePackage:
    """Read and validate a complete scene package."""
    source = load_scene_source(root)
    metadata, path = _read_metadata(source.root)
    reference = metadata.get("reference")
    if reference is None:
        raise ArtifactError(f"{path}: no reference camera yet; run ouroworld-reference")
    camera = _clipped(camera_from_record(reference["camera"], f"{path} reference"), metadata, path)
    image_path = source.root / reference["image"]
    if not image_path.is_file():
        raise ArtifactError(f"missing {image_path}")
    pivot = None if source.pivot is None else source.pivot.world
    return ScenePackage(
        source.root, source.gaussians_path, source.background, camera, image_path, pivot
    )


def save_scene_source(
    root: Path,
    background: tuple[float, float, float],
    import_camera: Camera,
    world_up: np.ndarray | None = None,
) -> None:
    """Write ``scene.json`` of a freshly imported scene; the PLY is written by the caller.

    Args:
        root: Package directory.
        background: RGB background colour in ``[0, 1]``.
        import_camera: The scene's camera; its clip planes become the scene's.
        world_up: Up direction, by default the import camera's up (``-y`` axis).
    """
    up = -import_camera.c2w[:3, 1] if world_up is None else np.asarray(world_up, np.float64)
    document = {
        "schema_version": SCHEMA_VERSION,
        "background": [float(value) for value in background],
        "clip": {"near": float(import_camera.znear), "far": float(import_camera.zfar)},
        "world_up": [float(value) for value in up],
        "import_camera": camera_to_record(import_camera),
        "pivot": None,
        "reference": None,
    }
    Path(root).mkdir(parents=True, exist_ok=True)
    _write_metadata(Path(root), document)


def save_pivot(root: Path, pivot: PivotAnnotation) -> None:
    """Store the pivot. The reference view depends on it, so it is cleared."""
    metadata, _ = _read_metadata(Path(root))
    metadata["pivot"] = {
        "world": [float(value) for value in pivot.world],
        "pixel": [float(value) for value in pivot.pixel],
        "depth": float(pivot.depth),
        "source": pivot.source,
    }
    metadata["reference"] = None
    _write_metadata(Path(root), metadata)


def save_reference(root: Path, camera: Camera, image: np.ndarray, selected: str) -> None:
    """Store the reference camera and its ``(H, W, 3)`` uint8 render.

    Args:
        root: Package directory.
        camera: Reference camera; its clip planes are ignored (the scene's apply).
        image: Render of the input 3DGS from ``camera``.
        selected: Name of the chosen candidate, ``"reference"`` for the import pose.
    """
    if image.shape != (camera.height, camera.width, 3):
        raise ValueError(f"image {image.shape} does not match the camera")
    metadata, _ = _read_metadata(Path(root))
    save_rgb8(image, Path(root) / REFERENCE_IMAGE_FILE)
    metadata["reference"] = {
        "camera": camera_to_record(camera),
        "image": REFERENCE_IMAGE_FILE,
        "selected": selected,
    }
    _write_metadata(Path(root), metadata)


def _read_metadata(root: Path) -> tuple[dict[str, Any], Path]:
    path = root / METADATA_FILE
    for required in (path, root / GAUSSIANS_FILE):
        if not required.is_file():
            raise ArtifactError(f"missing {required}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"{path}: expected schema_version {SCHEMA_VERSION}")
    return metadata, path


def _write_metadata(root: Path, document: dict[str, Any]) -> None:
    # Write then rename: an interrupted write must not leave a truncated package.
    temporary = root / f".{METADATA_FILE}.{os.getpid()}"
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(root / METADATA_FILE)


def _clipped(camera: Camera, metadata: dict[str, Any], path: Path) -> Camera:
    near, far = float(metadata["clip"]["near"]), float(metadata["clip"]["far"])
    if not 0.0 < near < far:
        raise ArtifactError(f"{path}: clip planes must satisfy 0 < near < far")
    return Camera(camera.width, camera.height, camera.K, camera.c2w, near, far)


def _pivot_from_record(record: dict[str, Any], path: Path) -> PivotAnnotation:
    depth = float(record["depth"])
    if not np.isfinite(depth) or depth <= 0 or record["source"] not in PIVOT_SOURCES:
        raise ArtifactError(f"{path}: pivot needs a positive depth and a source in {PIVOT_SOURCES}")
    return PivotAnnotation(
        _finite_vector(record["world"], 3, f"{path} pivot world"),
        _finite_vector(record["pixel"], 2, f"{path} pivot pixel"),
        depth,
        str(record["source"]),
    )


def _finite_vector(values: Any, size: int, where: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ArtifactError(f"{where}: expected {size} finite numbers")
    return array
