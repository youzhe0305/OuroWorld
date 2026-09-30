"""Scene importers (paper App. A).

Each importer copies (or rescales) the scene's PLY into a package and stores
the camera the scene comes with as the import camera:

* Mip-NeRF 360: a 3DGS trained on the COLMAP scene; the first training image.
* HY-World 2.0 and Marble: generated worlds without a recorded pose; a camera
  on a world axis, by default HY-World's own ``view0`` (origin, looking down
  ``-x`` with ``+z`` up, 120° x 90°).
* Lyra 2.0: the generator's keyframe camera, with clip planes measured from
  the Gaussians and the world scaled out of the rasterizer's near cull.

The paper's scenes are released already imported (``data/3dgs/<dataset>/<scene>``,
with the pivot and the reference view); :func:`import_released` turns one
into a complete package.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from ouroworld.datasets.colmap import read_cameras, read_images
from ouroworld.datasets.world_scale import (
    auto_world_scale,
    rescale_ply,
    sampled_points,
    visible_points,
)
from ouroworld.geometry.camera import DEFAULT_ZFAR, DEFAULT_ZNEAR, Camera, intrinsics_from_fov
from ouroworld.geometry.trajectories import normalize, project_world_point
from ouroworld.io.images import read_rgb8
from ouroworld.io.scene_package import (
    GAUSSIANS_FILE,
    REFERENCE_IMAGE_FILE,
    PivotAnnotation,
    save_pivot,
    save_reference,
    save_scene_source,
)

AXES = {
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}  # fmt: skip
BLACK = (0.0, 0.0, 0.0)


def axis_camera(
    forward: str,
    up: str,
    position: np.ndarray,
    size: tuple[int, int],
    fov_degrees: tuple[float, float],
    clip: tuple[float, float],
) -> Camera:
    """A camera at ``position`` looking down world axis ``forward`` with ``up`` up in the image."""
    forward_axis = normalize(np.array(AXES[forward]), "forward")
    up_axis = normalize(np.array(AXES[up]), "up")
    if abs(float(forward_axis @ up_axis)) > 1e-6:
        raise ValueError("the forward and up axes must be perpendicular")
    right = normalize(np.cross(forward_axis, up_axis), "right")
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = right, np.cross(forward_axis, right), forward_axis
    c2w[:3, 3] = np.asarray(position, dtype=np.float64)
    fov_x, fov_y = (math.radians(value) for value in fov_degrees)
    return Camera(size[0], size[1], intrinsics_from_fov(size[0], size[1], fov_x, fov_y), c2w, *clip)


def import_generated_world(ply: Path, out_dir: Path, camera: Camera) -> None:
    """HY-World 2.0 / Marble: copy the PLY and store ``camera`` (see :func:`axis_camera`)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ply, Path(out_dir) / GAUSSIANS_FILE)
    save_scene_source(out_dir, BLACK, camera)


def import_mipnerf360(
    ply: Path, colmap_dir: Path, images: str, out_dir: Path, holdout: int, clip: tuple[float, float]
) -> None:
    """Mip-NeRF 360: copy the trained 3DGS and store its first training camera.

    Args:
        ply: The trained 3DGS.
        colmap_dir: Scene directory with ``sparse/0`` and the image folders.
        images: Image folder the 3DGS was trained on, e.g. ``images_2``.
        out_dir: Package directory.
        holdout: Every ``holdout``-th image (sorted by name) was held out for
            testing and is skipped; 0 when all were trained on.
        clip: ``(near, far)`` clip planes.
    """
    sparse = Path(colmap_dir) / "sparse" / "0"
    cameras, frames = read_cameras(sparse / "cameras.bin"), read_images(sparse / "images.bin")
    ordered = sorted(frames.values(), key=lambda frame: frame.name)
    training = [frame for index, frame in enumerate(ordered) if not holdout or index % holdout]
    first = min(training, key=lambda frame: Path(frame.name).stem)
    intrinsics = cameras[first.camera_id]
    with Image.open(Path(colmap_dir) / images / first.name) as image:
        width, height = image.size
    # The 3DGS keeps COLMAP's field of view at whatever resolution it trains on.
    focal_x, focal_y = intrinsics.focal
    fov_x = 2.0 * math.atan(intrinsics.width / (2.0 * focal_x))
    fov_y = 2.0 * math.atan(intrinsics.height / (2.0 * focal_y))
    rotation = first.w2c()[:3, :3].T
    c2w = np.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = -rotation.dot(first.tvec)
    camera = Camera(width, height, intrinsics_from_fov(width, height, fov_x, fov_y), c2w, *clip)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ply, Path(out_dir) / GAUSSIANS_FILE)
    save_scene_source(out_dir, BLACK, camera)


@dataclass(frozen=True)
class LyraSettings:
    """Clip-plane measurement and rescaling of a Lyra 2.0 scene.

    Attributes:
        sample_stride: Every this-many Gaussian is measured.
        opacity_min: Only Gaussians at least this opaque are measured.
        near_margin: The measured near plane is multiplied by this.
        far_margin: The measured far plane is multiplied by this.
        cull_fraction: Target ratio of the near cull to the nearest surfaces.
        world_scale: A fixed scale, or ``None`` to derive it from ``cull_fraction``.
    """

    sample_stride: int = 50
    opacity_min: float = 0.5
    near_margin: float = 0.5
    far_margin: float = 1.2
    cull_fraction: float = 0.05
    world_scale: float | None = None


def import_lyra(source_dir: Path, out_dir: Path, settings: LyraSettings) -> float:
    """Lyra 2.0 (``scene.ply`` + ``camera.json`` keyframe); returns the applied world scale.

    The far plane is the largest distance from the camera, not the largest
    depth: the reference candidates orbit this single pose.
    """
    source_dir = Path(source_dir)
    record = json.loads((source_dir / "camera.json").read_text(encoding="utf-8"))
    intrinsics = record["intrinsics"]
    K = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ]
    )
    w2c = np.asarray(record["extrinsic_world_to_camera"]["matrix_4x4"], dtype=np.float64)
    c2w = np.linalg.inv(w2c)
    width, height = int(round(2.0 * K[0, 2])), int(round(2.0 * K[1, 2]))
    ply = source_dir / "scene.ply"
    xyz, logits = sampled_points(ply, settings.sample_stride)
    points = visible_points(xyz, logits, settings.opacity_min)
    depth = points.dot(w2c[2, :3]) + w2c[2, 3]
    near = float(np.percentile(depth[depth > 0], 0.1)) * settings.near_margin
    far = float(np.linalg.norm(points - c2w[:3, 3], axis=1).max()) * settings.far_margin
    near = max(near, 1e-4)
    if settings.world_scale is None:
        scale, _ = auto_world_scale(points, [(w2c, K, width, height)], settings.cull_fraction)
    else:
        scale = settings.world_scale
    c2w[:3, 3] *= scale
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rescale_ply(ply, Path(out_dir) / GAUSSIANS_FILE, scale)
    save_scene_source(out_dir, BLACK, Camera(width, height, K, c2w, near * scale, far * scale))
    return scale


RELEASED_FILES = (
    "point_cloud.ply",
    "metadata.yaml",
    "pivot.json",
    "reference_camera.json",
    "reference.png",
)


def import_released(source_dir: Path, out_dir: Path) -> str:
    """A released scene of the paper; returns the chosen reference candidate.

    A released scene holds the imported PLY (``point_cloud.ply``), its render
    settings (``metadata.yaml``), the clicked pivot (``pivot.json``) and the
    reference view (``reference_camera.json``, ``reference.png``). The pivot
    was clicked on the camera the scene was imported with, which is not
    released; the reference camera takes its place, and the pivot's pixel and
    depth are re-projected into it.
    """
    source_dir = Path(source_dir)
    missing = [name for name in RELEASED_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{source_dir}: missing {missing}")
    metadata = yaml.safe_load((source_dir / "metadata.yaml").read_text(encoding="utf-8"))
    background = tuple(float(value) for value in metadata["gaussians"]["background"])
    rendering = metadata["rendering"]
    world_up = rendering.get("world_up")
    record = json.loads((source_dir / "reference_camera.json").read_text(encoding="utf-8"))
    camera = Camera(
        int(record["width"]),
        int(record["height"]),
        np.asarray(record["K"], dtype=np.float64),
        np.asarray(record["c2w"], dtype=np.float64),
        # Packages written without clip planes used the renderer's defaults.
        float(rendering.get("near") or DEFAULT_ZNEAR),
        float(rendering.get("far") or DEFAULT_ZFAR),
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_dir / "point_cloud.ply", Path(out_dir) / GAUSSIANS_FILE)
    save_scene_source(
        out_dir,
        background,  # type: ignore[arg-type]
        camera,
        None if world_up is None else np.asarray(world_up, dtype=np.float64),
    )
    world = np.asarray(json.loads((source_dir / "pivot.json").read_text())["world_xyz"], float)
    pixel, depth = project_world_point(camera.K, camera.c2w, world)
    save_pivot(out_dir, PivotAnnotation(world, pixel, depth, "click"))
    selected = str(record.get("source_candidate", "reference"))
    save_reference(out_dir, camera, read_rgb8(source_dir / "reference.png"), selected)
    # Keep the released image byte for byte rather than a re-encoding.
    shutil.copyfile(source_dir / "reference.png", Path(out_dir) / REFERENCE_IMAGE_FILE)
    return selected
