"""Pivot selection on the import camera's depth.

The pivot is the median depth of a small patch around the chosen pixel,
lifted to world space; the median ignores a patch straddling a depth edge.
"""

from __future__ import annotations

import numpy as np

from ouroworld.geometry.camera import Camera
from ouroworld.geometry.trajectories import unproject_z_depth
from ouroworld.io.scene_package import PivotAnnotation


def pick_pivot(
    depth: np.ndarray, camera: Camera, pixel: tuple[float, float], patch_radius: int
) -> PivotAnnotation:
    """Pivot at a clicked ``pixel`` = (x, y) of ``camera``.

    Args:
        depth: ``(H, W)`` camera-z depth of ``camera``, 0 where not reliable.
        camera: The camera ``depth`` was rendered from.
        pixel: Clicked pixel.
        patch_radius: Half size of the square patch whose median depth is used.

    Raises:
        ValueError: If the pixel is outside the image or its patch has no depth.
    """
    x, y = float(pixel[0]), float(pixel[1])
    if not (0.0 <= x < camera.width and 0.0 <= y < camera.height):
        raise ValueError("the pixel lies outside the image")
    column, row = int(round(x)), int(round(y))
    patch = depth[
        max(0, row - patch_radius) : min(camera.height, row + patch_radius + 1),
        max(0, column - patch_radius) : min(camera.width, column + patch_radius + 1),
    ]
    return _annotation(patch, camera, np.array([x, y]), "click")


def center_pivot(depth: np.ndarray, camera: Camera, patch_fraction: float) -> PivotAnnotation:
    """Pivot at the image centre: the fallback when none was picked.

    Args:
        depth: ``(H, W)`` camera-z depth of ``camera``, 0 where not reliable.
        camera: The camera ``depth`` was rendered from.
        patch_fraction: Size of the central patch relative to the image.
    """
    width = max(1, int(round(camera.width * patch_fraction)))
    height = max(1, int(round(camera.height * patch_fraction)))
    left, top = max(0, (camera.width - width) // 2), max(0, (camera.height - height) // 2)
    patch = depth[top : top + height, left : left + width]
    centre = np.array([(camera.width - 1.0) / 2.0, (camera.height - 1.0) / 2.0])
    return _annotation(patch, camera, centre, "center_depth")


def _annotation(
    patch: np.ndarray, camera: Camera, pixel: np.ndarray, source: str
) -> PivotAnnotation:
    values = patch[np.isfinite(patch) & (patch > 0)]
    if values.size == 0:
        raise ValueError("no rendered surface around the pivot pixel; pick another one")
    depth = float(np.median(values))
    return PivotAnnotation(
        unproject_z_depth(camera.K, camera.c2w, pixel, depth), pixel, depth, source
    )
