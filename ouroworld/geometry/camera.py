"""Pinhole camera in the OpenCV convention and its rasterizer matrices.

World-to-camera matrices follow the OpenCV axes (x right, y down, z forward).
The rasterizer expects row-major (transposed) 4x4 matrices, as in 3D Gaussian
Splatting, which :meth:`Camera.raster_matrices` produces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

DEFAULT_ZNEAR = 0.01
DEFAULT_ZFAR = 100.0


@dataclass(frozen=True)
class RasterMatrices:
    """Tensors the rasterizer consumes for one camera (all float32, CPU)."""

    world_view: torch.Tensor  # (4, 4) transposed world-to-camera
    full_projection: torch.Tensor  # (4, 4) transposed world-to-clip
    camera_center: torch.Tensor  # (3,) world-space camera position
    tan_half_fov_x: float
    tan_half_fov_y: float


@dataclass(frozen=True)
class Camera:
    """Pinhole camera with a centred principal point.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        K: ``(3, 3)`` intrinsics in pixels.
        c2w: ``(4, 4)`` OpenCV camera-to-world pose.
        znear: Near clipping plane used by the projection matrix.
        zfar: Far clipping plane used by the projection matrix.
    """

    width: int
    height: int
    K: np.ndarray = field(repr=False)
    c2w: np.ndarray = field(repr=False)
    znear: float = DEFAULT_ZNEAR
    zfar: float = DEFAULT_ZFAR

    def __post_init__(self) -> None:
        K = np.asarray(self.K, dtype=np.float64)
        c2w = np.asarray(self.c2w, dtype=np.float64)
        if K.shape != (3, 3) or c2w.shape != (4, 4):
            raise ValueError(f"expected K (3, 3) and c2w (4, 4), got {K.shape} and {c2w.shape}")
        if not (np.isfinite(K).all() and np.isfinite(c2w).all()):
            raise ValueError("camera matrices must be finite")
        # The rasterizer only takes a field of view, so an off-centre principal
        # point would be silently ignored; reject it instead.
        if not (
            math.isclose(K[0, 2], self.width / 2.0, abs_tol=1e-3)
            and math.isclose(K[1, 2], self.height / 2.0, abs_tol=1e-3)
        ):
            raise ValueError(
                f"principal point {K[:2, 2].tolist()} is not the image centre "
                f"{[self.width / 2.0, self.height / 2.0]}"
            )
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "c2w", c2w)

    @property
    def fov_x(self) -> float:
        """Horizontal field of view in radians."""
        return 2.0 * math.atan(self.width / (2.0 * self.K[0, 0]))

    @property
    def fov_y(self) -> float:
        """Vertical field of view in radians."""
        return 2.0 * math.atan(self.height / (2.0 * self.K[1, 1]))

    @property
    def center(self) -> np.ndarray:
        """World-space camera position, ``(3,)``."""
        return self.c2w[:3, 3].copy()

    def w2c(self) -> np.ndarray:
        """OpenCV world-to-camera matrix, ``(4, 4)`` float64."""
        return np.linalg.inv(self.c2w)

    def with_pose(self, c2w: np.ndarray) -> Camera:
        """Return the same intrinsics at another pose."""
        return Camera(self.width, self.height, self.K, c2w, self.znear, self.zfar)

    def resized(self, width: int, height: int) -> Camera:
        """Return the same camera at another image resolution (no crop)."""
        return Camera(
            width,
            height,
            scale_intrinsics(self.K, (self.width, self.height), (width, height)),
            self.c2w,
            self.znear,
            self.zfar,
        )

    def raster_matrices(self) -> RasterMatrices:
        """Build the transposed matrices the rasterizer expects."""
        tan_x = math.tan(self.fov_x * 0.5)
        tan_y = math.tan(self.fov_y * 0.5)
        world_view = torch.tensor(np.float32(self.w2c())).transpose(0, 1)
        projection = perspective_projection(self.znear, self.zfar, tan_x, tan_y).transpose(0, 1)
        full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
        camera_center = world_view.inverse()[3, :3]
        return RasterMatrices(world_view, full_projection, camera_center, tan_x, tan_y)


def perspective_projection(
    znear: float, zfar: float, tan_half_fov_x: float, tan_half_fov_y: float
) -> torch.Tensor:
    """Return the symmetric-frustum projection used by 3D Gaussian Splatting.

    Depth is mapped to ``[0, 1]`` between ``znear`` and ``zfar``.
    """
    top = tan_half_fov_y * znear
    right = tan_half_fov_x * znear
    projection = torch.zeros(4, 4)
    projection[0, 0] = 2.0 * znear / (2.0 * right)
    projection[1, 1] = 2.0 * znear / (2.0 * top)
    projection[3, 2] = 1.0
    projection[2, 2] = zfar / (zfar - znear)
    projection[2, 3] = -(zfar * znear) / (zfar - znear)
    return projection


def scale_intrinsics(
    K: np.ndarray, source_size: tuple[float, float], target_size: tuple[float, float]
) -> np.ndarray:
    """Rescale pixel intrinsics from ``source_size`` to ``target_size`` (width, height)."""
    source_width, source_height = (float(value) for value in source_size)
    target_width, target_height = (float(value) for value in target_size)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("image dimensions must be positive")
    scaled = np.asarray(K, dtype=np.float64).copy()
    scaled[0, :] *= target_width / source_width
    scaled[1, :] *= target_height / source_height
    scaled[2, :] = (0.0, 0.0, 1.0)
    return scaled


def intrinsics_from_fov(width: int, height: int, fov_x: float, fov_y: float) -> np.ndarray:
    """Return ``K`` with a centred principal point for the given fields of view."""
    return np.array(
        [
            [width / (2.0 * math.tan(fov_x / 2.0)), 0.0, width / 2.0],
            [0.0, height / (2.0 * math.tan(fov_y / 2.0)), height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def center_crop_intrinsics(
    K: np.ndarray, source_size: tuple[float, float], target_size: tuple[float, float]
) -> np.ndarray:
    """Return ``K`` after a centred crop to the aspect of ``target_size``, then a resize to it."""
    source_width, source_height = (float(value) for value in source_size)
    target_width, target_height = (float(value) for value in target_size)
    target_aspect = target_width / target_height
    if source_width / source_height > target_aspect:
        crop_width, crop_height = source_height * target_aspect, source_height
    else:
        crop_width, crop_height = source_width, source_width / target_aspect
    cropped = np.asarray(K, dtype=np.float64).copy()
    cropped[0, 2] -= (source_width - crop_width) / 2.0
    cropped[1, 2] -= (source_height - crop_height) / 2.0
    return scale_intrinsics(cropped, (crop_width, crop_height), (target_width, target_height))
