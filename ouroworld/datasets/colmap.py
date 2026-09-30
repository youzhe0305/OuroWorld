"""Minimal reader of COLMAP binary models (`cameras.bin`, `images.bin`)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COLMAP camera model id -> number of parameters.
_MODEL_PARAMETERS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
_PINHOLE = 1


@dataclass(frozen=True)
class ColmapCamera:
    """Intrinsics of one COLMAP camera (parameters in the model's own order)."""

    model_id: int
    width: int
    height: int
    params: np.ndarray

    @property
    def focal(self) -> tuple[float, float]:
        """``(fx, fy)``; single-focal models repeat ``f``."""
        if self.model_id == _PINHOLE:
            return float(self.params[0]), float(self.params[1])
        return float(self.params[0]), float(self.params[0])


@dataclass(frozen=True)
class ColmapImage:
    """Pose of one registered image: world-to-camera quaternion and translation."""

    name: str
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int = 1

    def w2c(self) -> np.ndarray:
        """OpenCV world-to-camera matrix."""
        w, x, y, z = self.qvec
        rotation = np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
            ]
        )
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = self.tvec
        return matrix


def read_cameras(path: Path) -> dict[int, ColmapCamera]:
    """Read ``cameras.bin``, keyed by camera id."""
    with Path(path).open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        cameras = {}
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", handle.read(24))
            n = _MODEL_PARAMETERS[model_id]
            params = np.array(struct.unpack("<" + "d" * n, handle.read(8 * n)))
            cameras[camera_id] = ColmapCamera(model_id, int(width), int(height), params)
    return cameras


def read_images(path: Path) -> dict[int, ColmapImage]:
    """Read ``images.bin``, keyed by image id; 2D observations are skipped."""
    with Path(path).open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        images = {}
        for _ in range(count):
            image_id = struct.unpack("<i", handle.read(4))[0]
            qvec = np.array(struct.unpack("<dddd", handle.read(32)))
            tvec = np.array(struct.unpack("<ddd", handle.read(24)))
            camera_id = struct.unpack("<i", handle.read(4))[0]
            name = b""
            while (char := handle.read(1)) != b"\x00":
                name += char
            points = struct.unpack("<Q", handle.read(8))[0]
            handle.read(24 * points)  # 2D observations are not needed
            images[image_id] = ColmapImage(name.decode("utf-8"), qvec, tvec, camera_id)
    return images
