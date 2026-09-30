"""JSON (de)serialisation of :class:`~ouroworld.geometry.camera.Camera`.

A camera record is ``{"width", "height", "K", "c2w"}`` with ``K`` in pixels and
``c2w`` an OpenCV camera-to-world matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ouroworld.geometry.camera import Camera
from ouroworld.io.errors import ArtifactError


def camera_to_record(camera: Camera) -> dict[str, Any]:
    """Return a JSON-serialisable record of ``camera``."""
    return {
        "width": int(camera.width),
        "height": int(camera.height),
        "K": camera.K.tolist(),
        "c2w": camera.c2w.tolist(),
    }


def camera_from_record(record: dict[str, Any], where: str = "camera") -> Camera:
    """Build a camera from a record written by :func:`camera_to_record`."""
    missing = {"width", "height", "K", "c2w"} - record.keys()
    if missing:
        raise ArtifactError(f"{where}: missing keys {sorted(missing)}")
    try:
        return Camera(
            width=int(record["width"]),
            height=int(record["height"]),
            K=np.asarray(record["K"], dtype=np.float64),
            c2w=np.asarray(record["c2w"], dtype=np.float64),
        )
    except ValueError as error:
        raise ArtifactError(f"{where}: {error}") from error


def load_camera(path: Path) -> Camera:
    """Read a single camera JSON file."""
    with Path(path).open(encoding="utf-8") as handle:
        return camera_from_record(json.load(handle), where=str(path))


def save_camera(camera: Camera, path: Path) -> None:
    """Write a single camera JSON file."""
    Path(path).write_text(json.dumps(camera_to_record(camera), indent=2), encoding="utf-8")
