"""Image loading and saving shared by every stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

JPEG_SUFFIXES = (".jpg", ".jpeg")


def load_rgb(path: Path) -> torch.Tensor:
    """Read an image as a ``(3, H, W)`` float32 tensor in ``[0, 1]``.

    An alpha channel, if any, is dropped rather than composited: generated
    frames are opaque by construction.
    """
    with Image.open(path) as image:
        array = np.array(image if image.mode in ("RGB", "RGBA") else image.convert("RGB"))
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div(255)
    return tensor[:3]


def read_rgb8(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Read ``(H, W, 3)`` uint8, Lanczos-resized to ``size`` = (width, height) if given."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if size is not None and rgb.size != size:
            rgb = rgb.resize(size, Image.Resampling.LANCZOS)
        return np.array(rgb)


def to_rgb8(image: torch.Tensor) -> np.ndarray:
    """Round a ``(3, H, W)`` tensor in ``[0, 1]`` to ``(H, W, 3)`` uint8."""
    return image.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()


def save_rgb(image: torch.Tensor, path: Path) -> None:
    """Write a ``(3, H, W)`` tensor in ``[0, 1]`` as an 8-bit image."""
    save_rgb8(to_rgb8(image), path)


def save_rgb8(image: np.ndarray, path: Path) -> None:
    """Write ``(H, W, 3)`` uint8; JPEG files use the highest quality and no chroma subsampling."""
    if Path(path).suffix.lower() in JPEG_SUFFIXES:
        Image.fromarray(image).save(path, quality=100, subsampling=0)
    else:
        Image.fromarray(image).save(path)


def center_crop_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Centre-crop ``(H, W, 3)`` uint8 to the aspect of ``size``, then Lanczos-resize to it.

    ``size`` is ``(width, height)``.
    """
    width, height = size
    target_aspect = width / height
    rgb = Image.fromarray(image)
    source_width, source_height = rgb.size
    if source_width / source_height >= target_aspect:
        crop_width = int(round(source_height * target_aspect))
        left = (source_width - crop_width) // 2
        rgb = rgb.crop((left, 0, left + crop_width, source_height))
    else:
        crop_height = int(round(source_width / target_aspect))
        top = (source_height - crop_height) // 2
        rgb = rgb.crop((0, top, source_width, top + crop_height))
    return np.array(rgb.resize((width, height), Image.Resampling.LANCZOS))
