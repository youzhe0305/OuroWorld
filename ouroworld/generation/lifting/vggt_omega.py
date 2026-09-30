"""VGGT-Omega depth for the static views and the reference video together.

One forward pass over the ``t = 0`` renders of every view and the reference
video frames puts all depth maps in one frame of reference; the renders tie it
to the input 3DGS (see :mod:`ouroworld.generation.lifting.alignment`).
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ouroworld.adapters.vggt_omega import load_vggt_omega
from ouroworld.generation.lifting.base import DepthMaps

logger = logging.getLogger(__name__)

PATCH_SIZE = 16


class VggtOmegaLifter:
    """:class:`DepthLifter` backed by VGGT-Omega."""

    def __init__(self, checkpoint: Path, image_resolution: int) -> None:
        """Use ``checkpoint``; the longer image side is resized to ``image_resolution``."""
        self.checkpoint = Path(checkpoint)
        self.image_resolution = int(image_resolution)

    def lift(self, images: list[np.ndarray]) -> DepthMaps:
        """Return depth and confidence resized back to the input resolution."""
        height, width = images[0].shape[:2]
        if any(image.shape != images[0].shape for image in images):
            raise ValueError("VGGT-Omega inputs must share one resolution")
        batch = torch.stack([self.preprocess(image) for image in images]).cuda()
        logger.info("VGGT-Omega on %d images at %s", len(images), tuple(batch.shape[-2:]))
        model = load_vggt_omega(self.checkpoint)
        with torch.inference_mode():
            predictions = model(batch)
        depth = predictions["depth"][0, ..., 0].float().cpu()
        confidence = predictions["depth_conf"][0].float().cpu()
        del model, predictions, batch
        gc.collect()
        torch.cuda.empty_cache()
        return DepthMaps(
            depth=_resize_maps(depth, height, width),
            confidence=_resize_maps(confidence, height, width),
        )

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Resize so the longer side is ``image_resolution`` and both sides are patch multiples.

        Images are not cropped: every supported aspect ratio lies in ``[1/2, 2]``.
        """
        height, width = image.shape[:2]
        aspect = height / width
        if not 0.5 <= aspect <= 2.0:
            raise ValueError(f"aspect ratio {width}x{height} is outside [1:2, 2:1]")
        if aspect < 1.0:
            target = (self.image_resolution, _patch_multiple(self.image_resolution * aspect))
        else:
            target = (_patch_multiple(self.image_resolution / aspect), self.image_resolution)
        resized = Image.fromarray(image).resize(target, Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float().div(255.0)


def _patch_multiple(length: float) -> int:
    return max(PATCH_SIZE, round(length / PATCH_SIZE) * PATCH_SIZE)


def _resize_maps(maps: torch.Tensor, height: int, width: int) -> np.ndarray:
    resized = F.interpolate(maps[:, None], (height, width), mode="bilinear", align_corners=False)
    return resized[:, 0].numpy()
