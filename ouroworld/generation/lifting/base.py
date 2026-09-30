"""The ``DepthLifter`` interface: per-frame depth for a jointly reconstructed image set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class DepthMaps:
    """Depth predicted for a batch of images, at the images' own resolution.

    Attributes:
        depth: ``(N, H, W)`` float32 camera-z depth, up to an unknown affine map in disparity.
        confidence: ``(N, H, W)`` float32 per-pixel confidence; higher is better.
    """

    depth: np.ndarray
    confidence: np.ndarray


class DepthLifter(Protocol):
    """Predicts depth for images that are reconstructed together."""

    def lift(self, images: list[np.ndarray]) -> DepthMaps:
        """Return depth for ``images``, each ``(H, W, 3)`` uint8 of the same size."""
        ...
