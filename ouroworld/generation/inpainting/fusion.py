"""Put reliable warped pixels back over the video model's output.

With ``W`` the warp, ``G`` the generated video and ``M`` the known mask::

    E = erode(1[M >= known_threshold], erode_pixels)
    A = max_warp_alpha * 1[M >= known_threshold] * GaussianBlur(E, feather_sigma, feather_pixels)
    output = A * W + (1 - A) * G

``A`` is non-zero only where the warp is known, so holes keep the generated
content, and it fades out towards hole borders, where the warp is least
trustworthy (occlusion edges, depth errors).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FusionSettings:
    """Feathered warp-over-generation blend.

    Attributes:
        known_threshold: Known-mask values at or above this count as known.
        max_warp_alpha: Weight of the warp deep inside known regions.
        erode_pixels: Known regions are shrunk by this many pixels first.
        feather_pixels: Radius of the Gaussian feather kernel.
        feather_sigma: Standard deviation of the feather kernel, in pixels.
    """

    known_threshold: float
    max_warp_alpha: float
    erode_pixels: int
    feather_pixels: int
    feather_sigma: float


def fuse(
    generated: torch.Tensor, warped: torch.Tensor, known: torch.Tensor, settings: FusionSettings
) -> torch.Tensor:
    """Blend ``warped`` over ``generated``, both ``(T, 3, H, W)``; ``known`` is ``(T, 1, H, W)``."""
    if generated.shape != warped.shape or known.shape != (
        *generated.shape[:1],
        1,
        *generated.shape[2:],
    ):
        raise ValueError(
            f"shape mismatch: generated {tuple(generated.shape)}, warped {tuple(warped.shape)}, "
            f"known {tuple(known.shape)}"
        )
    is_known = (known >= settings.known_threshold).to(generated.dtype)
    alpha = is_known
    if settings.erode_pixels:
        alpha = 1.0 - F.max_pool2d(
            1.0 - is_known,
            kernel_size=2 * settings.erode_pixels + 1,
            stride=1,
            padding=settings.erode_pixels,
        )
    if settings.feather_pixels:
        radius = settings.feather_pixels
        kernel = gaussian_kernel(radius, settings.feather_sigma, generated.dtype, generated.device)
        alpha = F.conv2d(F.pad(alpha, (radius, radius, radius, radius), mode="replicate"), kernel)
    alpha = alpha.clamp(0, 1) * is_known * settings.max_warp_alpha
    return alpha * warped + (1.0 - alpha) * generated


def gaussian_kernel(
    radius: int, sigma: float, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return the normalised ``(1, 1, 2r+1, 2r+1)`` Gaussian kernel."""
    offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(offsets.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :])[None, None]
