"""Renders of the static input 3DGS, with opacity-normalised depth.

They are the reference image, the depth the pivot is picked on, and the
``t = 0`` frame of every training view: at ``t = 0`` the scene is exactly the
input 3DGS (paper §4.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ouroworld.gaussians.canonical import CanonicalGaussians
from ouroworld.geometry.camera import Camera
from ouroworld.io.ply import read_gaussian_ply
from ouroworld.render.rasterizer import RasterView, render

# Pixels whose accumulated opacity is below this have no reliable depth.
MIN_DEPTH_ALPHA = 0.5


@dataclass(frozen=True)
class StaticView:
    """One render of the input 3DGS.

    Attributes:
        image: ``(H, W, 3)`` uint8 colour.
        depth: ``(H, W)`` float32 camera-z depth, 0 where the opacity is below 0.5.
    """

    image: np.ndarray
    depth: np.ndarray


class StaticRenderer:
    """Renders the input 3DGS from arbitrary cameras."""

    def __init__(self, gaussians: CanonicalGaussians, background: torch.Tensor) -> None:
        """Keep ``gaussians`` (on the render device) and the ``(3,)`` background colour."""
        self.gaussians = gaussians
        self.background = background
        self._ones = torch.ones_like(gaussians.xyz)
        self._black = torch.zeros_like(background)

    @torch.no_grad()
    def image(self, camera: Camera) -> np.ndarray:
        """Return the ``(H, W, 3)`` uint8 render of ``camera``."""
        view = RasterView.from_camera(camera, self.background.device)
        output = render(
            self.gaussians.attributes(), view, self.background, self.gaussians.sh_degree
        )
        return quantize(output.image)

    @torch.no_grad()
    def view(self, camera: Camera) -> StaticView:
        """Return the render of ``camera`` with its opacity-normalised depth."""
        view = RasterView.from_camera(camera, self.background.device)
        attributes = self.gaussians.attributes()
        colour = render(attributes, view, self.background, self.gaussians.sh_degree)
        # Ones on black accumulate to the opacity: the weight that normalises the depth.
        opacity = render(
            attributes, view, self._black, self.gaussians.sh_degree, colors=self._ones
        ).image[0]
        depth = torch.where(
            opacity > 1e-8, colour.depth[0] / opacity.clamp_min(1e-8), torch.zeros_like(opacity)
        )
        depth = torch.where(opacity >= MIN_DEPTH_ALPHA, depth, torch.zeros_like(depth))
        return StaticView(quantize(colour.image), depth.float().cpu().numpy())


def load_static_renderer(
    gaussians_path: Path, background: tuple[float, float, float], device: str = "cuda"
) -> StaticRenderer:
    """Load the input 3DGS at its own SH degree onto ``device``."""
    arrays = read_gaussian_ply(gaussians_path)
    gaussians = CanonicalGaussians(arrays, arrays.sh_degree).to(device)
    return StaticRenderer(gaussians, torch.tensor(background, dtype=torch.float32, device=device))


def quantize(image: torch.Tensor) -> np.ndarray:
    """Convert a ``(3, H, W)`` render in ``[0, 1]`` to ``(H, W, 3)`` uint8.

    Values are truncated, not rounded: the paper's data was produced this way.
    """
    return (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
