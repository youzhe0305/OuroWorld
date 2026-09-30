"""The perceptual refinement loss of scene-view consistent optimisation (Eq. 5)."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ouroworld.training.refinement.base import DiffusionRefiner


class RefinementObjective:
    """``λ_p · LPIPS(render, refine(render, generated observation))``.

    The refiner and LPIPS network are built on first use, so their memory is
    only taken once refinement starts.
    """

    def __init__(
        self, weight: float, build_refiner: Callable[[], DiffusionRefiner], device: torch.device
    ):
        self.weight = weight
        self._build_refiner = build_refiner
        self.device = device
        self._refiner: DiffusionRefiner | None = None
        self._lpips: torch.nn.Module | None = None

    def __call__(self, render: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Loss for one ``(3, H, W)`` render and its generated observation."""
        if self._refiner is None:
            self._setup()
        assert self._refiner is not None and self._lpips is not None
        renders = render.unsqueeze(0)
        target = self._refiner.refine(renders.detach(), guide.unsqueeze(0))
        # Images in [0, 1] are fed to LPIPS without rescaling to [-1, 1],
        # exactly as in the paper runs.
        return self.weight * self._lpips(renders, target).mean()

    def _setup(self) -> None:
        import lpips

        self._refiner = self._build_refiner()
        self._lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device)
