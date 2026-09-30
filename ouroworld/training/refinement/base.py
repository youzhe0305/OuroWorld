"""Interface of the diffusion refiner used by scene-view consistent optimisation."""

from __future__ import annotations

from typing import Protocol

import torch


class DiffusionRefiner(Protocol):
    """Turns renders into cleaner images, guided by the generated observations."""

    def refine(self, renders: torch.Tensor, guides: torch.Tensor) -> torch.Tensor:
        """Refine ``(B, 3, H, W)`` renders in ``[0, 1]`` towards ``guides``; same shape out."""
        ...
