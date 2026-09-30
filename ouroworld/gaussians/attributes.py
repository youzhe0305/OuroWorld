"""Per-Gaussian attributes exchanged between the deformation fields and the renderer."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GaussianAttributes:
    """Gaussian attributes in their optimisation (pre-activation) space.

    Attributes:
        xyz: ``(N, 3)`` centres.
        log_scale: ``(N, 3)`` log standard deviations; the renderer applies ``exp``.
        rotation: ``(N, 4)`` scalar-first quaternions; the renderer normalises them.
        opacity_logit: ``(N, 1)`` opacities before the sigmoid.
        sh: ``(N, (degree + 1)^2, 3)`` spherical-harmonic colour coefficients.
    """

    xyz: torch.Tensor
    log_scale: torch.Tensor
    rotation: torch.Tensor
    opacity_logit: torch.Tensor
    sh: torch.Tensor

    def replace(self, **changes: torch.Tensor) -> GaussianAttributes:
        """Return a copy with some attributes replaced."""
        return dataclasses.replace(self, **changes)

    def detach(self) -> GaussianAttributes:
        """Return a copy cut from the autograd graph."""
        # dataclasses.astuple would deep-copy the tensors, so iterate fields directly.
        return GaussianAttributes(
            *(getattr(self, field.name).detach() for field in dataclasses.fields(self))
        )

    def __len__(self) -> int:
        return int(self.xyz.shape[0])
