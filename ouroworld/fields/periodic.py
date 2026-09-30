"""Periodic Deformation Field P(x, t) (paper §4.2).

P maps a canonical Gaussian to its state at loop phase ``t``. Geometry moves
by an SE(3) twist (optionally with a log-scale change); colour changes by a
residual on the SH coefficients, predicted by a separate appearance field.

Every output is the difference ``f(x, t) - f(x, 0)``, so P is the identity at
``t = 0`` by construction and the canonical Gaussians stay the static scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ouroworld.fields.encoding import Triplane, TriplaneSpec, contract_time, fourier_time_basis
from ouroworld.fields.heads import ChunkedEvaluator, input_layer, output_head

TWIST_WIDTH = 6
SCALE_WIDTH = 3
PERIODIC_HEADS = ("pos", "rot", "scale", "shs")


@dataclass(frozen=True)
class PeriodicFieldSpec:
    """Architecture of P.

    Attributes:
        heads: Subset of ``pos, rot, scale, shs``. ``pos``/``rot`` are the two
            halves of one SE(3) twist and are always built together.
        harmonics: Number of Fourier harmonics ``K``.
        period: Period of the time basis; 1 for the model, 2 for the ablation.
        motion_grid: Triplane of the geometry branch.
        appearance_grid: Triplane of the SH branch.
        width: Hidden width of both MLPs.
        max_log_scale_delta: Bound of the log-scale change, applied with ``tanh``.
        sh_degree: SH degree of the Gaussians.
    """

    heads: tuple[str, ...]
    harmonics: int
    period: float
    motion_grid: TriplaneSpec
    appearance_grid: TriplaneSpec
    width: int
    max_log_scale_delta: float
    sh_degree: int

    def __post_init__(self) -> None:
        unknown = set(self.heads) - set(PERIODIC_HEADS)
        if unknown:
            raise ValueError(f"unknown P heads {sorted(unknown)}; valid: {list(PERIODIC_HEADS)}")
        if not {"pos", "rot"} <= set(self.heads):
            raise ValueError("P needs both `pos` and `rot`: they form one SE(3) twist")
        if self.harmonics < 1:
            raise ValueError("P needs at least one harmonic")
        if self.max_log_scale_delta <= 0:
            raise ValueError("max_log_scale_delta must be positive")

    @property
    def basis_size(self) -> int:
        """Number of time-basis functions, ``1 + 2K``."""
        return 1 + 2 * self.harmonics

    @property
    def sh_coefficients(self) -> int:
        """Number of SH coefficients per colour channel."""
        return (self.sh_degree + 1) ** 2


@dataclass(frozen=True)
class PeriodicOutput:
    """P evaluated at one time, all zero at ``t = 0``.

    Attributes:
        twist: ``(N, 6)`` SE(3) twist ``[v, omega]``.
        log_scale_delta: ``(N, 3)`` additive log-scale change, or ``None``.
        sh_delta: ``(N, C, 3)`` SH residual, or ``None``.
    """

    twist: torch.Tensor
    log_scale_delta: torch.Tensor | None
    sh_delta: torch.Tensor | None


class PeriodicDeformationField(nn.Module):
    """P(x, t): periodic geometry and appearance change of canonical Gaussians."""

    def __init__(self, spec: PeriodicFieldSpec, evaluator: ChunkedEvaluator):
        super().__init__()
        self.spec = spec
        self.evaluator = evaluator
        self.has_scale = "scale" in spec.heads
        self.has_appearance = "shs" in spec.heads

        self.motion_grid = Triplane(spec.motion_grid, spec.basis_size)
        self.motion_input = input_layer(spec.motion_grid.feature_dim, spec.width)
        self.twist_head = output_head(spec.width, TWIST_WIDTH)
        if self.has_scale:
            self.scale_head = output_head(spec.width, SCALE_WIDTH)
        if self.has_appearance:
            self.appearance_grid = Triplane(spec.appearance_grid, spec.basis_size)
            self.appearance_input = input_layer(spec.appearance_grid.feature_dim, spec.width)
            self.sh_head = output_head(spec.width, spec.sh_coefficients * 3)

    def set_bounds(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        """Fit every triplane to the box ``[lower, upper]``."""
        self.motion_grid.set_bounds(lower, upper)
        if self.has_appearance:
            self.appearance_grid.set_bounds(lower, upper)

    def forward(self, points: torch.Tensor, time: torch.Tensor) -> PeriodicOutput:
        """Evaluate P at ``points`` ``(N, 3)`` and per-point times ``(N, 1)``."""
        geometry = self.evaluator(
            self._geometry_chunk, (points, time), self._geometry_width, self.training
        )
        twist = geometry[:, :TWIST_WIDTH]
        log_scale_delta = None
        if self.has_scale:
            raw = geometry[:, TWIST_WIDTH:]
            log_scale_delta = self.spec.max_log_scale_delta * torch.tanh(raw)
        return PeriodicOutput(twist, log_scale_delta, self.sh_delta(points, time))

    def twist(self, points: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Only the ``(N, 6)`` SE(3) twist."""
        return self.forward(points, time).twist

    def sh_delta(self, points: torch.Tensor, time: torch.Tensor) -> torch.Tensor | None:
        """The ``(N, C, 3)`` SH residual, or ``None`` without an appearance branch."""
        if not self.has_appearance:
            return None
        width = self.spec.sh_coefficients * 3
        flat = self.evaluator(self._appearance_chunk, (points, time), width, self.training)
        return flat.reshape(points.shape[0], self.spec.sh_coefficients, 3)

    @property
    def _geometry_width(self) -> int:
        return TWIST_WIDTH + (SCALE_WIDTH if self.has_scale else 0)

    def _basis_pair(self, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        basis = fourier_time_basis(time, self.spec.harmonics, self.spec.period)
        at_zero = fourier_time_basis(torch.zeros_like(time), self.spec.harmonics, self.spec.period)
        return basis, at_zero

    def _geometry_chunk(self, points: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        coefficients = self.motion_grid.coefficients(points)
        basis, at_zero = self._basis_pair(time)
        return self._geometry_heads(contract_time(coefficients, basis)) - self._geometry_heads(
            contract_time(coefficients, at_zero)
        )

    def _geometry_heads(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.motion_input(features)
        if not self.has_scale:
            return self.twist_head(hidden)
        return torch.cat((self.twist_head(hidden), self.scale_head(hidden)), dim=-1)

    def _appearance_chunk(self, points: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        coefficients = self.appearance_grid.coefficients(points)
        basis, at_zero = self._basis_pair(time)
        current = self.sh_head(self.appearance_input(contract_time(coefficients, basis)))
        initial = self.sh_head(self.appearance_input(contract_time(coefficients, at_zero)))
        return current - initial
