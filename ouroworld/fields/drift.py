"""Grounded Drift Field Δ_v(x, t) (paper §4.2, Eq. 2).

Δ absorbs how each generated view disagrees with the others. It is
conditioned on a learned per-view embedding and is used only while training:
novel-view inference renders the periodic field alone.

Grounding: Δ is exactly zero on the reference view (whose frames come from
the reference video) and at ``t = 0`` (where every view shows the static
scene). This module is the only place that rule is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ouroworld.fields.encoding import Triplane, TriplaneSpec, contract_time, fourier_time_basis
from ouroworld.fields.heads import ChunkedEvaluator, input_layer, output_head

DRIFT_HEADS = ("pos", "rot", "scale", "opacity", "shs")


@dataclass(frozen=True)
class DriftFieldSpec:
    """Architecture and grounding of Δ.

    Attributes:
        heads: Subset of ``pos, rot, scale, opacity, shs``.
        view_count: Number of training views.
        reference_view_id: The view holding the reference video.
        harmonics: Number of Fourier harmonics of the time basis.
        period: Period of the time basis. Drift accumulates along the loop
            rather than repeating, so the paper uses an aperiodic (period-2) basis.
        grid: Triplane of Δ.
        width: Hidden width.
        view_embedding_dim: Width of the per-view embedding.
        zero_reference_view: Ground Δ on the reference view.
        zero_t0: Ground Δ at ``t = 0``.
        sh_degree: SH degree of the Gaussians.
    """

    view_count: int
    reference_view_id: int
    heads: tuple[str, ...]
    harmonics: int
    period: float
    grid: TriplaneSpec
    width: int
    view_embedding_dim: int
    zero_reference_view: bool
    zero_t0: bool
    sh_degree: int

    def __post_init__(self) -> None:
        unknown = set(self.heads) - set(DRIFT_HEADS)
        if unknown or not self.heads:
            raise ValueError(f"Δ heads must be a non-empty subset of {list(DRIFT_HEADS)}")
        if not 0 <= self.reference_view_id < self.view_count:
            raise ValueError(
                f"reference view {self.reference_view_id} outside {self.view_count} views"
            )

    @property
    def is_fully_grounded(self) -> bool:
        """Whether both grounding constraints are on (the paper setting)."""
        return self.zero_reference_view and self.zero_t0

    def head_widths(self) -> dict[str, int]:
        """Output width of each built head, in evaluation order."""
        widths = {}
        if {"pos", "rot"} & set(self.heads):
            widths["twist"] = 6
        optional = {"scale": 3, "opacity": 1, "shs": 3 * (self.sh_degree + 1) ** 2}
        widths.update({name: width for name, width in optional.items() if name in self.heads})
        return widths


class GroundedDriftField(nn.Module):
    """Δ_v(x, t): per-view residual deformation used only during training."""

    def __init__(self, spec: DriftFieldSpec, evaluator: ChunkedEvaluator):
        super().__init__()
        self.spec = spec
        self.evaluator = evaluator
        self.grid = Triplane(spec.grid, 1 + 2 * spec.harmonics)
        self.view_embedding = nn.Embedding(spec.view_count, spec.view_embedding_dim)
        nn.init.normal_(self.view_embedding.weight, mean=0.0, std=0.01)
        self.input = input_layer(spec.grid.feature_dim + spec.view_embedding_dim, spec.width)
        self.heads = nn.ModuleDict(
            {name: output_head(spec.width, width) for name, width in spec.head_widths().items()}
        )
        self._out_width = sum(spec.head_widths().values())

    def set_bounds(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        """Fit the triplane to the box ``[lower, upper]``."""
        self.grid.set_bounds(lower, upper)

    def is_grounded(self, view_id: int, time: float) -> bool:
        """Whether Δ is identically zero for this observation (Eq. 2)."""
        on_reference = self.spec.zero_reference_view and view_id == self.spec.reference_view_id
        at_start = self.spec.zero_t0 and time == 0.0
        return on_reference or at_start

    def forward(
        self, points: torch.Tensor, time: torch.Tensor, view_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Evaluate Δ per point, with grounded rows set to zero.

        Args:
            points: ``(N, 3)`` canonical positions.
            time: ``(N, 1)`` loop phase.
            view_ids: ``(N,)`` view of each row.

        Returns:
            ``{head: (N, width)}`` for ``pos``, ``rot`` and the other declared
            heads; ``shs`` is returned flat, ``(N, 3 * C)``.
        """
        raw = self.evaluator(self._chunk, (points, time, view_ids), self._out_width, self.training)
        outputs = self._split(raw)
        grounded = self._grounded_rows(view_ids, time).unsqueeze(-1)
        return {name: value.masked_fill(grounded, 0.0) for name, value in outputs.items()}

    def _chunk(
        self, points: torch.Tensor, time: torch.Tensor, view_ids: torch.Tensor
    ) -> torch.Tensor:
        basis = fourier_time_basis(time, self.spec.harmonics, self.spec.period)
        features = contract_time(self.grid.coefficients(points), basis)
        hidden = self.input(torch.cat((features, self.view_embedding(view_ids)), dim=-1))
        parts = [head(hidden) for head in self.heads.values()]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _split(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        parts: dict[str, torch.Tensor] = {}
        cursor = 0
        for name, width in self.spec.head_widths().items():
            parts[name] = raw[:, cursor : cursor + width]
            cursor += width
        twist = parts.pop("twist", None)
        if twist is not None:
            # `pos` and `rot` share one twist head; keep only the declared halves.
            if "pos" in self.spec.heads:
                parts["pos"] = twist[:, :3]
            if "rot" in self.spec.heads:
                parts["rot"] = twist[:, 3:]
        return parts

    def _grounded_rows(self, view_ids: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        grounded = torch.zeros_like(view_ids, dtype=torch.bool)
        if self.spec.zero_reference_view:
            grounded |= view_ids == self.spec.reference_view_id
        if self.spec.zero_t0:
            grounded |= time[:, 0] == 0.0
        return grounded
