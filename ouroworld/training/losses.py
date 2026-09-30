"""Loss terms of the 4D optimisation (paper §4.2 and App. B).

Every regulariser evaluates the fields on a random subset of canonical
Gaussians at the time of the current observation, and carries its own weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ouroworld.fields.cinemagraph import CinemagraphModel
from ouroworld.fields.drift import DRIFT_HEADS
from ouroworld.fields.encoding import Triplane
from ouroworld.geometry.se3 import se3_exp_apply


class Regularizer(Protocol):
    """A weighted loss evaluated at one loop phase."""

    name: str

    def __call__(self, time: float) -> torch.Tensor | None:
        """Return the weighted loss, or ``None`` when the term is inactive."""
        ...


def photometric_l1(renders: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between ``(B, 3, H, W)`` renders and targets."""
    return torch.abs(renders - targets).mean()


def sample_indices(count: int, sample_size: int, device: torch.device) -> torch.Tensor:
    """Random subset of ``range(count)``; ``sample_size <= 0`` keeps everything."""
    size = count if sample_size <= 0 else min(sample_size, count)
    return torch.randperm(count, device=device)[:size]


def second_difference(
    before: torch.Tensor, center: torch.Tensor, after: torch.Tensor
) -> torch.Tensor:
    """Central second difference ``f(t + d) - 2 f(t) + f(t - d)``."""
    return after - 2.0 * center + before


@dataclass
class AppearanceResidualLoss:
    """Keep the SH residual of P small and smooth in time."""

    model: CinemagraphModel
    magnitude_weight: float
    smoothness_weight: float
    delta: float
    sample_size: int
    name: str = "appearance_residual"

    def __call__(self, time: float) -> torch.Tensor | None:
        """Weighted ``mean|dSH(t)| + mean|second difference of dSH|``."""
        points = _sampled_points(self.model, self.sample_size)
        times = points.new_full((points.shape[0], 1), time)
        center = self.model.periodic.sh_delta(points, times)
        assert center is not None
        loss = self.magnitude_weight * center.abs().mean()
        if self.smoothness_weight > 0:
            before = self.model.periodic.sh_delta(points, times - self.delta)
            after = self.model.periodic.sh_delta(points, times + self.delta)
            assert before is not None and after is not None
            smoothness = second_difference(before, center, after).abs().mean()
            loss = loss + self.smoothness_weight * smoothness
        return loss


@dataclass
class ScaleChangeLoss:
    """Make scale changes of P costly; inflating Gaussians is a cheap way to hide error."""

    model: CinemagraphModel
    weight: float
    sample_size: int
    name: str = "scale_change"

    def __call__(self, time: float) -> torch.Tensor | None:
        """Weighted mean absolute log-scale change."""
        points = _sampled_points(self.model, self.sample_size)
        times = points.new_full((points.shape[0], 1), time)
        delta = self.model.periodic(points, times).log_scale_delta
        assert delta is not None
        return self.weight * delta.abs().mean()


@dataclass
class MotionSmoothnessLoss:
    """Penalise the temporal second difference of the Gaussians deformed by P."""

    model: CinemagraphModel
    weight: float
    delta: float
    sample_size: int
    position_weight: float = 1.0
    rotation_weight: float = 1.0
    scale_weight: float = 0.5
    name: str = "motion_smoothness"

    def __call__(self, time: float) -> torch.Tensor | None:
        """Weighted squared second difference of position, rotation and log-scale."""
        canonical = self.model.canonical
        indices = sample_indices(len(canonical), self.sample_size, canonical.xyz.device)
        points = canonical.xyz[indices]
        rotations = canonical.rotation[indices]
        log_scales = canonical.log_scale[indices]

        def deform(at: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            times = points.new_full((points.shape[0], 1), at)
            output = self.model.periodic(points, times)
            moved, turned = se3_exp_apply(points, rotations, output.twist)
            scaled = (
                log_scales
                if output.log_scale_delta is None
                else log_scales + output.log_scale_delta
            )
            return moved, turned, scaled

        before, center, after = deform(time - self.delta), deform(time), deform(time + self.delta)
        weights = (self.position_weight, self.rotation_weight, self.scale_weight)
        terms = [
            weight * second_difference(b, c, a).pow(2).mean()
            for weight, b, c, a in zip(weights, before, center, after, strict=True)
            if weight > 0
        ]
        return self.weight * torch.stack(terms).sum() if terms else None


@dataclass
class TwistSmoothnessLoss:
    """Penalise the temporal second difference of P's SE(3) twist itself."""

    model: CinemagraphModel
    weight: float
    delta: float
    sample_size: int
    name: str = "twist_smoothness"

    def __call__(self, time: float) -> torch.Tensor | None:
        """Weighted squared second difference of the twist."""
        points = _sampled_points(self.model, self.sample_size)
        times = points.new_full((points.shape[0], 1), time)
        before = self.model.periodic.twist(points, times - self.delta)
        center = self.model.periodic.twist(points, times)
        after = self.model.periodic.twist(points, times + self.delta)
        return self.weight * second_difference(before, center, after).pow(2).mean()


@dataclass(frozen=True)
class DriftChannelWeights:
    """Relative weight of each Δ output channel in the magnitude penalty.

    Translation is divided by the scene radius so the penalty is scale-free.
    """

    translation: float = 1.0
    rotation: float = 1.0
    scale: float = 1.0
    opacity: float = 1.0
    sh: float = 1.0


class DriftMagnitudeLoss:
    """Keep Δ small so P explains everything the views agree on.

    The penalty averages over all non-grounded views, weighted by whether each
    sampled Gaussian is visible in that view's canonical render.
    """

    name = "drift_magnitude"

    def __init__(
        self,
        model: CinemagraphModel,
        weight: float,
        channel_weights: DriftChannelWeights,
        scene_radius: float,
        sample_size: int,
        view_chunk: int,
    ):
        if model.drift is None:
            raise ValueError("DriftMagnitudeLoss needs a drift field")
        self.model = model
        self.weight = weight
        self.channel_weights = channel_weights
        self.scene_radius = max(float(scene_radius), 1e-6)
        self.sample_size = sample_size
        self.view_chunk = max(1, view_chunk)
        self.visibility: torch.Tensor | None = None

    def set_visibility(self, visibility: torch.Tensor) -> None:
        """Store the ``(N, V)`` canonical visibility table (CPU, uint8)."""
        self.visibility = visibility

    def __call__(self, time: float) -> torch.Tensor | None:
        """Weighted, visibility-averaged squared Δ over the non-grounded views."""
        drift = self.model.drift
        assert drift is not None
        views = [view for view in range(drift.spec.view_count) if not drift.is_grounded(view, time)]
        if not views:
            return None
        canonical = self.model.canonical
        indices = sample_indices(len(canonical), self.sample_size, canonical.xyz.device)
        points = canonical.xyz[indices]
        outputs = torch.cat(
            [self._weighted_outputs(points, time, chunk) for chunk in self._chunks(views)], 1
        )
        weights = self._visibility_weights(indices, views, points)
        magnitude = ((outputs**2).sum(dim=-1) * weights).sum() / weights.sum().clamp_min(1.0)
        return self.weight * magnitude

    def _chunks(self, views: list[int]) -> list[list[int]]:
        return [
            views[start : start + self.view_chunk]
            for start in range(0, len(views), self.view_chunk)
        ]

    def _weighted_outputs(
        self, points: torch.Tensor, time: float, views: list[int]
    ) -> torch.Tensor:
        """Δ for every (point, view) pair, ``(S, len(views), channels)``, channel-weighted."""
        drift = self.model.drift
        assert drift is not None
        count, view_count = points.shape[0], len(views)
        view_ids = torch.tensor(views, device=points.device).repeat(count)
        repeated = points.repeat_interleave(view_count, dim=0)
        times = points.new_full((count * view_count, 1), time)
        outputs = drift(repeated, times, view_ids)
        heads = [name for name in DRIFT_HEADS if name in outputs]
        weights = torch.cat(
            [points.new_full((outputs[name].shape[-1],), self._head_weight(name)) for name in heads]
        )
        flat = torch.cat([outputs[name] for name in heads], dim=-1) * weights
        return flat.reshape(count, view_count, -1)

    def _head_weight(self, head: str) -> float:
        weights = self.channel_weights
        return {
            "pos": weights.translation / self.scene_radius,
            "rot": weights.rotation,
            "scale": weights.scale,
            "opacity": weights.opacity,
            "shs": weights.sh,
        }[head]

    def _visibility_weights(
        self, indices: torch.Tensor, views: list[int], points: torch.Tensor
    ) -> torch.Tensor:
        table = self.visibility
        # Densification changes the Gaussian count; a stale table is not used.
        if table is None or table.shape[0] != len(self.model.canonical):
            return points.new_ones((points.shape[0], len(views)))
        rows = table[indices.cpu()][:, views]
        return rows.to(device=points.device, dtype=points.dtype)


@dataclass
class PlaneSmoothnessLoss:
    """Second-difference smoothness of the motion triplane (K-Planes style)."""

    grid: Triplane
    weight: float
    name: str = "plane_smoothness"

    def __call__(self, time: float) -> torch.Tensor | None:
        """Weighted sum over planes of the mean squared second difference along rows."""
        del time  # the planes carry no time axis
        total = sum(
            (plane[..., 2:, :] - 2 * plane[..., 1:-1, :] + plane[..., :-2, :]).pow(2).mean()
            for plane in self.grid.planes()
        )
        return self.weight * total


def _sampled_points(model: CinemagraphModel, sample_size: int) -> torch.Tensor:
    xyz = model.canonical.xyz
    return xyz[sample_indices(xyz.shape[0], sample_size, xyz.device)]
