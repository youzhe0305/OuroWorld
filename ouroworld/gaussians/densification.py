"""Adaptive density control for the canonical Gaussians during 4D optimisation.

Only the pieces the paper runs use are kept: gradient-driven clone/split, a
capped "health" prune of transparent or non-finite Gaussians, and removal of
Gaussians whose rasterizer gradients were repeatedly non-finite.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ouroworld.gaussians.canonical import PARAMETER_NAMES, CanonicalGaussians


@dataclass(frozen=True)
class DensificationSettings:
    """Adaptive density control thresholds.

    Attributes:
        start_iteration: First iteration (exclusive) at which to densify and prune.
        stop_iteration: Densification stops at this iteration (exclusive).
        interval: Densify and prune every this many iterations.
        gradient_threshold: Mean screen-space gradient norm that triggers densification.
        percent_dense: Split Gaussians larger than this fraction of the scene extent,
            clone smaller ones.
        min_opacity: Prune Gaussians below this opacity.
        max_prune_fraction: Cap on the fraction removed by one health prune.
    """

    start_iteration: int
    stop_iteration: int
    interval: int
    gradient_threshold: float
    percent_dense: float
    min_opacity: float
    max_prune_fraction: float

    def is_active(self, iteration: int) -> bool:
        """Whether statistics are collected at ``iteration``."""
        return iteration < self.stop_iteration

    def is_due(self, iteration: int) -> bool:
        """Whether densification and pruning run at ``iteration``."""
        return (
            self.is_active(iteration)
            and iteration > self.start_iteration
            and iteration % self.interval == 0
        )


class ParameterSurgery:
    """Keep the optimiser state aligned with the Gaussians when rows change."""

    def __init__(self, gaussians: CanonicalGaussians, optimizer: torch.optim.Optimizer):
        self.gaussians = gaussians
        self.optimizer = optimizer

    def keep_rows(self, keep: torch.Tensor) -> None:
        """Drop every row where ``keep`` is false."""
        self._rewrite(lambda tensor, _name: tensor[keep], lambda state, _name: state[keep])

    def append_rows(self, rows: dict[str, torch.Tensor]) -> None:
        """Append new rows; their Adam moments start at zero."""
        self._rewrite(
            lambda tensor, name: torch.cat((tensor, rows[name]), dim=0),
            lambda state, name: torch.cat((state, torch.zeros_like(rows[name])), dim=0),
        )

    def _rewrite(self, edit_parameter, edit_state) -> None:  # noqa: ANN001
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name not in PARAMETER_NAMES:
                continue
            old = group["params"][0]
            state = self.optimizer.state.pop(old, None)
            new = self.gaussians.set_parameter(name, edit_parameter(old.detach(), name))
            group["params"][0] = new
            if state is not None:
                state["exp_avg"] = edit_state(state["exp_avg"], name)
                state["exp_avg_sq"] = edit_state(state["exp_avg_sq"], name)
                self.optimizer.state[new] = state


class DensificationState:
    """Per-Gaussian statistics accumulated between densification steps.

    ``strikes`` counts consecutive iterations in which a Gaussian's rasterizer
    gradient was non-finite; it survives :meth:`reset`.
    """

    def __init__(self, count: int, device: torch.device):
        self.gradient_sum = torch.zeros((count, 1), device=device)
        self.visible_count = torch.zeros((count, 1), device=device)
        self.strikes = torch.zeros(count, dtype=torch.int16, device=device)

    def record_bad_rows(self, bad: torch.Tensor, strike_limit: int) -> torch.Tensor:
        """Update strike counters; return the rows that reached ``strike_limit``."""
        self.strikes[bad] += 1
        self.strikes[~bad] = 0
        return self.strikes >= strike_limit

    def accumulate(self, viewspace_gradient: torch.Tensor, visible: torch.Tensor) -> None:
        """Add one iteration's screen-space gradient norms for the visible Gaussians."""
        self.gradient_sum[visible] += torch.norm(
            viewspace_gradient[visible, :2], dim=-1, keepdim=True
        )
        self.visible_count[visible] += 1

    def mean_gradient(self) -> torch.Tensor:
        """``(N, 1)`` mean gradient norm; zero for never-visible Gaussians."""
        mean = self.gradient_sum / self.visible_count
        mean[mean.isnan()] = 0.0
        return mean

    def keep_rows(self, keep: torch.Tensor) -> None:
        """Drop the statistics of removed Gaussians."""
        self.gradient_sum = self.gradient_sum[keep]
        self.visible_count = self.visible_count[keep]
        self.strikes = self.strikes[keep]

    def reset(self, count: int) -> None:
        """Clear the gradient statistics after rows were appended; new rows get no strikes."""
        device = self.gradient_sum.device
        self.gradient_sum = torch.zeros((count, 1), device=device)
        self.visible_count = torch.zeros((count, 1), device=device)
        extra = count - self.strikes.shape[0]
        self.strikes = torch.cat(
            (self.strikes, torch.zeros(extra, dtype=torch.int16, device=device))
        )


class DensityController:
    """Clone, split and prune the canonical Gaussians."""

    def __init__(
        self,
        gaussians: CanonicalGaussians,
        optimizer: torch.optim.Optimizer,
        settings: DensificationSettings,
        scene_extent: float,
    ):
        self.gaussians = gaussians
        self.surgery = ParameterSurgery(gaussians, optimizer)
        self.settings = settings
        self.scene_extent = float(scene_extent)
        self.state = DensificationState(len(gaussians), gaussians.xyz.device)

    def prune(self, remove: torch.Tensor) -> None:
        """Remove the rows where ``remove`` is true."""
        keep = ~remove
        self.surgery.keep_rows(keep)
        self.state.keep_rows(keep)

    def health_prune(self) -> int:
        """Remove transparent or non-finite Gaussians, at most a fixed fraction at once.

        Non-finite rows are always removed; the remaining budget goes to the
        most transparent candidates.

        Returns:
            The number of removed Gaussians.
        """
        gaussians = self.gaussians
        opacity = torch.sigmoid(gaussians.opacity.detach())
        invalid = torch.zeros(len(gaussians), dtype=torch.bool, device=opacity.device)
        for name in PARAMETER_NAMES:
            invalid |= ~torch.isfinite(gaussians.parameter_by_name(name).detach()).flatten(1).all(1)
        invalid |= ~torch.isfinite(torch.exp(gaussians.log_scale.detach())).all(dim=1)
        remove = (opacity < self.settings.min_opacity).squeeze(1) | invalid

        count = int(remove.sum().item())
        budget = max(1, int(len(gaussians) * self.settings.max_prune_fraction))
        if count > budget:
            capped = invalid.clone()
            candidates = torch.where(remove & ~invalid)[0]
            remaining = max(0, budget - int(invalid.sum().item()))
            if remaining and candidates.numel():
                lowest = torch.topk(
                    opacity[candidates, 0], k=min(remaining, candidates.numel()), largest=False
                ).indices
                capped[candidates[lowest]] = True
            remove = capped
            count = int(remove.sum().item())
        if count >= len(gaussians):
            raise RuntimeError("health pruning would remove every Gaussian")
        if count:
            self.prune(remove)
        return count

    def densify(self) -> int:
        """Clone small and split large Gaussians with a high mean screen-space gradient.

        Returns:
            The net number of added Gaussians.
        """
        before = len(self.gaussians)
        gradient = self.state.mean_gradient()
        self._clone(gradient)
        self._split(gradient)
        return len(self.gaussians) - before

    def _is_large(self) -> torch.Tensor:
        scale = torch.exp(self.gaussians.log_scale.detach())
        return scale.max(dim=1).values > self.settings.percent_dense * self.scene_extent

    def _clone(self, gradient: torch.Tensor) -> None:
        selected = (
            torch.norm(gradient, dim=-1) >= self.settings.gradient_threshold
        ) & ~self._is_large()
        self._append({name: self._rows(name)[selected] for name in PARAMETER_NAMES})

    def _split(self, gradient: torch.Tensor, children: int = 2) -> None:
        # `gradient` predates the clone step; the cloned rows count as zero.
        padded = torch.zeros(len(self.gaussians), device=gradient.device)
        padded[: gradient.shape[0]] = gradient.squeeze(1)
        selected = (padded >= self.settings.gradient_threshold) & self._is_large()
        if not selected.any():
            return
        scale = torch.exp(self.gaussians.log_scale.detach()[selected]).repeat(children, 1)
        offsets = torch.normal(mean=torch.zeros_like(scale), std=scale)
        rotations = quaternion_to_matrix(self._rows("rotation")[selected]).repeat(children, 1, 1)
        rows = {
            name: self._rows(name)[selected].repeat(children, *([1] * (self._rows(name).dim() - 1)))
            for name in PARAMETER_NAMES
        }
        rows["xyz"] = torch.bmm(rotations, offsets.unsqueeze(-1)).squeeze(-1) + rows["xyz"]
        rows["log_scale"] = torch.log(scale / (0.8 * children))
        self._append(rows)
        parents = torch.cat(
            (
                selected,
                torch.zeros(
                    children * int(selected.sum()), dtype=torch.bool, device=selected.device
                ),
            )
        )
        self.prune(parents)

    def _rows(self, name: str) -> torch.Tensor:
        return self.gaussians.parameter_by_name(name).detach()

    def _append(self, rows: dict[str, torch.Tensor]) -> None:
        self.surgery.append_rows(rows)
        self.state.reset(len(self.gaussians))


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert ``(N, 4)`` scalar-first quaternions (any norm) to ``(N, 3, 3)`` rotations."""
    q = quaternion / torch.linalg.norm(quaternion, dim=1, keepdim=True)
    w, x, y, z = q.unbind(dim=1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=1,
    ).reshape(-1, 3, 3)  # fmt: skip
