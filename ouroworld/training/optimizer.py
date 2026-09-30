"""Adam parameter groups and their learning-rate schedules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ouroworld.fields.cinemagraph import CinemagraphModel
from ouroworld.fields.encoding import Triplane
from ouroworld.training.schedules import ExponentialDecay


@dataclass(frozen=True)
class LearningRates:
    """Learning rates; position and field rates are multiplied by the scene extent.

    Attributes:
        xyz: Initial and final rate of the canonical centres.
        field_mlp: Initial and final rate of every field MLP and view embedding.
        field_grid: Initial and final rate of every triplane.
        decay_steps: Iterations over which the three decaying rates reach their final value.
        sh_dc: Rate of the degree-0 SH coefficients.
        sh_rest: Rate of the higher-order SH coefficients.
        opacity: Rate of the opacity logits.
        log_scale: Rate of the log-scales.
        rotation: Rate of the quaternions.
    """

    xyz: tuple[float, float]
    field_mlp: tuple[float, float]
    field_grid: tuple[float, float]
    decay_steps: int
    sh_dc: float
    sh_rest: float
    opacity: float
    log_scale: float
    rotation: float


FIELD_GROUPS = ("field_mlp", "field_grid")


class ScheduledOptimizer:
    """Adam over the canonical Gaussians and the fields, with per-group schedules."""

    def __init__(self, model: CinemagraphModel, rates: LearningRates, scene_extent: float):
        canonical = model.canonical
        mlp, grid = _split_field_parameters(model)
        groups = [
            ("xyz", [canonical.xyz], rates.xyz[0] * scene_extent),
            ("field_mlp", mlp, rates.field_mlp[0] * scene_extent),
            ("field_grid", grid, rates.field_grid[0] * scene_extent),
            ("sh_dc", [canonical.sh_dc], rates.sh_dc),
            ("sh_rest", [canonical.sh_rest], rates.sh_rest),
            ("opacity", [canonical.opacity], rates.opacity),
            ("log_scale", [canonical.log_scale], rates.log_scale),
            ("rotation", [canonical.rotation], rates.rotation),
        ]
        self.optimizer = torch.optim.Adam(
            [{"params": params, "lr": lr, "name": name} for name, params, lr in groups],
            lr=0.0,
            eps=1e-15,
        )
        self.schedules = {
            name: ExponentialDecay(initial * scene_extent, final * scene_extent, rates.decay_steps)
            for name, (initial, final) in (
                ("xyz", rates.xyz),
                ("field_mlp", rates.field_mlp),
                ("field_grid", rates.field_grid),
            )
        }

    def update_learning_rates(self, iteration: int) -> None:
        """Apply the decaying schedules for ``iteration``."""
        for group in self.optimizer.param_groups:
            schedule = self.schedules.get(group["name"])
            if schedule is not None:
                group["lr"] = schedule(iteration)

    def field_parameters(self) -> list[nn.Parameter]:
        """Parameters of the deformation fields, which share one gradient-norm clip."""
        return [
            parameter
            for group in self.optimizer.param_groups
            if group["name"] in FIELD_GROUPS
            for parameter in group["params"]
        ]

    def parameters_with_gradients(self) -> list[tuple[str, nn.Parameter]]:
        """``(group name, parameter)`` pairs that received a gradient."""
        return [
            (group["name"], parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]

    def step(self) -> None:
        """Apply one Adam step and clear gradients."""
        self.optimizer.step()
        self.zero_grad()

    def zero_grad(self) -> None:
        """Drop gradients (set to ``None`` so frozen tensors never see a zero-gradient step)."""
        self.optimizer.zero_grad(set_to_none=True)


def _split_field_parameters(
    model: CinemagraphModel,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return ``(mlp_parameters, grid_parameters)`` of P and Δ."""
    grid_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, Triplane):
            grid_ids.update(id(parameter) for parameter in module.parameters())
    fields = [model.periodic] + ([model.drift] if model.drift is not None else [])
    parameters = [parameter for field in fields for parameter in field.parameters()]
    mlp = [parameter for parameter in parameters if id(parameter) not in grid_ids]
    grid = [parameter for parameter in parameters if id(parameter) in grid_ids]
    return mlp, grid
