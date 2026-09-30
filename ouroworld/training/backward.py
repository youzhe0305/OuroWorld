"""Gradient accumulation across loss terms, with non-finite guards."""

from __future__ import annotations

import torch


class NonFiniteLossError(RuntimeError):
    """A loss term or a gradient became NaN or infinite; the step must be skipped."""


class BackwardAccumulator:
    """Backpropagate each loss term as soon as it is computed.

    Freeing each term's graph immediately keeps peak memory at one render
    graph, which matters once refinement adds a second render per iteration.
    Gradients still accumulate into a single optimiser step.
    """

    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def add(self, name: str, loss: torch.Tensor | None) -> None:
        """Backpropagate ``loss`` and remember its value under ``name``."""
        if loss is None:
            return
        if not torch.isfinite(loss).all():
            raise NonFiniteLossError(f"non-finite {name} loss")
        loss.backward()
        self.values[name] = self.values.get(name, 0.0) + float(loss.detach())

    @property
    def total(self) -> float:
        """Sum of every accumulated term."""
        return sum(self.values.values())


class GradientQuarantine:
    """Zero the gradient rows of Gaussians whose rasterizer gradient is non-finite.

    A rare projection singularity in the CUDA backward pass can yield NaN for a
    single Gaussian; zeroing that row keeps it from poisoning the shared field
    gradients. Offending rows are remembered so repeat offenders can be pruned.
    """

    def __init__(self) -> None:
        self.bad_rows: torch.Tensor | None = None

    def begin_iteration(self, count: int, device: torch.device) -> None:
        """Reset the record for an iteration over ``count`` Gaussians."""
        self.bad_rows = torch.zeros(count, dtype=torch.bool, device=device)

    def __call__(self, gradient: torch.Tensor) -> torch.Tensor:
        """Gradient hook: record and zero non-finite rows."""
        bad = ~torch.isfinite(gradient.reshape(gradient.shape[0], -1)).all(dim=1)
        if self.bad_rows is not None and self.bad_rows.shape[0] == bad.shape[0]:
            self.bad_rows |= bad
        return torch.where(
            bad.view(-1, *([1] * (gradient.dim() - 1))), torch.zeros_like(gradient), gradient
        )


def first_nonfinite_gradient(named_parameters: list[tuple[str, torch.nn.Parameter]]) -> str | None:
    """Return the group name of the first parameter with a non-finite gradient."""
    if not named_parameters:
        return None
    flags = torch.stack([torch.isfinite(parameter.grad).all() for _, parameter in named_parameters])
    if bool(flags.all()):
        return None
    for (name, _), is_finite in zip(named_parameters, flags.tolist(), strict=True):
        if not is_finite:
            return name
    return None
