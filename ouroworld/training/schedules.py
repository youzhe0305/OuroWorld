"""Iteration-dependent schedules: learning rates, drift activation, refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ExponentialDecay:
    """Log-linear interpolation from ``initial`` to ``final`` over ``steps`` iterations."""

    initial: float
    final: float
    steps: int

    def __call__(self, iteration: int) -> float:
        """Learning rate at ``iteration``; constant at ``final`` afterwards."""
        if iteration < 0 or (self.initial == 0.0 and self.final == 0.0):
            return 0.0
        progress = min(max(iteration / self.steps, 0.0), 1.0)
        return math.exp(math.log(self.initial) * (1 - progress) + math.log(self.final) * progress)


@dataclass(frozen=True)
class PhaseSchedule:
    """When the two training phases and the drift field start.

    Attributes:
        refinement_start: First iteration with diffusion refinement, or ``None``
            when refinement is disabled.
        drift_start: First iteration in which Δ is applied, or ``None`` without Δ.
    """

    refinement_start: int | None
    drift_start: int | None

    def is_refining(self, iteration: int) -> bool:
        """Whether refinement runs at ``iteration``."""
        return self.refinement_start is not None and iteration >= self.refinement_start

    def is_drift_active(self, iteration: int) -> bool:
        """Whether Δ is applied at ``iteration``."""
        return self.drift_start is not None and iteration >= self.drift_start


def phase_schedule(
    iterations: int,
    refinement_after: int | None,
    has_drift: bool,
    drift_waits_for_refinement: bool,
    drift_warmup: int,
) -> PhaseSchedule:
    """Resolve the phase boundaries.

    Args:
        iterations: Total number of iterations.
        refinement_after: Last iteration without refinement, or ``None`` to disable it.
        has_drift: Whether the model has a drift field.
        drift_waits_for_refinement: Start Δ with refinement. This is the paper
            setting: with full grounding every pixel-supervised observation is
            grounded, so Δ first matters on the refined (generated) observations.
        drift_warmup: Otherwise Δ starts after this many iterations.
    """
    refinement_start = None
    if refinement_after is not None and refinement_after < iterations:
        refinement_start = refinement_after + 1
    drift_start = None
    if has_drift:
        if drift_waits_for_refinement:
            if refinement_start is None:
                raise ValueError(
                    "a drift field grounded on every pixel observation needs refinement"
                )
            drift_start = refinement_start
        else:
            drift_start = drift_warmup + 1
    return PhaseSchedule(refinement_start, drift_start)
