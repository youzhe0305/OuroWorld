"""Supervision policies: which observations get a pixel loss and which get refinement.

Scene-view consistent optimisation (paper §4.2) trusts only the observations
that are consistent by construction -- the reference video and the ``t = 0``
renders of the static scene -- with a pixel loss. Generated frames of other
views are used only through diffusion refinement.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from ouroworld.io.multiview import MultiviewVideos, Observation, Role


class SupervisionMode(enum.StrEnum):
    """How observations are supervised."""

    SVCO = "svco"  # scene-view consistent optimisation (the method)
    NAIVE_L1 = "naive_l1"  # ablation: pixel loss on every observation, no refinement


@dataclass(frozen=True)
class SupervisionSplit:
    """Observations partitioned by the loss that consumes them.

    Attributes:
        pixel: Observations supervised with an L1 loss.
        refinement: Observations supervised through diffusion refinement.
    """

    pixel: tuple[Observation, ...]
    refinement: tuple[Observation, ...]


def split_observations(videos: MultiviewVideos, mode: SupervisionMode) -> SupervisionSplit:
    """Partition the observations of ``videos`` according to ``mode``."""
    if mode is SupervisionMode.NAIVE_L1:
        return SupervisionSplit(pixel=videos.observations, refinement=())
    pixel = tuple(videos.with_role(Role.REFERENCE, Role.T0))
    refinement = tuple(videos.with_role(Role.GENERATED))
    return SupervisionSplit(pixel=pixel, refinement=refinement)
