"""Random choice of the observations used at each iteration."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from ouroworld.io.multiview import Observation


class ViewBalancedSampler:
    """Pick a view, then an observation of that view.

    The reference view is chosen with probability ``reference_probability``;
    the rest is shared uniformly by the other views, regardless of how many
    observations each view has.
    """

    def __init__(
        self,
        observations: Sequence[Observation],
        reference_view_id: int,
        reference_probability: float,
        rng: random.Random,
    ):
        if not 0.0 <= reference_probability <= 1.0:
            raise ValueError("reference_probability must lie in [0, 1]")
        by_view: dict[int, list[Observation]] = defaultdict(list)
        for observation in observations:
            by_view[observation.view_id].append(observation)
        if reference_view_id not in by_view:
            raise ValueError(f"reference view {reference_view_id} has no observations")
        self.groups = dict(sorted(by_view.items()))
        self.reference_view_id = reference_view_id
        self.others = [view for view in self.groups if view != reference_view_id]
        if not self.others and reference_probability < 1.0:
            raise ValueError("reference_probability must be 1 when only the reference view exists")
        self.reference_probability = reference_probability
        self.rng = rng

    def sample(self, count: int) -> list[Observation]:
        """Draw ``count`` observations with replacement."""
        batch = []
        for _ in range(count):
            use_reference = not self.others or self.rng.random() < self.reference_probability
            view = self.reference_view_id if use_reference else self.rng.choice(self.others)
            batch.append(self.rng.choice(self.groups[view]))
        return batch


class ShuffledCycleSampler:
    """Draw without replacement, refilling once every observation was used."""

    def __init__(self, observations: Sequence[Observation], rng: random.Random):
        if not observations:
            raise ValueError("cannot sample from an empty observation set")
        self.pool = list(observations)
        self.remaining = list(observations)
        self.rng = rng

    def sample(self, count: int) -> list[Observation]:
        """Draw ``count`` observations."""
        batch = []
        for _ in range(count):
            batch.append(self.remaining.pop(self.rng.randint(0, len(self.remaining) - 1)))
            if not self.remaining:
                self.remaining = list(self.pool)
        return batch
