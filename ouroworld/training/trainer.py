"""The 4D optimisation loop (paper §4.2).

Each iteration renders a few pixel-supervised observations, adds the
regularisers, and -- in the refinement phase -- one generated observation
supervised through diffusion refinement. Every loss term is backpropagated
as soon as it is computed (:class:`BackwardAccumulator`); one optimiser step
follows.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ouroworld.fields.cinemagraph import CinemagraphModel, ViewContext
from ouroworld.fields.serialization import save_model
from ouroworld.gaussians.densification import DensityController
from ouroworld.io.multiview import Observation
from ouroworld.render.rasterizer import RenderOutput, render
from ouroworld.training.backward import (
    BackwardAccumulator,
    GradientQuarantine,
    NonFiniteLossError,
    first_nonfinite_gradient,
)
from ouroworld.training.checkpoint import checkpoint_dir, save_training_state
from ouroworld.training.data import TrainingData
from ouroworld.training.logging import MetricsLog
from ouroworld.training.losses import DriftMagnitudeLoss, Regularizer, photometric_l1
from ouroworld.training.optimizer import ScheduledOptimizer
from ouroworld.training.refinement.objective import RefinementObjective
from ouroworld.training.sampler import ShuffledCycleSampler, ViewBalancedSampler
from ouroworld.training.schedules import PhaseSchedule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainSettings:
    """Loop-level settings.

    Attributes:
        iterations: Number of iterations. The last one updates statistics but
            takes no optimiser step, as in the paper runs.
        batch_size: Pixel-supervised observations per iteration.
        field_grad_clip: Global gradient-norm bound of the field parameters.
        max_consecutive_skips: Abort after this many consecutive non-finite steps.
        strike_limit: Prune a Gaussian after this many consecutive non-finite gradients.
        checkpoint_iterations: Iterations after which a checkpoint is written.
        log_interval: Write metrics every this many iterations.
        drift_regularized_every_iteration: Evaluate the drift regularisers on the
            pixel batch every iteration (naive L1 ablation) instead of on the
            refined observation (scene-view consistent optimisation).
    """

    iterations: int
    batch_size: int
    field_grad_clip: float
    max_consecutive_skips: int
    strike_limit: int
    checkpoint_iterations: tuple[int, ...]
    log_interval: int
    drift_regularized_every_iteration: bool


@dataclass
class Regularizers:
    """Regularisers of the fields; ``None`` disables a term."""

    appearance: Regularizer | None = None
    scale_change: Regularizer | None = None
    motion_smoothness: Regularizer | None = None
    twist_smoothness: Regularizer | None = None
    drift_magnitude: DriftMagnitudeLoss | None = None
    plane_smoothness: Regularizer | None = None


@dataclass
class TrainingSetup:
    """Everything the loop needs, assembled by :mod:`ouroworld.cli.build`."""

    model: CinemagraphModel
    data: TrainingData
    pixel_sampler: ViewBalancedSampler
    refinement_sampler: ShuffledCycleSampler | None
    schedule: PhaseSchedule
    optimizer: ScheduledOptimizer
    regularizers: Regularizers
    refinement: RefinementObjective | None
    density: DensityController | None
    background: torch.Tensor
    settings: TrainSettings
    run_dir: Path
    metrics: MetricsLog
    quarantine: GradientQuarantine = field(default_factory=GradientQuarantine)


@dataclass
class _Iteration:
    """Outputs of one forward/backward pass needed after the optimiser step."""

    losses: dict[str, float]
    pixel_renders: list[RenderOutput]
    psnr: float


class Trainer:
    """Runs :class:`TrainingSetup` from a start iteration to the end."""

    def __init__(self, setup: TrainingSetup):
        self.setup = setup
        self.consecutive_skips = 0
        self.loss_ema = 0.0
        self.started = time.monotonic()

    def run(self, start_iteration: int = 1) -> None:
        """Train from ``start_iteration`` to ``settings.iterations`` inclusive."""
        setup = self.setup
        setup.model.train()
        if start_iteration > 1 and setup.schedule.is_refining(start_iteration):
            self._refresh_drift_visibility()
        for iteration in range(start_iteration, setup.settings.iterations + 1):
            self._begin(iteration)
            try:
                result = self._forward_backward(iteration)
                offenders = self._record_bad_gradients()
                self._check_gradients()
            except NonFiniteLossError as error:
                self._skip(iteration, str(error))
                continue
            self._step(iteration)
            self._adapt_density(iteration, result, offenders)
            self._log(iteration, result)
            if iteration in setup.settings.checkpoint_iterations:
                self.save_checkpoint(iteration)

    def _begin(self, iteration: int) -> None:
        setup = self.setup
        setup.optimizer.update_learning_rates(iteration)
        setup.quarantine.begin_iteration(len(setup.model.canonical), setup.background.device)
        drift_loss = setup.regularizers.drift_magnitude
        if drift_loss is None:
            return
        starts_refinement = iteration == setup.schedule.refinement_start
        first_iteration = iteration == 1 and setup.settings.drift_regularized_every_iteration
        if starts_refinement or first_iteration:
            self._refresh_drift_visibility()

    def _forward_backward(self, iteration: int) -> _Iteration:
        setup = self.setup
        backward = BackwardAccumulator()
        drift_active = setup.schedule.is_drift_active(iteration)
        batch = setup.pixel_sampler.sample(setup.settings.batch_size)
        contexts = [self._context(observation, drift_active) for observation in batch]
        renders = [
            self._render(observation, context)
            for observation, context in zip(batch, contexts, strict=True)
        ]
        images = torch.stack([output.image for output in renders])
        targets = torch.stack([setup.data.image(observation) for observation in batch])
        psnr = _psnr(images.detach(), targets)
        backward.add("l1", photometric_l1(images, targets))
        del images

        time = batch[0].time
        self._drift_regularization_on_pixels(backward, batch, contexts, drift_active)
        regularizers = setup.regularizers
        for term in (
            regularizers.appearance,
            regularizers.scale_change,
            regularizers.motion_smoothness,
        ):
            if term is not None:
                backward.add(term.name, term(time))
        if setup.schedule.is_refining(iteration):
            self._refine(backward, drift_active)
        if regularizers.plane_smoothness is not None:
            backward.add(regularizers.plane_smoothness.name, regularizers.plane_smoothness(time))
        return _Iteration(backward.values, renders, psnr)

    def _drift_regularization_on_pixels(
        self,
        backward: BackwardAccumulator,
        batch: list[Observation],
        contexts: list[ViewContext],
        drift_active: bool,
    ) -> None:
        regularizers = self.setup.regularizers
        magnitude = regularizers.drift_magnitude
        if self.setup.settings.drift_regularized_every_iteration:
            if regularizers.twist_smoothness is not None:
                backward.add(
                    regularizers.twist_smoothness.name, regularizers.twist_smoothness(batch[0].time)
                )
            if magnitude is not None and drift_active:
                backward.add(magnitude.name, magnitude(batch[0].time))
            return
        # Scene-view consistent optimisation: Δ reaches pixel observations only
        # when grounding is disabled (ablation); regularise it where it acted.
        drifted = [context for context in contexts if self.setup.model.applies_drift(context)]
        if magnitude is not None and drifted:
            if magnitude.visibility is None:
                self._refresh_drift_visibility()
            backward.add(magnitude.name, magnitude(drifted[0].time))

    def _refine(self, backward: BackwardAccumulator, drift_active: bool) -> None:
        setup = self.setup
        assert setup.refinement is not None and setup.refinement_sampler is not None
        observation = setup.refinement_sampler.sample(1)[0]
        output = self._render(observation, self._context(observation, drift_active))
        backward.add("refinement", setup.refinement(output.image, setup.data.image(observation)))
        if setup.settings.drift_regularized_every_iteration:
            return
        regularizers = setup.regularizers
        if regularizers.drift_magnitude is not None:
            backward.add(
                regularizers.drift_magnitude.name, regularizers.drift_magnitude(observation.time)
            )
        if regularizers.twist_smoothness is not None:
            backward.add(
                regularizers.twist_smoothness.name, regularizers.twist_smoothness(observation.time)
            )

    def _context(self, observation: Observation, drift_active: bool) -> ViewContext:
        return ViewContext(observation.time, observation.view_id if drift_active else None)

    def _render(self, observation: Observation, context: ViewContext) -> RenderOutput:
        setup = self.setup
        return render(
            setup.model.deformed(context),
            setup.data.view(observation),
            setup.background,
            setup.model.canonical.sh_degree,
            gradient_filter=setup.quarantine,
        )

    def _record_bad_gradients(self) -> torch.Tensor | None:
        bad = self.setup.quarantine.bad_rows
        density = self.setup.density
        if bad is None or density is None:
            return None
        count = int(bad.sum())
        if count:
            logger.warning("quarantined non-finite gradients of %d Gaussian(s)", count)
        return density.state.record_bad_rows(bad, self.setup.settings.strike_limit)

    def _check_gradients(self) -> None:
        name = first_nonfinite_gradient(self.setup.optimizer.parameters_with_gradients())
        if name is not None:
            raise NonFiniteLossError(f"non-finite gradient in {name}")

    def _skip(self, iteration: int, reason: str) -> None:
        self.setup.optimizer.zero_grad()
        self.consecutive_skips += 1
        logger.warning(
            "skipping iteration %d: %s (%d in a row)", iteration, reason, self.consecutive_skips
        )
        if self.consecutive_skips >= self.setup.settings.max_consecutive_skips:
            raise RuntimeError(
                f"{self.consecutive_skips} consecutive non-finite steps; last: {reason}"
            )

    def _step(self, iteration: int) -> None:
        setup = self.setup
        if setup.settings.field_grad_clip > 0:
            parameters = [p for p in setup.optimizer.field_parameters() if p.grad is not None]
            if parameters:
                torch.nn.utils.clip_grad_norm_(parameters, setup.settings.field_grad_clip)
        if iteration < setup.settings.iterations:
            setup.optimizer.step()
        self.consecutive_skips = 0

    @torch.no_grad()
    def _adapt_density(
        self, iteration: int, result: _Iteration, offenders: torch.Tensor | None
    ) -> None:
        density = self.setup.density
        if density is None or not density.settings.is_active(iteration):
            return
        viewspace_gradient = sum(output.viewspace_points.grad for output in result.pixel_renders)
        visible = torch.stack([output.visible for output in result.pixel_renders]).any(dim=0)
        density.state.accumulate(viewspace_gradient, visible)
        if offenders is not None and bool(offenders.any()):
            logger.info(
                "pruning %d Gaussian(s) with repeated non-finite gradients", int(offenders.sum())
            )
            density.prune(offenders)
        if density.settings.is_due(iteration):
            density.health_prune()
            density.densify()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _refresh_drift_visibility(self) -> None:
        """Cache which Gaussians each view sees in the canonical scene."""
        setup = self.setup
        magnitude = setup.regularizers.drift_magnitude
        if magnitude is None:
            return
        canonical = setup.model.canonical.attributes()
        black = torch.zeros_like(setup.background)
        columns = [
            (render(canonical, view, black, setup.model.canonical.sh_degree).radii > 0).to(
                "cpu", torch.uint8
            )
            for _, view in sorted(setup.data.views.items())
        ]
        magnitude.set_visibility(torch.stack(columns, dim=1))

    def _log(self, iteration: int, result: _Iteration) -> None:
        settings = self.setup.settings
        self.loss_ema = 0.4 * sum(result.losses.values()) + 0.6 * self.loss_ema
        if iteration % settings.log_interval and iteration != settings.iterations:
            return
        record = {
            "iteration": iteration,
            "loss_ema": self.loss_ema,
            "psnr": result.psnr,
            "gaussians": len(self.setup.model.canonical),
            "drift_active": self.setup.schedule.is_drift_active(iteration),
            "elapsed_seconds": time.monotonic() - self.started,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
            **result.losses,
        }
        self.setup.metrics.write(record)
        logger.info("iteration %d: loss %.5f, psnr %.2f", iteration, self.loss_ema, result.psnr)

    def save_checkpoint(self, iteration: int) -> Path:
        """Write the model and the resumable training state."""
        setup = self.setup
        directory = checkpoint_dir(setup.run_dir, iteration)
        save_model(setup.model, directory)
        density = setup.density
        state = {
            "iteration": iteration,
            "optimizer": setup.optimizer.optimizer.state_dict(),
            "density": None if density is None else vars(density.state),
            "pixel_sampler_rng": setup.pixel_sampler.rng.getstate(),
            "refinement_sampler": None
            if setup.refinement_sampler is None
            else (setup.refinement_sampler.rng.getstate(), setup.refinement_sampler.remaining),
        }
        save_training_state(state, directory)
        logger.info("saved checkpoint %s", directory)
        return directory


def _psnr(images: torch.Tensor, targets: torch.Tensor) -> float:
    mse = ((images - targets) ** 2).flatten(1).mean(dim=1)
    return float((20 * torch.log10(1.0 / torch.sqrt(mse))).mean())
