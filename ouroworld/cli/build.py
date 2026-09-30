"""Composition root: turn a :class:`Config` into ready-to-run objects.

This is the only module that reads the configuration schema; everything
below it receives small, module-owned parameter objects.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ouroworld.config.schema import Config, ModelConfig, TriplaneConfig
from ouroworld.fields.cinemagraph import CinemagraphModel
from ouroworld.fields.drift import DriftFieldSpec, GroundedDriftField
from ouroworld.fields.encoding import TriplaneSpec
from ouroworld.fields.heads import ChunkedEvaluator
from ouroworld.fields.periodic import PeriodicDeformationField, PeriodicFieldSpec
from ouroworld.fields.serialization import load_model
from ouroworld.gaussians.canonical import CanonicalGaussians
from ouroworld.gaussians.densification import DensificationSettings, DensityController
from ouroworld.io.multiview import MultiviewVideos, load_multiview
from ouroworld.io.ply import read_gaussian_ply
from ouroworld.io.scene_package import ScenePackage, load_scene_package
from ouroworld.rendering.loop_render import LoopRenderSettings
from ouroworld.training.checkpoint import checkpoint_dir, latest_checkpoint, load_training_state
from ouroworld.training.data import TrainingData, scene_extent
from ouroworld.training.logging import MetricsLog
from ouroworld.training.losses import (
    AppearanceResidualLoss,
    DriftChannelWeights,
    DriftMagnitudeLoss,
    MotionSmoothnessLoss,
    PlaneSmoothnessLoss,
    ScaleChangeLoss,
    TwistSmoothnessLoss,
)
from ouroworld.training.optimizer import LearningRates, ScheduledOptimizer
from ouroworld.training.refinement.objective import RefinementObjective
from ouroworld.training.refinement.sd15_lcm import LcmRefiner, RefinerSettings
from ouroworld.training.sampler import ShuffledCycleSampler, ViewBalancedSampler
from ouroworld.training.schedules import phase_schedule
from ouroworld.training.supervision import SupervisionMode, split_observations
from ouroworld.training.trainer import Regularizers, TrainingSetup, TrainSettings

PERIODIC_PERIOD = 1.0
APERIODIC_PERIOD = 2.0


def triplane_spec(config: TriplaneConfig) -> TriplaneSpec:
    """Triplane shape from its config block."""
    return TriplaneSpec(tuple(config.resolution), tuple(config.multires), config.channels)  # type: ignore[arg-type]


def evaluator(config: ModelConfig) -> ChunkedEvaluator:
    """Chunking policy shared by all fields."""
    return ChunkedEvaluator(config.point_chunk_size, config.activation_checkpointing)


def periodic_spec(config: ModelConfig) -> PeriodicFieldSpec:
    """Architecture of P."""
    periodic = config.periodic
    return PeriodicFieldSpec(
        heads=tuple(periodic.heads),
        harmonics=periodic.harmonics,
        period=PERIODIC_PERIOD if periodic.is_periodic else APERIODIC_PERIOD,
        motion_grid=triplane_spec(periodic.motion_grid),
        appearance_grid=triplane_spec(periodic.appearance_grid),
        width=periodic.width,
        max_log_scale_delta=periodic.max_log_scale_delta,
        sh_degree=config.sh_degree,
    )


def drift_spec(config: ModelConfig, videos: MultiviewVideos) -> DriftFieldSpec | None:
    """Architecture of Δ, or ``None`` when it is disabled."""
    drift = config.drift
    if not drift.enabled:
        return None
    if videos.view_ids != list(range(len(videos.view_ids))):
        raise ValueError(f"view ids must be 0..V-1, got {videos.view_ids}")
    return DriftFieldSpec(
        view_count=len(videos.view_ids),
        reference_view_id=videos.reference_view_id,
        heads=tuple(drift.heads),
        harmonics=config.periodic.harmonics,
        period=PERIODIC_PERIOD if drift.is_periodic else APERIODIC_PERIOD,
        grid=triplane_spec(drift.grid),
        width=drift.width,
        view_embedding_dim=drift.view_embedding_dim,
        zero_reference_view=drift.zero_reference_view,
        zero_t0=drift.zero_t0,
        sh_degree=config.sh_degree,
    )


def build_model(config: Config, package: ScenePackage, videos: MultiviewVideos) -> CinemagraphModel:
    """A fresh model whose canonical Gaussians are the input 3DGS."""
    chunks = evaluator(config.model)
    canonical = CanonicalGaussians(
        read_gaussian_ply(package.gaussians_path), config.model.sh_degree, config.model.frozen
    )
    periodic = PeriodicDeformationField(periodic_spec(config.model), chunks)
    spec = drift_spec(config.model, videos)
    drift = None if spec is None else GroundedDriftField(spec, chunks)
    model = CinemagraphModel(canonical, periodic, drift)
    model.fit_fields_to_canonical()
    return model


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainingRun:
    """A training setup and the iteration to start at."""

    setup: TrainingSetup
    start_iteration: int


def build_training(config: Config, resume_from: Path | None = None) -> TrainingRun:
    """Assemble a training run, optionally resuming from a checkpoint directory."""
    device = torch.device("cuda")
    train = config.train
    seed_everything(train.seed)
    package = load_scene_package(Path(config.paths.scene_dir))
    videos = load_multiview(Path(config.paths.multiview_dir))
    mode = SupervisionMode(train.supervision)
    split = split_observations(videos, mode)
    extent = scene_extent([videos.cameras[observation.view_id] for observation in split.pixel])

    if resume_from is None:
        model = build_model(config, package, videos)
    else:
        model = load_model(resume_from, evaluator(config.model), tuple(config.model.frozen))
    model = model.to(device)
    rates = train.learning_rates
    optimizer = ScheduledOptimizer(
        model,
        LearningRates(
            xyz=tuple(rates.xyz),  # type: ignore[arg-type]
            field_mlp=tuple(rates.field_mlp),  # type: ignore[arg-type]
            field_grid=tuple(rates.field_grid),  # type: ignore[arg-type]
            decay_steps=rates.decay_steps,
            sh_dc=rates.sh_dc,
            sh_rest=rates.sh_rest,
            opacity=rates.opacity,
            log_scale=rates.log_scale,
            rotation=rates.rotation,
        ),
        extent,
    )
    refinement_after = train.refinement.start_after if train.refinement.enabled else None
    fully_grounded = model.drift is not None and model.drift.spec.is_fully_grounded
    schedule = phase_schedule(
        iterations=train.iterations,
        refinement_after=refinement_after,
        has_drift=model.drift is not None,
        drift_waits_for_refinement=mode is SupervisionMode.SVCO and fully_grounded,
        drift_warmup=config.model.drift.warmup_iterations,
    )
    run_dir = Path(config.paths.run_dir)
    checkpoints = set(train.checkpoint_iterations) | {train.iterations}
    if refinement_after is not None and refinement_after > 0:
        checkpoints.add(refinement_after)
    setup = TrainingSetup(
        model=model,
        data=TrainingData(videos, device),
        pixel_sampler=ViewBalancedSampler(
            split.pixel,
            videos.reference_view_id,
            train.reference_view_probability,
            random.Random(train.seed),
        ),
        refinement_sampler=ShuffledCycleSampler(split.refinement, random.Random(train.seed + 1))
        if split.refinement
        else None,
        schedule=schedule,
        optimizer=optimizer,
        regularizers=build_regularizers(config, model, extent),
        refinement=_refinement(config, device),
        density=_density(config, model, optimizer, extent),
        background=torch.tensor(package.background, dtype=torch.float32, device=device),
        settings=TrainSettings(
            iterations=train.iterations,
            batch_size=train.batch_size,
            field_grad_clip=train.field_grad_clip,
            max_consecutive_skips=train.max_consecutive_skips,
            strike_limit=train.strike_limit,
            checkpoint_iterations=tuple(sorted(checkpoints)),
            log_interval=train.log_interval,
            drift_regularized_every_iteration=mode is SupervisionMode.NAIVE_L1,
        ),
        run_dir=run_dir,
        metrics=MetricsLog(run_dir / "metrics.jsonl", resume=resume_from is not None),
    )
    start = 1 if resume_from is None else _restore(setup, resume_from) + 1
    return TrainingRun(setup, start)


def build_regularizers(config: Config, model: CinemagraphModel, extent: float) -> Regularizers:
    """The regularisers enabled by ``config`` (terms with zero weight are left out)."""
    losses = config.train.losses
    samples = losses.sample_gaussians
    heads = set(config.model.periodic.heads)
    regularizers = Regularizers()
    if "shs" in heads and (losses.appearance_magnitude > 0 or losses.appearance_smoothness > 0):
        regularizers.appearance = AppearanceResidualLoss(
            model,
            losses.appearance_magnitude,
            losses.appearance_smoothness,
            losses.appearance_delta,
            samples,
        )
    if "scale" in heads and losses.scale_change > 0:
        regularizers.scale_change = ScaleChangeLoss(model, losses.scale_change, samples)
    if losses.motion_smoothness > 0:
        weights = losses.motion_weights
        regularizers.motion_smoothness = MotionSmoothnessLoss(
            model, losses.motion_smoothness, losses.motion_delta, samples,
            weights.position, weights.rotation, weights.scale,
        )  # fmt: skip
    if model.drift is not None and losses.twist_smoothness > 0:
        regularizers.twist_smoothness = TwistSmoothnessLoss(
            model, losses.twist_smoothness, losses.twist_delta, samples
        )
    if model.drift is not None and losses.drift_magnitude > 0:
        channels = losses.drift_channel_weights
        regularizers.drift_magnitude = DriftMagnitudeLoss(
            model,
            losses.drift_magnitude,
            DriftChannelWeights(
                channels.translation,
                channels.rotation,
                channels.scale,
                channels.opacity,
                channels.sh,
            ),
            extent,
            samples,
            losses.drift_view_chunk,
        )
    if losses.plane_smoothness > 0:
        regularizers.plane_smoothness = PlaneSmoothnessLoss(
            model.periodic.motion_grid, losses.plane_smoothness
        )
    return regularizers


def _refinement(config: Config, device: torch.device) -> RefinementObjective | None:
    refinement = config.train.refinement
    if not refinement.enabled:
        return None
    settings = RefinerSettings(
        stable_diffusion=refinement.stable_diffusion,
        lcm_lora=refinement.lcm_lora,
        inference_steps=refinement.inference_steps,
        rectification_weights=tuple(refinement.rectification_weights),
        prompt=refinement.prompt,
        dtype=refinement.dtype,
    )
    if not Path(settings.lcm_lora).is_file():
        raise FileNotFoundError(f"LCM-LoRA weights not found: {settings.lcm_lora}")
    return RefinementObjective(refinement.weight, lambda: LcmRefiner(settings, device), device)


def _density(
    config: Config, model: CinemagraphModel, optimizer: ScheduledOptimizer, extent: float
) -> DensityController | None:
    density = config.train.densification
    if not density.enabled:
        return None
    settings = DensificationSettings(
        start_iteration=density.start_iteration,
        stop_iteration=density.stop_iteration,
        interval=density.interval,
        gradient_threshold=density.gradient_threshold,
        percent_dense=density.percent_dense,
        min_opacity=density.min_opacity,
        max_prune_fraction=density.max_prune_fraction,
    )
    return DensityController(model.canonical, optimizer.optimizer, settings, extent)


def _restore(setup: TrainingSetup, directory: Path) -> int:
    """Load the resumable state into ``setup``; return the checkpoint's iteration."""
    state = load_training_state(directory)
    setup.optimizer.optimizer.load_state_dict(state["optimizer"])
    if setup.density is not None and state["density"] is not None:
        for name, tensor in state["density"].items():
            setattr(setup.density.state, name, tensor.to(setup.background.device))
    setup.pixel_sampler.rng.setstate(state["pixel_sampler_rng"])
    if setup.refinement_sampler is not None and state["refinement_sampler"] is not None:
        rng_state, remaining = state["refinement_sampler"]
        setup.refinement_sampler.rng.setstate(rng_state)
        setup.refinement_sampler.remaining = list(remaining)
    return int(state["iteration"])


def resolve_checkpoint(run_dir: Path, which: str) -> Path:
    """``latest`` or an iteration number to a checkpoint directory."""
    if which == "latest":
        return latest_checkpoint(run_dir)
    directory = checkpoint_dir(run_dir, int(which))
    if not directory.is_dir():
        raise FileNotFoundError(f"no checkpoint {directory}")
    return directory


def loop_render_settings(config: Config) -> LoopRenderSettings:
    """Loop video settings."""
    render = config.render
    return LoopRenderSettings(
        n_loops=render.n_loops,
        fps=render.fps,
        cycle_seconds=render.cycle_seconds,
        orbit_degrees=(render.orbit_degrees[0], render.orbit_degrees[1]),
        orbit_radius_scale=render.orbit_radius_scale,
        orbit_camera_loops=render.orbit_camera_loops,
    )
