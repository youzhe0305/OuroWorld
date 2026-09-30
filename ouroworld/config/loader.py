"""Load, merge, validate and snapshot configuration files.

Merge order: the schema, then each YAML file in the given order, then
``key=value`` overrides. Missing values or wrong types fail at load time.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ouroworld.config.schema import Config, EvaluationConfig

SUPERVISION_MODES = ("svco", "naive_l1")
FREEZABLE = ("xyz", "opacity", "scale", "rotation", "sh")


class ConfigError(ValueError):
    """The configuration is incomplete or inconsistent."""


def load_config(files: Sequence[Path], overrides: Sequence[str] = ()) -> Config:
    """Merge ``files`` and ``overrides`` onto the schema and validate the result."""
    try:
        merged = OmegaConf.merge(
            OmegaConf.structured(Config),
            *(OmegaConf.load(Path(path)) for path in files),
            OmegaConf.from_dotlist(list(overrides)),
        )
        OmegaConf.resolve(merged)
        config = OmegaConf.to_object(merged)
    except OmegaConfBaseException as error:
        raise ConfigError(str(error)) from error
    assert isinstance(config, Config)
    validate(config)
    return config


def load_evaluation_config(
    files: Sequence[Path], overrides: Sequence[str] = ()
) -> EvaluationConfig:
    """Merge the evaluation ``files`` and ``overrides`` onto their schema."""
    try:
        merged = OmegaConf.merge(
            OmegaConf.structured(EvaluationConfig),
            *(OmegaConf.load(Path(path)) for path in files),
            OmegaConf.from_dotlist(list(overrides)),
        )
        OmegaConf.resolve(merged)
        config = OmegaConf.to_object(merged)
    except OmegaConfBaseException as error:
        raise ConfigError(str(error)) from error
    assert isinstance(config, EvaluationConfig)
    if len(config.views) != 2:
        raise ConfigError("views must name the static and the orbit render")
    return config


def save_config(config: Config | EvaluationConfig, path: Path) -> None:
    """Write the resolved configuration of a run."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(config), Path(path))


def validate(config: Config) -> None:
    """Reject combinations the method does not define."""
    train = config.train
    problems = []
    if train.supervision not in SUPERVISION_MODES:
        problems.append(f"train.supervision must be one of {SUPERVISION_MODES}")
    if train.supervision == "naive_l1" and train.refinement.enabled:
        problems.append(
            "train.supervision=naive_l1 supervises every frame with L1; disable refinement"
        )
    if train.supervision == "svco" and not train.refinement.enabled:
        problems.append("train.supervision=svco needs refinement for the generated frames")
    if train.refinement.enabled and not 0 <= train.refinement.start_after < train.iterations:
        problems.append("train.refinement.start_after must lie in [0, train.iterations)")
    if not 0.0 <= train.reference_view_probability <= 1.0:
        problems.append("train.reference_view_probability must lie in [0, 1]")
    unknown = set(config.model.frozen) - set(FREEZABLE)
    if unknown:
        problems.append(f"model.frozen has unknown entries {sorted(unknown)}; valid: {FREEZABLE}")
    if train.densification.enabled and set(config.model.frozen) & {
        "xyz",
        "opacity",
        "scale",
        "rotation",
    }:
        problems.append("densification cannot be combined with frozen geometry")
    reference_video = config.reference_video
    if reference_video.backend not in ("seedance", "wan"):
        problems.append("reference_video.backend must be seedance or wan")
    if (reference_video.frame_count - 1) % (config.generation.frame_count - 1):
        problems.append(
            "reference_video.frame_count - 1 must be a multiple of generation.frame_count - 1"
        )
    if reference_video.wan.memory_mode not in ("model", "sequential", "cuda"):
        problems.append("reference_video.wan.memory_mode must be model, sequential or cuda")
    generation = config.generation
    if generation.orbit.view_count < 3 or generation.orbit.view_count % 2 == 0:
        problems.append("generation.orbit.view_count must be odd and >= 3")
    frames = generation.frame_count + generation.inpainting.sweep_frames
    if frames > 49 or (frames - 1) % 4:
        problems.append(
            "generation.frame_count + generation.inpainting.sweep_frames must be 4k + 1 <= 49"
        )
    if generation.inpainting.cpu_offload not in ("model", "sequential", "none"):
        problems.append("generation.inpainting.cpu_offload must be model, sequential or none")
    if len(train.learning_rates.xyz) != 2 or len(config.render.orbit_degrees) != 2:
        problems.append("learning-rate ranges and render.orbit_degrees need exactly two values")
    if problems:
        raise ConfigError("; ".join(problems))
