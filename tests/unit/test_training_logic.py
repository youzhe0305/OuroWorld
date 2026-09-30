import random
from pathlib import Path

import pytest
import torch

from ouroworld.cli.common import DEFAULT_CONFIG, REPOSITORY_ROOT
from ouroworld.config.loader import ConfigError, load_config
from ouroworld.gaussians.canonical import CanonicalGaussians
from ouroworld.gaussians.densification import ParameterSurgery
from ouroworld.io.multiview import Observation, Role
from ouroworld.training.losses import PlaneSmoothnessLoss
from ouroworld.training.sampler import ShuffledCycleSampler, ViewBalancedSampler
from ouroworld.training.schedules import ExponentialDecay, phase_schedule
from tests.helpers import random_gaussians, small_model

SCENE = ["dataset=Test", "scene=test"]


def test_default_config_loads_and_resolves_paths() -> None:
    config = load_config([DEFAULT_CONFIG], SCENE)
    assert config.paths.run_dir == "output/Test/test/model"
    assert config.train.refinement.weight == 0.1  # λ_p
    assert config.model.periodic.heads == ["pos", "rot", "scale", "shs"]


@pytest.mark.parametrize(
    "overrides",
    [
        ["train.supervision=bogus"],
        ["train.supervision=naive_l1"],
        ["train.refinement.start_after=13000"],
    ],
)
def test_invalid_configs_fail_at_load(overrides: list[str]) -> None:
    with pytest.raises(ConfigError):
        load_config([DEFAULT_CONFIG], SCENE + overrides)


@pytest.mark.parametrize(
    "ablation", sorted((REPOSITORY_ROOT / "configs" / "ablations").glob("*.yaml"))
)
def test_ablations_load(ablation: Path) -> None:
    load_config([DEFAULT_CONFIG, ablation], SCENE)


def test_exponential_decay_endpoints() -> None:
    decay = ExponentialDecay(1e-3, 1e-5, 100)
    assert decay(0) == pytest.approx(1e-3) and decay(100) == pytest.approx(1e-5)
    assert decay(1000) == pytest.approx(1e-5) and decay(50) == pytest.approx(1e-4)


def test_phase_schedule_paper_setting() -> None:
    schedule = phase_schedule(13000, 12000, True, drift_waits_for_refinement=True, drift_warmup=500)
    assert not schedule.is_refining(12000) and schedule.is_refining(12001)
    assert not schedule.is_drift_active(12000) and schedule.is_drift_active(12001)
    ablation = phase_schedule(13000, None, True, drift_waits_for_refinement=False, drift_warmup=500)
    assert ablation.refinement_start is None and ablation.drift_start == 501


def _observations(views: int, frames: int) -> list[Observation]:
    return [
        Observation(view, frame / frames, Path(f"{view}_{frame}.png"), Role.GENERATED)
        for view in range(views)
        for frame in range(frames)
    ]


def test_view_balanced_sampler_honours_reference_share() -> None:
    sampler = ViewBalancedSampler(_observations(5, 3), 2, 0.5, random.Random(0))
    draws = sampler.sample(20000)
    share = sum(o.view_id == 2 for o in draws) / len(draws)
    assert share == pytest.approx(0.5, abs=0.02)
    assert {o.view_id for o in draws} == set(range(5))


def test_shuffled_cycle_uses_everything_once_per_cycle() -> None:
    pool = _observations(3, 4)
    sampler = ShuffledCycleSampler(pool, random.Random(0))
    assert sorted(sampler.sample(len(pool)), key=str) == sorted(pool, key=str)


def test_plane_smoothness_vanishes_on_linear_planes() -> None:
    model = small_model(with_drift=False)
    grid = model.periodic.motion_grid
    with torch.no_grad():
        for plane in grid.planes():
            rows = torch.arange(plane.shape[-2], dtype=plane.dtype).view(-1, 1)
            plane.copy_(rows.expand_as(plane[0, 0]).expand_as(plane))
    assert float(PlaneSmoothnessLoss(grid, 1.0)(0.0)) == pytest.approx(0.0, abs=1e-10)


def test_parameter_surgery_keeps_adam_state_and_freeze() -> None:
    canonical = CanonicalGaussians(random_gaussians(count=10), sh_degree=3, frozen=("sh",))
    optimizer = torch.optim.Adam(
        [
            {"params": [canonical.parameter_by_name(name)], "name": name}
            for name in ("xyz", "sh_dc", "sh_rest", "opacity", "log_scale", "rotation")
        ]  # fmt: skip
    )
    canonical.xyz.sum().backward()
    optimizer.step()
    surgery = ParameterSurgery(canonical, optimizer)
    keep = torch.ones(10, dtype=torch.bool)
    keep[3] = False
    surgery.keep_rows(keep)
    names = ("xyz", "sh_dc", "sh_rest", "opacity", "log_scale", "rotation")
    surgery.append_rows({name: canonical.parameter_by_name(name).detach()[:2] for name in names})
    assert len(canonical) == 11
    assert optimizer.state[canonical.xyz]["exp_avg"].shape == (11, 3)
    assert not canonical.sh_dc.requires_grad and canonical.xyz.requires_grad
    assert optimizer.param_groups[0]["params"][0] is canonical.xyz
