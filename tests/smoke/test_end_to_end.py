"""Tiny end-to-end runs: train, resume and render on a synthetic scene."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ouroworld.cli import render as render_cli
from ouroworld.cli import train as train_cli
from ouroworld.cli.build import build_training
from ouroworld.cli.common import DEFAULT_CONFIG
from ouroworld.config.loader import load_config
from ouroworld.fields.cinemagraph import CinemagraphModel, ViewContext
from ouroworld.fields.heads import ChunkedEvaluator
from ouroworld.fields.periodic import PeriodicDeformationField
from ouroworld.gaussians.canonical import CanonicalGaussians
from ouroworld.geometry.camera import Camera
from ouroworld.geometry.trajectories import axis_angle_rotation
from ouroworld.io.images import save_rgb
from ouroworld.io.multiview import save_multiview_index
from ouroworld.io.ply import write_gaussian_ply
from ouroworld.render.rasterizer import RasterView, render
from ouroworld.training.refinement.objective import RefinementObjective
from ouroworld.training.trainer import Trainer
from tests.helpers import (
    look_at_camera,
    periodic_spec,
    random_gaussians,
    randomize_heads,
    write_scene_metadata,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

VIEWS, FRAMES, WIDTH, HEIGHT = 3, 5, 64, 36


@pytest.fixture(scope="module")
def scene_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A random static scene and its 'videos': renders of a randomly moving copy."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    root = tmp_path_factory.mktemp("output") / "Toy" / "toy"
    arrays = random_gaussians(count=2000, seed=1)
    write_gaussian_ply(arrays, root / "scene" / "gaussians.ply")
    write_scene_metadata(root / "scene", look_at_camera(), np.zeros(3))

    truth = CinemagraphModel(
        CanonicalGaussians(arrays, 3),
        PeriodicDeformationField(periodic_spec(), ChunkedEvaluator(0, False)),
        None,
    )
    truth.fit_fields_to_canonical()
    randomize_heads(truth.periodic, scale=0.3)
    truth = truth.cuda()
    cameras, records = {}, []
    for view in range(VIEWS):
        c2w = np.eye(4)
        c2w[:3, :3] = axis_angle_rotation(np.array([0.0, 1.0, 0.0]), np.radians(10 * (view - 1)))
        c2w[:3, 3] = c2w[:3, :3] @ np.array([0.0, 0.0, -4.0])
        K = np.array([[0.8 * WIDTH, 0, WIDTH / 2], [0, 0.8 * WIDTH, HEIGHT / 2], [0, 0, 1.0]])
        cameras[view] = Camera(WIDTH, HEIGHT, K, c2w)
        for frame in range(FRAMES):
            time = frame / FRAMES
            with torch.no_grad():
                image = render(
                    truth.deformed(ViewContext(time)),
                    RasterView.from_camera(cameras[view]),
                    torch.zeros(3, device="cuda"),
                    3,
                ).image
            path = f"frames/view_{view:02d}/frame_{frame:04d}.png"
            (root / "multiview" / path).parent.mkdir(parents=True, exist_ok=True)
            save_rgb(image, root / "multiview" / path)
            records.append({"view_id": view, "time": time, "path": path})
    save_multiview_index(root / "multiview", cameras, records, reference_view_id=1)
    return root


def _overrides(root: Path, run: str) -> list[str]:
    return [
        "dataset=Toy",
        "scene=toy",
        f"paths.output_root={root.parent.parent}",
        f"paths.run_dir={root / run}",
        f"paths.render_dir={root / run}_render",
        "model.point_chunk_size=512",
        "model.periodic.motion_grid.resolution=[8,8,8]",
        "model.periodic.appearance_grid.resolution=[8,8,8]",
        "model.periodic.appearance_grid.multires=[1]",
        "model.drift.grid.resolution=[8,8,8]",
        "train.log_interval=10",
        "train.losses.sample_gaussians=256",
        "render.n_loops=1",
        "render.fps=4",
        "render.cycle_seconds=2.0",
    ]


def _metrics(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def test_naive_l1_train_resume_render(scene_root: Path) -> None:
    overrides = _overrides(scene_root, "naive") + [
        "train.supervision=naive_l1",
        "train.refinement.enabled=false",
        "train.iterations=60",
        "train.checkpoint_iterations=[30]",
        "train.densification.enabled=true",
        "train.densification.start_iteration=10",
        "train.densification.interval=10",
        "train.densification.stop_iteration=50",
        "model.drift.warmup_iterations=20",
    ]
    train_cli.main(["--config", str(DEFAULT_CONFIG), *overrides])
    run_dir = scene_root / "naive"
    metrics = _metrics(run_dir)
    assert metrics[-1]["iteration"] == 60 and metrics[-1]["drift_active"]
    assert metrics[-1]["gaussians"] != metrics[0]["gaussians"]  # densification ran
    assert (run_dir / "checkpoints" / "iter_00030" / "fields.pt").is_file()

    train_cli.main(["--resume", str(run_dir), "train.iterations=80"])
    assert _metrics(run_dir)[-1]["iteration"] == 80

    render_cli.main(overrides)
    out = scene_root / "naive_render" / "iter_00080"
    assert (out / "static.mp4").stat().st_size > 0 and (out / "orbit.mp4").stat().st_size > 0
    report = json.loads((out / "loop_report.json").read_text())
    assert report["seam"]["psnr_t0_vs_t1"] > 60


def _mean_l1(setup) -> float:  # noqa: ANN001
    total = 0.0
    with torch.no_grad():
        for observation in setup.data.videos.observations:
            image = render(
                setup.model.deformed(ViewContext(observation.time)),
                setup.data.view(observation),
                setup.background,
                setup.model.canonical.sh_degree,
            ).image
            total += float((image - setup.data.image(observation)).abs().mean())
    return total / len(setup.data.videos.observations)


def test_fitting_improves_without_densification(scene_root: Path) -> None:
    config = load_config(
        [DEFAULT_CONFIG],
        _overrides(scene_root, "fit")
        + ["train.supervision=naive_l1", "train.refinement.enabled=false", "train.iterations=300"],
    )
    run = build_training(config)
    before = _mean_l1(run.setup)
    Trainer(run.setup).run(run.start_iteration)
    assert _mean_l1(run.setup) < 0.8 * before


class _IdentityRefiner:
    def refine(self, renders: torch.Tensor, guides: torch.Tensor) -> torch.Tensor:
        return guides


def test_svco_reaches_refinement_with_drift(scene_root: Path) -> None:
    config = load_config(
        [DEFAULT_CONFIG],
        _overrides(scene_root, "svco") + ["train.iterations=40", "train.refinement.start_after=20"],
    )
    run = build_training(config)
    setup = run.setup
    setup.refinement = RefinementObjective(0.1, _IdentityRefiner, setup.background.device)
    Trainer(setup).run(run.start_iteration)
    metrics = _metrics(Path(config.paths.run_dir))
    refining = [m for m in metrics if m["iteration"] > 20]
    assert refining and all("refinement" in m and "drift_magnitude" in m for m in refining)
    assert all(not m["drift_active"] for m in metrics if m["iteration"] <= 20)
