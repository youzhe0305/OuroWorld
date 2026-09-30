"""Small builders shared by the tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ouroworld.fields.cinemagraph import CinemagraphModel
from ouroworld.fields.drift import DriftFieldSpec, GroundedDriftField
from ouroworld.fields.encoding import TriplaneSpec
from ouroworld.fields.heads import ChunkedEvaluator
from ouroworld.fields.periodic import PeriodicDeformationField, PeriodicFieldSpec
from ouroworld.gaussians.canonical import CanonicalGaussians
from ouroworld.geometry.camera import Camera
from ouroworld.io.ply import GaussianArrays
from ouroworld.io.scene_package import (
    PivotAnnotation,
    save_pivot,
    save_reference,
    save_scene_source,
)

SMALL_GRID = TriplaneSpec((8, 8, 8), (1, 2), 4)


def random_gaussians(count: int = 64, sh_degree: int = 3, seed: int = 0) -> GaussianArrays:
    rng = np.random.default_rng(seed)
    rest = (sh_degree + 1) ** 2 - 1
    return GaussianArrays(
        xyz=rng.uniform(-1, 1, (count, 3)).astype(np.float32),
        sh_dc=rng.normal(0, 0.5, (count, 1, 3)).astype(np.float32),
        sh_rest=rng.normal(0, 0.1, (count, rest, 3)).astype(np.float32),
        opacity=rng.normal(0, 1, (count, 1)).astype(np.float32),
        log_scale=rng.uniform(-4, -2, (count, 3)).astype(np.float32),
        rotation=rng.normal(0, 1, (count, 4)).astype(np.float32),
    )


def periodic_spec(
    heads: tuple[str, ...] = ("pos", "rot", "scale", "shs"), period: float = 1.0
) -> PeriodicFieldSpec:
    return PeriodicFieldSpec(
        heads=heads,
        harmonics=2,
        period=period,
        motion_grid=SMALL_GRID,
        appearance_grid=SMALL_GRID,
        width=16,
        max_log_scale_delta=0.693,
        sh_degree=3,
    )


def drift_spec(
    heads: tuple[str, ...] = ("pos", "rot", "scale", "shs"),
    zero_reference_view: bool = True,
    zero_t0: bool = True,
) -> DriftFieldSpec:
    return DriftFieldSpec(
        view_count=4,
        reference_view_id=1,
        heads=heads,
        harmonics=2,
        period=2.0,
        grid=SMALL_GRID,
        width=16,
        view_embedding_dim=4,
        zero_reference_view=zero_reference_view,
        zero_t0=zero_t0,
        sh_degree=3,
    )


def randomize_heads(module: torch.nn.Module, seed: int = 0, scale: float = 0.1) -> None:
    """Zero-initialised heads make every field the identity; perturb them for testing."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(scale * torch.randn(parameter.shape, generator=generator))


def small_model(with_drift: bool = True, **drift_options: bool) -> CinemagraphModel:
    evaluator = ChunkedEvaluator(chunk_size=16, use_checkpointing=False)
    canonical = CanonicalGaussians(random_gaussians(), sh_degree=3)
    periodic = PeriodicDeformationField(periodic_spec(), evaluator)
    drift = GroundedDriftField(drift_spec(**drift_options), evaluator) if with_drift else None
    model = CinemagraphModel(canonical, periodic, drift)
    model.fit_fields_to_canonical()
    randomize_heads(periodic)
    if drift is not None:
        randomize_heads(drift, seed=1)
    return model


def look_at_camera(width: int = 64, height: int = 36, distance: float = 4.0) -> Camera:
    focal = 0.8 * width
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    c2w = np.eye(4)
    c2w[2, 3] = -distance  # camera on -z looking along +z towards the origin
    return Camera(width, height, K, c2w)


def write_scene_metadata(root: Path, camera: Camera, pivot: np.ndarray | None) -> None:
    """A complete package around an existing PLY: imported at, and referenced from, ``camera``."""
    save_scene_source(root, (0.0, 0.0, 0.0), camera)
    if pivot is not None:
        save_pivot(root, PivotAnnotation(np.asarray(pivot, float), np.zeros(2), 1.0, "click"))
    save_reference(root, camera, np.zeros((camera.height, camera.width, 3), np.uint8), "reference")
