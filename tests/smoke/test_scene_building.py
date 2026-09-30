"""Import -> pivot fallback -> reference view on a synthetic world."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ouroworld.datasets.importers import axis_camera, import_generated_world
from ouroworld.io.ply import write_gaussian_ply
from ouroworld.io.scene_package import load_scene_source
from ouroworld.reference.candidates import CandidateSettings
from ouroworld.reference.reference_view import ReferenceSettings, choose_reference
from ouroworld.render.static import load_static_renderer
from tests.helpers import random_gaussians

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_reference_view_from_a_fresh_import(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    arrays = random_gaussians(count=4000, seed=3)
    arrays.xyz[:, 2] += 5.0  # in front of a camera at the origin looking down +z
    arrays.log_scale[:] = -2.0
    arrays.opacity[:] = 3.0
    write_gaussian_ply(arrays, tmp_path / "raw.ply")
    camera = axis_camera("+z", "-y", np.zeros(3), (160, 90), (60.0, 36.0), (0.01, 100.0))
    import_generated_world(tmp_path / "raw.ply", tmp_path / "scene", camera)
    source = load_scene_source(tmp_path / "scene")
    settings = ReferenceSettings(
        "candidate_003", 64, 36, 0.05, CandidateSettings(10.0, 10.0, 3, 3, (0.0, 0.2)), 40, 0.5
    )
    renderer = load_static_renderer(source.gaussians_path, source.background)
    package = choose_reference(source, renderer, settings, tmp_path / "candidates")
    assert (
        package.pivot is not None
        and load_scene_source(tmp_path / "scene").pivot.source == "center_depth"
    )
    records = json.loads((tmp_path / "candidates" / "candidates.json").read_text())
    assert len(records) == 1 + 2 * 9 - 1 and (tmp_path / "candidates" / "candidates.png").is_file()
    chosen = next(record for record in records if record["name"] == "candidate_003")
    assert np.allclose(package.reference_camera.c2w, chosen["c2w"])
    assert package.reference_camera.width == 64 and package.reference_image_path.is_file()
