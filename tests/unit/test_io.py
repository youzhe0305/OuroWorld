import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ouroworld.gaussians.canonical import CanonicalGaussians
from ouroworld.geometry.camera import Camera
from ouroworld.io.errors import ArtifactError
from ouroworld.io.images import read_rgb8, save_rgb
from ouroworld.io.multiview import Role, load_multiview, save_multiview_index
from ouroworld.io.ply import read_gaussian_ply, write_gaussian_ply
from ouroworld.io.scene_package import (
    PivotAnnotation,
    load_scene_package,
    load_scene_source,
    save_pivot,
    save_reference,
    save_scene_source,
)
from tests.helpers import look_at_camera, random_gaussians


def _write_videos(root: Path, frames: int = 3, views: int = 3, reference: int = 1) -> None:
    records = []
    for view in range(views):
        for frame in range(frames):
            path = f"frames/view_{view:02d}/frame_{frame:04d}.png"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            save_rgb(torch.rand(3, 36, 64), root / path)
            records.append({"view_id": view, "time": frame / frames, "path": path})
    cameras = {view: look_at_camera() for view in range(views)}
    save_multiview_index(root, cameras, records, reference)


def test_multiview_round_trip_assigns_roles(tmp_path: Path) -> None:
    _write_videos(tmp_path)
    videos = load_multiview(tmp_path)
    assert videos.reference_view_id == 1 and videos.view_ids == [0, 1, 2]
    roles = {(o.view_id, o.time): o.role for o in videos.observations}
    assert roles[(1, 0.0)] is Role.REFERENCE
    assert roles[(0, 0.0)] is Role.T0
    assert roles[(2, 1 / 3)] is Role.GENERATED
    assert np.allclose(videos.cameras[0].c2w, look_at_camera().c2w)


def test_multiview_rejects_inconsistent_role(tmp_path: Path) -> None:
    _write_videos(tmp_path)
    document = json.loads((tmp_path / "observations.json").read_text())
    document["observations"][0]["role"] = "generated"
    (tmp_path / "observations.json").write_text(json.dumps(document))
    with pytest.raises(ArtifactError, match="marked generated"):
        load_multiview(tmp_path)


@pytest.mark.parametrize("degree", [0, 2, 3])
def test_ply_round_trip(tmp_path: Path, degree: int) -> None:
    arrays = random_gaussians(sh_degree=degree)
    write_gaussian_ply(arrays, tmp_path / "g.ply")
    loaded = read_gaussian_ply(tmp_path / "g.ply")
    assert loaded.sh_degree == degree
    for name in ("xyz", "sh_dc", "sh_rest", "opacity", "log_scale", "rotation"):
        assert np.array_equal(getattr(loaded, name), getattr(arrays, name)), name


def test_canonical_pads_sh_degree() -> None:
    arrays = random_gaussians(sh_degree=0)
    canonical = CanonicalGaussians(arrays, sh_degree=3)
    assert canonical.sh.shape == (len(arrays), 16, 3)
    assert torch.count_nonzero(canonical.sh_rest) == 0


def test_scene_package_round_trip(tmp_path: Path) -> None:
    write_gaussian_ply(random_gaussians(), tmp_path / "gaussians.ply")
    camera = look_at_camera()
    camera = Camera(camera.width, camera.height, camera.K, camera.c2w, 0.05, 80.0)
    save_scene_source(tmp_path, (0.0, 0.0, 0.0), camera)
    with pytest.raises(ArtifactError, match="no reference camera"):
        load_scene_package(tmp_path)
    source = load_scene_source(tmp_path)
    assert source.pivot is None and np.allclose(source.world_up, [0.0, -1.0, 0.0])
    pivot = PivotAnnotation(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0]), 2.5, "click")
    save_pivot(tmp_path, pivot)
    reference = camera.resized(32, 18)
    save_reference(tmp_path, reference, np.full((18, 32, 3), 7, np.uint8), "candidate_003")
    package = load_scene_package(tmp_path)
    assert package.background == (0.0, 0.0, 0.0) and np.array_equal(package.pivot, [1.0, 2.0, 3.0])
    loaded = package.reference_camera
    assert np.array_equal(loaded.K, reference.K) and np.array_equal(loaded.c2w, camera.c2w)
    assert (loaded.znear, loaded.zfar) == (0.05, 80.0)
    assert np.all(read_rgb8(package.reference_image_path) == 7)
    save_pivot(tmp_path, pivot)  # a new pivot invalidates the reference view
    with pytest.raises(ArtifactError, match="no reference camera"):
        load_scene_package(tmp_path)
