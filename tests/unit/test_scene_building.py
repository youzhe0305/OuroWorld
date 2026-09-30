"""Scene package building: importers, pivot and reference candidates (CPU)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from ouroworld.cli.scene import import_main
from ouroworld.datasets.importers import axis_camera, import_released
from ouroworld.datasets.world_scale import auto_world_scale, ply_layout, rescale_ply
from ouroworld.geometry.camera import Camera, center_crop_intrinsics
from ouroworld.geometry.trajectories import project_world_point
from ouroworld.io.images import save_rgb8
from ouroworld.io.ply import read_gaussian_ply, write_gaussian_ply
from ouroworld.io.scene_package import PivotAnnotation, load_scene_package, load_scene_source
from ouroworld.reference.candidates import CandidateSettings, candidate_poses
from ouroworld.reference.pivot import center_pivot, pick_pivot
from tests.helpers import look_at_camera, random_gaussians

GRID = CandidateSettings(30.0, 30.0, 5, 5, (0.0, 0.1, 0.5))


def test_centre_crop_keeps_the_principal_point_and_the_cropped_field_of_view() -> None:
    K = np.array([[500.0, 0, 866.0], [0, 500.0, 500.0], [0, 0, 1]])
    cropped = center_crop_intrinsics(K, (1732, 1000), (1024, 576))
    assert np.allclose(cropped[:2, 2], [512.0, 288.0])
    # 1732x1000 is narrower than 16:9, so the full width is kept.
    assert math.isclose(1024 / cropped[0, 0], 1732 / K[0, 0])


def test_axis_camera_matches_hy_world_view0() -> None:
    camera = axis_camera("-x", "+z", np.zeros(3), (1732, 1000), (120.0, 90.0), (0.01, 100.0))
    assert np.allclose(camera.c2w[:3, 2], [-1, 0, 0]) and np.allclose(camera.c2w[:3, 1], [0, 0, -1])
    assert math.isclose(camera.fov_x, math.radians(120.0)) and camera.zfar == 100.0
    with pytest.raises(ValueError, match="perpendicular"):
        axis_camera("-x", "+x", np.zeros(3), (8, 8), (90.0, 90.0), (0.01, 1.0))


def test_world_import_takes_a_camera_file(tmp_path: Path) -> None:
    write_gaussian_ply(random_gaussians(count=10), tmp_path / "in.ply")
    camera = look_at_camera(width=64, height=36)
    record = {"width": 64, "height": 36, "K": camera.K.tolist(), "c2w": camera.c2w.tolist()}
    (tmp_path / "camera.json").write_text(json.dumps(record))
    out = tmp_path / "scene"
    arguments = [
        "world",
        "--ply",
        str(tmp_path / "in.ply"),
        "--camera",
        str(tmp_path / "camera.json"),
    ]
    import_main([*arguments, "--clip", "0.1", "50", "--out", str(out)])
    source = load_scene_source(out)
    assert np.allclose(source.import_camera.c2w, camera.c2w)
    assert np.allclose(source.import_camera.K, camera.K)
    assert (source.import_camera.znear, source.import_camera.zfar) == (0.1, 50.0)
    with pytest.raises(SystemExit):
        import_main([*arguments, "--forward=-z", "--out", str(out)])


def test_pivot_is_the_median_depth_of_the_patch() -> None:
    camera = look_at_camera()
    depth = np.full((camera.height, camera.width), 4.0, np.float32)
    depth[10, 10] = 100.0  # an outlier inside the patch
    pivot = pick_pivot(depth, camera, (10.2, 9.8), patch_radius=2)
    assert pivot.depth == 4.0 and pivot.source == "click"
    pixel, z = project_world_point(camera.K, camera.c2w, pivot.world)
    assert np.allclose(pixel, [10.2, 9.8]) and math.isclose(z, 4.0)
    with pytest.raises(ValueError, match="no rendered surface"):
        pick_pivot(np.zeros_like(depth), camera, (10, 10), 2)
    with pytest.raises(ValueError, match="outside"):
        pick_pivot(depth, camera, (camera.width, 0), 2)
    centre = center_pivot(depth, camera, 0.1)
    assert centre.source == "center_depth" and np.allclose(centre.pixel, [31.5, 17.5])


def test_candidates_keep_the_pivot_at_its_pixel() -> None:
    camera = look_at_camera()
    pivot = PivotAnnotation(np.array([0.3, -0.2, 0.5]), np.zeros(2), 0.0, "click")
    pivot = PivotAnnotation(
        pivot.world, project_world_point(camera.K, camera.c2w, pivot.world)[0], 0.0, "click"
    )
    candidates = candidate_poses(camera, pivot, np.array([0.0, -1.0, 0.0]), GRID)
    assert len(candidates) == 1 + 3 * 25 - 1 and candidates[0].name == "reference"
    assert candidates[1].name == "candidate_001"
    radius = np.linalg.norm(camera.center - pivot.world)
    for candidate in candidates:
        if candidate.forward_fraction == 0.0:
            pixel, _ = project_world_point(camera.K, candidate.c2w, pivot.world)
            assert np.allclose(pixel, pivot.pixel, atol=1e-6)
            assert math.isclose(np.linalg.norm(candidate.c2w[:3, 3] - pivot.world), radius)
            # No roll: image up, the ray to the pivot and the world up are coplanar.
            ray = pivot.world - candidate.c2w[:3, 3]
            assert abs(np.linalg.det(np.stack([-candidate.c2w[:3, 1], ray, [0, -1, 0]]))) < 1e-9
        else:  # moved along its own viewing direction
            start = (
                candidate.c2w[:3, 3] - candidate.c2w[:3, 2] * radius * candidate.forward_fraction
            )
            assert math.isclose(np.linalg.norm(start - pivot.world), radius)


def test_rescaled_ply_moves_positions_and_scales_together(tmp_path: Path) -> None:
    arrays = random_gaussians(count=300)
    write_gaussian_ply(arrays, tmp_path / "in.ply")
    count, properties, _ = ply_layout(tmp_path / "in.ply")
    assert count == 300 and properties[:3] == ["x", "y", "z"]
    rescale_ply(tmp_path / "in.ply", tmp_path / "out.ply", 4.0)
    scaled = read_gaussian_ply(tmp_path / "out.ply")
    assert np.allclose(scaled.xyz, 4.0 * arrays.xyz)
    assert np.allclose(scaled.log_scale, arrays.log_scale + math.log(4.0), atol=1e-6)
    assert np.array_equal(scaled.sh_dc, arrays.sh_dc)


def test_auto_world_scale_pushes_the_cull_behind_the_near_surfaces() -> None:
    camera = look_at_camera(width=64, height=64)
    rng = np.random.default_rng(0)
    points = np.column_stack([rng.uniform(-0.2, 0.2, (5000, 2)), rng.uniform(0.5, 1.0, 5000)])
    points[:, 2] -= 4.0  # 0.5 .. 1.0 in front of the camera at z = -4
    scale, near = auto_world_scale(points, [(camera.w2c(), camera.K, 64, 64)], 0.05)
    assert 0.5 < near < 0.6 and scale == math.ceil(0.2 / (0.05 * near))
    # The same scene 1000 times larger needs no rescaling.
    far_c2w = camera.c2w.copy()
    far_c2w[:3, 3] *= 1000.0
    far_w2c = np.linalg.inv(far_c2w)
    assert auto_world_scale(points * 1000.0, [(far_w2c, camera.K, 64, 64)], 0.05)[0] == 1.0


def test_released_scene_imports_with_its_pivot_and_reference(tmp_path: Path) -> None:
    source = tmp_path / "released"
    source.mkdir()
    write_gaussian_ply(random_gaussians(count=10), source / "point_cloud.ply")
    camera = look_at_camera(width=64, height=36)
    metadata = {
        "gaussians": {"background": [0.0, 0.0, 0.0]},
        "rendering": {"near": 0.1, "far": 50.0, "world_up": None},
    }
    (source / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    record = {"width": 64, "height": 36, "K": camera.K.tolist(), "c2w": camera.c2w.tolist()}
    record["source_candidate"] = "candidate_003"
    (source / "reference_camera.json").write_text(json.dumps(record))
    save_rgb8(np.full((36, 64, 3), 7, np.uint8), source / "reference.png")
    pixel = np.array([40.0, 10.0])
    world = camera.center + 2.0 * normalize_ray(camera, pixel)
    (source / "pivot.json").write_text(json.dumps({"world_xyz": world.tolist()}))

    assert import_released(source, tmp_path / "scene") == "candidate_003"
    package = load_scene_package(tmp_path / "scene")
    pivot = load_scene_source(tmp_path / "scene").pivot
    assert np.allclose(package.reference_camera.c2w, camera.c2w)
    assert (package.reference_camera.znear, package.reference_camera.zfar) == (0.1, 50.0)
    assert package.pivot is not None and np.allclose(package.pivot, world)
    assert pivot is not None and np.allclose(pivot.pixel, pixel)
    assert package.reference_image_path.read_bytes() == (source / "reference.png").read_bytes()


def normalize_ray(camera: Camera, pixel: np.ndarray) -> np.ndarray:
    direction = np.linalg.inv(camera.K) @ np.append(pixel, 1.0)
    return camera.c2w[:3, :3] @ (direction / direction[2])
