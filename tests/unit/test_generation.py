import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ouroworld.generation.inpainting.direct_prefix import assemble_condition
from ouroworld.generation.inpainting.fusion import FusionSettings, fuse
from ouroworld.generation.inpainting.warp import forward_warp, grow_holes
from ouroworld.generation.lifting.alignment import (
    AlignmentSettings,
    DisparityAffine,
    confident_pixels,
    fit_disparity_affine,
)
from ouroworld.generation.lifting.vggt_omega import VggtOmegaLifter
from ouroworld.geometry.trajectories import pivot_orbit, pivot_sweep, uniform_yaw_angles
from ouroworld.io.errors import ArtifactError
from ouroworld.io.reference_video import load_reference_video, save_reference_video_index
from tests.helpers import look_at_camera

PIVOT = np.array([0.2, -0.1, 0.5])


def test_yaw_angles_are_symmetric_and_centred() -> None:
    angles = uniform_yaw_angles(21, 20.0)
    assert angles[10] == 0.0 and np.allclose(np.diff(angles), 2.0)
    with pytest.raises(ValueError, match="odd"):
        uniform_yaw_angles(20, 20.0)


def test_orbit_turns_about_the_pivot() -> None:
    reference = look_at_camera().c2w
    poses = pivot_orbit(reference, PIVOT, uniform_yaw_angles(5, 20.0), 5.0)
    assert np.allclose(poses[2], reference, atol=1e-12)
    radius = np.linalg.norm(reference[:3, 3] - PIVOT)
    for pose in poses:
        assert np.isclose(np.linalg.norm(pose[:3, 3] - PIVOT), radius)
        # The pivot keeps its position in every camera: rotation about an axis through it.
        seen = pose[:3, :3].T @ (PIVOT - pose[:3, 3])
        assert np.allclose(seen, reference[:3, :3].T @ (PIVOT - reference[:3, 3]))


def test_sweep_ends_on_its_inputs_and_keeps_the_pivot_distance() -> None:
    reference = look_at_camera().c2w
    orbit = pivot_orbit(reference, PIVOT, np.array([0.0, 20.0]), 5.0)
    sweep = pivot_sweep(orbit[0], orbit[1], PIVOT, 24)
    assert np.array_equal(sweep[0], orbit[0]) and np.array_equal(sweep[-1], orbit[1])
    radius = np.linalg.norm(reference[:3, 3] - PIVOT)
    assert np.allclose(np.linalg.norm(sweep[:, :3, 3] - PIVOT, axis=1), radius)


def _write_video(root: Path, count: int) -> None:
    records = []
    (root / "frames").mkdir(parents=True)
    for index in range(count):
        name = f"frames/frame_{index:04d}.png"
        Image.new("RGB", (8, 4)).save(root / name)
        records.append({"path": name, "time": index / (count - 1)})
    save_reference_video_index(root, records, 10.0)


def test_reference_video_subsampling(tmp_path: Path) -> None:
    _write_video(tmp_path, 241)
    video = load_reference_video(tmp_path)
    indices = video.evenly_spaced(25)
    assert indices[0] == 0 and indices[-1] == 240 and np.allclose(np.diff(indices), 10)
    with pytest.raises(ArtifactError, match="equal steps"):
        video.evenly_spaced(26)


def test_reference_video_must_close_the_loop(tmp_path: Path) -> None:
    _write_video(tmp_path, 5)
    document = json.loads((tmp_path / "reference_video.json").read_text())
    document["frames"][-1]["time"] = 0.9
    (tmp_path / "reference_video.json").write_text(json.dumps(document))
    with pytest.raises(ArtifactError, match="from 0 to 1"):
        load_reference_video(tmp_path)


def test_disparity_fit_ignores_outlier_regions() -> None:
    generator = torch.Generator().manual_seed(0)
    predicted = 1.0 + 4.0 * torch.rand(64, 64, generator=generator)
    target = 1.0 / (0.5 / predicted + 0.05)
    target[:8] *= 3.0  # a wrong region, as at depth discontinuities
    affine = fit_disparity_affine(target, predicted, AlignmentSettings(8, 1.345, 100))
    assert abs(affine.scale - 0.5) < 1e-6 and abs(affine.shift - 0.05) < 1e-6
    assert torch.allclose(affine.apply(predicted)[8:], target[8:], rtol=1e-5)


def test_negative_disparity_maps_to_invalid_depth() -> None:
    depth = DisparityAffine(1.0, -0.5, 0.0, 1.0, 1).apply(torch.tensor([1.0, 4.0]))
    assert depth[0] == 2.0 and depth[1] == 0.0


def test_confidence_drop_removes_the_lowest_percentile() -> None:
    depth = torch.ones(1, 10, 10)
    confidence = torch.arange(100.0).reshape(1, 10, 10)
    valid = confident_pixels(depth, confidence, 10.0)
    assert int(valid.sum()) == 90  # threshold 9.9 (linear interpolation)


def test_warp_with_identical_cameras_is_identity() -> None:
    image = torch.rand(1, 3, 12, 16) * 2 - 1
    depth = torch.full((1, 1, 12, 16), 2.0)
    K = torch.tensor([[[20.0, 0, 8], [0, 20.0, 6], [0, 0, 1]]])
    pose = torch.eye(4)[None]
    warped, known = forward_warp(image, torch.ones(1, 1, 12, 16), depth, pose, pose, K)
    assert torch.all(known == 1)
    assert torch.allclose(warped, image, atol=1e-6)


def test_hole_growth_blanks_the_border_of_holes() -> None:
    known = torch.ones(1, 1, 9, 9)
    known[..., 4, 4] = 0
    image, grown = grow_holes(torch.zeros(1, 3, 9, 9), known)
    assert grown.dtype == torch.float64 and int((grown == 0).sum()) == 25
    assert torch.all(image[..., 2:7, 2:7] == 0) and torch.all(image[..., 0, 0] == 0.5)


def test_fusion_keeps_generation_in_holes_and_warp_inside() -> None:
    settings = FusionSettings(0.5, 1.0, 1, 2, 1.0)
    known = torch.zeros(1, 1, 20, 20)
    known[..., :, :10] = 1
    warped, generated = torch.ones(1, 3, 20, 20), torch.zeros(1, 3, 20, 20)
    fused = fuse(generated, warped, known, settings)
    assert torch.all(fused[..., 10:] == 0)
    assert torch.allclose(fused[..., :5], torch.ones(1, 3, 20, 5))
    assert torch.all(fused[..., 8] < 1)  # feathered towards the hole


def test_condition_prefix_is_fully_known() -> None:
    prefix = [np.full((12, 16, 3), 255, np.uint8)] * 4
    warped = torch.rand(5, 3, 12, 16, dtype=torch.float64)
    known = (torch.rand(5, 1, 12, 16) > 0.5).double()
    condition = assemble_condition(prefix, warped, known, 6, 8)
    assert condition.frames.shape == (9, 3, 6, 8) and condition.frames.dtype == torch.float64
    assert torch.all(condition.known[:4] == 1) and torch.all(condition.frames[:4] == 1)


def test_lifter_resizes_to_patch_multiples() -> None:
    lifter = VggtOmegaLifter(Path("unused.pt"), 512)
    assert lifter.preprocess(np.zeros((576, 1024, 3), np.uint8)).shape == (3, 288, 512)
    assert lifter.preprocess(np.zeros((1024, 768, 3), np.uint8)).shape == (3, 512, 384)
    with pytest.raises(ValueError, match="aspect"):
        lifter.preprocess(np.zeros((100, 300, 3), np.uint8))
