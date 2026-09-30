"""Evaluation metrics on synthetic inputs (CPU)."""

import json
from pathlib import Path

import numpy as np
import pytest

from ouroworld.evaluation.loop_seam import (
    LoopSeamSettings,
    loop_seam_metrics,
    ring_lags,
    sample_times,
)
from ouroworld.evaluation.naturalness import (
    NaturalnessSettings,
    clip_frame_indices,
    clip_start_times,
    kvd,
    polynomial_mmd2,
)
from ouroworld.evaluation.sharpness import variance_of_laplacian
from ouroworld.evaluation.tables import TABLE2, build_table, markdown, write_result
from ouroworld.evaluation.vividness import illumination_mask, motion_mask
from ouroworld.io.video import write_mp4


def _texture(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (48, 64, 3), dtype=np.uint8)


def test_sampling_is_tied_to_the_period() -> None:
    assert sample_times(300, 600, 10) == list(range(0, 600, 30))
    assert sample_times(15, 31, 10) == [
        0,
        2,
        3,
        4,
        6,
        8,
        9,
        10,
        12,
        14,
        15,
        16,
        18,
        20,
        21,
        22,
        24,
        26,
        27,
        28,
        30,
    ]
    lags = ring_lags(300, 600, 16)
    assert len(lags) == 16 and all(150 <= lag <= 270 or 330 <= lag <= 450 for lag in lags)


def test_a_repeating_video_scores_high_and_a_still_one_zero(tmp_path: Path) -> None:
    cycle = [_texture(seed) for seed in range(8)]
    write_mp4(tmp_path / "loop.mp4", cycle * 4, fps=8, crf=0)
    loop = loop_seam_metrics(tmp_path / "loop.mp4", 8, LoopSeamSettings(4, 16))
    assert loop["seam_count"] == 3 and loop["malf"] > 0.5
    write_mp4(tmp_path / "still.mp4", [cycle[0]] * 32, fps=8, crf=0)
    still = loop_seam_metrics(tmp_path / "still.mp4", 8, LoopSeamSettings(4, 16))
    assert still["seam_ssim"] > 0.99 and abs(still["malf"]) < 1e-6


def test_blur_lowers_the_variance_of_the_laplacian() -> None:
    import cv2

    sharp = _texture(0)
    assert variance_of_laplacian(cv2.GaussianBlur(sharp, (9, 9), 3)) < 0.1 * variance_of_laplacian(
        sharp
    )


def test_vividness_masks() -> None:
    flow = np.zeros((2, 4, 4), np.float32)
    flow[0, :2] = 2.0
    assert motion_mask(flow, 1.0).mean() == 0.5
    previous = np.full((4, 4), 0.5, np.float32)
    current = previous.copy()
    current[0, 0] = 0.7  # +40%
    changed, valid = illumination_mask(previous, current, np.zeros((2, 4, 4)), 0.2, 1e-6)
    assert changed.sum() == 1 and changed[0, 0] and valid.all()


def test_kvd_separates_distributions_and_is_seeded() -> None:
    rng = np.random.default_rng(0)
    first, second = rng.normal(0, 1, (300, 400)), rng.normal(0, 1, (300, 400))
    shifted = rng.normal(0.5, 1, (300, 400))
    assert (
        abs(polynomial_mmd2(first, second, 3, 1.0)) < 0.2 < polynomial_mmd2(first, shifted, 3, 1.0)
    )
    settings = NaturalnessSettings(subset_size=50, subset_count=5)
    assert kvd(first, shifted, settings) == kvd(first, shifted, settings)


def test_clips_resample_by_time() -> None:
    settings = NaturalnessSettings()
    assert clip_start_times(20.0, settings) == [0.0, 10.0] and clip_start_times(9.9, settings) == []
    indices = clip_frame_indices(0.0, 30.0, 600, settings)
    assert indices[:3] == [0, 5, 10] and indices[-1] == 295
    with pytest.raises(ValueError, match="I3D"):
        NaturalnessSettings(frame_size=256)


def test_table_cells_average_the_scenes(tmp_path: Path) -> None:
    for group, values in (("ours", [0.2, 0.4]), ("base", [0.1, 0.1])):
        videos = [{"vividness": value, "malf": value, "seam_ssim": 1.0} for value in values]
        write_result(tmp_path / group / "static" / "vividness.json", {"videos": videos})
        write_result(tmp_path / group / "static" / "loop_seam.json", {"videos": videos})
        (tmp_path / group / "static" / "naturalness.json").write_text(json.dumps({"kvd": 10.0}))
    records = build_table(
        tmp_path, [("base", "Base"), ("ours", "Ours")], ["static", "orbit"], TABLE2
    )
    ours = records[1]
    assert ours["Vividness Degree ↑"] == pytest.approx(0.3) and ours["KVD ↓"] == 10.0
    assert records[3]["Seam SSIM ↑"] is None  # static camera only
    text = markdown(records, TABLE2, "Table 2")
    assert "**0.3000**" in text and "| orbit | Ours |" in text
