"""Loop Seam Coherence: Seam SSIM and Motion-Aware Loop Fidelity (paper §4.4, Eq. 4).

* Seam SSIM: SSIM between the last frame of a cycle and the first of the
  next, averaged over the seams of the video.
* MALF: similarity of frames one period apart minus the similarity of frames
  a comparable but wrong distance apart (0.5-0.9 and 1.1-1.5 periods). A
  static video matches itself at every distance and scores ~0, so standing
  still does not win. Both similarities are sampled at 10 time points per
  period, so short and long loops are measured on the same footing.

Both come in an SSIM (reported) and a PSNR flavour, on grey frames (SSIM)
and colour frames (PSNR). The static camera is used: an orbit's camera motion
is not periodic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ouroworld.evaluation.videos import read_bgr_frames

SSIM_WINDOW = (11, 11)  # Wang et al. 2004: 11x11 Gaussian, sigma 1.5
SSIM_SIGMA = 1.5
SSIM_C1 = (0.01 * 255.0) ** 2
SSIM_C2 = (0.03 * 255.0) ** 2
IDENTICAL_PSNR = 99.0


@dataclass(frozen=True)
class LoopSeamSettings:
    """Sampling of the loop metrics.

    Attributes:
        samples_per_cycle: Time points per period (M in Eq. 4).
        max_ring_lags: Wrong distances averaged for the MALF reference term.
    """

    samples_per_cycle: int = 10
    max_ring_lags: int = 16


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    """PSNR of two uint8 images; 99 dB when identical."""
    mse = float(np.mean((first.astype(np.float32) - second.astype(np.float32)) ** 2))
    return IDENTICAL_PSNR if mse <= 1e-12 else 10.0 * math.log10(255.0**2 / mse)


def _blur(plane: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(plane, SSIM_WINDOW, SSIM_SIGMA)


class _SsimTerms:
    """Per-frame SSIM terms, computed once however many pairs use a frame."""

    def __init__(self, grey: list[np.ndarray]) -> None:
        self.grey = grey
        self.cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __call__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if index not in self.cache:
            plane = self.grey[index].astype(np.float32)
            mean = _blur(plane)
            self.cache[index] = (plane, mean, _blur(plane * plane) - mean * mean)
        return self.cache[index]


def ssim(first: tuple, second: tuple) -> float:
    """Gaussian-windowed SSIM from two frames' :class:`_SsimTerms`."""
    plane_a, mean_a, var_a = first
    plane_b, mean_b, var_b = second
    covariance = _blur(plane_a * plane_b) - mean_a * mean_b
    numerator = (2.0 * mean_a * mean_b + SSIM_C1) * (2.0 * covariance + SSIM_C2)
    denominator = (mean_a**2 + mean_b**2 + SSIM_C1) * (var_a + var_b + SSIM_C2)
    return float(np.mean(numerator / denominator))


def sample_times(period: int, limit: int, samples_per_cycle: int) -> list[int]:
    """Frame indices below ``limit``, ``samples_per_cycle`` per period, duplicates removed."""
    step = period / samples_per_cycle
    times: list[int] = []
    index = 0
    while (time := int(round(index * step))) < limit:
        if not times or time != times[-1]:
            times.append(time)
        index += 1
    return times


def ring_lags(period: int, frame_count: int, max_lags: int) -> list[int]:
    """Distances comparable to one period but at the wrong phase."""
    low = range(max(1, math.ceil(0.5 * period)), math.floor(0.9 * period) + 1)
    high = range(math.ceil(1.1 * period), math.floor(1.5 * period) + 1)
    lags = [lag for lag in [*low, *high] if lag < frame_count - 1]
    if len(lags) > max_lags:
        step = len(lags) / max_lags
        lags = [lags[int(index * step)] for index in range(max_lags)]
    return lags


def loop_seam_metrics(path: Path, period: int, settings: LoopSeamSettings) -> dict[str, float]:
    """Seam and MALF of one video whose loop lasts ``period`` frames."""
    import cv2

    colour = [frame for _, frame in read_bgr_frames(path)]
    grey = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in colour]
    count = len(colour)
    if period >= count:
        raise ValueError(f"{path}: a {period}-frame period needs more than {count} frames")
    terms = _SsimTerms(grey)
    adjacent_psnr = np.array([psnr(colour[t], colour[t + 1]) for t in range(count - 1)])
    adjacent_ssim = np.array([ssim(terms(t), terms(t + 1)) for t in range(count - 1)])
    # (f_{kP-1}, f_{kP}) is the transition the loop has to hide.
    at_seam = ((np.arange(count - 1) + 1) % period) == 0

    def lag_similarity(lag: int) -> tuple[float, float]:
        times = sample_times(period, count - lag, settings.samples_per_cycle)
        pairs = [(psnr(colour[t], colour[t + lag]), ssim(terms(t), terms(t + lag))) for t in times]
        return float(np.mean([p for p, _ in pairs])), float(np.mean([s for _, s in pairs]))

    aligned_psnr, aligned_ssim = lag_similarity(period)
    ring = [lag_similarity(lag) for lag in ring_lags(period, count, settings.max_ring_lags)]
    ring_psnr = float(np.mean([p for p, _ in ring]))
    ring_ssim = float(np.mean([s for _, s in ring]))
    return {
        "period_frames": period,
        "seam_count": int(at_seam.sum()),
        "seam_ssim": float(adjacent_ssim[at_seam].mean()),
        "interior_ssim": float(adjacent_ssim[~at_seam].mean()),
        "malf": aligned_ssim - ring_ssim,
        "aligned_ssim": aligned_ssim,
        "ring_ssim": ring_ssim,
        "seam_psnr": float(adjacent_psnr[at_seam].mean()),
        "interior_psnr": float(adjacent_psnr[~at_seam].mean()),
        "malf_psnr": aligned_psnr - ring_psnr,
    }
