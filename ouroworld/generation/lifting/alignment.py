"""Bring predicted depth to the scale of the input 3DGS.

Predicted depth relates to the 3DGS depth by a scale *and* a shift in
disparity, so one affine map ``1 / z_3dgs = a / z_pred + b`` is fitted on the
reference view at ``t = 0``, where both are known, and applied to every frame:
the temporal variation of the prediction passes through untouched.

The residuals are heavy tailed (the prediction errs in whole regions: depth
discontinuities, sky, untextured walls), so the fit is iteratively reweighted
with Huber weights instead of plain least squares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Normal-consistent scale of the median absolute deviation.
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class AlignmentSettings:
    """Robust affine fit in disparity.

    Attributes:
        iterations: Reweighting iterations.
        huber_k: Huber threshold in units of the robust residual scale.
        min_valid_pixels: Fewer pixels with both depths valid is an error.
    """

    iterations: int
    huber_k: float
    min_valid_pixels: int


@dataclass(frozen=True)
class DisparityAffine:
    """``1 / z_aligned = scale / z + shift``, plus fit diagnostics."""

    scale: float
    shift: float
    residual_sigma: float
    inlier_fraction: float
    fit_pixels: int

    def apply(self, depth: torch.Tensor) -> torch.Tensor:
        """Map depth through the fit; pixels whose disparity becomes non-positive get 0."""
        disparity = self.scale / depth + self.shift
        return torch.where(disparity > 0, 1.0 / disparity.clamp_min(1e-6), torch.zeros_like(depth))


def fit_disparity_affine(
    target_depth: torch.Tensor, predicted_depth: torch.Tensor, settings: AlignmentSettings
) -> DisparityAffine:
    """Fit the affine map taking ``predicted_depth`` onto ``target_depth`` (both ``(H, W)``).

    Raises:
        ValueError: If too few pixels are valid in both maps or the fit is degenerate.
    """
    valid = (
        torch.isfinite(target_depth)
        & (target_depth > 0)
        & torch.isfinite(predicted_depth)
        & (predicted_depth > 0)
    )
    count = int(valid.sum())
    if count < settings.min_valid_pixels:
        raise ValueError(f"only {count} pixels have both depths valid")
    target = (1.0 / target_depth[valid]).numpy().astype(np.float64)
    source = (1.0 / predicted_depth[valid]).numpy().astype(np.float64)
    scale, shift, residual, sigma = robust_line_fit(
        target, source, settings.iterations, settings.huber_k
    )
    if not np.isfinite([scale, shift]).all() or scale <= 0.0:
        raise ValueError(f"degenerate disparity fit (scale={scale:.6g}, shift={shift:.6g})")
    inliers = float(np.mean(residual <= settings.huber_k * sigma)) if sigma > 0 else float("nan")
    return DisparityAffine(scale, shift, sigma, inliers, count)


def robust_line_fit(
    target: np.ndarray, source: np.ndarray, iterations: int, huber_k: float
) -> tuple[float, float, np.ndarray, float]:
    """Huber-IRLS fit of ``target ≈ a * source + b``; return ``(a, b, |residual|, sigma)``."""
    design = np.stack([source, np.ones_like(source)], axis=1)
    weights = np.ones_like(source)
    params = np.array([1.0, 0.0])
    residual = np.abs(design @ params - target)
    sigma = float("nan")
    for _ in range(iterations):
        solution, *_ = np.linalg.lstsq(design * weights[:, None], target * weights, rcond=None)
        if not np.isfinite(solution).all():
            break
        params = solution
        residual = np.abs(design @ params - target)
        sigma = MAD_TO_SIGMA * float(np.median(residual))
        if not np.isfinite(sigma) or sigma <= 0.0:
            break
        weights = np.minimum(1.0, huber_k * sigma / np.maximum(residual, 1e-12))
    return float(params[0]), float(params[1]), residual, sigma


def confident_pixels(
    depth: torch.Tensor, confidence: torch.Tensor, drop_percentile: float
) -> torch.Tensor:
    """Return the ``(N, H, W)`` mask of valid pixels above each frame's confidence percentile."""
    masks = []
    for frame_depth, frame_confidence in zip(depth, confidence, strict=True):
        finite = torch.isfinite(frame_depth) & (frame_depth > 0) & torch.isfinite(frame_confidence)
        if not finite.any():
            raise ValueError("a frame has no valid depth")
        threshold = torch.quantile(frame_confidence[finite], drop_percentile / 100.0)
        masks.append(finite & (frame_confidence >= threshold))
    return torch.stack(masks)
