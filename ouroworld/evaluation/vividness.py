"""Vividness Degree (paper §4.4, Eq. 3; App. E).

Frames are sampled at 3 FPS. Between each pair of sampled frames:

* Motion Variation (MV): fraction of pixels whose SEA-RAFT flow is longer
  than 0.3% of the short image side;
* Illumination Variation (IV): fraction of pixels whose grey level, after
  warping the previous frame with the backward flow, changes by more than 20%
  (relative to the darker value);
* Visual Variation (VV): 1 - SSIM of the grey frames.

Each is averaged over the video, and Vividness = (MV + IV + VV) / 3.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ouroworld.evaluation.videos import read_bgr_frames, video_properties

logger = logging.getLogger(__name__)

# (image1, image2) -> flow, (B, 3, H, W) RGB in [0, 255] -> (B, 2, H, W).
FlowModel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class VividnessSettings:
    """Thresholds and sampling of the metric.

    Attributes:
        sample_fps: Frames per second the video is sampled at.
        motion_threshold_percent: Flow threshold, percent of the short image side.
        illumination_threshold_percent: Relative grey-level change threshold, percent.
        illumination_epsilon: Guards the ratio near black.
    """

    sample_fps: float = 3.0
    motion_threshold_percent: float = 0.3
    illumination_threshold_percent: float = 20.0
    illumination_epsilon: float = 1e-6


def motion_mask(flow: np.ndarray, threshold_pixels: float) -> np.ndarray:
    """Pixels of a ``(2, H, W)`` flow that move farther than ``threshold_pixels``."""
    return np.linalg.norm(np.asarray(flow, dtype=np.float32), axis=0) > float(threshold_pixels)


def illumination_mask(
    previous: np.ndarray,
    current: np.ndarray,
    backward_flow: np.ndarray,
    threshold: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixels whose relative grey-level change exceeds ``threshold``, and the warp's valid pixels.

    Args:
        previous: ``(H, W)`` grey frame in ``[0, 1]``.
        current: ``(H, W)`` grey frame in ``[0, 1]``.
        backward_flow: ``(2, H, W)`` flow from ``current`` to ``previous``.
        threshold: Relative change, e.g. 0.2.
        epsilon: Added to the darker value.
    """
    import cv2

    height, width = previous.shape
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = xx + backward_flow[0].astype(np.float32)
    map_y = yy + backward_flow[1].astype(np.float32)
    valid = (map_x >= 0.0) & (map_x <= width - 1) & (map_y >= 0.0) & (map_y <= height - 1)
    warped = cv2.remap(  # type: ignore[call-overload]
        previous.astype(np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    darker, brighter = np.minimum(current, warped), np.maximum(current, warped)
    change = np.maximum(brighter / (darker + float(epsilon)) - 1.0, 0.0)
    return valid & (change > float(threshold)), valid


def grey_ssim(previous: np.ndarray, current: np.ndarray) -> float:
    """Standard SSIM (Gaussian window, sigma 1.5) of two BGR frames' grey images, in ``[0, 1]``."""
    import cv2
    from skimage.metrics import structural_similarity

    value = structural_similarity(
        cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(current, cv2.COLOR_BGR2GRAY),
        data_range=255,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    return float(np.clip(value, 0.0, 1.0))


def video_vividness(
    path: Path, flow_model: FlowModel, device: str, settings: VividnessSettings
) -> dict[str, float]:
    """MV, IV, VV and the Vividness Degree of one video."""
    import cv2

    _, fps = video_properties(path)
    interval = max(1, int(round(fps / settings.sample_fps)))
    frames = read_bgr_frames(path, every=interval)
    if len(frames) < 2:
        raise ValueError(f"{path} has fewer than two frames at {fps / interval:.3g} FPS")
    height, width = frames[0][1].shape[:2]
    motion_threshold = min(height, width) * settings.motion_threshold_percent / 100.0
    illumination_threshold = settings.illumination_threshold_percent / 100.0
    motion, illumination, visual = [], [], []
    with torch.inference_mode():
        for (_, first), (_, second) in zip(frames[:-1], frames[1:], strict=True):
            first_rgb, second_rgb = _rgb(first), _rgb(second)
            # Both directions in one batch: forward for motion, backward to warp.
            flows = (
                flow_model(
                    torch.stack((first_rgb, second_rgb)).to(device),
                    torch.stack((second_rgb, first_rgb)).to(device),
                )
                .float()
                .cpu()
                .numpy()
            )
            previous = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            current = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            motion.append(float(np.mean(motion_mask(flows[0], motion_threshold))))
            changed, _ = illumination_mask(
                previous, current, flows[1], illumination_threshold, settings.illumination_epsilon
            )
            illumination.append(float(np.mean(changed)))
            visual.append(1.0 - grey_ssim(first, second))
    scores = {
        "motion_variation": float(np.mean(motion)),
        "illumination_variation": float(np.mean(illumination)),
        "visual_variation": float(np.mean(visual)),
    }
    scores["vividness"] = sum(scores.values()) / 3.0
    scores["frame_pairs"] = len(motion)
    return scores


def sea_raft_flow(model: torch.nn.Module) -> FlowModel:
    """Wrap a loaded SEA-RAFT as a :data:`FlowModel`."""

    def flow(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        return model(image1, image2, test_mode=True)["final"]

    return flow


def _rgb(frame: np.ndarray) -> torch.Tensor:
    import cv2

    return torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
