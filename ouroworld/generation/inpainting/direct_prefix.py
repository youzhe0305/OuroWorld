"""The camera-sweep prefix of TrajectoryCrafter's condition ("direct" mode).

TrajectoryCrafter was trained on videos whose camera moves from the source
view to the target view. The condition of a fixed target camera therefore
starts with a sweep from the reference camera to the target at ``t = 0``,
rendered from the input 3DGS (fully known), followed by the warped reference
video at the target camera. Only the frames after the sweep are kept.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ouroworld.generation.inpainting.base import ConditionVideo
from ouroworld.geometry.camera import Camera
from ouroworld.geometry.trajectories import pivot_sweep


def sweep_cameras(
    reference: Camera, target: Camera, pivot: np.ndarray, frame_count: int
) -> list[Camera]:
    """Return the ``frame_count`` sweep cameras; the first is ``reference``, the last ``target``."""
    poses = pivot_sweep(reference.c2w, target.c2w, pivot, frame_count)
    return [reference.with_pose(pose) for pose in poses]


def assemble_condition(
    prefix: list[np.ndarray],
    warped: torch.Tensor,
    warped_known: torch.Tensor,
    height: int,
    width: int,
) -> ConditionVideo:
    """Concatenate the sweep and the warped video at the video model's resolution.

    Args:
        prefix: Sweep renders, ``(H, W, 3)`` uint8 each; every pixel is known.
        warped: ``(T, 3, H, W)`` warped reference video at the target camera.
        warped_known: ``(T, 1, H, W)`` its known mask.
        height: Output height.
        width: Output width.
    """
    sweep = torch.stack(
        [torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 for image in prefix]
    )
    frames = torch.cat((resize_frames(sweep, height, width), resize_frames(warped, height, width)))
    known = torch.cat(
        (
            torch.ones((len(prefix), 1, height, width)),
            F.interpolate(warped_known, (height, width), mode="nearest"),
        )
    )
    return ConditionVideo(frames, known)


def resize_frames(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Bilinearly resize ``(T, C, H, W)`` frames (no-op at the same size)."""
    if tuple(frames.shape[-2:]) == (height, width):
        return frames
    return F.interpolate(frames, (height, width), mode="bilinear", align_corners=False)
