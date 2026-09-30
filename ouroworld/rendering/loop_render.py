"""Seamless loop videos of a trained cinemagraph (paper §4.3).

* ``static.mp4`` -- the reference camera, fixed.
* ``orbit.mp4`` -- a closed horizontal orbit around the scene pivot.

Both play ``n_loops`` deformation cycles. A periodic model is queried at
``t`` beyond 1 on purpose, so any seam would show up in the video.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ouroworld.fields.cinemagraph import CinemagraphModel, ViewContext
from ouroworld.geometry.camera import Camera
from ouroworld.geometry.trajectories import (
    horizontal_orbit,
    project_world_point,
    uniform_horizontal_angles,
)
from ouroworld.io.video import write_mp4
from ouroworld.render.rasterizer import RasterView, render

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopRenderSettings:
    """Video settings.

    Attributes:
        n_loops: Deformation cycles per video.
        fps: Output frame rate.
        cycle_seconds: Duration of one cycle.
        orbit_degrees: ``(negative, positive)`` yaw limits of the orbit.
        orbit_radius_scale: Orbit radius relative to the reference-to-pivot distance.
        orbit_camera_loops: Camera orbits per video.
    """

    n_loops: int
    fps: int
    cycle_seconds: float
    orbit_degrees: tuple[float, float]
    orbit_radius_scale: float
    orbit_camera_loops: int

    @property
    def cycle_frames(self) -> int:
        """Frames per deformation cycle."""
        return max(1, round(self.cycle_seconds * self.fps))

    @property
    def total_frames(self) -> int:
        """Frames per video."""
        return self.n_loops * self.cycle_frames


class LoopRenderer:
    """Render loop videos with the periodic field only (no drift)."""

    def __init__(
        self,
        model: CinemagraphModel,
        background: torch.Tensor,
        settings: LoopRenderSettings,
        is_periodic: bool,
    ):
        self.model = model.eval()
        self.background = background
        self.settings = settings
        self.is_periodic = is_periodic

    def time_of_frame(self, frame: int) -> float:
        """Loop phase of output ``frame``; wrapped only for the aperiodic ablation."""
        time = frame / self.settings.cycle_frames
        return time if self.is_periodic else time % 1.0

    @torch.no_grad()
    def render_frame(self, camera: RasterView, time: float) -> torch.Tensor:
        """``(3, H, W)`` render at loop phase ``time``."""
        attributes = self.model.deformed(ViewContext(time))
        return render(attributes, camera, self.background, self.model.canonical.sh_degree).image

    def static_frames(self, camera: Camera) -> Iterator[np.ndarray]:
        """Frames of the fixed-camera video."""
        view = RasterView.from_camera(camera, self.background.device)
        for frame in tqdm(range(self.settings.total_frames), desc="static"):
            yield _to_uint8(self.render_frame(view, self.time_of_frame(frame)))

    def orbit_frames(self, poses: np.ndarray, camera: Camera) -> Iterator[np.ndarray]:
        """Frames of the orbit video; ``poses`` is one camera loop."""
        for frame in tqdm(range(self.settings.total_frames), desc="orbit"):
            pose = poses[frame % len(poses)]
            view = RasterView.from_camera(camera.with_pose(pose), self.background.device)
            yield _to_uint8(self.render_frame(view, self.time_of_frame(frame)))

    def orbit_poses(self, reference: Camera, pivot: np.ndarray) -> np.ndarray:
        """One closed camera loop around ``pivot`` (0 -> right -> left -> 0)."""
        settings = self.settings
        if settings.total_frames % settings.orbit_camera_loops:
            raise ValueError("frames per video must be divisible by orbit_camera_loops")
        frames_per_loop = settings.total_frames // settings.orbit_camera_loops
        angles = uniform_horizontal_angles(frames_per_loop, *settings.orbit_degrees)
        pivot_pixel, _ = project_world_point(reference.K, reference.c2w, pivot)
        return horizontal_orbit(
            reference.c2w, reference.K, pivot, pivot_pixel, angles, settings.orbit_radius_scale
        )

    @torch.no_grad()
    def seam_report(self, camera: Camera) -> dict[str, float]:
        """PSNR between ``t = 0`` and ``t = 1`` / ``t = n_loops``; infinite for a periodic model."""
        view = RasterView.from_camera(camera, self.background.device)
        start = self.render_frame(view, 0.0)
        end_time = float(self.settings.n_loops) if self.is_periodic else 1.0
        return {
            "psnr_t0_vs_t1": _psnr(start, self.render_frame(view, 1.0)),
            "endpoint_time": end_time,
            "psnr_t0_vs_endpoint": _psnr(start, self.render_frame(view, end_time)),
        }


def render_loop_videos(
    renderer: LoopRenderer, reference: Camera, pivot: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """Write ``static.mp4``, ``orbit.mp4`` and ``loop_report.json`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = renderer.settings.fps
    write_mp4(out_dir / "static.mp4", renderer.static_frames(reference), fps)
    poses = renderer.orbit_poses(reference, pivot)
    write_mp4(out_dir / "orbit.mp4", renderer.orbit_frames(poses, reference), fps)
    report: dict[str, object] = {
        "frames": renderer.settings.total_frames,
        "fps": fps,
        "cycle_frames": renderer.settings.cycle_frames,
        "pivot": [float(value) for value in pivot],
        "seam": renderer.seam_report(reference),
    }
    (out_dir / "loop_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote loop videos to %s", out_dir)
    return report


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    """``(3, H, W)`` float image to ``(H, W, 3)`` uint8, truncating like the paper runs."""
    return (255 * np.clip(image.detach().cpu().numpy(), 0, 1)).astype(np.uint8).transpose(1, 2, 0)


def _psnr(first: torch.Tensor, second: torch.Tensor) -> float:
    mse = float(((first - second) ** 2).mean())
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
