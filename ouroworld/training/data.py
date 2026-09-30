"""Supervision images and render views, loaded once and kept in memory."""

from __future__ import annotations

import numpy as np
import torch

from ouroworld.geometry.camera import Camera
from ouroworld.io.images import load_rgb
from ouroworld.io.multiview import MultiviewVideos, Observation
from ouroworld.render.rasterizer import RasterView


class TrainingData:
    """Cameras prepared for rendering and lazily cached target images."""

    def __init__(self, videos: MultiviewVideos, device: torch.device):
        self.videos = videos
        self.device = device
        self.views = {
            view: RasterView.from_camera(camera, device) for view, camera in videos.cameras.items()
        }
        self._images: dict[Observation, torch.Tensor] = {}

    def view(self, observation: Observation) -> RasterView:
        """Render view of an observation."""
        return self.views[observation.view_id]

    def image(self, observation: Observation) -> torch.Tensor:
        """``(3, H, W)`` target image on the training device."""
        cached = self._images.get(observation)
        if cached is None:
            cached = load_rgb(observation.image_path).clamp(0.0, 1.0)
            self._images[observation] = cached
        return cached.to(self.device, non_blocking=True)


def scene_extent(cameras: list[Camera]) -> float:
    """1.1 times the largest camera distance from the mean camera centre (NeRF++ radius).

    Each entry counts once, so pass one camera per supervised observation to
    weight views by how often they are supervised.
    """
    centers = np.stack([camera.center for camera in cameras])
    distances = np.linalg.norm(centers - centers.mean(axis=0, keepdims=True), axis=1)
    return float(distances.max() * 1.1)
