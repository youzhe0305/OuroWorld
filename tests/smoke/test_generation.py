"""Multi-view generation end to end on a synthetic scene, with stand-in models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ouroworld.generation.inpainting.base import ConditionVideo
from ouroworld.generation.inpainting.fusion import FusionSettings
from ouroworld.generation.lifting.alignment import AlignmentSettings
from ouroworld.generation.lifting.base import DepthMaps
from ouroworld.generation.pipeline import (
    GenerationSettings,
    MultiviewGenerator,
    _static_renderer,
)
from ouroworld.io.multiview import Role
from ouroworld.io.ply import write_gaussian_ply
from ouroworld.io.reference_video import load_reference_video, save_reference_video_index
from ouroworld.io.scene_package import load_scene_package
from tests.helpers import look_at_camera, random_gaussians, write_scene_metadata

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

SETTINGS = GenerationSettings(
    view_count=3,
    max_yaw_degrees=10.0,
    axis_tilt_degrees=5.0,
    frame_count=5,
    sweep_frames=4,
    appearance_frames=2,
    confidence_drop_percentile=10.0,
    seed=0,
    alignment=AlignmentSettings(4, 1.345, 16),
    fusion=FusionSettings(0.5, 1.0, 1, 2, 1.0),
    save_diagnostics=True,
)


class TrueDepth:
    """Stand-in lifter: the reference render's own depth for every image."""

    def __init__(self, depth: np.ndarray) -> None:
        self.depth = np.where(depth > 0, depth, 1.0).astype(np.float32)

    def lift(self, images: list[np.ndarray]) -> DepthMaps:
        stack = np.stack([self.depth] * len(images))
        return DepthMaps(stack, np.ones_like(stack))


class GreyInpainter:
    """Stand-in video model: grey everywhere; records its calls."""

    def __init__(self) -> None:
        self.calls: list[ConditionVideo] = []

    def inpaint(
        self, condition: ConditionVideo, reference: torch.Tensor, prompt: str, seed: int
    ) -> torch.Tensor:
        assert prompt == "a scene" and len(reference) == SETTINGS.appearance_frames
        self.calls.append(condition)
        return torch.full(condition.frames.shape, 0.5)

    def release(self) -> None:
        pass


class FixedCaption:
    def caption(self, image: np.ndarray) -> str:
        return "a scene"


@pytest.fixture
def scene(tmp_path: Path) -> tuple[Path, Path]:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    arrays = random_gaussians(count=4000, seed=3)
    arrays.log_scale[:] = -2.0
    arrays.opacity[:] = 3.0
    write_gaussian_ply(arrays, tmp_path / "scene" / "gaussians.ply")
    write_scene_metadata(tmp_path / "scene", look_at_camera(), np.zeros(3))
    video = tmp_path / "reference_video"
    (video / "frames").mkdir(parents=True)
    records = []
    for index in range(9):
        name = f"frames/frame_{index:04d}.png"
        Image.fromarray(np.full((36, 64, 3), 20 * index, np.uint8)).save(video / name)
        records.append({"path": name, "time": index / 8})
    save_reference_video_index(video, records, 10.0)
    return tmp_path / "scene", video


def test_generation_writes_a_complete_artifact(scene: tuple[Path, Path], tmp_path: Path) -> None:
    package = load_scene_package(scene[0])
    video = load_reference_video(scene[1])
    reference_depth = _static_renderer(package).view(package.reference_camera).depth
    inpainter = GreyInpainter()
    generator = MultiviewGenerator(SETTINGS, TrueDepth(reference_depth), inpainter, FixedCaption())
    out = tmp_path / "multiview"
    videos = generator.run(package, video, out)

    assert len(inpainter.calls) == 2  # every view but the reference
    condition = inpainter.calls[0]
    assert condition.frames.shape == (9, 3, 384, 672) and torch.all(condition.known[:4] == 1)
    assert videos.reference_view_id == 1 and sorted(videos.cameras) == [0, 1, 2]
    assert len([o for o in videos.observations if o.view_id == 1]) == 8
    generated = [o for o in videos.observations if o.view_id == 0]
    assert [o.time for o in generated] == [0.0, 0.25, 0.5, 0.75]
    assert generated[0].role is Role.T0 and generated[1].role is Role.GENERATED
    assert (out / "intermediate" / "views" / "view_00" / "condition.mp4").is_file()

    generator.run(package, video, out)  # resumes: nothing left to generate
    assert len(inpainter.calls) == 2
