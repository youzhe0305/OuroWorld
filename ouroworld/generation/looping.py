"""Looping reference video from the reference image (paper §3.1).

1. A vision-language model writes a motion prompt for the reference image.
2. A video model animates the image, which is both its first and last frame.
3. The first two frames are checked for a global jump.
4. The video is centre-cropped to the reference camera and saved as a
   ``ReferenceVideo``, whose last frame closes the loop at ``t = 1``.

The prompt and the raw video cost money, so both are kept in the output
directory and reused when the stage is run again; delete them to regenerate.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ouroworld.generation.prompt.base import PromptWriter
from ouroworld.generation.video.base import VideoGenerator
from ouroworld.generation.video.continuity import ContinuityLimits, start_continuity
from ouroworld.io.images import center_crop_resize, read_rgb8, save_rgb8
from ouroworld.io.reference_video import (
    ReferenceVideo,
    load_reference_video,
    save_reference_video_index,
)
from ouroworld.io.scene_package import ScenePackage
from ouroworld.io.video import read_mp4

logger = logging.getLogger(__name__)

PROMPT_FILE = "prompt.txt"
PROMPT_METADATA_FILE = "prompt.json"
RAW_VIDEO_FILE = "raw.mp4"
GENERATION_FILE = "generation.json"
CONTINUITY_FILE = "continuity.json"
FRAMES_DIR = "frames"


class ContinuityError(RuntimeError):
    """The generated video jumps between its first two frames."""


@dataclass(frozen=True)
class LoopingVideoSettings:
    """Stage settings.

    Attributes:
        frame_count: Frames of the saved loop, both ends included.
        continuity: Limits of the start-continuity check, ``None`` to skip it.
    """

    frame_count: int
    continuity: ContinuityLimits | None


class LoopingVideoGenerator:
    """Runs the stage for one scene."""

    def __init__(
        self,
        settings: LoopingVideoSettings,
        prompt_writer: PromptWriter | None,
        video_generator: VideoGenerator | None,
    ) -> None:
        """Compose the stage from its two models; ``None`` when its output is reused."""
        self.settings = settings
        self.prompt_writer = prompt_writer
        self.video_generator = video_generator

    def run(self, package: ScenePackage, out_dir: Path) -> ReferenceVideo:
        """Generate (or reuse) the video of ``package`` and write the ``ReferenceVideo``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt = self._prompt(package.reference_image_path, out_dir)
        raw = out_dir / RAW_VIDEO_FILE
        if raw.is_file():
            logger.info("reusing %s", raw)
        else:
            partial = out_dir / f"partial_{RAW_VIDEO_FILE}"
            if self.video_generator is None:
                raise RuntimeError(f"no video model configured and no {raw} to reuse")
            image = read_rgb8(package.reference_image_path)
            provenance = self.video_generator.generate(image, prompt, partial)
            partial.replace(raw)
            _write_json(out_dir / GENERATION_FILE, provenance)
        if self.settings.continuity is not None:
            report = check_start_continuity(raw, self.settings.continuity)
            _write_json(out_dir / CONTINUITY_FILE, report)
            if not report["passed"]:
                raise ContinuityError(
                    f"{raw} jumps between its first two frames ({report}); delete it to "
                    "generate a new video"
                )
        camera = package.reference_camera
        return write_loop_frames(
            raw, (camera.width, camera.height), self.settings.frame_count, out_dir
        )

    def _prompt(self, image_path: Path, out_dir: Path) -> str:
        path = out_dir / PROMPT_FILE
        if path.is_file():
            logger.info("reusing %s", path)
            return path.read_text(encoding="utf-8").strip()
        if self.prompt_writer is None:
            raise RuntimeError(f"no prompt writer configured and no {path} to reuse")
        prompt = self.prompt_writer.write(image_path)
        path.write_text(prompt.text + "\n", encoding="utf-8")
        _write_json(out_dir / PROMPT_METADATA_FILE, prompt.metadata)
        logger.info("prompt: %s", prompt.text)
        return prompt.text


def check_start_continuity(video_path: Path, limits: ContinuityLimits) -> dict:
    """Run :func:`start_continuity` on the first two frames of ``video_path``."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    frames = [capture.read() for _ in range(2)]
    capture.release()
    if not all(ok for ok, _ in frames):
        return {"passed": False, "reason": "cannot decode the first two frames"}
    first, second = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for _, frame in frames)
    return start_continuity(first, second, limits)


def write_loop_frames(
    video_path: Path, size: tuple[int, int], frame_count: int, out_dir: Path
) -> ReferenceVideo:
    """Save ``frame_count`` frames of ``video_path``, cropped and resized to ``size``.

    A video with more frames is sampled at the nearest-lower frame of uniform
    phase steps; the first and last frame are always kept.

    Raises:
        ValueError: If the video has fewer than ``frame_count`` frames.
    """
    video = read_mp4(video_path)
    total = len(video.frames)
    if total < frame_count:
        raise ValueError(
            f"{video_path} has {total} frames, fewer than reference_video.frame_count={frame_count}"
        )
    indices = (
        np.arange(total)
        if total == frame_count
        else np.linspace(0, total - 1, frame_count).astype(int)
    )
    frames_dir = Path(out_dir) / FRAMES_DIR
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    records = []
    for position, index in enumerate(indices):
        name = f"{FRAMES_DIR}/frame_{position:04d}.png"
        save_rgb8(center_crop_resize(video.frames[index], size), Path(out_dir) / name)
        records.append({"path": name, "time": float(index) / (total - 1)})
    save_reference_video_index(Path(out_dir), records, video.span_seconds)
    return load_reference_video(Path(out_dir))


def _write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
