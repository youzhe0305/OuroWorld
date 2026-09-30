"""MP4 reading and writing with FFmpeg."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DecodedVideo:
    """Every frame of a video file.

    Attributes:
        frames: ``(H, W, 3)`` uint8 RGB frames in display order.
        fps: Frames per second of the container.
    """

    frames: list[np.ndarray]
    fps: float

    @property
    def span_seconds(self) -> float:
        """Time from the first to the last frame (one frame interval less than the duration)."""
        return (len(self.frames) - 1) / self.fps


def read_mp4(path: Path) -> DecodedVideo:
    """Decode all frames of ``path`` with the FFmpeg shipped by ``imageio-ffmpeg``."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    try:
        fps = float(reader.get_meta_data()["fps"])
        frames = [np.asarray(frame) for frame in reader]  # type: ignore[attr-defined]
    finally:
        reader.close()
    if not frames or not fps > 0:
        raise ValueError(f"{path}: no frames or no frame rate")
    return DecodedVideo(frames, fps)


def write_mp4(path: Path, frames: Iterable[np.ndarray], fps: int, crf: int = 17) -> int:
    """Encode ``(H, W, 3)`` uint8 frames to ``path`` without holding the video in memory.

    Returns:
        The number of frames written.
    """
    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        raise ValueError("cannot encode an empty video")
    height, width = first.shape[:2]
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]  # fmt: skip
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None and process.stderr is not None
    count = 0
    try:
        for frame in _chain(first, iterator):
            if frame.shape != (height, width, 3):
                raise ValueError(f"frame shape {frame.shape} != {(height, width, 3)}")
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            count += 1
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with {return_code}: {stderr.strip()}")
    return count


def _chain(first: np.ndarray, rest: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
    yield first
    yield from rest
