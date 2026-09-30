"""Scene sharpness: variance of the Laplacian (VoL, paper Table 3).

The second derivative's energy collapses when detail is smeared. VoL scales
with scene contrast, so it compares variants of the same scene; the table
averages it over scenes. Frames are sampled evenly over one loop period:
renders repeat the period verbatim.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np


def sample_period_frames(path: Path, count: int, period_frames: int) -> list[np.ndarray]:
    """``count`` BGR frames spread evenly over the first ``period_frames`` of ``path``."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    span = min(period_frames, total)
    if span < 1:
        capture.release()
        raise ValueError(f"{path}: no frames to sample")
    wanted = sorted(
        {int(round(index)) % span for index in np.linspace(0, span, count, endpoint=False)}
    )
    frames = []
    for index in wanted:
        # Seeking keeps a 600-frame render from being decoded in full.
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"cannot decode any frame of {path}")
    return frames


def variance_of_laplacian(frame: np.ndarray) -> float:
    """VoL of a BGR frame's grey image."""
    import cv2

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def video_sharpness(path: Path, count: int, period_frames: int) -> dict[str, float]:
    """Mean VoL over the sampled frames of one video."""
    frames = sample_period_frames(path, count, period_frames)
    values = [variance_of_laplacian(frame) for frame in frames]
    return {"frames": len(frames), "vol": statistics.fmean(values)}
