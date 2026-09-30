"""The videos under evaluation and how they are read.

Renders are collected as ``<results>/<group>/<dataset>/<scene>/<view>.mp4``,
where ``group`` is ``ours``, ``baselines/<method>`` or ``ablation/<name>`` and
``view`` is ``static`` or ``orbit``. The scenes reported in the paper are
listed in ``<results>/used_scene.json``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCENE_LIST = "used_scene.json"


@dataclass(frozen=True)
class EvaluatedVideo:
    """One render of one scene.

    Attributes:
        path: The MP4 file.
        dataset: Dataset of the scene.
        scene: Scene name.
    """

    path: Path
    dataset: str
    scene: str


def reported_scenes(results_root: Path) -> dict[str, list[str]]:
    """``{dataset: [scene, ...]}`` of the scenes the paper reports on."""
    document = json.loads((Path(results_root) / SCENE_LIST).read_text(encoding="utf-8"))
    return {dataset: list(scenes) for dataset, scenes in document["scenes"].items()}


def group_videos(results_root: Path, group: str, view: str) -> list[EvaluatedVideo]:
    """The ``view`` render of every reported scene of ``group``, in dataset/scene order.

    Raises:
        FileNotFoundError: If a reported scene has no such render.
    """
    videos, missing = [], []
    for dataset, scenes in sorted(reported_scenes(results_root).items()):
        for scene in sorted(scenes):
            path = Path(results_root) / group / dataset / scene / f"{view}.mp4"
            if path.is_file():
                videos.append(EvaluatedVideo(path.resolve(), dataset, scene))
            else:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"{len(missing)} renders are missing, e.g. {missing[0]}")
    # Path order, as the paper's evaluation listed them: KVD's random subsets
    # index into the concatenated features, so the order is part of the result.
    return sorted(videos, key=lambda video: str(video.path))


def video_properties(path: Path) -> tuple[int, float]:
    """``(frame count, fps)`` from the container."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"cannot read the frame count and rate of {path}")
    return count, fps


def read_bgr_frames(path: Path, every: int = 1) -> list[tuple[int, np.ndarray]]:
    """``(index, (H, W, 3) uint8 BGR frame)`` of every ``every``-th frame, decoded by OpenCV."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames, index = [], 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % every == 0:
                frames.append((index, frame))
            index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"cannot decode any frame of {path}")
    return frames
