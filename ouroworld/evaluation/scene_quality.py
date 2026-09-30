"""Scene quality with VBench (paper §4.4, Tables 2 and 3).

Aesthetic Quality and Overall Consistency (Table 2), Subject and Background
Consistency (Table 3). Overall Consistency scores each render against the
motion prompt of its scene, the same text for every method.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ouroworld.adapters.vbench import run_vbench
from ouroworld.evaluation.videos import EvaluatedVideo

TABLE2_DIMENSIONS = ("aesthetic_quality", "overall_consistency")
TABLE3_DIMENSIONS = ("subject_consistency", "background_consistency")


def scene_quality(
    videos: list[EvaluatedVideo],
    prompts: dict[str, str],
    work_dir: Path,
    weights_dir: Path,
    device: str,
    dimensions: tuple[str, ...] = TABLE2_DIMENSIONS + TABLE3_DIMENSIONS,
) -> list[dict[str, float]]:
    """Per-video scores, in the order of ``videos``.

    Args:
        videos: The renders.
        prompts: Motion prompt by ``"<dataset>/<scene>"``.
        work_dir: Scratch directory (VBench reads a flat folder of videos).
        weights_dir: VBench's pretrained models.
        device: ``cuda`` or ``cpu``.
        dimensions: VBench dimensions to score.
    """
    staged = Path(work_dir) / "videos"
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir(parents=True)
    names = []
    for video in videos:
        # Every render of a group is called <view>.mp4; the flat name keeps them apart.
        name = f"{video.dataset}__{video.scene}__{video.path.stem}.mp4"
        (staged / name).symlink_to(video.path)
        names.append(name)
    texts = {
        name: prompts[f"{video.dataset}/{video.scene}"]
        for name, video in zip(names, videos, strict=True)
    }
    scores = run_vbench(
        staged, texts, list(dimensions), Path(work_dir) / "vbench", weights_dir, device
    )
    return [{dimension: scores[dimension][name] for dimension in dimensions} for name in names]
