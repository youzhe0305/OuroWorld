"""``ouroworld-evaluate``: the metrics of paper §4.4 and Tables 2 and 3.

Each metric is run per result group and view and written to
``<output_root>/<group>/<view>/<metric>.json``; existing files are kept unless
``--overwrite`` is given. ``tables`` assembles Tables 2 and 3 from them.

Examples::

    ouroworld-evaluate loop-seam                      # every group of both tables
    ouroworld-evaluate vividness --groups ours --views static
    ouroworld-evaluate tables

Metrics: vividness, naturalness (KVD), loop-seam (MALF, Seam SSIM; static
camera only), sharpness (VoL), scene-quality (VBench).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ouroworld.cli.common import REPOSITORY_ROOT, setup_logging
from ouroworld.config.loader import load_evaluation_config
from ouroworld.config.schema import EvaluationConfig
from ouroworld.evaluation.tables import result_path, write_result, write_tables
from ouroworld.evaluation.videos import EvaluatedVideo, group_videos, video_properties

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "evaluation.yaml"
# Which rows each metric is needed for: Table 2 (methods), Table 3 (ablations + ours).
TABLE2_METRICS = ("vividness", "naturalness", "loop_seam", "scene_quality")
TABLE3_METRICS = ("vividness", "sharpness", "scene_quality", "loop_seam")


def _loop_seam(config: EvaluationConfig) -> Callable[..., dict[str, Any]]:
    from ouroworld.evaluation.loop_seam import LoopSeamSettings, loop_seam_metrics

    settings = LoopSeamSettings(config.loop_seam.samples_per_cycle, config.loop_seam.max_ring_lags)

    def run(videos: list[EvaluatedVideo], cycle_seconds: float, **_: Any) -> dict[str, Any]:
        rows = []
        for video in videos:
            _, fps = video_properties(video.path)
            period = int(round(cycle_seconds * fps))
            rows.append({**_identity(video), **loop_seam_metrics(video.path, period, settings)})
        return {"cycle_seconds": cycle_seconds, "videos": rows}

    return run


def _sharpness(config: EvaluationConfig) -> Callable[..., dict[str, Any]]:
    from ouroworld.evaluation.sharpness import video_sharpness

    def run(videos: list[EvaluatedVideo], **_: Any) -> dict[str, Any]:
        settings = config.sharpness
        return {
            "videos": [
                {
                    **_identity(video),
                    **video_sharpness(video.path, settings.frames, settings.period_frames),
                }
                for video in videos
            ]
        }

    return run


def _vividness(config: EvaluationConfig) -> Callable[..., dict[str, Any]]:
    from ouroworld.adapters.sea_raft import load_sea_raft
    from ouroworld.evaluation.vividness import VividnessSettings, sea_raft_flow, video_vividness

    options = config.vividness
    settings = VividnessSettings(
        options.sample_fps,
        options.motion_threshold_percent,
        options.illumination_threshold_percent,
        options.illumination_epsilon,
    )
    flow = sea_raft_flow(load_sea_raft(_path(options.sea_raft), config.device))

    def run(videos: list[EvaluatedVideo], **_: Any) -> dict[str, Any]:
        rows = []
        for index, video in enumerate(videos, 1):
            rows.append(
                {**_identity(video), **video_vividness(video.path, flow, config.device, settings)}
            )
            logger.info("vividness %d/%d %s/%s", index, len(videos), video.dataset, video.scene)
        return {"videos": rows}

    return run


def _naturalness(config: EvaluationConfig) -> Callable[..., dict[str, Any]]:
    from ouroworld.evaluation.naturalness import I3dFeatures, NaturalnessSettings, naturalness

    options = config.naturalness
    settings = NaturalnessSettings(
        options.clip_seconds,
        options.clip_stride_seconds,
        options.clip_frames,
        options.frame_size,
        options.max_clips_per_video,
        options.subset_size,
        options.subset_count,
        options.kernel_degree,
        options.kernel_coefficient,
        options.seed,
    )
    features = I3dFeatures(
        _path(options.i3d), config.device, settings, _path(options.feature_cache)
    )

    def run(videos: list[EvaluatedVideo], **_: Any) -> dict[str, Any]:
        return naturalness(
            videos,
            _path(options.reference_root),
            options.reference_pattern,
            features,
            settings,
            dict(options.reference_dataset_dirs),
        )

    return run


def _scene_quality(config: EvaluationConfig) -> Callable[..., dict[str, Any]]:
    from ouroworld.evaluation.scene_quality import (
        TABLE2_DIMENSIONS,
        TABLE3_DIMENSIONS,
        scene_quality,
    )

    prompts = json.loads(_path(config.scene_quality.prompts).read_text(encoding="utf-8"))
    table2 = {row.group for row in config.methods}
    table3 = {row.group for row in config.ablations} | {"ours"}

    def run(videos: list[EvaluatedVideo], work_dir: Path, group: str, **_: Any) -> dict[str, Any]:
        # Only the dimensions of the tables the group appears in.
        dimensions = (TABLE2_DIMENSIONS if group in table2 else ()) + (
            TABLE3_DIMENSIONS if group in table3 else ()
        )
        scores = scene_quality(
            videos,
            prompts,
            work_dir,
            _path(config.scene_quality.vbench_weights),
            config.device,
            dimensions or TABLE2_DIMENSIONS + TABLE3_DIMENSIONS,
        )
        return {
            "videos": [
                {**_identity(video), **row} for video, row in zip(videos, scores, strict=True)
            ]
        }

    return run


METRICS = {
    "vividness": _vividness,
    "naturalness": _naturalness,
    "loop_seam": _loop_seam,
    "sharpness": _sharpness,
    "scene_quality": _scene_quality,
}


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("metric", choices=[*(name.replace("_", "-") for name in METRICS), "tables"])
    parser.add_argument("--groups", nargs="+", help="default: every group the tables need")
    parser.add_argument("--views", nargs="+", help="default: the configured views")
    parser.add_argument("--overwrite", action="store_true", help="recompute existing results")
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    arguments = parser.parse_args(argv)
    setup_logging()
    config = load_evaluation_config([DEFAULT_CONFIG, *arguments.config], arguments.overrides)
    output_root = _path(config.output_root)
    methods = [(row.group, row.label) for row in config.methods]
    ablations = [(row.group, row.label) for row in config.ablations]
    if arguments.metric == "tables":
        period = (config.period_ablation.group, config.period_ablation.label)
        path = write_tables(output_root, methods, ablations, period, config.views)
        print(path.read_text(encoding="utf-8"))
        return
    metric = arguments.metric.replace("-", "_")
    rows = [*config.methods, *config.ablations, config.period_ablation]
    cycles = {row.group: row.cycle_seconds for row in rows}
    groups = arguments.groups or _groups_for(metric, config)
    views = arguments.views or (config.views[:1] if metric == "loop_seam" else config.views)
    jobs = [
        (group, view)
        for group in groups
        for view in views
        if arguments.overwrite or not result_path(output_root, group, view, metric).is_file()
    ]
    if not jobs:
        logger.info("%s: every result exists; --overwrite recomputes", metric)
        return
    run = METRICS[metric](config)
    for group, view in jobs:
        videos = group_videos(_path(config.results_root), group, view)
        logger.info("%s: %s/%s, %d videos", metric, group, view, len(videos))
        document = run(
            videos,
            group=group,
            cycle_seconds=cycles[group],
            work_dir=output_root / group / view / f"{metric}_work",
        )
        write_result(result_path(output_root, group, view, metric), {"metric": metric, **document})


def _groups_for(metric: str, config: EvaluationConfig) -> list[str]:
    groups = []
    if metric in TABLE2_METRICS:
        groups += [row.group for row in config.methods]
    if metric in TABLE3_METRICS:
        groups += [row.group for row in config.ablations] + ["ours"]
    if metric == "loop_seam":
        groups.append(config.period_ablation.group)
    return list(dict.fromkeys(groups))


def _identity(video: EvaluatedVideo) -> dict[str, str]:
    return {"dataset": video.dataset, "scene": video.scene, "path": str(video.path)}


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


if __name__ == "__main__":
    main()
