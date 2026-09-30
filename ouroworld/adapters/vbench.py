"""VBench (``third_party/VBench``, with two local patches listed in its UPSTREAM_COMMIT)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

VBENCH_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "VBench"


def run_vbench(
    videos_dir: Path,
    prompts: dict[str, str],
    dimensions: list[str],
    out_dir: Path,
    weights_dir: Path,
    device: str,
) -> dict[str, dict[str, float]]:
    """Score every MP4 in ``videos_dir`` on ``dimensions``.

    Args:
        videos_dir: Directory of the videos (VBench's ``custom_input`` mode).
        prompts: Text of each video, by file name; needed by ``overall_consistency``.
        dimensions: VBench dimension names.
        out_dir: Where VBench writes its own result files.
        weights_dir: VBench's pretrained-model directory (``VBENCH_CACHE_DIR``).
        device: ``cuda`` or ``cpu``.

    Returns:
        ``{dimension: {file name: score}}``.
    """
    # VBench resolves every checkpoint from this variable at import time.
    os.environ["VBENCH_CACHE_DIR"] = str(weights_dir)
    if str(VBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(VBENCH_ROOT))
    import torch
    from vbench import VBench

    name = "vbench"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    VBench(
        torch.device(device), str(VBENCH_ROOT / "vbench" / "VBench_full_info.json"), str(out_dir)
    ).evaluate(
        videos_path=str(videos_dir),
        name=name,
        prompt_list=prompts,
        dimension_list=dimensions,
        local=True,
        read_frame=False,
        mode="custom_input",
        imaging_quality_preprocessing_mode="longer",
    )
    results = json.loads((out_dir / f"{name}_eval_results.json").read_text(encoding="utf-8"))
    return {
        dimension: {Path(item["video_path"]).name: float(item["video_results"]) for item in entries}
        for dimension, (_, entries) in results.items()
    }
