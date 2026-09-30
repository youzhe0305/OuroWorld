"""``ouroworld-reference-video``: generate the looping reference video (paper §3.1).

Reads the scene package's reference image, asks GPT for a motion prompt and
Seedance (or Wan) for the loop, and writes the ``ReferenceVideo``. API keys
come from the environment or the repository's ``.env``. The prompt and the raw
video are kept and reused on a rerun, so a failed later step never pays twice.

Example::

    ouroworld-reference-video dataset=Lyra_2.0 scene=palace_rooftops
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from ouroworld.cli.common import (
    REPOSITORY_ROOT,
    add_config_arguments,
    config_from_arguments,
    load_dotenv,
    require_env,
    setup_logging,
)
from ouroworld.config.loader import save_config
from ouroworld.config.schema import Config
from ouroworld.generation.looping import (
    PROMPT_FILE,
    RAW_VIDEO_FILE,
    LoopingVideoGenerator,
    LoopingVideoSettings,
)
from ouroworld.generation.prompt.responses import ResponsesPromptWriter, ResponsesSettings
from ouroworld.generation.video.base import VideoGenerator
from ouroworld.generation.video.continuity import ContinuityLimits
from ouroworld.generation.video.seedance import SeedanceGenerator, SeedanceSettings
from ouroworld.generation.video.wan import WanGenerator, WanSettings
from ouroworld.io.scene_package import load_scene_package

logger = logging.getLogger(__name__)

RUN_CONFIG = "config.yaml"


def build_generator(
    config: Config, need_prompt: bool = True, need_video: bool = True
) -> LoopingVideoGenerator:
    """Compose the stage from ``config.reference_video``.

    Only the models whose output is still missing are built, so a rerun that
    reuses ``prompt.txt`` and ``raw.mp4`` needs no API key.
    """
    import requests

    settings = config.reference_video
    session = requests.Session()
    prompt = settings.prompt
    writer = (
        None
        if not need_prompt
        else ResponsesPromptWriter(
            ResponsesSettings(
                base_url=os.environ.get(prompt.base_url_env, "").strip() or prompt.default_base_url,
                model=prompt.model,
                reasoning_effort=prompt.reasoning_effort,
                image_detail=prompt.image_detail,
                max_output_tokens=prompt.max_output_tokens,
                timeout_seconds=prompt.timeout_seconds,
                additional_instruction=prompt.additional_instruction,
            ),
            require_env(prompt.api_key_env),
            session,
        )
    )
    video: VideoGenerator | None = None
    if need_video and settings.backend == "seedance":
        seedance = settings.seedance
        video = SeedanceGenerator(
            SeedanceSettings(
                base_url=seedance.base_url,
                model=seedance.model,
                resolution=seedance.resolution,
                ratio=seedance.ratio,
                width=seedance.width,
                height=seedance.height,
                duration_seconds=seedance.duration_seconds,
                additional_instruction=seedance.additional_instruction,
                poll_interval_seconds=seedance.poll_interval_seconds,
                task_timeout_seconds=seedance.task_timeout_seconds,
                request_timeout_seconds=seedance.request_timeout_seconds,
            ),
            require_env(seedance.api_key_env),
            session,
        )
    elif need_video:
        wan = settings.wan
        video = WanGenerator(
            WanSettings(
                model=wan.model,
                cache_dir=REPOSITORY_ROOT / wan.cache_dir,
                memory_mode=wan.memory_mode,
                max_area=wan.max_area,
                frame_count=wan.frame_count,
                inference_steps=wan.inference_steps,
                guidance_scale=wan.guidance_scale,
                fps=wan.fps,
                seed=wan.seed,
            )
        )
    continuity = settings.continuity
    limits = (
        ContinuityLimits(
            continuity.max_scale_change_percent,
            continuity.max_translation_pixels,
            continuity.max_rotation_degrees,
        )
        if continuity.enabled
        else None
    )
    return LoopingVideoGenerator(LoopingVideoSettings(settings.frame_count, limits), writer, video)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_arguments(parser)
    arguments = parser.parse_args(argv)
    setup_logging()
    load_dotenv()
    config = config_from_arguments(arguments)
    out_dir = Path(config.paths.reference_video_dir)
    package = load_scene_package(Path(config.paths.scene_dir))
    generator = build_generator(
        config,
        need_prompt=not (out_dir / PROMPT_FILE).is_file(),
        need_video=not (out_dir / RAW_VIDEO_FILE).is_file(),
    )
    save_config(config, out_dir / RUN_CONFIG)
    video = generator.run(package, out_dir)
    logger.info("wrote %s: %d frames over %.2f s", out_dir, len(video), video.cycle_seconds)


if __name__ == "__main__":
    main()
