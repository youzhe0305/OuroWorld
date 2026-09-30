"""``ouroworld-multiview``: generate the multi-view videos of a scene (paper §4.1).

Reads the scene package and the reference video, writes ``MultiviewVideos``.
An interrupted run can be restarted with the same command: finished views are
kept.

Example::

    ouroworld-multiview dataset=Lyra_2.0 scene=palace_rooftops
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from ouroworld.cli.common import add_config_arguments, config_from_arguments, setup_logging
from ouroworld.config.loader import save_config
from ouroworld.config.schema import Config
from ouroworld.generation.inpainting.caption import Blip2Captioner
from ouroworld.generation.inpainting.fusion import FusionSettings
from ouroworld.generation.inpainting.trajectorycrafter import (
    TrajectoryCrafterInpainter,
    TrajectoryCrafterSettings,
)
from ouroworld.generation.lifting.alignment import AlignmentSettings
from ouroworld.generation.lifting.vggt_omega import VggtOmegaLifter
from ouroworld.generation.pipeline import GenerationSettings, MultiviewGenerator
from ouroworld.io.reference_video import load_reference_video
from ouroworld.io.scene_package import load_scene_package

logger = logging.getLogger(__name__)

RUN_CONFIG = "config.yaml"


def build_generator(config: Config) -> MultiviewGenerator:
    """Compose the generator and its models from ``config.generation``."""
    generation = config.generation
    lifting, inpainting = generation.lifting, generation.inpainting
    fusion = inpainting.fusion
    settings = GenerationSettings(
        view_count=generation.orbit.view_count,
        max_yaw_degrees=generation.orbit.max_yaw_degrees,
        axis_tilt_degrees=generation.orbit.axis_tilt_degrees,
        frame_count=generation.frame_count,
        sweep_frames=inpainting.sweep_frames,
        appearance_frames=inpainting.appearance_frames,
        confidence_drop_percentile=lifting.confidence_drop_percentile,
        seed=inpainting.seed,
        alignment=AlignmentSettings(
            lifting.alignment_iterations, lifting.alignment_huber_k, lifting.min_valid_pixels
        ),
        fusion=FusionSettings(
            fusion.known_threshold,
            fusion.max_warp_alpha,
            fusion.erode_pixels,
            fusion.feather_pixels,
            fusion.feather_sigma,
        ),
        save_diagnostics=generation.save_diagnostics,
    )
    inpainter = TrajectoryCrafterInpainter(
        TrajectoryCrafterSettings(
            base_model_dir=Path(inpainting.base_model),
            transformer_dir=Path(inpainting.transformer),
            inference_steps=inpainting.inference_steps,
            guidance_scale=inpainting.guidance_scale,
            negative_prompt=inpainting.negative_prompt,
            cpu_offload=inpainting.cpu_offload,
            attention_query_chunk=inpainting.attention_query_chunk,
        )
    )
    return MultiviewGenerator(
        settings,
        VggtOmegaLifter(Path(lifting.checkpoint), lifting.image_resolution),
        inpainter,
        Blip2Captioner(Path(inpainting.caption_model), inpainting.caption_suffix),
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_arguments(parser)
    arguments = parser.parse_args(argv)
    setup_logging()
    # Must be set before CUDA initialises; the video model's activations fragment memory.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    config = config_from_arguments(arguments)
    out_dir = Path(config.paths.multiview_dir)
    save_config(config, out_dir / "intermediate" / RUN_CONFIG)
    package = load_scene_package(Path(config.paths.scene_dir))
    video = load_reference_video(Path(config.paths.reference_video_dir))
    videos = build_generator(config).run(package, video, out_dir)
    logger.info(
        "wrote %s: %d views, %d observations",
        out_dir,
        len(videos.cameras),
        len(videos.observations),
    )


if __name__ == "__main__":
    main()
