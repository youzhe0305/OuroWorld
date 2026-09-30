"""Wan 2.2 image-to-video (A14B MoE) through diffusers: the open-source option.

The reference image is passed as both the first and the last image. Wan 2.2
I2V is not a dedicated first-last-frame model, so the loop closes less
reliably than with Seedance; the paper uses Seedance.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ouroworld.adapters.torch_compat import allow_sdpa_enable_gqa
from ouroworld.io.video import write_mp4

logger = logging.getLogger(__name__)

MEMORY_MODES = ("model", "sequential", "cuda")


@dataclass(frozen=True)
class WanSettings:
    """Model, memory and sampling settings.

    Attributes:
        model: Hugging Face id of the diffusers weights.
        cache_dir: Download cache of the weights.
        memory_mode: ``model`` / ``sequential`` CPU offload, or ``cuda`` (all on the GPU).
        max_area: Output pixel budget; the resolution follows the image's aspect.
        frame_count: Frames to generate (4k + 1).
        inference_steps: Denoising steps.
        guidance_scale: Classifier-free guidance.
        fps: Frame rate of the written video.
        seed: Sampling seed.
    """

    model: str
    cache_dir: Path
    memory_mode: str
    max_area: int
    frame_count: int
    inference_steps: int
    guidance_scale: float
    fps: int
    seed: int


class WanGenerator:
    """Generates the looping video with Wan 2.2 I2V."""

    def __init__(self, settings: WanSettings) -> None:
        """Validate ``settings``; the pipeline loads on first use."""
        if settings.memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {MEMORY_MODES}")
        if (settings.frame_count - 1) % 4:
            raise ValueError("Wan needs 4k + 1 frames")
        self.settings = settings

    def generate(self, image: np.ndarray, prompt: str, out_path: Path) -> dict[str, Any]:
        """Write the loop animating ``image`` to ``out_path`` and return its provenance."""
        import torch
        from diffusers import WanImageToVideoPipeline
        from PIL import Image

        allow_sdpa_enable_gqa()
        settings = self.settings
        pipe = WanImageToVideoPipeline.from_pretrained(
            settings.model, torch_dtype=torch.bfloat16, cache_dir=str(settings.cache_dir)
        )
        if settings.memory_mode == "model":
            pipe.enable_model_cpu_offload()
        elif settings.memory_mode == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.vae.enable_tiling()
        # Both the VAE's spatial stride and the transformer's patch must divide the size.
        multiple = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        height, width = image.shape[:2]
        size = wan_resolution(width, height, settings.max_area, multiple)
        frame = Image.fromarray(image).resize(size, Image.Resampling.LANCZOS)
        logger.info("Wan: %d frames at %dx%d", settings.frame_count, *size)
        frames = pipe(
            image=frame,
            last_image=frame,
            prompt=prompt,
            negative_prompt="",
            width=size[0],
            height=size[1],
            num_frames=settings.frame_count,
            guidance_scale=settings.guidance_scale,
            num_inference_steps=settings.inference_steps,
            generator=torch.Generator("cuda").manual_seed(settings.seed),
            output_type="np",
        ).frames[0]
        write_mp4(
            out_path,
            ((np.clip(item, 0, 1) * 255).round().astype(np.uint8) for item in frames),
            settings.fps,
        )
        return {
            "backend": "wan",
            "model": settings.model,
            "seed": settings.seed,
            "size": list(size),
            "frame_count": settings.frame_count,
            "fps": settings.fps,
            "inference_steps": settings.inference_steps,
            "guidance_scale": settings.guidance_scale,
        }


def wan_resolution(width: int, height: int, max_area: int, multiple: int) -> tuple[int, int]:
    """Largest ``(width, height)`` of the image's aspect within ``max_area``, snapped down."""
    aspect = height / width
    out_height = round(math.sqrt(max_area * aspect)) // multiple * multiple
    out_width = round(math.sqrt(max_area / aspect)) // multiple * multiple
    return int(out_width), int(out_height)
