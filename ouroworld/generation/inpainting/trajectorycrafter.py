"""TrajectoryCrafter as the :class:`VideoInpainter` (paper §4.1)."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ouroworld.adapters.trajectorycrafter import load_trajectorycrafter
from ouroworld.generation.inpainting.base import ConditionVideo

# The video model works at this resolution (its training size).
HEIGHT, WIDTH = 384, 672
# It generates at most this many frames, of the form 4k + 1.
MAX_FRAMES = 49


@dataclass(frozen=True)
class TrajectoryCrafterSettings:
    """Sampling settings of TrajectoryCrafter.

    Attributes:
        base_model_dir: CogVideoX-Fun-V1.1-5b-InP checkpoint.
        transformer_dir: TrajectoryCrafter transformer checkpoint.
        inference_steps: DDIM steps.
        guidance_scale: Classifier-free guidance scale.
        negative_prompt: Negative text prompt.
        cpu_offload: ``model``, ``sequential`` or ``none``.
        attention_query_chunk: Query rows per cross-attention block (0: unchunked).
    """

    base_model_dir: Path
    transformer_dir: Path
    inference_steps: int
    guidance_scale: float
    negative_prompt: str
    cpu_offload: str
    attention_query_chunk: int


class TrajectoryCrafterInpainter:
    """Loads the pipeline on first use and keeps it for the following views."""

    def __init__(self, settings: TrajectoryCrafterSettings) -> None:
        """Store ``settings``; nothing is loaded yet."""
        self.settings = settings
        self._pipeline: Any = None

    def inpaint(
        self, condition: ConditionVideo, reference: torch.Tensor, prompt: str, seed: int
    ) -> torch.Tensor:
        """See :meth:`VideoInpainter.inpaint`."""
        frames = len(condition.frames)
        if frames > MAX_FRAMES or (frames - 1) % 4:
            raise ValueError(f"TrajectoryCrafter needs 4k + 1 <= {MAX_FRAMES} frames, got {frames}")
        if tuple(condition.frames.shape[-2:]) != (HEIGHT, WIDTH):
            raise ValueError(f"condition must be {HEIGHT}x{WIDTH}")
        pipeline = self._load()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        # The pipeline also perturbs the condition with noise from the global generator
        # (``add_noise_to_reference_video``); seeding it makes every view reproducible.
        torch.manual_seed(seed)
        with torch.inference_mode():
            output = pipeline(
                prompt,
                num_frames=frames,
                negative_prompt=self.settings.negative_prompt,
                height=HEIGHT,
                width=WIDTH,
                generator=generator,
                guidance_scale=self.settings.guidance_scale,
                num_inference_steps=self.settings.inference_steps,
                # (1, 3, T, H, W) in [0, 1]; the mask is 255 in holes.
                video=condition.frames.permute(1, 0, 2, 3).unsqueeze(0),
                mask_video=(1.0 - condition.known.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0,
                reference=reference.permute(1, 0, 2, 3).unsqueeze(0),
            ).videos
        video = output[0, :, :frames].permute(1, 0, 2, 3).float().cpu()
        if len(video) != frames:
            raise RuntimeError(f"TrajectoryCrafter decoded {len(video)} frames, expected {frames}")
        return video

    def release(self) -> None:
        """Free the pipeline's memory."""
        self._pipeline = None
        gc.collect()
        torch.cuda.empty_cache()

    def _load(self) -> Any:  # noqa: ANN401 - the upstream pipeline class
        if self._pipeline is None:
            self._pipeline = load_trajectorycrafter(
                self.settings.base_model_dir,
                self.settings.transformer_dir,
                self.settings.cpu_offload,
                self.settings.attention_query_chunk,
            )
        return self._pipeline
