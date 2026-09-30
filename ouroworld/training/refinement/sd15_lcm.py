"""SD1.5 + LCM-LoRA refinement with noise rectification (App. C).

A render is encoded, noised to an early timestep, and denoised in a few LCM
steps. At every step the predicted noise is blended with the noise that
would reproduce the guide image (a generated observation), with weights
``w_i`` that decay over the steps. The last clean-image prediction is the
refined target.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefinerSettings:
    """Diffusion refiner configuration.

    Attributes:
        stable_diffusion: Hugging Face id or local path of Stable Diffusion 1.5.
        lcm_lora: Path of the LCM-LoRA weights for SD1.5.
        inference_steps: Length of the LCM timestep schedule.
        rectification_weights: ``w_i`` of each denoising step; its length is the
            number of steps, which also fixes the starting noise level.
        prompt: Text prompt of the denoiser.
        dtype: ``float16`` or ``float32`` for the diffusion model.
    """

    stable_diffusion: str
    lcm_lora: str
    inference_steps: int
    rectification_weights: tuple[float, ...]
    prompt: str
    dtype: str

    @property
    def denoise_steps(self) -> int:
        """Number of denoising steps per refinement."""
        return len(self.rectification_weights)


class LcmRefiner:
    """Implements :class:`~ouroworld.training.refinement.base.DiffusionRefiner`."""

    def __init__(self, settings: RefinerSettings, device: torch.device | str = "cuda"):
        from diffusers import LCMScheduler, StableDiffusionPipeline

        dtypes = {"float16": torch.float16, "float32": torch.float32}
        if settings.dtype not in dtypes:
            raise ValueError(f"refiner dtype must be one of {sorted(dtypes)}")
        if not Path(settings.lcm_lora).is_file():
            raise FileNotFoundError(f"LCM-LoRA weights not found: {settings.lcm_lora}")
        self.settings = settings
        self.device = torch.device(device)
        pipeline = StableDiffusionPipeline.from_pretrained(
            settings.stable_diffusion, torch_dtype=dtypes[settings.dtype]
        )
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        pipeline.load_lora_weights(settings.lcm_lora)
        pipeline.fuse_lora()
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except (ImportError, ModuleNotFoundError, ValueError):
            logger.info("xformers unavailable; using default attention")
        pipeline.to(self.device)
        # LCM runs without classifier-free guidance, so only the positive prompt matters.
        self.prompt_embeds = pipeline.encode_prompt(settings.prompt, self.device, 1, False)[0]
        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self.scheduler = pipeline.scheduler
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        self.scheduler.set_timesteps(settings.inference_steps, device=self.device)
        self.timesteps = self.scheduler.timesteps[-settings.denoise_steps :]

    @torch.no_grad()
    def refine(self, renders: torch.Tensor, guides: torch.Tensor) -> torch.Tensor:
        """Refine ``(B, 3, H, W)`` renders towards ``guides``, both in ``[0, 1]``."""
        self.scheduler.set_timesteps(self.settings.inference_steps, device=self.device)
        latents = self._encode(renders)
        noise = torch.randn_like(latents)
        start = self.timesteps[0].repeat(latents.shape[0]).to(self.device)
        latents = self.scheduler.add_noise(latents, noise, start)
        guide_latents = None
        refined = renders
        for step, weight in enumerate(self.settings.rectification_weights):
            timestep = self.timesteps[step]
            predicted_noise = self._predict_noise(latents, timestep)
            refined = self._predict_clean_image(latents, predicted_noise, timestep)
            if guide_latents is None:
                guide_latents = self._encode(guides)
            rectified = self._rectify(latents, predicted_noise, guide_latents, timestep, weight)
            latents = self.scheduler.step(rectified, timestep, latents).prev_sample
        return refined.to(renders.dtype)

    def _predict_noise(self, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        embeds = self.prompt_embeds.repeat(latents.shape[0], 1, 1)
        model_input = self.scheduler.scale_model_input(latents, timestep)
        return self.unet(
            model_input, timestep.reshape(1), encoder_hidden_states=embeds, return_dict=False
        )[0]

    def _predict_clean_image(
        self, latents: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """LCM's clean-image estimate at ``timestep``, decoded to ``[0, 1]``.

        ``scheduler.step`` is the only public way to get LCM's boundary-condition
        estimate, so it is called and its step counter rolled back.
        """
        denoised = self.scheduler.step(noise, timestep, latents).denoised
        self.scheduler._step_index -= 1  # noqa: SLF001 - undo the probing step
        return (self._decode(denoised) + 1) / 2

    def _rectify(
        self,
        latents: torch.Tensor,
        predicted_noise: torch.Tensor,
        guide_latents: torch.Tensor,
        timestep: torch.Tensor,
        weight: float,
    ) -> torch.Tensor:
        """Blend the predicted noise with the noise that leads to the guide (Eq. 5)."""
        alphas = self.scheduler.alphas_cumprod.to(dtype=latents.dtype, device=latents.device)
        alpha = alphas[timestep.reshape(1)].flatten().view(-1, 1, 1, 1)
        guide_noise = (latents - alpha**0.5 * guide_latents) / (1 - alpha) ** 0.5
        dims = list(range(1, guide_noise.ndim))
        guide_noise = (
            guide_noise
            / guide_noise.std(dim=dims, keepdim=True)
            * predicted_noise.std(dim=dims, keepdim=True)
        )
        return predicted_noise * (1.0 - weight) + guide_noise * weight

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` images in ``[0, 1]`` to scaled latent means."""
        pixels = (images * 2 - 1).to(dtype=self.vae.dtype)
        moments = self.vae.quant_conv(self.vae.encoder(pixels))
        mean, _ = torch.chunk(moments, 2, dim=1)
        return mean * self.vae.config.scaling_factor

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Scaled latents to images in ``[-1, 1]``."""
        return self.vae.decoder(self.vae.post_quant_conv(latents / self.vae.config.scaling_factor))
