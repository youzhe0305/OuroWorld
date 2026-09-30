"""TrajectoryCrafter (``third_party/TrajectoryCrafter``): the video inpainting diffusion model.

Two local patches are applied to the vendored modules, without editing them:

* diffusers >= 0.33 returns torch tensors from ``get_3d_sincos_pos_embed``;
  the upstream constructor expects numpy.
* the Perceiver cross-attention is evaluated in query chunks. This is the same
  softmax, computed blockwise, and avoids materialising the full attention
  matrix (out of memory on 24 GB cards).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

REPOSITORY = Path(__file__).resolve().parents[2] / "third_party" / "TrajectoryCrafter"


def load_trajectorycrafter(
    base_model_dir: Path,
    transformer_dir: Path,
    cpu_offload: str,
    attention_query_chunk: int,
) -> Any:  # noqa: ANN401 - the upstream pipeline class
    """Build the TrajectoryCrafter pipeline in bfloat16.

    Args:
        base_model_dir: CogVideoX-Fun-V1.1-5b-InP (VAE, T5 encoder, scheduler).
        transformer_dir: TrajectoryCrafter's cross-attention transformer.
        cpu_offload: ``model``, ``sequential`` or ``none`` (diffusers offloading).
        attention_query_chunk: Query rows per Perceiver attention block; 0 disables chunking.
    """
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))
    from diffusers import DDIMScheduler
    from models import crosstransformer3d
    from models.autoencoder_magvit import AutoencoderKLCogVideoX
    from models.pipeline_trajectorycrafter import TrajCrafter_Pipeline
    from transformers import T5EncoderModel

    _patch_position_embedding(crosstransformer3d)
    if attention_query_chunk > 0:
        crosstransformer3d.PerceiverCrossAttention.forward = _chunked_perceiver_attention(
            crosstransformer3d.reshape_tensor, attention_query_chunk
        )
    dtype = torch.bfloat16
    transformer = crosstransformer3d.CrossTransformer3DModel.from_pretrained(
        str(transformer_dir)
    ).to(dtype)
    vae = AutoencoderKLCogVideoX.from_pretrained(str(base_model_dir), subfolder="vae").to(dtype)
    text_encoder = T5EncoderModel.from_pretrained(
        str(base_model_dir), subfolder="text_encoder", torch_dtype=dtype
    )
    scheduler = DDIMScheduler.from_pretrained(str(base_model_dir), subfolder="scheduler")
    pipeline = TrajCrafter_Pipeline.from_pretrained(
        str(base_model_dir),
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=dtype,
    )
    if cpu_offload == "model":
        pipeline.enable_model_cpu_offload()
    elif cpu_offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    elif cpu_offload == "none":
        pipeline.to("cuda")
    else:
        raise ValueError(f"unknown cpu_offload mode {cpu_offload!r}")
    return pipeline


def _patch_position_embedding(module: Any) -> None:  # noqa: ANN401
    original = module.get_3d_sincos_pos_embed
    if getattr(original, "returns_numpy", False):
        return
    if "output_type" not in original.__code__.co_varnames:
        return

    def numpy_position_embedding(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        kwargs["output_type"] = "pt"
        return original(*args, **kwargs).cpu().numpy()

    numpy_position_embedding.returns_numpy = True  # type: ignore[attr-defined]
    module.get_3d_sincos_pos_embed = numpy_position_embedding


def _chunked_perceiver_attention(
    reshape_tensor: Callable[[torch.Tensor, int], torch.Tensor], query_chunk: int
) -> Callable[..., torch.Tensor]:
    def forward(module: Any, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:  # noqa: ANN401
        x = module.norm1(x)
        latents = module.norm2(latents)
        batch, length, _ = latents.shape
        queries = reshape_tensor(module.to_q(latents), module.heads)
        keys, values = module.to_kv(x).chunk(2, dim=-1)
        keys = reshape_tensor(keys, module.heads).transpose(-2, -1)
        values = reshape_tensor(values, module.heads)
        # Split the 1/sqrt(d) scale between queries and keys, as upstream does, for fp16 range.
        scale = module.dim_head**-0.25
        blocks = []
        for start in range(0, length, query_chunk):
            weight = (queries[:, :, start : start + query_chunk] * scale) @ (keys * scale)
            weight = torch.softmax(weight.float(), dim=-1).to(weight.dtype)
            blocks.append(weight @ values)
        output = torch.cat(blocks, dim=2).permute(0, 2, 1, 3).reshape(batch, length, -1)
        return module.to_out(output)

    return forward
