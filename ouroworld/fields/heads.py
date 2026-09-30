"""MLP building blocks shared by the deformation fields."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def output_head(width: int, out_dim: int) -> nn.Sequential:
    """Two-layer head whose last layer starts at zero, so the field starts as identity."""
    head = nn.Sequential(nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, out_dim))
    _xavier(head)
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def input_layer(in_dim: int, width: int) -> nn.Linear:
    """Xavier-initialised first layer of a field MLP."""
    layer = nn.Linear(in_dim, width)
    _xavier(layer)
    return layer


def _xavier(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class ChunkedEvaluator:
    """Evaluate a per-point function in chunks, optionally with activation checkpointing.

    Per-point fields over millions of Gaussians would otherwise keep every
    intermediate activation alive for backward.
    """

    def __init__(self, chunk_size: int, use_checkpointing: bool):
        self.chunk_size = int(chunk_size)
        self.use_checkpointing = bool(use_checkpointing)

    def __call__(
        self,
        function: Callable[..., torch.Tensor],
        inputs: Sequence[torch.Tensor],
        out_width: int,
        is_training: bool,
    ) -> torch.Tensor:
        """Apply ``function`` to aligned row chunks of ``inputs`` and concatenate."""
        count = inputs[0].shape[0]
        if count == 0:
            return inputs[0].new_zeros((0, out_width))
        size = self.chunk_size if self.chunk_size > 0 else count
        use_checkpoint = self.use_checkpointing and is_training and torch.is_grad_enabled()
        outputs = []
        for start in range(0, count, size):
            chunk = tuple(tensor[start : start + size] for tensor in inputs)
            if use_checkpoint:
                outputs.append(checkpoint(function, *chunk, use_reentrant=False))
            else:
                outputs.append(function(*chunk))
        return torch.cat(outputs, dim=0)
