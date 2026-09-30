"""The ``VideoInpainter`` interface: fill the holes of a warped condition video."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch


@dataclass(frozen=True)
class ConditionVideo:
    """What the video model is asked to complete.

    Attributes:
        frames: ``(T, 3, H, W)`` condition colour in ``[0, 1]``.
        known: ``(T, 1, H, W)`` 1 where ``frames`` is to be kept, 0 in holes.
    """

    frames: torch.Tensor
    known: torch.Tensor


class VideoInpainter(Protocol):
    """A video diffusion model that completes a partially known video."""

    def inpaint(
        self, condition: ConditionVideo, reference: torch.Tensor, prompt: str, seed: int
    ) -> torch.Tensor:
        """Return ``(T, 3, H, W)`` float32 frames in ``[0, 1]``.

        Args:
            condition: The video to complete.
            reference: ``(R, 3, H, W)`` frames of the source video, for appearance.
            prompt: Text description of the scene.
            seed: Seed of the initial noise.
        """
        ...

    def release(self) -> None:
        """Free the model's memory; it is reloaded on the next call."""
        ...


class Captioner(Protocol):
    """Describes an image in text, for the video model's prompt."""

    def caption(self, image: np.ndarray) -> str:
        """Return the prompt for the ``(H, W, 3)`` uint8 ``image``."""
        ...
