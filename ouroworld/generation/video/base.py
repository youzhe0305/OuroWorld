"""The video-generator interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np


class VideoGenerator(Protocol):
    """Animates a still image into a video that starts and ends on it."""

    def generate(self, image: np.ndarray, prompt: str, out_path: Path) -> dict[str, Any]:
        """Write an MP4 animating ``(H, W, 3)`` uint8 ``image`` to ``out_path``.

        Returns:
            JSON-serialisable provenance of the generation.
        """
        ...
