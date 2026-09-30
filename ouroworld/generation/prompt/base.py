"""The prompt-writer interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class MotionPrompt:
    """Text sent to the video model, with how it was obtained.

    Attributes:
        text: The prompt, used verbatim.
        metadata: Provenance (model, request id, token usage, ...), JSON-serialisable.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptWriter(Protocol):
    """Describes the motion that brings a still image to life."""

    def write(self, image_path: Path) -> MotionPrompt:
        """Return the video prompt for the reference image at ``image_path``."""
        ...
