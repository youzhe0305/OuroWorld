"""GPT motion prompts through the OpenAI Responses API (paper §3.1, Fig. 8).

The model sees the reference image and the instructions of Fig. 8
(``instructions.txt``) and its whole answer is the video prompt; nothing is
added or parsed out of it.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ouroworld.generation.prompt.base import MotionPrompt

INSTRUCTIONS_PATH = Path(__file__).with_name("instructions.txt")
REQUEST_TEXT = "Use this reference image to write the final video-generation prompt."
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@dataclass(frozen=True)
class ResponsesSettings:
    """One Responses API request.

    Attributes:
        base_url: API root, e.g. ``https://api.openai.com/v1``.
        model: Model name.
        reasoning_effort: ``reasoning.effort`` of the request.
        image_detail: Detail level of the input image.
        max_output_tokens: Upper bound on the answer, reasoning included.
        timeout_seconds: HTTP timeout.
        additional_instruction: Extra direction appended to the request, may be empty.
    """

    base_url: str
    model: str
    reasoning_effort: str
    image_detail: str
    max_output_tokens: int
    timeout_seconds: float
    additional_instruction: str


class ResponsesPromptWriter:
    """Writes the prompt with one ``POST /responses`` call."""

    def __init__(self, settings: ResponsesSettings, api_key: str, session: Any) -> None:
        """Use ``session`` (a ``requests.Session``) to reach the API with ``api_key``."""
        if not api_key:
            raise ValueError("the Responses API needs an API key")
        self.settings = settings
        self.api_key = api_key
        self.session = session

    def write(self, image_path: Path) -> MotionPrompt:
        """Return the video prompt for the reference image at ``image_path``."""
        import requests

        instructions = INSTRUCTIONS_PATH.read_text(encoding="utf-8").strip()
        payload = build_request(instructions, Path(image_path), self.settings)
        try:
            response = self.session.post(
                f"{self.settings.base_url.rstrip('/')}/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
        except requests.RequestException as error:
            raise RuntimeError(f"prompt request failed: {error}") from error
        try:
            data = response.json()
        except ValueError as error:
            raise RuntimeError(
                f"prompt request returned non-JSON HTTP {response.status_code}: "
                f"{response.text[:500]}"
            ) from error
        if not response.ok:
            raise RuntimeError(f"prompt request returned HTTP {response.status_code}: {data}")
        return MotionPrompt(
            response_text(data),
            {
                "model": str(data.get("model") or self.settings.model),
                "response_id": data.get("id"),
                "reasoning_effort": self.settings.reasoning_effort,
                "image_detail": self.settings.image_detail,
                "additional_instruction": self.settings.additional_instruction,
                "input_image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
                "usage": data.get("usage") if isinstance(data.get("usage"), dict) else {},
            },
        )


def build_request(
    instructions: str, image_path: Path, settings: ResponsesSettings
) -> dict[str, Any]:
    """Return the JSON body of the request."""
    mime = IMAGE_TYPES.get(image_path.suffix.lower())
    if mime is None:
        raise ValueError(f"unsupported image type: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    text = REQUEST_TEXT
    if settings.additional_instruction.strip():
        text += f" Additional direction for this request: {settings.additional_instruction.strip()}"
    return {
        "model": settings.model,
        "instructions": instructions,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{encoded}",
                        "detail": settings.image_detail,
                    },
                ],
            }
        ],
        "reasoning": {"effort": settings.reasoning_effort},
        "max_output_tokens": settings.max_output_tokens,
    }


def response_text(data: dict[str, Any]) -> str:
    """Return the answer of a Responses API reply.

    Raises:
        RuntimeError: If the reply holds no text, e.g. when reasoning used up
            ``max_output_tokens``.
    """
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    for item in data.get("output", []):
        if isinstance(item, dict) and item.get("type") == "message":
            for content in item.get("content", []):
                text = content.get("text") if isinstance(content, dict) else None
                if content.get("type") == "output_text" and isinstance(text, str) and text.strip():
                    return text.strip()
    raise RuntimeError(f"the reply holds no output text (status {data.get('status')!r})")
