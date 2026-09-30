"""Seedance 2.0 through the BytePlus ModelArk task API (paper §3.1).

One task generates the whole loop: the reference image is both the first and
the last frame. A task costs money, so once it is created it is only polled,
never resubmitted, and its video is downloaded with retries before the
temporary URL expires.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

FAILED_STATES = ("failed", "cancelled", "canceled")


@dataclass(frozen=True)
class SeedanceSettings:
    """Generation and transport settings.

    Attributes:
        base_url: API root, e.g. ``https://ark.ap-southeast.bytepluses.com/api/v3``.
        model: Model id.
        resolution: Output resolution class, e.g. ``720p``.
        ratio: Aspect ratio, ``adaptive`` to follow the image.
        width: Width the image is resized to before upload (the output canvas).
        height: Height the image is resized to before upload.
        duration_seconds: Video length.
        additional_instruction: Direction appended to the prompt, may be empty.
        poll_interval_seconds: Wait between status requests.
        task_timeout_seconds: Give up on a task after this long.
        request_timeout_seconds: HTTP timeout of a single request.
        download_attempts: Tries to download the finished video.
    """

    base_url: str
    model: str
    resolution: str
    ratio: str
    width: int
    height: int
    duration_seconds: int
    additional_instruction: str
    poll_interval_seconds: float
    task_timeout_seconds: float
    request_timeout_seconds: float
    download_attempts: int = 3


class SeedanceGenerator:
    """Generates the looping video with one Seedance task."""

    def __init__(
        self,
        settings: SeedanceSettings,
        api_key: str,
        session: Any,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Reach the API through ``session`` (a ``requests.Session``) with ``api_key``.

        ``sleep`` and ``clock`` are replaced in tests.
        """
        if not api_key:
            raise ValueError("Seedance needs an API key")
        self.settings = settings
        self.api_key = api_key
        self.session = session
        self.sleep = sleep
        self.clock = clock

    def generate(self, image: np.ndarray, prompt: str, out_path: Path) -> dict[str, Any]:
        """Write the loop animating ``image`` to ``out_path`` and return its provenance."""
        conditioning = conditioning_png(image, self.settings.width, self.settings.height)
        image_url = "data:image/png;base64," + base64.b64encode(conditioning).decode("ascii")
        payload = build_payload(self.settings, prompt, image_url)
        task_id, task = self._run_task(payload, Path(out_path))
        return {
            "backend": "seedance",
            "model": self.settings.model,
            "task_id": task_id,
            "prompt": payload["content"][0]["text"],
            "resolution": self.settings.resolution,
            "ratio": self.settings.ratio,
            "duration_seconds": self.settings.duration_seconds,
            "conditioning_size": [self.settings.width, self.settings.height],
            "conditioning_sha256": hashlib.sha256(conditioning).hexdigest(),
            "completion_tokens": (task.get("usage") or {}).get("completion_tokens"),
        }

    def _run_task(self, payload: dict[str, Any], out_path: Path) -> tuple[str, dict[str, Any]]:
        import requests

        tasks_url = f"{self.settings.base_url.rstrip('/')}/contents/generations/tasks"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        timeout = self.settings.request_timeout_seconds
        try:
            response = self.session.post(tasks_url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as error:
            raise RuntimeError(f"Seedance task creation failed: {error}") from error
        task_id = _json(response, "task creation").get("id")
        if not task_id:
            raise RuntimeError("Seedance task creation returned no task id")
        logger.info("Seedance task %s created", task_id)
        task = self._wait(f"{tasks_url}/{task_id}", headers, task_id)
        video_url = (task.get("content") or {}).get("video_url")
        if not video_url:
            raise RuntimeError(f"Seedance task {task_id} succeeded without a video URL")
        self._download(video_url, out_path)
        return str(task_id), task

    def _wait(self, url: str, headers: dict[str, str], task_id: str) -> dict[str, Any]:
        import requests

        deadline = self.clock() + self.settings.task_timeout_seconds
        while self.clock() < deadline:
            try:
                response = self.session.get(
                    url, headers=headers, timeout=self.settings.request_timeout_seconds
                )
                task = _json(response, "task retrieval")
            except requests.RequestException as error:
                # The task exists remotely; a failed status request must not lead
                # to a second, paid, submission. Keep polling the same task.
                logger.warning("Seedance task %s: status request failed (%s)", task_id, error)
                self.sleep(self.settings.poll_interval_seconds)
                continue
            status = str(task.get("status", "")).lower()
            if status == "succeeded":
                return task
            if status in FAILED_STATES:
                raise RuntimeError(f"Seedance task {task_id} {status}: {task.get('error')}")
            logger.info("Seedance task %s: %s", task_id, status or "pending")
            self.sleep(self.settings.poll_interval_seconds)
        raise TimeoutError(
            f"Seedance task {task_id} did not finish within {self.settings.task_timeout_seconds} s"
        )

    def _download(self, url: str, out_path: Path) -> None:
        import requests

        for attempt in range(1, self.settings.download_attempts + 1):
            try:
                response = self.session.get(
                    url, stream=True, timeout=self.settings.request_timeout_seconds
                )
                response.raise_for_status()
                with out_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
                break
            except requests.RequestException as error:
                if attempt == self.settings.download_attempts:
                    raise RuntimeError(f"Seedance download failed: {error}") from error
                logger.warning("Seedance download failed (%s); retrying", error)
                self.sleep(min(self.settings.poll_interval_seconds, 10.0))
        if out_path.stat().st_size == 0:
            raise RuntimeError("Seedance returned an empty video")


def seedance_text(prompt: str, additional_instruction: str) -> str:
    """Seedance has no separate fields for extra direction; it goes after the prompt."""
    parts = [prompt.strip()]
    if additional_instruction.strip():
        parts.append("Additional direction: " + additional_instruction.strip())
    return " ".join(parts)


def build_payload(settings: SeedanceSettings, prompt: str, image_url: str) -> dict[str, Any]:
    """Return the task body: the image as both first and last frame."""
    frame = {"type": "image_url", "image_url": {"url": image_url}}
    return {
        "model": settings.model,
        "content": [
            {"type": "text", "text": seedance_text(prompt, settings.additional_instruction)},
            {**frame, "role": "first_frame"},
            {**frame, "role": "last_frame"},
        ],
        "resolution": settings.resolution,
        "ratio": settings.ratio,
        "duration": int(settings.duration_seconds),
        "generate_audio": False,
        "watermark": False,
    }


def conditioning_png(image: np.ndarray, width: int, height: int) -> bytes:
    """Resize ``image`` to the output canvas so the model does not reframe it; PNG bytes.

    Raises:
        ValueError: If that would change the aspect ratio.
    """
    source_height, source_width = image.shape[:2]
    if abs(source_width / source_height - width / height) > 1e-3:
        raise ValueError(
            f"the {source_width}x{source_height} image would be distorted to {width}x{height}"
        )
    buffer = io.BytesIO()
    Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def _json(response: Any, operation: str) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        data = None
    if not response.ok:
        detail = data if data is not None else response.text[:1000]
        raise RuntimeError(f"Seedance {operation} failed: HTTP {response.status_code}: {detail}")
    if not isinstance(data, dict):
        raise RuntimeError(f"Seedance {operation} returned a non-JSON response")
    return data
