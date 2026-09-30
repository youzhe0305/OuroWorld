"""Reference-video stage with stand-in HTTP sessions and models (no network, no GPU)."""

import json
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

from ouroworld.generation.looping import (
    ContinuityError,
    LoopingVideoGenerator,
    LoopingVideoSettings,
    write_loop_frames,
)
from ouroworld.generation.prompt.base import MotionPrompt
from ouroworld.generation.prompt.responses import (
    ResponsesPromptWriter,
    ResponsesSettings,
    response_text,
)
from ouroworld.generation.video.continuity import ContinuityLimits, start_continuity
from ouroworld.generation.video.seedance import (
    SeedanceGenerator,
    SeedanceSettings,
    build_payload,
    conditioning_png,
)
from ouroworld.io.images import center_crop_resize
from ouroworld.io.ply import write_gaussian_ply
from ouroworld.io.scene_package import load_scene_package
from ouroworld.io.video import write_mp4
from tests.helpers import look_at_camera, random_gaussians, write_scene_metadata

SEEDANCE = SeedanceSettings(
    base_url="https://ark.test/api/v3",
    model="seedance-test",
    resolution="720p",
    ratio="adaptive",
    width=64,
    height=36,
    duration_seconds=10,
    additional_instruction="Larger motion.",
    poll_interval_seconds=1.0,
    task_timeout_seconds=30.0,
    request_timeout_seconds=5.0,
)
LIMITS = ContinuityLimits(0.1, 1.0, 0.05)


class Response:
    def __init__(self, status: int = 200, payload: object = None, body: bytes = b"") -> None:
        self.status_code, self.payload, self.body = status, payload, body
        self.ok = status < 400
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self) -> object:
        if self.payload is None:
            raise ValueError("no JSON")
        return self.payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self.body]


class Session:
    """Replays scripted responses (or raises scripted exceptions) and records the calls."""

    def __init__(self, posts: list, gets: list) -> None:
        self.posts, self.gets, self.calls = list(posts), list(gets), []

    def _next(self, queue: list, call: tuple) -> Response:
        self.calls.append(call)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def post(self, url: str, **kwargs: object) -> Response:
        return self._next(self.posts, ("POST", url, kwargs.get("json")))

    def get(self, url: str, **kwargs: object) -> Response:
        return self._next(self.gets, ("GET", url, None))


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _generator(session: Session, clock: Clock | None = None) -> SeedanceGenerator:
    clock = clock or Clock()
    return SeedanceGenerator(SEEDANCE, "key", session, sleep=clock.sleep, clock=clock)


def _image(width: int = 64, height: int = 36) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)


def test_seedance_payload_uses_the_image_as_both_ends() -> None:
    payload = build_payload(SEEDANCE, "Leaves sway. ", "data:image/png;base64,AAA")
    text, first, last = payload["content"]
    assert text["text"] == "Leaves sway. Additional direction: Larger motion."
    assert (first["role"], last["role"]) == ("first_frame", "last_frame")
    assert first["image_url"] == last["image_url"] == {"url": "data:image/png;base64,AAA"}
    assert payload["duration"] == 10 and payload["generate_audio"] is False
    with pytest.raises(ValueError, match="distorted"):
        conditioning_png(_image(64, 48), 64, 36)


def test_seedance_polls_through_errors_without_resubmitting(tmp_path: Path) -> None:
    session = Session(
        posts=[Response(payload={"id": "task-1"})],
        gets=[
            Response(payload={"status": "running"}),
            requests.ConnectionError("dropped"),
            Response(payload={"status": "succeeded", "content": {"video_url": "https://v"}}),
            requests.ConnectionError("download dropped"),
            Response(body=b"mp4"),
        ],
    )
    metadata = _generator(session).generate(_image(), "Leaves sway.", tmp_path / "raw.mp4")
    assert [call[0] for call in session.calls].count("POST") == 1
    assert (tmp_path / "raw.mp4").read_bytes() == b"mp4"
    assert metadata["task_id"] == "task-1" and metadata["conditioning_size"] == [64, 36]


@pytest.mark.parametrize(
    ("gets", "error", "match"),
    [
        ([Response(payload={"status": "failed", "error": "policy"})], RuntimeError, "failed"),
        ([Response(payload={"status": "succeeded", "content": {}})], RuntimeError, "video URL"),
        ([Response(payload={"status": "running"})] * 40, TimeoutError, "did not finish"),
        (
            [Response(payload={"status": "succeeded", "content": {"video_url": "u"}}), Response()],
            RuntimeError,
            "empty",
        ),
    ],
)
def test_seedance_failures(tmp_path: Path, gets: list, error: type, match: str) -> None:
    session = Session(posts=[Response(payload={"id": "task-1"})], gets=gets)
    with pytest.raises(error, match=match):
        _generator(session).generate(_image(), "x", tmp_path / "raw.mp4")


def test_seedance_rejected_task_creation(tmp_path: Path) -> None:
    session = Session(posts=[Response(401, {"error": "bad key"})], gets=[])
    with pytest.raises(RuntimeError, match="HTTP 401"):
        _generator(session).generate(_image(), "x", tmp_path / "raw.mp4")


def test_prompt_request_and_answer(tmp_path: Path) -> None:
    image = tmp_path / "reference.png"
    Image.fromarray(_image()).save(image)
    settings = ResponsesSettings("https://api.test/v1/", "gpt-test", "medium", "high", 800, 30, "")
    answer = {
        "id": "r1",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": " Waves roll. "}]}
        ],
    }
    session = Session(posts=[Response(payload=answer)], gets=[])
    prompt = ResponsesPromptWriter(settings, "key", session).write(image)
    assert prompt.text == "Waves roll." and prompt.metadata["response_id"] == "r1"
    method, url, body = session.calls[0]
    assert url == "https://api.test/v1/responses" and body["model"] == "gpt-test"
    content = body["input"][0]["content"]
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert body["instructions"].startswith("You write the final English prompt")
    with pytest.raises(RuntimeError, match="no output text"):
        response_text({"status": "incomplete", "output": [{"type": "reasoning"}]})
    failing = Session(posts=[Response(429, {"error": "rate"})], gets=[])
    with pytest.raises(RuntimeError, match="HTTP 429"):
        ResponsesPromptWriter(settings, "key", failing).write(image)


def _textured(shift: int = 0) -> np.ndarray:
    base = np.random.default_rng(1).integers(0, 255, (120, 200), dtype=np.uint8)
    import cv2

    base = cv2.GaussianBlur(base, (3, 3), 0)
    return np.roll(base, shift, axis=1)


def test_continuity_accepts_a_still_start_and_rejects_a_jump() -> None:
    assert start_continuity(_textured(), _textured(), LIMITS)["passed"]
    jump = start_continuity(_textured(), _textured(4), LIMITS)
    assert not jump["passed"] and jump["translation_pixels"] > 3
    assert not start_continuity(np.zeros((60, 80), np.uint8), np.zeros((60, 80), np.uint8), LIMITS)[
        "passed"
    ]


def test_centre_crop_keeps_the_middle() -> None:
    image = np.zeros((40, 100, 3), np.uint8)
    image[:, 30:70] = 255  # a 40x40 square in the middle
    out = center_crop_resize(image, (20, 20))
    assert out.shape == (20, 20, 3) and out.min() > 250


def test_frames_are_sampled_evenly_and_close_the_loop(tmp_path: Path) -> None:
    frames = [np.full((36, 64, 3), 10 * index, np.uint8) for index in range(9)]
    write_mp4(tmp_path / "raw.mp4", frames, fps=4)
    video = write_loop_frames(tmp_path / "raw.mp4", (32, 18), 5, tmp_path / "out")
    assert len(video) == 5 and video.cycle_seconds == 2.0
    assert np.allclose(video.times, [0, 0.25, 0.5, 0.75, 1.0])
    with pytest.raises(ValueError, match="fewer than"):
        write_loop_frames(tmp_path / "raw.mp4", (32, 18), 10, tmp_path / "out")


class FakePrompt:
    def __init__(self) -> None:
        self.calls = 0

    def write(self, image_path: Path) -> MotionPrompt:
        self.calls += 1
        return MotionPrompt("The river flows.", {"model": "fake"})


class FakeVideo:
    def __init__(self, jump: bool = False) -> None:
        self.calls, self.jump = 0, jump

    def generate(self, image: np.ndarray, prompt: str, out_path: Path) -> dict:
        self.calls += 1
        texture = np.repeat(_textured()[..., None], 3, axis=2)
        frames = [texture, np.roll(texture, 4 if self.jump else 0, axis=1)] + [texture] * 7
        write_mp4(out_path, frames, fps=4)
        return {"backend": "fake", "prompt": prompt}


def _package(tmp_path: Path):  # noqa: ANN202
    write_gaussian_ply(random_gaussians(), tmp_path / "scene" / "gaussians.ply")
    write_scene_metadata(tmp_path / "scene", look_at_camera(), np.zeros(3))
    return load_scene_package(tmp_path / "scene")


def test_stage_writes_the_artifact_and_never_pays_twice(tmp_path: Path) -> None:
    package = _package(tmp_path)
    prompt, video = FakePrompt(), FakeVideo()
    stage = LoopingVideoGenerator(LoopingVideoSettings(5, LIMITS), prompt, video)
    result = stage.run(package, tmp_path / "reference_video")
    assert len(result) == 5 and (tmp_path / "reference_video" / "prompt.txt").is_file()
    assert (
        json.loads((tmp_path / "reference_video" / "generation.json").read_text())["prompt"]
        == "The river flows."
    )
    frame = np.asarray(Image.open(result.frame_paths[0]))
    assert frame.shape == (36, 64, 3)
    stage.run(package, tmp_path / "reference_video")
    assert (prompt.calls, video.calls) == (1, 1)


def test_stage_rejects_a_jumping_start(tmp_path: Path) -> None:
    stage = LoopingVideoGenerator(
        LoopingVideoSettings(5, LIMITS), FakePrompt(), FakeVideo(jump=True)
    )
    with pytest.raises(ContinuityError):
        stage.run(_package(tmp_path), tmp_path / "reference_video")
    assert not (tmp_path / "reference_video" / "reference_video.json").exists()
