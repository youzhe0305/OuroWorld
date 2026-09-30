"""Naturalness: KVD to a natural-video distribution (paper §4.4).

KVD (Kernel Video Distance) is the unbiased polynomial-kernel MMD² between
I3D features of two video sets, what KID is to FID. The reference set is 20
MiniMax-H3 videos per scene, generated from the scene's reference image and
motion prompt. Each video is cut into 10 s clips (one loop), each resampled
to 60 frames at 224x224 (6 FPS, which lands on real frames of both the 24 FPS
references and the 30 FPS renders), and KVD is averaged over 100 random
equal-sized subsets.

I3D features are cached per video file, keyed by its path, size and mtime.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ouroworld.evaluation.videos import EvaluatedVideo

logger = logging.getLogger(__name__)

FEATURE_DIM = 400
I3D_MIN_FRAMES = 16  # shorter clips fail inside the TorchScript
I3D_FRAME_SIZE = 224  # larger inputs skip the final pooling
CACHE_FORMAT = 1


@dataclass(frozen=True)
class NaturalnessSettings:
    """Clip sampling and KVD estimation.

    Attributes:
        clip_seconds: Clip length.
        clip_stride_seconds: Distance between clip starts.
        clip_frames: Frames per clip, evenly spaced in time.
        frame_size: I3D input side.
        max_clips_per_video: Clip limit per video, 0 for none.
        subset_size: Clips per set in one KVD estimate.
        subset_count: KVD estimates averaged.
        kernel_degree: Polynomial kernel degree.
        kernel_coefficient: Polynomial kernel constant.
        seed: Seed of the subset draws.
    """

    clip_seconds: float = 10.0
    clip_stride_seconds: float = 10.0
    clip_frames: int = 60
    frame_size: int = 224
    max_clips_per_video: int = 0
    subset_size: int = 100
    subset_count: int = 100
    kernel_degree: int = 3
    kernel_coefficient: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.clip_frames < I3D_MIN_FRAMES or self.frame_size != I3D_FRAME_SIZE:
            raise ValueError(
                f"I3D needs >= {I3D_MIN_FRAMES} frames of {I3D_FRAME_SIZE}x{I3D_FRAME_SIZE}"
            )


def clip_start_times(duration: float, settings: NaturalnessSettings) -> list[float]:
    """Start times of the clips that fit in ``duration`` seconds."""
    starts: list[float] = []
    start = 0.0
    while start + settings.clip_seconds <= duration + 1e-6:
        starts.append(start)
        if settings.max_clips_per_video and len(starts) >= settings.max_clips_per_video:
            break
        start += settings.clip_stride_seconds
    return starts


def clip_frame_indices(
    start: float, fps: float, frame_count: int, settings: NaturalnessSettings
) -> list[int]:
    """Nearest frames of ``clip_frames`` instants evenly spaced over one clip."""
    step = settings.clip_seconds / settings.clip_frames
    indices = []
    for position in range(settings.clip_frames):
        index = int(math.floor((start + position * step) * fps + 0.5))
        indices.append(min(max(index, 0), frame_count - 1))
    return indices


def read_clips(path: Path, settings: NaturalnessSettings) -> list[np.ndarray]:
    """The ``(T, S, S, 3)`` uint8 RGB clips of one video (aspect not kept, as in FVD)."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    plans = [
        clip_frame_indices(start, fps, frame_count, settings)
        for start in clip_start_times(frame_count / fps, settings)
    ]
    wanted = sorted({index for plan in plans for index in plan})
    frames: dict[int, np.ndarray] = {}
    index, cursor = -1, 0
    try:
        while cursor < len(wanted):
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            while cursor < len(wanted) and wanted[cursor] == index:
                size = (settings.frame_size, settings.frame_size)
                resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                frames[index] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                cursor += 1
    finally:
        capture.release()
    if cursor < len(wanted):
        raise ValueError(f"{path}: decoding stopped before frame {wanted[cursor]}")
    return [np.stack([frames[i] for i in plan]) for plan in plans]


class I3dFeatures:
    """I3D (Kinetics-400, StyleGAN-V TorchScript) logits of video clips, with a disk cache."""

    def __init__(
        self,
        weights: Path,
        device: str,
        settings: NaturalnessSettings,
        cache_dir: Path | None,
        batch_size: int = 16,
    ) -> None:
        """Load the TorchScript at ``weights``; ``cache_dir`` None disables the cache."""
        self.model = torch.jit.load(str(weights)).eval().to(torch.device(device))
        self.weights, self.device, self.settings = Path(weights), device, settings
        self.batch_size = batch_size
        self.cache_dir = None if cache_dir is None else Path(cache_dir) / self._namespace()

    def __call__(self, path: Path) -> np.ndarray:
        """``(clips, 400)`` float64 features of one video."""
        entry = None if self.cache_dir is None else self.cache_dir / _cache_key(path)
        if entry is not None and entry.is_file():
            with np.load(entry, allow_pickle=False) as bundle:
                return bundle["features"].astype(np.float64)
        clips = read_clips(path, self.settings)
        features = self._features(clips)
        if entry is not None:
            entry.parent.mkdir(parents=True, exist_ok=True)
            partial = entry.with_name(f".{entry.stem}.partial.npz")
            np.savez_compressed(partial, features=features.astype(np.float32))
            partial.replace(entry)
        # Cached features are float32; return the same precision either way.
        return features.astype(np.float32).astype(np.float64)

    def _features(self, clips: list[np.ndarray]) -> np.ndarray:
        if not clips:
            return np.zeros((0, FEATURE_DIM))
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(clips), self.batch_size):
                batch = torch.from_numpy(np.stack(clips[start : start + self.batch_size]))
                # (B, T, H, W, C) uint8 -> (B, C, T, H, W) in [-1, 1].
                batch = batch.to(self.device).float().permute(0, 4, 1, 2, 3) / 127.5 - 1.0
                output = self.model(batch, rescale=False, resize=False, return_features=True)
                outputs.append(output.float().cpu().numpy())
        return np.concatenate(outputs).astype(np.float64)

    def _namespace(self) -> str:
        """Directory name covering everything that changes a video's features."""
        payload = {
            key: value
            for key, value in asdict(self.settings).items()
            if key.startswith(("clip_", "frame_", "max_"))
        }
        payload.update(device=self.device, weights=self.weights.name, format=CACHE_FORMAT)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        return f"T{self.settings.clip_frames}_S{self.settings.frame_size}_{digest}"


def polynomial_mmd2(
    reference: np.ndarray, generated: np.ndarray, degree: int, coefficient: float
) -> float:
    """Unbiased MMD² under ``k(x, y) = (<x, y> / d + coefficient) ** degree``.

    The within-set diagonals are dropped, so the estimate can dip below zero
    for two samples of one distribution.
    """
    gamma = 1.0 / reference.shape[1]

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return (gamma * (left @ right.T) + coefficient) ** degree

    m, n = reference.shape[0], generated.shape[0]
    k_rr, k_ff = kernel(reference, reference), kernel(generated, generated)
    sum_rr = float(k_rr.sum() - np.trace(k_rr))
    sum_ff = float(k_ff.sum() - np.trace(k_ff))
    return (
        sum_rr / (m * (m - 1))
        + sum_ff / (n * (n - 1))
        - 2.0 * float(kernel(reference, generated).mean())
    )


def kvd(
    reference: np.ndarray, generated: np.ndarray, settings: NaturalnessSettings
) -> dict[str, Any]:
    """Mean and standard deviation of KVD over random subsets of both sets."""
    size = min(settings.subset_size, reference.shape[0], generated.shape[0])
    if size < 2:
        raise ValueError(
            f"KVD needs two clips per set, got {reference.shape[0]} and {generated.shape[0]}"
        )
    generator = np.random.default_rng(settings.seed)
    values = []
    for _ in range(settings.subset_count):
        reference_subset = reference[generator.choice(reference.shape[0], size, replace=False)]
        generated_subset = generated[generator.choice(generated.shape[0], size, replace=False)]
        values.append(
            polynomial_mmd2(
                reference_subset,
                generated_subset,
                settings.kernel_degree,
                settings.kernel_coefficient,
            )
        )
    array = np.asarray(values)
    return {
        "kvd": float(array.mean()),
        "kvd_std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "subset_size": size,
        "reference_clips": int(reference.shape[0]),
        "generated_clips": int(generated.shape[0]),
    }


def reference_videos(
    root: Path,
    pattern: str,
    scenes: set[tuple[str, str]],
    dataset_dirs: Mapping[str, str] | None = None,
) -> list[EvaluatedVideo]:
    """Reference videos ``<root>/<dataset>/<scene>/<pattern>`` of the given scenes.

    ``dataset_dirs`` maps a dataset name to its directory under ``root`` where
    they differ; the order stays that of the dataset names, as KVD needs.
    """
    dataset_dirs = dataset_dirs or {}
    videos = []
    for dataset, scene in sorted(scenes):
        directory = Path(root) / dataset_dirs.get(dataset, dataset) / scene
        for path in sorted(directory.glob(pattern)):
            videos.append(EvaluatedVideo(path.resolve(), dataset, scene))
    return videos


def naturalness(
    generated: list[EvaluatedVideo],
    reference_root: Path,
    reference_pattern: str,
    features: I3dFeatures,
    settings: NaturalnessSettings,
    dataset_dirs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """KVD of ``generated`` against the reference videos of the same scenes.

    Scenes without reference videos are left out of both sets.
    """
    scenes = {(video.dataset, video.scene) for video in generated}
    references = reference_videos(reference_root, reference_pattern, scenes, dataset_dirs)
    covered = {(video.dataset, video.scene) for video in references}
    kept = [video for video in generated if (video.dataset, video.scene) in covered]
    logger.info("KVD: %d generated and %d reference videos", len(kept), len(references))
    reference_features = np.concatenate([features(video.path) for video in references])
    generated_features = np.concatenate([features(video.path) for video in kept])
    return {
        **kvd(reference_features, generated_features, settings),
        "scenes": len(covered),
        "scenes_without_reference": sorted(f"{d}/{s}" for d, s in scenes - covered),
    }


def _cache_key(path: Path) -> str:
    status = Path(path).stat()
    identity = json.dumps([str(Path(path).resolve()), status.st_size, status.st_mtime_ns])
    return hashlib.sha256(identity.encode()).hexdigest() + ".npz"
