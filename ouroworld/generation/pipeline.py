"""Multi-view video generation (paper §4.1).

``ScenePackage`` + ``ReferenceVideo`` -> ``MultiviewVideos`` in four steps:

1. Training cameras: the reference camera turned about the pivot.
2. ``t = 0`` frames: renders of the input 3DGS from every camera.
3. Depth of the reference video, jointly with the ``t = 0`` renders, aligned
   to the 3DGS depth of the reference view.
4. Per novel view: warp the reference video into it, prepend a camera sweep,
   let the video model fill the holes, blend the reliable warp back in.

Everything the steps compute is kept under ``intermediate/`` for inspection;
the depth and the prompt are reused when a run is resumed, and finished views
are skipped.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ouroworld.generation.inpainting.base import Captioner, ConditionVideo, VideoInpainter
from ouroworld.generation.inpainting.direct_prefix import (
    assemble_condition,
    resize_frames,
    sweep_cameras,
)
from ouroworld.generation.inpainting.fusion import FusionSettings, fuse
from ouroworld.generation.inpainting.trajectorycrafter import HEIGHT, WIDTH
from ouroworld.generation.inpainting.warp import warp_frames
from ouroworld.generation.lifting.alignment import (
    AlignmentSettings,
    DisparityAffine,
    confident_pixels,
    fit_disparity_affine,
)
from ouroworld.generation.lifting.base import DepthLifter
from ouroworld.geometry.camera import Camera
from ouroworld.geometry.trajectories import pivot_orbit, uniform_yaw_angles
from ouroworld.io.images import read_rgb8, save_rgb8, to_rgb8
from ouroworld.io.multiview import MultiviewVideos, load_multiview, save_multiview_index
from ouroworld.io.reference_video import ReferenceVideo
from ouroworld.io.scene_package import ScenePackage
from ouroworld.io.video import write_mp4
from ouroworld.render.static import StaticRenderer, StaticView, load_static_renderer

logger = logging.getLogger(__name__)

INTERMEDIATE = "intermediate"
DIAGNOSTICS_FPS = 10


@dataclass(frozen=True)
class GenerationSettings:
    """Multi-view generation settings.

    Attributes:
        view_count: Number of training cameras (odd; the middle one is the reference).
        max_yaw_degrees: The cameras span ``[-max, max]`` degrees of yaw.
        axis_tilt_degrees: Tilt of the orbit axis (see :func:`pivot_orbit`).
        frame_count: Frames per generated video, both loop ends included (4k + 1).
        sweep_frames: Frames of the camera-sweep prefix of the condition.
        appearance_frames: Leading reference frames shown to the video model.
        confidence_drop_percentile: Per frame, the least confident depth pixels
            (this percentile) are not warped.
        seed: Initial noise seed, shared by every view.
        alignment: Depth alignment.
        fusion: Warp-over-generation blend.
        save_diagnostics: Also write condition and raw output videos per view.
    """

    view_count: int
    max_yaw_degrees: float
    axis_tilt_degrees: float
    frame_count: int
    sweep_frames: int
    appearance_frames: int
    confidence_drop_percentile: float
    seed: int
    alignment: AlignmentSettings
    fusion: FusionSettings
    save_diagnostics: bool


@dataclass(frozen=True)
class ReferenceGeometry:
    """The reference video at the reference camera, ready to warp.

    Attributes:
        frames: ``(T, 3, H, W)`` float32 in ``[0, 1]``, one per generated time.
        depth: ``(T, H, W)`` depth aligned to the input 3DGS.
        valid: ``(T, H, W)`` pixels that may be warped.
        affine: The disparity alignment that was applied.
    """

    frames: torch.Tensor
    depth: torch.Tensor
    valid: torch.Tensor
    affine: DisparityAffine


class MultiviewGenerator:
    """Runs the steps of the module docstring for one scene."""

    def __init__(
        self,
        settings: GenerationSettings,
        lifter: DepthLifter,
        inpainter: VideoInpainter,
        captioner: Captioner,
    ) -> None:
        """Compose the generator from its models."""
        self.settings = settings
        self.lifter = lifter
        self.inpainter = inpainter
        self.captioner = captioner

    def run(self, package: ScenePackage, video: ReferenceVideo, out_dir: Path) -> MultiviewVideos:
        """Write the ``MultiviewVideos`` of the scene to ``out_dir`` and return it."""
        if package.pivot is None:
            raise ValueError(f"{package.root} has no pivot; multi-view generation needs one")
        out_dir = Path(out_dir)
        work = out_dir / INTERMEDIATE
        work.mkdir(parents=True, exist_ok=True)
        cameras = training_cameras(package.reference_camera, package.pivot, self.settings)
        reference_id = len(cameras) // 2
        renderer = _static_renderer(package)
        static = [renderer.view(camera) for camera in cameras]
        for view_id, view in enumerate(static):
            save_rgb8(view.image, _t0_path(work, view_id))

        times = video.evenly_spaced(self.settings.frame_count)
        size = (package.reference_camera.width, package.reference_camera.height)
        frames = [read_rgb8(video.frame_paths[index], size) for index in times]
        geometry = self._reference_geometry(static, reference_id, frames, work)
        prompt = self._prompt(frames[len(frames) // 2], work)

        for view_id in range(len(cameras)):
            if view_id == reference_id or _is_complete(out_dir, view_id, len(times) - 1):
                continue
            logger.info("generating view %d/%d", view_id, len(cameras) - 1)
            images = self._generate_view(
                renderer,
                cameras,
                static,
                (reference_id, view_id),
                package.pivot,
                geometry,
                prompt,
                work,
            )
            _write_view(out_dir, view_id, images[: len(times) - 1])
        self.inpainter.release()
        if not _is_complete(out_dir, reference_id, len(video) - 1):
            _write_view(out_dir, reference_id, _reference_frames(video, static[reference_id]))

        output_cameras = {
            view_id: camera.resized(WIDTH, HEIGHT) for view_id, camera in enumerate(cameras)
        }
        records = _observation_records(video, times, output_cameras, reference_id)
        save_multiview_index(out_dir, output_cameras, records, reference_id)
        _write_report(work, self.settings, geometry.affine, prompt)
        return load_multiview(out_dir)

    def _reference_geometry(
        self,
        static: list[StaticView],
        reference_id: int,
        frames: list[np.ndarray],
        work: Path,
    ) -> ReferenceGeometry:
        cache = work / "reference_depth.npz"
        if cache.is_file():
            stored = np.load(cache)
            depth, confidence = stored["depth"], stored["confidence"]
        else:
            # t = 0 renders of every view, then the reference video after its first frame
            # (the reference render stands in for it).
            maps = self.lifter.lift([view.image for view in static] + frames[1:])
            order = [reference_id, *range(len(static), len(static) + len(frames) - 1)]
            depth, confidence = maps.depth[order], maps.confidence[order]
            np.savez(cache, depth=depth, confidence=confidence)
        affine = fit_disparity_affine(
            torch.from_numpy(static[reference_id].depth),
            torch.from_numpy(depth[0]),
            self.settings.alignment,
        )
        aligned = affine.apply(torch.from_numpy(depth))
        valid = confident_pixels(
            aligned, torch.from_numpy(confidence), self.settings.confidence_drop_percentile
        )
        logger.info(
            "depth alignment 1/z = %.6g/z_pred %+.6g (%.1f%% inliers)",
            affine.scale,
            affine.shift,
            100 * affine.inlier_fraction,
        )
        rgb = torch.stack(
            [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames]
        )
        return ReferenceGeometry(rgb, aligned, valid, affine)

    def _prompt(self, frame: np.ndarray, work: Path) -> str:
        cache = work / "prompt.txt"
        if cache.is_file():
            return cache.read_text(encoding="utf-8")
        prompt = self.captioner.caption(frame)
        cache.write_text(prompt, encoding="utf-8")
        logger.info("prompt: %s", prompt)
        return prompt

    def _generate_view(
        self,
        renderer: StaticRenderer,
        cameras: list[Camera],
        static: list[StaticView],
        view_pair: tuple[int, int],
        pivot: np.ndarray,
        geometry: ReferenceGeometry,
        prompt: str,
        work: Path,
    ) -> list[np.ndarray]:
        """Return the frames of one novel view, ``t = 0`` to ``t = 1`` inclusive."""
        reference_id, view_id = view_pair
        reference, target = cameras[reference_id], cameras[view_id]
        K = torch.tensor(np.float32(reference.K))
        warped, known = warp_frames(
            geometry.frames,
            geometry.depth,
            geometry.valid,
            K,
            reference.c2w,
            [target.c2w] * len(geometry.frames),
        )
        # At t = 0 the target view is known exactly: it is the 3DGS render.
        warped[0] = torch.from_numpy(static[view_id].image).permute(2, 0, 1).float() / 255.0
        known[0] = 1.0
        sweep = sweep_cameras(reference, target, pivot, self.settings.sweep_frames)
        prefix = [static[reference_id].image]
        prefix += [renderer.image(camera) for camera in sweep[1:-1]]
        prefix.append(static[view_id].image)
        condition = assemble_condition(prefix, warped, known, HEIGHT, WIDTH)
        appearance = resize_frames(
            geometry.frames[: self.settings.appearance_frames], HEIGHT, WIDTH
        )
        output = self.inpainter.inpaint(condition, appearance, prompt, self.settings.seed)
        kept = slice(len(prefix), None)
        fused = fuse(
            output[kept], condition.frames[kept], condition.known[kept], self.settings.fusion
        )
        fused[0] = condition.frames[len(prefix)]
        if self.settings.save_diagnostics:
            _save_diagnostics(work / "views" / f"view_{view_id:02d}", condition, output)
        return [to_rgb8(frame) for frame in fused]


def training_cameras(
    reference: Camera, pivot: np.ndarray, settings: GenerationSettings
) -> list[Camera]:
    """Return the training cameras at the reference camera's resolution."""
    yaws = uniform_yaw_angles(settings.view_count, settings.max_yaw_degrees)
    poses = pivot_orbit(reference.c2w, pivot, yaws, settings.axis_tilt_degrees)
    return [reference.with_pose(pose) for pose in poses]


def _static_renderer(package: ScenePackage) -> StaticRenderer:
    return load_static_renderer(package.gaussians_path, package.background)


def _reference_frames(video: ReferenceVideo, reference_t0: StaticView) -> list[np.ndarray]:
    """The reference video at the output size; its first frame is the 3DGS render."""
    frames = [read_rgb8(path, (WIDTH, HEIGHT)) for path in video.frame_paths[:-1]]
    render = torch.from_numpy(reference_t0.image).permute(2, 0, 1).float()[None] / 255.0
    frames[0] = to_rgb8(resize_frames(render, HEIGHT, WIDTH)[0])
    return frames


def _observation_records(
    video: ReferenceVideo, times: list[int], cameras: dict[int, Camera], reference_id: int
) -> list[dict[str, object]]:
    # The last frame closes the loop (t = 1, an alias of t = 0) and is not an observation.
    records = []
    for view_id in cameras:
        indices = range(len(video) - 1) if view_id == reference_id else times[:-1]
        for frame, index in enumerate(indices):
            records.append(
                {
                    "view_id": view_id,
                    "time": float(video.times[index]),
                    "path": _frame_name(view_id, frame),
                }
            )
    return records


def _frame_name(view_id: int, frame: int) -> str:
    return f"frames/view_{view_id:02d}/frame_{frame:04d}.jpg"


def _t0_path(work: Path, view_id: int) -> Path:
    path = work / "t0" / f"view_{view_id:02d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _is_complete(out_dir: Path, view_id: int, frame_count: int) -> bool:
    folder = out_dir / "frames" / f"view_{view_id:02d}"
    return folder.is_dir() and len(list(folder.glob("frame_*.jpg"))) == frame_count


def _write_view(out_dir: Path, view_id: int, images: list[np.ndarray]) -> None:
    """Write through a ``.partial`` folder so an interrupted view is never taken as done."""
    folder = out_dir / "frames" / f"view_{view_id:02d}"
    partial = folder.with_name(folder.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for frame, image in enumerate(images):
        save_rgb8(image, partial / f"frame_{frame:04d}.jpg")
    shutil.rmtree(folder, ignore_errors=True)
    partial.replace(folder)


def _save_diagnostics(folder: Path, condition: ConditionVideo, output: torch.Tensor) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    frames = (to_rgb8(frame) for frame in condition.frames)
    known = (to_rgb8(mask.expand(3, -1, -1)) for mask in condition.known)
    write_mp4(folder / "condition.mp4", frames, DIAGNOSTICS_FPS)
    write_mp4(folder / "known.mp4", known, DIAGNOSTICS_FPS)
    write_mp4(folder / "generated.mp4", (to_rgb8(frame) for frame in output), DIAGNOSTICS_FPS)


def _write_report(
    work: Path, settings: GenerationSettings, affine: DisparityAffine, prompt: str
) -> None:
    report = {
        "settings": asdict(settings),
        "depth_alignment": asdict(affine),
        "prompt": prompt,
    }
    (work / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
