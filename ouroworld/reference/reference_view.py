"""Choose the reference camera and render the reference image.

The reference camera keeps the import camera's field of view, centre-cropped
to the working aspect (16:9) and resized, at the pose of the selected
candidate. Every candidate is rendered as a thumbnail with a contact sheet so
the choice can be made by eye; the choice is a config value
(``reference.selected``), so rerunning the stage reproduces it.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ouroworld.geometry.camera import Camera, center_crop_intrinsics
from ouroworld.geometry.trajectories import project_world_point
from ouroworld.io.images import save_rgb8
from ouroworld.io.scene_package import (
    ScenePackage,
    SceneSource,
    load_scene_package,
    save_pivot,
    save_reference,
)
from ouroworld.reference.candidates import Candidate, CandidateSettings, candidate_poses
from ouroworld.reference.pivot import center_pivot
from ouroworld.render.static import StaticRenderer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReferenceSettings:
    """Reference-view settings.

    Attributes:
        selected: ``"reference"`` (the import pose) or a candidate name.
        width: Reference image width.
        height: Reference image height.
        center_patch_fraction: Patch of the centre-depth pivot fallback.
        candidates: The candidate grid.
        thumbnail_width: Width of the candidate thumbnails.
        min_coverage: A thumbnail with less rendered surface is marked invalid.
    """

    selected: str
    width: int
    height: int
    center_patch_fraction: float
    candidates: CandidateSettings
    thumbnail_width: int
    min_coverage: float


def choose_reference(
    source: SceneSource,
    renderer: StaticRenderer,
    settings: ReferenceSettings,
    diagnostics_dir: Path | None,
) -> ScenePackage:
    """Store the reference camera and image of ``source`` and return the complete package.

    Without a picked pivot, the centre-depth pivot of the import camera is
    stored first.

    Raises:
        ValueError: If ``settings.selected`` names no candidate.
    """
    camera = source.import_camera
    pivot = source.pivot
    if pivot is None:
        pivot = center_pivot(renderer.view(camera).depth, camera, settings.center_patch_fraction)
        save_pivot(source.root, pivot)
        logger.info("no pivot picked; using the centre depth %.4f", pivot.depth)
    candidates = candidate_poses(camera, pivot, source.world_up, settings.candidates)
    if diagnostics_dir is not None:
        write_candidate_sheet(candidates, camera, pivot.pixel, renderer, settings, diagnostics_dir)
    chosen = {candidate.name: candidate for candidate in candidates}.get(settings.selected)
    if chosen is None:
        raise ValueError(f"reference.selected={settings.selected!r} is not one of the candidates")
    size = (settings.width, settings.height)
    K = center_crop_intrinsics(camera.K, (camera.width, camera.height), size)
    reference = Camera(settings.width, settings.height, K, chosen.c2w, camera.znear, camera.zfar)
    save_reference(source.root, reference, renderer.image(reference), chosen.name)
    return load_scene_package(source.root)


def write_candidate_sheet(
    candidates: list[Candidate],
    camera: Camera,
    pivot_pixel: np.ndarray,
    renderer: StaticRenderer,
    settings: ReferenceSettings,
    out_dir: Path,
) -> None:
    """Render every candidate small; write the thumbnails, ``candidates.json`` and a sheet."""
    height = max(1, round(settings.thumbnail_width * camera.height / camera.width))
    thumbnail = camera.resized(settings.thumbnail_width, height)
    scale = np.array([thumbnail.width / camera.width, thumbnail.height / camera.height])
    column, row = np.clip(
        np.round(pivot_pixel * scale), 0, [thumbnail.width - 1, height - 1]
    ).astype(int)
    out_dir = Path(out_dir)
    (out_dir / "candidates").mkdir(parents=True, exist_ok=True)
    records, tiles = [], []
    for candidate in candidates:
        view = renderer.view(thumbnail.with_pose(candidate.c2w))
        save_rgb8(view.image, out_dir / "candidates" / f"{candidate.name}.png")
        coverage = float(np.mean(view.depth > 0))
        valid = coverage >= settings.min_coverage and bool(view.depth[row, column] > 0)
        records.append(
            {
                "name": candidate.name,
                "yaw_degrees": candidate.yaw,
                "pitch_degrees": candidate.pitch,
                "forward_fraction": candidate.forward_fraction,
                "coverage": coverage,
                "valid": valid,
                "c2w": candidate.c2w.tolist(),
            }
        )
        label = (
            f"{candidate.name} yaw={candidate.yaw:+.0f} pitch={candidate.pitch:+.0f} "
            f"f={candidate.forward_fraction:.2f}{'' if valid else ' INVALID'}"
        )
        tiles.append((view.image, label))
    (out_dir / "candidates.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    _contact_sheet(tiles).save(out_dir / "candidates.png")


def pivot_preview(image: np.ndarray, camera: Camera, pivot_world: np.ndarray) -> np.ndarray:
    """Mark the pivot on ``image`` of ``camera`` with a red ring."""
    pixel, _ = project_world_point(camera.K, camera.c2w, pivot_world)
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    radius = max(8, round(min(canvas.size) * 0.015))
    x, y = float(pixel[0]), float(pixel[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 40, 30), width=3)
    return np.array(canvas)


def _contact_sheet(tiles: list[tuple[np.ndarray, str]], columns: int = 5) -> Image.Image:
    height, width = tiles[0][0].shape[:2]
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * width, rows * (height + 20)), "black")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(tiles):
        x, y = (index % columns) * width, (index // columns) * (height + 20)
        sheet.paste(Image.fromarray(image), (x, y))
        draw.text((x + 4, y + height + 4), label, fill="white")
    return sheet
