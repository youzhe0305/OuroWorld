"""Candidate reference cameras around the pivot.

The import camera is often not a good reference view: too close, too far, or
seeing the scene from an edge. Candidates sit on a yaw/pitch grid around the
pivot at the import camera's distance, each optionally moved forward along
its viewing direction, and keep the pivot at the pixel it was picked at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ouroworld.geometry.camera import Camera
from ouroworld.geometry.trajectories import look_at_pivot, normalize, yaw_pitch_directions
from ouroworld.io.scene_package import PivotAnnotation

IMPORT_POSE = "reference"


@dataclass(frozen=True)
class CandidateSettings:
    """The candidate grid.

    Attributes:
        horizontal_degrees: Yaw range ``[-h, h]``.
        vertical_degrees: Pitch range ``[-v, v]``.
        horizontal_steps: Odd number of yaw samples, so the import pose is on the grid.
        vertical_steps: Odd number of pitch samples.
        forward_fractions: Moves along the viewing direction, as fractions of the
            pivot distance; each gives a full grid.
    """

    horizontal_degrees: float
    vertical_degrees: float
    horizontal_steps: int
    vertical_steps: int
    forward_fractions: tuple[float, ...]


@dataclass(frozen=True)
class Candidate:
    """One candidate pose.

    Attributes:
        name: ``"reference"`` for the import pose, ``"candidate_XXX"`` otherwise.
        c2w: ``(4, 4)`` camera-to-world pose.
        yaw: Yaw about the pivot in degrees.
        pitch: Pitch about the pivot in degrees.
        forward_fraction: Move along the viewing direction.
    """

    name: str
    c2w: np.ndarray
    yaw: float
    pitch: float
    forward_fraction: float


def candidate_poses(
    camera: Camera, pivot: PivotAnnotation, world_up: np.ndarray, settings: CandidateSettings
) -> list[Candidate]:
    """The import pose followed by the grid candidates, numbered in order."""
    if settings.horizontal_steps % 2 == 0 or settings.vertical_steps % 2 == 0:
        raise ValueError("candidate grid steps must be odd so the import pose is on the grid")
    if not all(0.0 <= fraction < 1.0 for fraction in settings.forward_fractions):
        raise ValueError("forward fractions must lie in [0, 1)")
    radial = normalize(camera.center - pivot.world, "pivot-to-camera direction")
    radius = float(np.linalg.norm(camera.center - pivot.world))
    samples = yaw_pitch_directions(
        radial,
        world_up,
        np.linspace(
            -settings.horizontal_degrees, settings.horizontal_degrees, settings.horizontal_steps
        ),
        np.linspace(-settings.vertical_degrees, settings.vertical_degrees, settings.vertical_steps),
    )
    candidates = [Candidate(IMPORT_POSE, camera.c2w.copy(), 0.0, 0.0, 0.0)]
    for fraction in settings.forward_fractions:
        for yaw, pitch, direction in samples:
            if fraction == 0.0 and yaw == 0.0 and pitch == 0.0:
                continue  # the import pose itself
            c2w = look_at_pivot(
                camera.c2w,
                camera.K,
                pivot.pixel,
                pivot.world + radius * direction,
                pivot.world,
                world_up,
            )
            # Column 2 of an OpenCV c2w is the viewing direction.
            c2w[:3, 3] += c2w[:3, 2] * radius * fraction
            name = f"candidate_{len(candidates):03d}"
            candidates.append(Candidate(name, c2w, yaw, pitch, fraction))
    return candidates
