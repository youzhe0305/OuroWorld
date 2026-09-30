"""Camera trajectories around a scene pivot.

The training orbit turns the reference camera about the pivot to get the
views the multi-view generator fills in (paper §4.1). The evaluation orbit
swings horizontally around the pivot at a fraction of the reference camera's
distance, following 0 -> +max -> -max -> 0 so the path is closed (§4.3).
"""

from __future__ import annotations

import math

import numpy as np


def normalize(vector: np.ndarray, name: str = "vector") -> np.ndarray:
    """Return ``vector`` scaled to unit length; reject zero or non-finite input."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"{name} has zero or invalid norm")
    return vector / norm


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the ``(3, 3)`` rotation by ``angle`` radians about ``axis`` (Rodrigues)."""
    axis = normalize(axis, "rotation axis")
    skew = _skew(axis)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the minimal rotation taking direction ``source`` onto ``target``."""
    source = normalize(source, "source direction")
    target = normalize(target, "target direction")
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        axis = _any_perpendicular(source)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = _skew(cross / sine)
    return np.eye(3) + sine * skew + (1.0 - cosine) * skew @ skew


def pixel_ray(K: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    """Return the unit camera-space ray through ``pixel``."""
    homogeneous = np.linalg.solve(np.asarray(K, dtype=np.float64), [pixel[0], pixel[1], 1.0])
    return normalize(homogeneous, "pixel ray")


def unproject_z_depth(
    K: np.ndarray, c2w: np.ndarray, pixel: np.ndarray, depth: float
) -> np.ndarray:
    """Lift ``pixel`` at camera-z ``depth`` to a world-space point."""
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError("depth must be finite and positive")
    camera_point = np.linalg.solve(np.asarray(K, dtype=np.float64), [pixel[0], pixel[1], 1.0])
    camera_point = camera_point * float(depth)
    c2w = np.asarray(c2w, dtype=np.float64)
    return c2w[:3, :3] @ camera_point + c2w[:3, 3]


def project_world_point(
    K: np.ndarray, c2w: np.ndarray, point: np.ndarray
) -> tuple[np.ndarray, float]:
    """Project a world point; return its pixel and camera-z depth."""
    c2w = np.asarray(c2w, dtype=np.float64)
    camera_point = c2w[:3, :3].T @ (np.asarray(point, dtype=np.float64) - c2w[:3, 3])
    depth = float(camera_point[2])
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("world point is not in front of the camera")
    homogeneous = np.asarray(K, dtype=np.float64) @ camera_point
    return homogeneous[:2] / homogeneous[2], depth


def uniform_horizontal_angles(
    frame_count: int, negative_degrees: float, positive_degrees: float
) -> np.ndarray:
    """Sample the closed path 0 -> positive -> negative -> 0 at uniform arc length.

    Returns:
        ``(frame_count,)`` yaw angles in degrees; the first and last are 0.
    """
    if frame_count < 2:
        raise ValueError("an orbit needs at least two frames")
    if negative_degrees > 0 or positive_degrees < 0:
        raise ValueError("orbit angles must satisfy negative <= 0 <= positive")
    first = positive_degrees
    second = positive_degrees - negative_degrees
    total = first + second - negative_degrees
    distances = np.linspace(0.0, total, frame_count)
    return np.where(
        distances <= first,
        distances,
        np.where(
            distances <= first + second,
            positive_degrees - (distances - first),
            negative_degrees + (distances - first - second),
        ),
    )


def look_at_pivot(
    reference_c2w: np.ndarray,
    K: np.ndarray,
    target_pixel: np.ndarray,
    center: np.ndarray,
    pivot: np.ndarray,
    world_up: np.ndarray,
) -> np.ndarray:
    """Place a camera at ``center`` so ``target_pixel`` sees ``pivot``.

    The reference orientation is rotated minimally, then rolled about the
    viewing ray so the image "up" stays aligned with ``world_up``.
    """
    reference_c2w = np.asarray(reference_c2w, dtype=np.float64)
    source_ray = reference_c2w[:3, :3] @ pixel_ray(K, target_pixel)
    target_ray = normalize(np.asarray(pivot) - np.asarray(center), "viewing ray")
    rotation = rotation_between(source_ray, target_ray) @ reference_c2w[:3, :3]

    current_up = -rotation[:, 1]
    current_up = current_up - target_ray * (current_up @ target_ray)
    desired_up = np.asarray(world_up, dtype=np.float64)
    desired_up = desired_up - target_ray * (desired_up @ target_ray)
    if np.linalg.norm(current_up) > 1e-8 and np.linalg.norm(desired_up) > 1e-8:
        current_up = normalize(current_up)
        desired_up = normalize(desired_up)
        roll = math.atan2(
            float(target_ray @ np.cross(current_up, desired_up)),
            float(np.clip(current_up @ desired_up, -1.0, 1.0)),
        )
        rotation = axis_angle_rotation(target_ray, roll) @ rotation

    c2w = np.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = np.asarray(center, dtype=np.float64)
    return c2w


def horizontal_orbit(
    reference_c2w: np.ndarray,
    K: np.ndarray,
    pivot: np.ndarray,
    target_pixel: np.ndarray,
    angles_degrees: np.ndarray,
    radius_scale: float,
) -> np.ndarray:
    """Return ``(N, 4, 4)`` c2w poses orbiting ``pivot`` about the reference up axis.

    Args:
        reference_c2w: Pose the orbit is centred on.
        K: Intrinsics shared by every pose.
        pivot: World point every pose keeps at ``target_pixel``.
        target_pixel: Pixel of the reference image that sees ``pivot``.
        angles_degrees: Yaw of each pose relative to the reference.
        radius_scale: Orbit radius as a fraction of the reference-to-pivot distance.
    """
    if not np.isfinite(radius_scale) or radius_scale <= 0.0:
        raise ValueError("orbit radius_scale must be finite and positive")
    reference_c2w = np.asarray(reference_c2w, dtype=np.float64)
    pivot = np.asarray(pivot, dtype=np.float64)
    offset = (reference_c2w[:3, 3] - pivot) * radius_scale
    world_up = normalize(-reference_c2w[:3, 1], "reference camera up")
    poses = []
    for angle_degrees in angles_degrees:
        angle = math.radians(float(angle_degrees))
        if abs(angle) < 1e-12 and abs(radius_scale - 1.0) < 1e-12:
            poses.append(reference_c2w.copy())
            continue
        center = pivot + axis_angle_rotation(world_up, angle) @ offset
        poses.append(look_at_pivot(reference_c2w, K, target_pixel, center, pivot, world_up))
    return np.stack(poses)


def yaw_pitch_directions(
    radial: np.ndarray, world_up: np.ndarray, yaws: np.ndarray, pitches: np.ndarray
) -> list[tuple[float, float, np.ndarray]]:
    """Turn the unit direction ``radial`` by every (yaw, pitch) pair, in degrees.

    Yaw turns about ``world_up``; pitch then tilts towards it. Pitch varies
    slowest in the returned ``(yaw, pitch, direction)`` list.
    """
    radial = normalize(radial, "radial direction")
    world_up = normalize(world_up, "world up")
    samples = []
    for pitch in pitches:
        for yaw in yaws:
            yawed = normalize(axis_angle_rotation(world_up, math.radians(float(yaw))) @ radial)
            pitch_axis = np.cross(world_up, yawed)
            if np.linalg.norm(pitch_axis) < 1e-8:
                pitch_axis = _any_perpendicular(yawed)
            direction = axis_angle_rotation(pitch_axis, math.radians(float(pitch))) @ yawed
            samples.append((float(yaw), float(pitch), normalize(direction)))
    return samples


def uniform_yaw_angles(view_count: int, max_degrees: float) -> np.ndarray:
    """Return ``view_count`` yaw angles evenly spaced in ``[-max_degrees, max_degrees]``."""
    if view_count < 2 or view_count % 2 == 0:
        raise ValueError("the training orbit needs an odd number of views (>= 3)")
    if not np.isfinite(max_degrees) or max_degrees <= 0:
        raise ValueError("max_degrees must be finite and positive")
    return np.linspace(-max_degrees, max_degrees, view_count)


def pivot_orbit(
    reference_c2w: np.ndarray,
    pivot: np.ndarray,
    yaw_degrees: np.ndarray,
    axis_tilt_degrees: float,
) -> np.ndarray:
    """Return the ``(N, 4, 4)`` training cameras: the reference camera turned about the pivot.

    The rotation axis passes through ``pivot`` and is the reference camera's
    vertical axis tilted by ``axis_tilt_degrees`` about its x axis (the
    "elevation" of the ViewCrafter trajectory convention the multi-view
    generator was built on). A yaw of 0 reproduces the reference camera.
    """
    reference_c2w = np.asarray(reference_c2w, dtype=np.float64)
    tilt = math.radians(180.0 - axis_tilt_degrees)
    # Orbit frame -> reference camera frame; its origin is the pivot.
    orbit_to_camera = np.eye(4)
    orbit_to_camera[1:3, 1:3] = [
        [math.cos(tilt), math.sin(tilt)],
        [-math.sin(tilt), math.cos(tilt)],
    ]
    orbit_to_camera[:3, 3] = (np.linalg.inv(reference_c2w) @ np.r_[pivot, 1.0])[:3]
    camera_in_orbit = np.linalg.inv(orbit_to_camera)
    poses = []
    for yaw in yaw_degrees:
        turn = np.eye(4)
        turn[:3, :3] = _rotation_y(math.radians(float(yaw)))
        poses.append(reference_c2w @ orbit_to_camera @ (turn @ camera_in_orbit))
    return np.stack(poses)


def pivot_sweep(
    source_c2w: np.ndarray, target_c2w: np.ndarray, pivot: np.ndarray, frame_count: int
) -> np.ndarray:
    """Return ``(frame_count, 4, 4)`` poses rotating ``source`` onto ``target`` about ``pivot``.

    ``R(a) = exp(a w) R_s`` and ``C(a) = p + exp(a w) (C_s - p)`` with
    ``exp(w) = R_t R_s^T`` and ``a`` uniform in ``[0, 1]``. The endpoints are
    the input poses themselves, so their renders match the fixed views exactly.
    """
    source = np.asarray(source_c2w, dtype=np.float64)
    target = np.asarray(target_c2w, dtype=np.float64)
    if frame_count < 2:
        raise ValueError("a sweep needs at least two frames")
    rotation_vector = _rotation_vector(target[:3, :3] @ source[:3, :3].T)
    offset = source[:3, 3] - np.asarray(pivot, dtype=np.float64)
    poses = [source.copy()]
    for index in range(1, frame_count - 1):
        rotation = _rotation_from_vector(rotation_vector * (index / (frame_count - 1)))
        pose = np.eye(4)
        pose[:3, :3] = rotation @ source[:3, :3]
        pose[:3, 3] = pivot + rotation @ offset
        poses.append(pose)
    poses.append(target.copy())
    return np.stack(poses)


def _rotation_y(angle: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(matrix).as_rotvec()


def _rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-15:
        return np.eye(3)
    skew = _skew(vector / angle)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _skew(vector: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )


def _any_perpendicular(axis: np.ndarray) -> np.ndarray:
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    return normalize(np.cross(helper, axis))
