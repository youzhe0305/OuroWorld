"""Differentiable SE(3) exponential map acting on Gaussian centres and orientations.

Quaternions are scalar-first ``(w, x, y, z)``. A twist is ``[v, omega]`` with
translational part ``v`` and rotational part ``omega`` (axis times angle).
"""

from __future__ import annotations

import torch

# Below this squared angle the closed-form coefficients lose precision, so a
# fourth-order Taylor series is used instead.
_SMALL_ANGLE_SQUARED = 1e-8


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the Hamilton product ``left * right`` of ``(..., 4)`` quaternions."""
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Scale ``(..., 4)`` quaternions to unit norm."""
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(eps)


def _so3_coefficients(
    theta_squared: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the Rodrigues / left-Jacobian coefficients ``(A, B, C)``.

    ``A = sin(t)/t``, ``B = (1 - cos(t))/t^2`` and ``C = (t - sin(t))/t^3``,
    switching to their Taylor series near zero.
    """
    small = theta_squared < _SMALL_ANGLE_SQUARED
    theta_fourth = theta_squared * theta_squared
    safe_theta_squared = theta_squared.clamp_min(_SMALL_ANGLE_SQUARED)
    safe_theta = torch.sqrt(safe_theta_squared)
    a_exact = torch.sin(safe_theta) / safe_theta
    b_exact = (1.0 - torch.cos(safe_theta)) / safe_theta_squared
    c_exact = (safe_theta - torch.sin(safe_theta)) / (safe_theta_squared * safe_theta)
    a_series = 1.0 - theta_squared / 6.0 + theta_fourth / 120.0
    b_series = 0.5 - theta_squared / 24.0 + theta_fourth / 720.0
    c_series = 1.0 / 6.0 - theta_squared / 120.0 + theta_fourth / 5040.0
    return (
        torch.where(small, a_series, a_exact),
        torch.where(small, b_series, b_exact),
        torch.where(small, c_series, c_exact),
    )


def _delta_quaternion(omega: torch.Tensor, theta_squared: torch.Tensor) -> torch.Tensor:
    """Return the unit quaternion of the rotation ``exp(omega)``."""
    safe_theta = torch.sqrt(theta_squared.clamp_min(_SMALL_ANGLE_SQUARED))
    small = theta_squared < _SMALL_ANGLE_SQUARED
    xyz_exact = omega * (torch.sin(0.5 * safe_theta) / safe_theta)
    xyz_series = omega * (0.5 - theta_squared / 48.0 + theta_squared * theta_squared / 3840.0)
    scalar_exact = torch.cos(0.5 * safe_theta)
    scalar_series = 1.0 - theta_squared / 8.0 + theta_squared * theta_squared / 384.0
    return torch.cat(
        (
            torch.where(small, scalar_series, scalar_exact),
            torch.where(small.expand_as(omega), xyz_series, xyz_exact),
        ),
        dim=-1,
    )


def se3_exp_apply(
    points: torch.Tensor, quaternions: torch.Tensor, twist: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``exp([v, omega])`` to points and to Gaussian orientations.

    The point transform is ``p' = R p + J(omega) v`` and orientations are
    left-multiplied by the same rotation, ``q' = q_delta * q``.

    Args:
        points: ``(..., 3)`` positions.
        quaternions: ``(..., 4)`` orientations; need not be normalised.
        twist: ``(..., 6)`` twist ``[v, omega]``.

    Returns:
        The transformed ``(..., 3)`` points and unit ``(..., 4)`` quaternions.
    """
    if twist.shape[-1] != 6:
        raise ValueError(f"an SE(3) twist has six channels [v, omega], got {twist.shape[-1]}")
    velocity, omega = twist[..., :3], twist[..., 3:]
    theta_squared = (omega * omega).sum(dim=-1, keepdim=True)
    a, b, c = _so3_coefficients(theta_squared)

    omega_cross_point = torch.cross(omega, points, dim=-1)
    rotated = points + a * omega_cross_point + b * torch.cross(omega, omega_cross_point, dim=-1)
    omega_cross_velocity = torch.cross(omega, velocity, dim=-1)
    translation = (
        velocity + b * omega_cross_velocity + c * torch.cross(omega, omega_cross_velocity, dim=-1)
    )

    delta = normalize_quaternion(_delta_quaternion(omega, theta_squared))
    rotated_quaternions = quaternion_multiply(delta, normalize_quaternion(quaternions))
    return rotated + translation, normalize_quaternion(rotated_quaternions)
