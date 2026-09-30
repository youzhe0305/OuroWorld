"""Spatial triplane encoding and Fourier time bases (paper §4.2, App. B).

Time never enters the triplane. Each triplane channel is instead a bank of
Fourier coefficients, contracted with a time basis afterwards:

    feature(x, t) = sum_b coefficient_b(x) * basis_b(t)

With a period-1 basis this makes the field exactly periodic in ``t``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# The three axis-aligned planes xy, xz and yz.
PLANE_AXES = tuple(itertools.combinations(range(3), 2))


def fourier_time_basis(time: torch.Tensor, harmonics: int, period: float = 1.0) -> torch.Tensor:
    """Return the DC plus sine/cosine basis ``(N, 1 + 2K)`` of ``time`` ``(N, 1)``.

    ``period=1`` gives an exactly periodic basis. The aperiodic ablation uses
    ``period=2``: the same harmonics traversed over half a cycle, so the width
    and smoothness are unchanged but ``basis(0) != basis(1)``.
    """
    phase = (2.0 / period) * math.pi * time
    features = [torch.ones_like(time)]
    for k in range(1, harmonics + 1):
        features.append(torch.sin(k * phase))
        features.append(torch.cos(k * phase))
    return torch.cat(features, dim=-1)


@dataclass(frozen=True)
class TriplaneSpec:
    """Shape of a multi-resolution triplane.

    Attributes:
        resolution: Base ``(x, y, z)`` resolution of the planes.
        multires: Resolution multiplier of each level; levels are concatenated.
        channels: Feature channels per level (before the Fourier coefficient bank).
    """

    resolution: tuple[int, int, int]
    multires: tuple[int, ...]
    channels: int

    def __post_init__(self) -> None:
        if len(self.resolution) != 3 or min(self.resolution) <= 0:
            raise ValueError("triplane resolution must be three positive integers")
        if not self.multires or min(self.multires) <= 0:
            raise ValueError("triplane multires must be a non-empty list of positive integers")

    @property
    def feature_dim(self) -> int:
        """Width of the time-contracted feature."""
        return self.channels * len(self.multires)


class Triplane(nn.Module):
    """Multi-resolution xy/xz/yz feature planes over an axis-aligned box.

    Plane features are multiplied across the three planes (K-Planes style)
    and concatenated across levels.
    """

    def __init__(self, spec: TriplaneSpec, basis_size: int):
        """Allocate the planes.

        Args:
            spec: Plane resolutions and channel count.
            basis_size: Number of time-basis functions each channel carries.
        """
        super().__init__()
        self.spec = spec
        self.basis_size = int(basis_size)
        out_channels = spec.channels * self.basis_size
        self.levels = nn.ModuleList()
        for multiplier in spec.multires:
            size = [r * multiplier for r in spec.resolution]
            planes = nn.ParameterList()
            for first, second in PLANE_AXES:
                plane = nn.Parameter(torch.empty(1, out_channels, size[second], size[first]))
                nn.init.uniform_(plane, a=0.1, b=0.5)
                planes.append(plane)
            self.levels.append(planes)
        # Box the planes span; set from the canonical Gaussians before training.
        self.register_buffer("lower", -torch.ones(3))
        self.register_buffer("upper", torch.ones(3))

    def set_bounds(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        """Set the world-space box mapped onto ``[-1, 1]^3``."""
        self.lower.copy_(lower)
        self.upper.copy_(upper)

    def coefficients(self, points: torch.Tensor) -> torch.Tensor:
        """Sample the coefficient bank at ``points`` ``(N, 3)``.

        Returns:
            ``(N, levels, basis_size, channels)`` coefficients.
        """
        normalized = (points - self.lower) * (2.0 / (self.upper - self.lower)) - 1.0
        per_level = []
        for planes in self.levels:
            product = 1.0
            for plane, axes in zip(planes, PLANE_AXES, strict=True):
                product = product * _sample_plane(plane, normalized[:, axes])
            per_level.append(product)
        stacked = torch.cat(per_level, dim=-1)
        return stacked.reshape(
            points.shape[0], len(self.levels), self.basis_size, self.spec.channels
        )

    def planes(self) -> list[nn.Parameter]:
        """All plane tensors, for regularisation."""
        return [plane for planes in self.levels for plane in planes]


def contract_time(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Combine ``(N, L, B, C)`` coefficients with a ``(N, B)`` basis into ``(N, L * C)``."""
    count = coefficients.shape[0]
    return torch.einsum("nlbc,nb->nlc", coefficients, basis).reshape(count, -1)


def _sample_plane(plane: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``plane`` ``(1, C, H, W)`` at ``(N, 2)`` coordinates in ``[-1, 1]``."""
    grid = coordinates.view(1, 1, -1, 2)
    sampled = F.grid_sample(plane, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.view(plane.shape[1], -1).transpose(0, 1)
