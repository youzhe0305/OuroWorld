"""The canonical (static, ``t = 0``) Gaussians being optimised."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from ouroworld.gaussians.attributes import GaussianAttributes
from ouroworld.io.ply import GaussianArrays

# Names of the learnable per-Gaussian tensors, also used as optimiser group names.
PARAMETER_NAMES = ("xyz", "sh_dc", "sh_rest", "opacity", "log_scale", "rotation")

# User-facing groups that can be frozen as a whole.
FREEZABLE = {
    "xyz": ("xyz",),
    "opacity": ("opacity",),
    "scale": ("log_scale",),
    "rotation": ("rotation",),
    "sh": ("sh_dc", "sh_rest"),
}


class CanonicalGaussians(nn.Module):
    """Learnable canonical Gaussians.

    Frozen attributes keep ``requires_grad=False`` for the whole run, including
    rows that densification appends later.
    """

    def __init__(self, arrays: GaussianArrays, sh_degree: int, frozen: Iterable[str] = ()):
        """Build parameters from raw arrays.

        Args:
            arrays: Raw per-Gaussian arrays, e.g. from a PLY.
            sh_degree: SH degree to optimise at. Input with a lower degree is
                zero-padded; input with a higher degree is truncated.
            frozen: Keys of :data:`FREEZABLE` to keep fixed.
        """
        super().__init__()
        unknown = set(frozen) - FREEZABLE.keys()
        if unknown:
            raise ValueError(
                f"unknown frozen attributes {sorted(unknown)}; valid: {sorted(FREEZABLE)}"
            )
        self.sh_degree = int(sh_degree)
        self.frozen_parameters = frozenset(name for key in frozen for name in FREEZABLE[key])
        tensors = {
            "xyz": arrays.xyz,
            "sh_dc": arrays.sh_dc,
            "sh_rest": _resize_sh_rest(arrays.sh_rest, self.sh_degree),
            "opacity": arrays.opacity,
            "log_scale": arrays.log_scale,
            "rotation": arrays.rotation,
        }
        for name in PARAMETER_NAMES:
            self.set_parameter(name, torch.as_tensor(np.ascontiguousarray(tensors[name])).float())

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def set_parameter(self, name: str, tensor: torch.Tensor) -> nn.Parameter:
        """Install ``tensor`` as parameter ``name``, honouring the freeze list."""
        parameter = nn.Parameter(tensor.detach(), requires_grad=name not in self.frozen_parameters)
        setattr(self, name, parameter)
        return parameter

    def parameter_by_name(self, name: str) -> nn.Parameter:
        """Return the parameter called ``name``."""
        return getattr(self, name)

    @property
    def sh(self) -> torch.Tensor:
        """``(N, (degree + 1)^2, 3)`` SH coefficients."""
        return torch.cat((self.sh_dc, self.sh_rest), dim=1)

    def attributes(self) -> GaussianAttributes:
        """Return the canonical attributes (differentiable)."""
        return GaussianAttributes(
            xyz=self.xyz,
            log_scale=self.log_scale,
            rotation=self.rotation,
            opacity_logit=self.opacity,
            sh=self.sh,
        )

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(min, max)`` corners of the centres' bounding box."""
        xyz = self.xyz.detach()
        return xyz.min(dim=0).values, xyz.max(dim=0).values

    def to_arrays(self) -> GaussianArrays:
        """Export the current parameters for PLY writing."""

        def numpy(tensor: torch.Tensor) -> np.ndarray:
            return tensor.detach().float().cpu().numpy()

        return GaussianArrays(
            xyz=numpy(self.xyz),
            sh_dc=numpy(self.sh_dc),
            sh_rest=numpy(self.sh_rest),
            opacity=numpy(self.opacity),
            log_scale=numpy(self.log_scale),
            rotation=numpy(self.rotation),
        )


def _resize_sh_rest(sh_rest: np.ndarray, degree: int) -> np.ndarray:
    """Zero-pad or truncate ``(N, C - 1, 3)`` coefficients to ``degree``."""
    target = (degree + 1) ** 2 - 1
    resized = np.zeros((sh_rest.shape[0], target, 3), dtype=np.float32)
    kept = min(target, sh_rest.shape[1])
    resized[:, :kept] = sh_rest[:, :kept]
    return resized
