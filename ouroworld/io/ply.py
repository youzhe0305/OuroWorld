"""Read and write 3D Gaussians in the INRIA 3DGS PLY layout.

Attributes are stored *before* activation: log-scales, logit opacities and
unnormalised quaternions, exactly as the optimiser sees them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from ouroworld.io.errors import ArtifactError


@dataclass(frozen=True)
class GaussianArrays:
    """Raw per-Gaussian arrays, float32.

    Attributes:
        xyz: ``(N, 3)`` centres.
        sh_dc: ``(N, 1, 3)`` degree-0 SH coefficients.
        sh_rest: ``(N, C - 1, 3)`` higher-order SH coefficients, ``C = (degree + 1)^2``.
        opacity: ``(N, 1)`` opacity logits.
        log_scale: ``(N, 3)`` log standard deviations.
        rotation: ``(N, 4)`` scalar-first quaternions (not necessarily unit).
    """

    xyz: np.ndarray
    sh_dc: np.ndarray
    sh_rest: np.ndarray
    opacity: np.ndarray
    log_scale: np.ndarray
    rotation: np.ndarray

    @property
    def sh_degree(self) -> int:
        """SH degree implied by the number of coefficients."""
        return math.isqrt(self.sh_rest.shape[1] + 1) - 1

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


def read_gaussian_ply(path: Path) -> GaussianArrays:
    """Read a 3DGS PLY file."""
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"missing Gaussian PLY {path}")
    vertex = PlyData.read(str(path)).elements[0]
    names = [prop.name for prop in vertex.properties]

    def stack(prefix: str) -> np.ndarray:
        columns = sorted(
            (name for name in names if name.startswith(prefix)),
            key=lambda name: int(name.rsplit("_", 1)[-1]),
        )
        if not columns:
            return np.zeros((len(vertex.data), 0), dtype=np.float32)
        return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in columns], axis=1)

    rest = stack("f_rest_")
    coefficient_count = rest.shape[1] // 3
    degree = math.isqrt(coefficient_count + 1) - 1
    if rest.shape[1] % 3 or (degree + 1) ** 2 - 1 != coefficient_count:
        raise ArtifactError(f"{path}: invalid number of f_rest columns ({rest.shape[1]})")
    # PLY stores the SH rest block channel-major: (N, 3, C - 1) flattened.
    sh_rest = rest.reshape(rest.shape[0], 3, coefficient_count).transpose(0, 2, 1)
    return GaussianArrays(
        xyz=np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32),
        sh_dc=stack("f_dc_").reshape(-1, 1, 3),
        sh_rest=np.ascontiguousarray(sh_rest),
        opacity=np.asarray(vertex["opacity"], dtype=np.float32)[:, None],
        log_scale=stack("scale_"),
        rotation=stack("rot"),
    )


def write_gaussian_ply(gaussians: GaussianArrays, path: Path) -> None:
    """Write a 3DGS PLY file readable by standard 3DGS viewers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(gaussians)
    sh_rest = gaussians.sh_rest.transpose(0, 2, 1).reshape(count, -1)
    columns = {
        "x": gaussians.xyz[:, 0],
        "y": gaussians.xyz[:, 1],
        "z": gaussians.xyz[:, 2],
        "nx": np.zeros(count),
        "ny": np.zeros(count),
        "nz": np.zeros(count),
    }
    columns.update({f"f_dc_{i}": gaussians.sh_dc.reshape(count, 3)[:, i] for i in range(3)})
    columns.update({f"f_rest_{i}": sh_rest[:, i] for i in range(sh_rest.shape[1])})
    columns["opacity"] = gaussians.opacity[:, 0]
    columns.update({f"scale_{i}": gaussians.log_scale[:, i] for i in range(3)})
    columns.update({f"rot_{i}": gaussians.rotation[:, i] for i in range(4)})
    elements = np.empty(count, dtype=[(name, "f4") for name in columns])
    for name, values in columns.items():
        elements[name] = values
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))
