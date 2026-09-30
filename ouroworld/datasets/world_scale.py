"""Global rescaling of a 3DGS world out of the rasterizer's near cull.

The rasterizer drops every Gaussian with view-space ``z <= 0.2``. Generators
that normalise their world (Lyra 2.0 scenes are ~5 units across) would lose
their whole foreground, so such a scene is scaled up: Gaussian positions and
scales, camera positions and clip planes all by the same factor, which the
rest of the method is invariant to. The PLY is streamed, never fully loaded.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

NEAR_CULL = 0.2  # hard-coded in diff-gaussian-rasterization
ROWS_PER_CHUNK = 1 << 18


def ply_layout(path: Path) -> tuple[int, list[str], int]:
    """``(vertex count, property names, payload offset)`` of a float32 binary PLY."""
    path = Path(path)
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        encoding, count, properties, in_vertex = None, None, [], False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            text = line.decode("ascii").strip()
            if text.startswith("format "):
                encoding = text.split()[1]
            elif text.startswith("element "):
                parts = text.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif text.startswith("property ") and in_vertex:
                parts = text.split()
                if parts[1] != "float":
                    raise ValueError(f"only float32 vertex properties are supported: {text}")
                properties.append(parts[-1])
            elif text == "end_header":
                break
        offset = handle.tell()
    if encoding != "binary_little_endian" or not count:
        raise ValueError(f"expected a non-empty binary_little_endian PLY: {path}")
    if os.path.getsize(path) - offset != 4 * count * len(properties):
        raise ValueError(f"PLY payload does not match its header: {path}")
    return count, properties, offset


def vertex_table(path: Path) -> tuple[np.ndarray, list[str]]:
    """Memory-map the vertices as ``(count, properties)`` float32."""
    count, properties, offset = ply_layout(path)
    table = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(count, len(properties)))
    return table, properties


def sampled_points(path: Path, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Every ``stride``-th Gaussian's position ``(N, 3)`` and opacity logit ``(N,)``."""
    table, properties = vertex_table(path)
    x, opacity = properties.index("x"), properties.index("opacity")
    sample = table[::stride]
    return np.asarray(sample[:, x : x + 3], np.float64), np.asarray(sample[:, opacity], np.float64)


def visible_points(xyz: np.ndarray, opacity_logit: np.ndarray, opacity_min: float) -> np.ndarray:
    """Positions of the opaque Gaussians; all of them if fewer than 1000 are opaque."""
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))
    visible = xyz[opacity >= opacity_min]
    return xyz if visible.shape[0] < 1000 else visible


def auto_world_scale(
    points: np.ndarray,
    cameras: list[tuple[np.ndarray, np.ndarray, int, int]],
    cull_fraction: float,
    depth_percentile: float = 5.0,
) -> tuple[float, float]:
    """Smallest integer scale putting the near cull below ``cull_fraction`` of the near surfaces.

    Args:
        points: Opaque Gaussian positions, ``(N, 3)``.
        cameras: ``(w2c, K, width, height)`` of the scene's cameras. Only points
            inside an image count: floaters just in front of a camera but
            outside its view would otherwise ask for an absurd scale.
        cull_fraction: Target ratio of the cull distance to the nearest surfaces.
        depth_percentile: Depth percentile taken as "the nearest surfaces".

    Returns:
        ``(scale, nearest visible depth)``; the scale is 1 for a large enough world.
    """
    nears = []
    for w2c, K, width, height in cameras:
        camera_points = points @ np.asarray(w2c)[:3, :3].T + np.asarray(w2c)[:3, 3]
        depth = camera_points[:, 2]
        front = depth > 0
        if not front.any():
            continue
        camera_points, depth = camera_points[front], depth[front]
        uv = camera_points @ np.asarray(K).T
        uv = uv[:, :2] / depth[:, None]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        if inside.sum() >= 1000:
            nears.append(float(np.percentile(depth[inside], depth_percentile)))
    if not nears:
        raise ValueError("no Gaussian is visible from any camera")
    near = min(nears)
    scale = NEAR_CULL / (cull_fraction * near)
    return (1.0, near) if scale <= 1.0 else (float(math.ceil(scale)), near)


def rescale_ply(source: Path, destination: Path, scale: float) -> None:
    """Copy ``source`` with positions times ``scale`` (log-scales plus ``log(scale)``)."""
    if not scale > 0.0:
        raise ValueError("the world scale must be positive")
    count, properties, offset = ply_layout(source)
    xyz = [properties.index(name) for name in ("x", "y", "z")]
    scales = [properties.index(f"scale_{axis}") for axis in range(3)]
    stride, log_scale = len(properties), math.log(scale)
    with Path(source).open("rb") as reader, Path(destination).open("wb") as writer:
        writer.write(reader.read(offset))
        remaining = count
        while remaining:
            rows = min(ROWS_PER_CHUNK, remaining)
            chunk = np.fromfile(reader, dtype="<f4", count=rows * stride).reshape(rows, stride)
            chunk[:, xyz] *= np.float32(scale)
            chunk[:, scales] += np.float32(log_scale)
            chunk.tofile(writer)
            remaining -= rows
