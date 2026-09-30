"""Forward warp of the reference video into a novel view: the video model's condition.

Each reference pixel is lifted with its aligned depth and splatted bilinearly
into the target camera; where several pixels land on one target pixel the
nearest wins softly (weights fall off exponentially with depth). Target pixels
nobody lands on are holes for the video model to fill.

Ported from TrajectoryCrafter's ``Warper`` (``third_party/TrajectoryCrafter``)
with the same arithmetic, so the condition is identical to the paper runs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# Depth is saturated here before the soft z-buffer weights are computed.
MAX_SPLAT_DEPTH = 1000.0
# Sharpness of the soft z-buffer: weights fall by e^50 across the depth range.
Z_BUFFER_SHARPNESS = 50.0
# Holes are grown by this many pixels: splats at hole borders are unreliable.
HOLE_DILATION = 2


def world_to_camera(c2w: np.ndarray) -> torch.Tensor:
    """Return the ``(1, 4, 4)`` float32 world-to-camera matrix of ``c2w``.

    The pose is cast to float32 before the inversion, as in the paper runs.
    """
    return torch.from_numpy(np.linalg.inv(np.asarray(c2w, dtype=np.float32))).float()[None]


def warp_frames(
    frames: torch.Tensor,
    depth: torch.Tensor,
    valid: torch.Tensor,
    K: torch.Tensor,
    source_c2w: np.ndarray,
    target_c2ws: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp ``frames[i]`` from the source camera to ``target_c2ws[i]``.

    Args:
        frames: ``(T, 3, H, W)`` source colour in ``[0, 1]``.
        depth: ``(T, H, W)`` aligned source depth.
        valid: ``(T, H, W)`` pixels whose depth may be splatted.
        K: ``(3, 3)`` float32 intrinsics shared by source and target.
        source_c2w: Source camera pose.
        target_c2ws: One target pose per frame.

    Returns:
        ``(T, 3, H, W)`` warped colour in ``[0, 1]`` and ``(T, 1, H, W)`` known
        mask, both float64 (the precision the paper's conditions were built in).
    """
    if len(target_c2ws) != len(frames):
        raise ValueError(f"{len(frames)} frames but {len(target_c2ws)} target cameras")
    source_w2c = world_to_camera(source_c2w).cuda()
    intrinsics = K[None].cuda()
    images, known_masks = [], []
    for index, target_c2w in enumerate(target_c2ws):
        mask = valid[index : index + 1, None].float().cuda()
        frame_depth = depth[index : index + 1, None].cuda()
        frame_depth = torch.where(mask.bool(), frame_depth, torch.ones_like(frame_depth))
        warped, known = forward_warp(
            frames[index : index + 1].cuda().mul(2).sub(1),
            mask,
            frame_depth,
            source_w2c,
            world_to_camera(target_c2w).cuda(),
            intrinsics,
        )
        image, known = grow_holes(warped, known)
        images.append(image[0].cpu())
        known_masks.append(known[0].cpu())
    return torch.stack(images), torch.stack(known_masks)


def forward_warp(
    image: torch.Tensor,
    mask: torch.Tensor,
    depth: torch.Tensor,
    source_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Splat ``image`` (``(B, 3, H, W)`` in ``[-1, 1]``) into the target camera.

    Returns:
        Warped image in ``[-1, 1]`` (``-1`` in holes) and ``(B, 1, H, W)`` known mask.
    """
    batch, _, height, width = image.shape
    points = _target_points(depth, source_w2c, target_w2c, K)
    coordinates = points[:, :, :, :2, 0] / points[:, :, :, 2:3, 0]
    target_depth = points[:, :, :, 2, 0]
    grid = _pixel_grid(batch, height, width).to(coordinates)
    flow = coordinates.permute(0, 3, 1, 2) - grid
    return _bilinear_splat(image, mask, target_depth, flow)


def grow_holes(image: torch.Tensor, known: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dilate the holes by :data:`HOLE_DILATION`; return the image in ``[0, 1]`` and the mask."""
    holes = 1.0 - known
    kernel = 2 * HOLE_DILATION + 1
    grown = F.max_pool2d(holes, kernel, stride=1, padding=HOLE_DILATION).double()
    cleaned = (image + 1.0) / 2.0 * (1.0 - grown)
    return cleaned, 1.0 - grown


def _target_points(
    depth: torch.Tensor, source_w2c: torch.Tensor, target_w2c: torch.Tensor, K: torch.Tensor
) -> torch.Tensor:
    """Return ``(B, H, W, 3, 1)`` homogeneous target-image coordinates of every source pixel."""
    batch, _, height, width = depth.shape
    transformation = torch.bmm(target_w2c, torch.linalg.inv(source_w2c))
    x = torch.arange(0, width)[None].repeat([height, 1]).to(depth)
    y = torch.arange(0, height)[:, None].repeat([1, width]).to(depth)
    ones = torch.ones(size=(height, width)).to(depth)
    pixels = torch.stack([x, y, ones], dim=2)[None, :, :, :, None]
    rays = torch.matmul(torch.linalg.inv(K)[:, None, None], pixels)
    camera_points = depth[:, 0][:, :, :, None, None] * rays
    homogeneous = torch.cat(
        [camera_points, ones[None, :, :, None, None].repeat([batch, 1, 1, 1, 1])], 3
    )
    moved = torch.matmul(transformation[:, None, None], homogeneous)[:, :, :, :3]
    return torch.matmul(K[:, None, None], moved)


def _pixel_grid(batch: int, height: int, width: int) -> torch.Tensor:
    x = torch.arange(0, width)[None].repeat([height, 1])
    y = torch.arange(0, height)[:, None].repeat([1, width])
    return torch.stack([x, y], dim=0)[None].repeat([batch, 1, 1, 1])


def _bilinear_splat(
    image: torch.Tensor, mask: torch.Tensor, depth: torch.Tensor, flow: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, height, width = image.shape
    position = flow + _pixel_grid(batch, height, width).to(image)
    # Work in a frame padded by one pixel so splats just outside still have a cell.
    offset = position + 1
    floor = torch.floor(offset).long()
    ceil = torch.ceil(offset).long()
    offset = torch.stack(
        [offset[:, 0].clamp(0, width + 1), offset[:, 1].clamp(0, height + 1)], dim=1
    )
    floor = torch.stack([floor[:, 0].clamp(0, width + 1), floor[:, 1].clamp(0, height + 1)], dim=1)
    ceil = torch.stack([ceil[:, 0].clamp(0, width + 1), ceil[:, 1].clamp(0, height + 1)], dim=1)

    below_y = 1 - (offset[:, 1:2] - floor[:, 1:2])
    above_y = 1 - (ceil[:, 1:2] - offset[:, 1:2])
    below_x = 1 - (offset[:, 0:1] - floor[:, 0:1])
    above_x = 1 - (ceil[:, 0:1] - offset[:, 0:1])
    saturated = torch.clamp(depth, min=0, max=MAX_SPLAT_DEPTH)
    log_depth = torch.log(1 + saturated)
    depth_weight = torch.exp(log_depth / log_depth.max() * Z_BUFFER_SHARPNESS).unsqueeze(1)
    corners = (
        (below_y * below_x, floor[:, 1], floor[:, 0]),
        (above_y * below_x, ceil[:, 1], floor[:, 0]),
        (below_y * above_x, floor[:, 1], ceil[:, 0]),
        (above_y * above_x, ceil[:, 1], ceil[:, 0]),
    )
    splatted = torch.zeros(size=(batch, height + 2, width + 2, channels), dtype=torch.float32).to(
        image
    )
    weights = torch.zeros(size=(batch, height + 2, width + 2, 1), dtype=torch.float32).to(image)
    channels_last = torch.moveaxis(image, [0, 1, 2, 3], [0, 3, 1, 2])
    batch_index = torch.arange(batch)[:, None, None].to(image.device)
    corner_weights = []
    for proximity, row, column in corners:
        weight = torch.moveaxis(proximity * mask / depth_weight, [0, 1, 2, 3], [0, 3, 1, 2])
        splatted.index_put_((batch_index, row, column), channels_last * weight, accumulate=True)
        corner_weights.append((weight, row, column))
    for weight, row, column in corner_weights:
        weights.index_put_((batch_index, row, column), weight, accumulate=True)

    splatted = torch.moveaxis(splatted, [0, 1, 2, 3], [0, 2, 3, 1])[:, :, 1:-1, 1:-1]
    weights = torch.moveaxis(weights, [0, 1, 2, 3], [0, 2, 3, 1])[:, :, 1:-1, 1:-1]
    known = weights > 0
    warped = torch.where(known, splatted / weights, torch.tensor(-1.0, device=image.device))
    return warped.clamp(-1, 1), known.to(image)
