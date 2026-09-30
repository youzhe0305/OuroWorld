"""Thin wrapper over the differentiable Gaussian rasterizer.

The renderer only sees :class:`GaussianAttributes`; it knows nothing about the
deformation fields that may have produced them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ouroworld.gaussians.attributes import GaussianAttributes
from ouroworld.geometry.camera import Camera

GradientFilter = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class RasterView:
    """A camera prepared for the rasterizer on a given device."""

    width: int
    height: int
    tan_half_fov_x: float
    tan_half_fov_y: float
    world_view: torch.Tensor  # (4, 4) transposed world-to-camera
    full_projection: torch.Tensor  # (4, 4) transposed world-to-clip
    camera_center: torch.Tensor  # (3,)

    @classmethod
    def from_camera(cls, camera: Camera, device: torch.device | str = "cuda") -> RasterView:
        """Precompute the rasterizer matrices of ``camera`` on ``device``."""
        matrices = camera.raster_matrices()
        return cls(
            width=camera.width,
            height=camera.height,
            tan_half_fov_x=matrices.tan_half_fov_x,
            tan_half_fov_y=matrices.tan_half_fov_y,
            world_view=matrices.world_view.to(device),
            full_projection=matrices.full_projection.to(device),
            camera_center=matrices.camera_center.to(device),
        )


@dataclass(frozen=True)
class RenderOutput:
    """Result of one rasterisation.

    Attributes:
        image: ``(3, H, W)`` rendered colour.
        depth: ``(1, H, W)`` alpha-weighted camera-z depth.
        radii: ``(N,)`` screen-space radii; zero for culled Gaussians.
        viewspace_points: ``(N, 3)`` tensor whose ``.grad`` holds the screen-space
            position gradients after ``backward()``, used by densification.
    """

    image: torch.Tensor
    depth: torch.Tensor
    radii: torch.Tensor
    viewspace_points: torch.Tensor

    @property
    def visible(self) -> torch.Tensor:
        """``(N,)`` mask of Gaussians that touched at least one pixel."""
        return self.radii > 0


def render(
    attributes: GaussianAttributes,
    view: RasterView,
    background: torch.Tensor,
    sh_degree: int,
    gradient_filter: GradientFilter | None = None,
    colors: torch.Tensor | None = None,
) -> RenderOutput:
    """Rasterise Gaussians given in optimisation space.

    Args:
        attributes: Pre-activation attributes; ``exp``, ``normalize`` and
            ``sigmoid`` are applied here.
        view: Target camera.
        background: ``(3,)`` background colour on the render device.
        sh_degree: Active SH degree.
        gradient_filter: Optional hook applied to the gradient of every
            per-Gaussian rasterizer input, e.g. to quarantine non-finite rows.
        colors: Optional ``(N, 3)`` view-independent colours replacing the SH,
            e.g. ones on a black background to render the accumulated opacity.
    """
    rasterization = _rasterizer_module()
    xyz = attributes.xyz
    viewspace_points = torch.zeros_like(xyz, requires_grad=True) + 0
    if viewspace_points.requires_grad:
        viewspace_points.retain_grad()
    settings = rasterization.GaussianRasterizationSettings(
        image_height=view.height,
        image_width=view.width,
        tanfovx=view.tan_half_fov_x,
        tanfovy=view.tan_half_fov_y,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=view.world_view,
        projmatrix=view.full_projection,
        sh_degree=sh_degree,
        campos=view.camera_center,
        prefiltered=False,
        debug=False,
    )
    scales = torch.exp(attributes.log_scale)
    rotations = F.normalize(attributes.rotation)
    opacities = torch.sigmoid(attributes.opacity_logit)
    shs = attributes.sh if colors is None else None
    inputs = tuple(
        tensor
        for tensor in (viewspace_points, xyz, scales, rotations, opacities, shs, colors)
        if tensor is not None
    )
    if gradient_filter is not None and torch.is_grad_enabled():
        for tensor in inputs:
            if tensor.requires_grad:
                tensor.register_hook(gradient_filter)
    image, radii, depth = rasterization.GaussianRasterizer(raster_settings=settings)(
        means3D=xyz,
        means2D=viewspace_points,
        shs=shs,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    return RenderOutput(image=image, depth=depth, radii=radii, viewspace_points=viewspace_points)


def _rasterizer_module():  # noqa: ANN202 - returns a CUDA extension module
    try:
        import diff_gaussian_rasterization
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the CUDA rasterizer is not installed; run "
            "`pip install ./third_party/diff-gaussian-rasterization`"
        ) from error
    return diff_gaussian_rasterization
