"""The 3D cinemagraph: canonical Gaussians deformed by P, plus Δ during training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ouroworld.fields.drift import GroundedDriftField
from ouroworld.fields.periodic import PeriodicDeformationField
from ouroworld.gaussians.attributes import GaussianAttributes
from ouroworld.gaussians.canonical import CanonicalGaussians
from ouroworld.geometry.se3 import se3_exp_apply


@dataclass(frozen=True)
class ViewContext:
    """The observation being rendered.

    Attributes:
        time: Normalised loop phase; the only time coordinate in the model.
        view_id: Training view of the observation. ``None`` renders the model
            as at inference (P only); a training view also applies Δ unless
            that observation is grounded.
    """

    time: float
    view_id: int | None = None


class CinemagraphModel(nn.Module):
    """Canonical Gaussians with a Periodic Deformation Field and an optional drift field."""

    def __init__(
        self,
        canonical: CanonicalGaussians,
        periodic: PeriodicDeformationField,
        drift: GroundedDriftField | None,
    ):
        super().__init__()
        self.canonical = canonical
        self.periodic = periodic
        self.drift = drift

    def fit_fields_to_canonical(self) -> None:
        """Span every triplane over the bounding box of the canonical centres."""
        lower, upper = self.canonical.bounds()
        self.periodic.set_bounds(lower, upper)
        if self.drift is not None:
            self.drift.set_bounds(lower, upper)

    def applies_drift(self, context: ViewContext) -> bool:
        """Whether Δ contributes to this observation."""
        return (
            self.drift is not None
            and context.view_id is not None
            and not self.drift.is_grounded(context.view_id, context.time)
        )

    def deformed(self, context: ViewContext) -> GaussianAttributes:
        """Return the Gaussians at ``context.time`` as seen from ``context.view_id``."""
        canonical = self.canonical.attributes()
        count = len(canonical)
        time = canonical.xyz.new_full((count, 1), float(context.time))
        attributes = self._apply_periodic(canonical, time)
        if self.applies_drift(context):
            assert context.view_id is not None
            view_ids = torch.full((count,), context.view_id, dtype=torch.long, device=time.device)
            attributes = self._apply_drift(attributes, canonical.xyz, time, view_ids)
        return attributes

    def _apply_periodic(
        self, canonical: GaussianAttributes, time: torch.Tensor
    ) -> GaussianAttributes:
        output = self.periodic(canonical.xyz, time)
        xyz, rotation = se3_exp_apply(canonical.xyz, canonical.rotation, output.twist)
        log_scale = canonical.log_scale
        if output.log_scale_delta is not None:
            log_scale = log_scale + output.log_scale_delta
        sh = canonical.sh if output.sh_delta is None else canonical.sh + output.sh_delta
        return canonical.replace(xyz=xyz, rotation=rotation, log_scale=log_scale, sh=sh)

    def _apply_drift(
        self,
        attributes: GaussianAttributes,
        canonical_xyz: torch.Tensor,
        time: torch.Tensor,
        view_ids: torch.Tensor,
    ) -> GaussianAttributes:
        """Compose Δ after P. Δ is queried at the canonical position, like P."""
        assert self.drift is not None
        drift = self.drift(canonical_xyz, time, view_ids)
        changes: dict[str, torch.Tensor] = {}
        if "pos" in drift or "rot" in drift:
            zero = torch.zeros_like(attributes.xyz)
            twist = torch.cat((drift.get("pos", zero), drift.get("rot", zero)), dim=-1)
            xyz, rotation = se3_exp_apply(attributes.xyz, attributes.rotation, twist)
            # With only one half declared, `se3_exp_apply` still moves the other
            # attribute (a pure rotation turns points about the origin), so adopt
            # only what a head claims.
            if "pos" in drift:
                changes["xyz"] = xyz
            if "rot" in drift:
                changes["rotation"] = rotation
        if "scale" in drift:
            changes["log_scale"] = attributes.log_scale + drift["scale"]
        if "opacity" in drift:
            changes["opacity_logit"] = attributes.opacity_logit + drift["opacity"]
        if "shs" in drift:
            changes["sh"] = attributes.sh + drift["shs"].reshape(attributes.sh.shape)
        return attributes.replace(**changes)
