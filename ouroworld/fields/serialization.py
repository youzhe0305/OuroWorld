"""Save and load a renderable cinemagraph model.

A model directory holds::

    gaussians.ply   canonical Gaussians (standard 3DGS PLY)
    fields.pt       {"periodic_spec", "drift_spec", "state_dict"}
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

from ouroworld.fields.cinemagraph import CinemagraphModel
from ouroworld.fields.drift import DriftFieldSpec, GroundedDriftField
from ouroworld.fields.encoding import TriplaneSpec
from ouroworld.fields.heads import ChunkedEvaluator
from ouroworld.fields.periodic import PeriodicDeformationField, PeriodicFieldSpec
from ouroworld.gaussians.canonical import CanonicalGaussians
from ouroworld.io.ply import read_gaussian_ply, write_gaussian_ply

GAUSSIANS_FILE = "gaussians.ply"
FIELDS_FILE = "fields.pt"


def save_model(model: CinemagraphModel, directory: Path) -> None:
    """Write the renderable part of a checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    write_gaussian_ply(model.canonical.to_arrays(), directory / GAUSSIANS_FILE)
    fields = {"periodic": model.periodic.state_dict()}
    if model.drift is not None:
        fields["drift"] = model.drift.state_dict()
    torch.save(
        {
            "periodic_spec": dataclasses.asdict(model.periodic.spec),
            "drift_spec": None if model.drift is None else dataclasses.asdict(model.drift.spec),
            "state_dict": fields,
        },
        directory / FIELDS_FILE,
    )


def load_model(
    directory: Path,
    evaluator: ChunkedEvaluator,
    frozen: tuple[str, ...] = (),
    with_drift: bool = True,
) -> CinemagraphModel:
    """Rebuild a model from :func:`save_model` output (on CPU).

    Args:
        directory: Checkpoint directory.
        evaluator: Chunking policy of the fields.
        frozen: Canonical attributes to keep frozen, when resuming training.
        with_drift: Load Δ too; rendering does not need it.
    """
    payload = torch.load(Path(directory) / FIELDS_FILE, map_location="cpu", weights_only=True)
    periodic_spec = periodic_spec_from_dict(payload["periodic_spec"])
    canonical = CanonicalGaussians(
        read_gaussian_ply(Path(directory) / GAUSSIANS_FILE), periodic_spec.sh_degree, frozen
    )
    periodic = PeriodicDeformationField(periodic_spec, evaluator)
    periodic.load_state_dict(payload["state_dict"]["periodic"])
    drift = None
    if with_drift and payload["drift_spec"] is not None:
        drift = GroundedDriftField(drift_spec_from_dict(payload["drift_spec"]), evaluator)
        drift.load_state_dict(payload["state_dict"]["drift"])
    return CinemagraphModel(canonical, periodic, drift)


def periodic_spec_from_dict(data: dict[str, Any]) -> PeriodicFieldSpec:
    """Inverse of ``dataclasses.asdict`` for :class:`PeriodicFieldSpec`."""
    return PeriodicFieldSpec(
        **{
            **data,
            "heads": tuple(data["heads"]),
            "motion_grid": _triplane_spec(data["motion_grid"]),
            "appearance_grid": _triplane_spec(data["appearance_grid"]),
        }
    )


def drift_spec_from_dict(data: dict[str, Any]) -> DriftFieldSpec:
    """Inverse of ``dataclasses.asdict`` for :class:`DriftFieldSpec`."""
    return DriftFieldSpec(
        **{**data, "heads": tuple(data["heads"]), "grid": _triplane_spec(data["grid"])}
    )


def _triplane_spec(data: dict[str, Any]) -> TriplaneSpec:
    return TriplaneSpec(
        resolution=tuple(data["resolution"]),  # type: ignore[arg-type]
        multires=tuple(data["multires"]),
        channels=int(data["channels"]),
    )
