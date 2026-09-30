"""VGGT-Omega (``third_party/vggt-omega``)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPOSITORY = Path(__file__).resolve().parents[2] / "third_party" / "vggt-omega"


def load_vggt_omega(checkpoint: Path, device: str = "cuda") -> torch.nn.Module:
    """Return the VGGT-Omega model in eval mode with ``checkpoint`` loaded strictly."""
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device)
