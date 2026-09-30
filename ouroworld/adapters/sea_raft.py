"""SEA-RAFT optical flow (``third_party/SEA-RAFT``), the flow of the Vividness metric."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SEA_RAFT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "SEA-RAFT"
CONFIG = "config/eval/spring-M.json"


def load_sea_raft(weights_dir: Path, device: str) -> torch.nn.Module:
    """Load SEA-RAFT (spring-M) from ``weights_dir`` (a Hugging Face snapshot) onto ``device``.

    Call the model as ``model(image1, image2, test_mode=True)["final"]`` with
    ``(B, 3, H, W)`` float RGB in ``[0, 255]``; the result is ``(B, 2, H, W)``.
    """
    # SEA-RAFT imports its modules as top-level names from core/.
    for path in (SEA_RAFT_ROOT, SEA_RAFT_ROOT / "core"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from config.parser import json_to_args
    from raft import RAFT

    arguments = json_to_args(str(SEA_RAFT_ROOT / CONFIG))
    model = RAFT.from_pretrained(str(weights_dir), args=arguments)
    return model.eval().to(torch.device(device))
