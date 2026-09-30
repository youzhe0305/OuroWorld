"""Training checkpoints: the model plus everything needed to resume.

Layout of ``<run>/checkpoints/iter_XXXXX``::

    gaussians.ply, fields.pt   the model (see :mod:`ouroworld.fields.serialization`)
    training_state.pt          optimiser, densification and RNG state
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

TRAINING_STATE_FILE = "training_state.pt"


def checkpoint_dir(run_dir: Path, iteration: int) -> Path:
    """Directory of the checkpoint written at ``iteration``."""
    return Path(run_dir) / "checkpoints" / f"iter_{iteration:05d}"


def latest_checkpoint(run_dir: Path) -> Path:
    """The checkpoint with the highest iteration."""
    candidates = sorted((Path(run_dir) / "checkpoints").glob("iter_*"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoints under {run_dir}")
    return candidates[-1]


def save_training_state(state: dict[str, Any], directory: Path) -> None:
    """Write the optimiser/densification state together with all RNG states."""
    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save({**state, "rng": rng}, Path(directory) / TRAINING_STATE_FILE)


def load_training_state(directory: Path) -> dict[str, Any]:
    """Read :func:`save_training_state` output and restore the global RNG states."""
    state = torch.load(
        Path(directory) / TRAINING_STATE_FILE, map_location="cpu", weights_only=False
    )
    rng = state.pop("rng")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return state
