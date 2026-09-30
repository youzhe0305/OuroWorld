"""``ouroworld-train``: optimise a 3D cinemagraph from multi-view videos.

Examples::

    ouroworld-train dataset=mip-nerf scene=room
    ouroworld-train dataset=mip-nerf scene=room train.iterations=2000
    ouroworld-train --resume output/mip-nerf/room/model train.iterations=15000
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from ouroworld.cli.build import build_training
from ouroworld.cli.common import add_config_arguments, config_from_arguments, setup_logging
from ouroworld.config.loader import ConfigError, load_config, save_config
from ouroworld.training.checkpoint import latest_checkpoint
from ouroworld.training.trainer import Trainer

logger = logging.getLogger(__name__)

RUN_CONFIG = "config.yaml"
# A resumed run keeps its model and data; only how long it runs may change.
RESUMABLE_OVERRIDES = ("train.iterations", "train.checkpoint_iterations", "train.log_interval")


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--resume", type=Path, help="run directory to continue from its latest checkpoint"
    )
    arguments = parser.parse_args(argv)
    setup_logging()
    # Must be set before CUDA initialises; reduces fragmentation from large render graphs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if arguments.resume is None:
        config = config_from_arguments(arguments)
        resume_from = None
    else:
        config = _resumed_config(arguments.resume, arguments.overrides)
        resume_from = latest_checkpoint(arguments.resume)
    save_config(config, Path(config.paths.run_dir) / RUN_CONFIG)
    run = build_training(config, resume_from)
    logger.info(
        "training %s/%s from iteration %d", config.dataset, config.scene, run.start_iteration
    )
    Trainer(run.setup).run(run.start_iteration)


def _resumed_config(run_dir: Path, overrides: list[str]):  # noqa: ANN202 - Config
    rejected = [item for item in overrides if item.split("=", 1)[0] not in RESUMABLE_OVERRIDES]
    if rejected:
        raise ConfigError(
            f"a resumed run only accepts overrides of {RESUMABLE_OVERRIDES}; got {rejected}"
        )
    return load_config([run_dir / RUN_CONFIG], overrides)


if __name__ == "__main__":
    main()
