"""``ouroworld-render``: render the static and orbit loop videos of a trained cinemagraph.

Example::

    ouroworld-render dataset=mip-nerf scene=room render.checkpoint=13000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from ouroworld.cli.build import evaluator, loop_render_settings, resolve_checkpoint
from ouroworld.cli.common import add_config_arguments, config_from_arguments, setup_logging
from ouroworld.fields.serialization import load_model
from ouroworld.io.errors import ArtifactError
from ouroworld.io.multiview import load_multiview
from ouroworld.io.scene_package import load_scene_package
from ouroworld.rendering.loop_render import LoopRenderer, render_loop_videos

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_arguments(parser)
    arguments = parser.parse_args(argv)
    setup_logging()
    config = config_from_arguments(arguments)

    checkpoint = resolve_checkpoint(Path(config.paths.run_dir), config.render.checkpoint)
    package = load_scene_package(Path(config.paths.scene_dir))
    if package.pivot is None:
        raise ArtifactError(f"{package.root} has no pivot; the orbit needs one")
    videos = load_multiview(Path(config.paths.multiview_dir))
    model = load_model(checkpoint, evaluator(config.model), with_drift=False).cuda()
    background = torch.tensor(package.background, dtype=torch.float32, device="cuda")
    renderer = LoopRenderer(
        model,
        background,
        loop_render_settings(config),
        is_periodic=model.periodic.spec.period == 1.0,
    )
    reference = videos.cameras[videos.reference_view_id]
    out_dir = Path(config.paths.render_dir) / checkpoint.name
    report = render_loop_videos(renderer, reference, package.pivot, out_dir)
    logger.info("seam: %s", report["seam"])


if __name__ == "__main__":
    main()
