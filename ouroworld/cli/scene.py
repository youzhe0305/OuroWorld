r"""Build a scene package: import, pick the pivot, choose the reference view (App. A).

``ouroworld-import`` copies a released scene into a package::

    ouroworld-import mipnerf360 --ply point_cloud.ply --colmap mipnerf360/room \\
        --images images_2 --out output/mip-nerf/room/scene
    ouroworld-import world --ply scene.ply --out output/HY_World_2.0/jungle/scene
    ouroworld-import world --ply world.ply --forward=-z --up=+y --position 0,1.5,-1 \\
        --out output/Marble/world/scene
    ouroworld-import world --ply my_scene.ply --camera my_camera.json \\
        --out output/Custom/my_scene/scene
    ouroworld-import lyra --source lyra2_3dgs_scenes/02 --out output/Lyra_2.0/palace_rooftops/scene
    ouroworld-import released --source data/3dgs/Lyra_2.0/palace_rooftops \\
        --out output/Lyra_2.0/palace_rooftops/scene   # pivot and reference included

``ouroworld-pivot`` serves a page to click the pivot on (or takes ``--pixel``),
and ``ouroworld-reference`` renders the reference candidates and stores the
camera selected by ``reference.selected``::

    ouroworld-pivot dataset=Marble scene=world
    ouroworld-reference dataset=Marble scene=world
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from ouroworld.cli.common import add_config_arguments, config_from_arguments, setup_logging
from ouroworld.datasets.importers import (
    AXES,
    LyraSettings,
    axis_camera,
    import_generated_world,
    import_lyra,
    import_mipnerf360,
    import_released,
)
from ouroworld.geometry.camera import DEFAULT_ZFAR, DEFAULT_ZNEAR, Camera
from ouroworld.io.cameras import load_camera
from ouroworld.io.images import save_rgb8
from ouroworld.io.scene_package import load_scene_source, save_pivot
from ouroworld.reference.candidates import CandidateSettings
from ouroworld.reference.pivot import pick_pivot
from ouroworld.reference.pivot_ui import serve_pivot_picker
from ouroworld.reference.reference_view import (
    ReferenceSettings,
    choose_reference,
    pivot_preview,
)
from ouroworld.render.static import load_static_renderer

logger = logging.getLogger(__name__)

CANDIDATES_DIR = "reference_candidates"


def import_main(argv: list[str] | None = None) -> None:
    """Entry point of ``ouroworld-import``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="dataset", required=True)
    mipnerf = commands.add_parser("mipnerf360", help="3DGS trained on a Mip-NeRF 360 scene")
    mipnerf.add_argument("--ply", type=Path, required=True)
    mipnerf.add_argument("--colmap", type=Path, required=True, help="scene dir with sparse/0")
    mipnerf.add_argument("--images", required=True, help="image folder trained on, e.g. images_2")
    mipnerf.add_argument("--holdout", type=int, default=8, help="test hold-out step (0: none)")
    world = commands.add_parser("world", help="HY-World 2.0 or Marble generated world")
    world.add_argument("--ply", type=Path, required=True)
    world.add_argument(
        "--camera",
        type=Path,
        help='camera JSON {"width", "height", "K", "c2w"} (OpenCV, camera-to-world); '
        "replaces the axis camera below",
    )
    world.add_argument("--forward", choices=AXES, help="camera looks down this axis (-x)")
    world.add_argument("--up", choices=AXES, help="axis up in the image (+z)")
    world.add_argument("--position", help="camera position x,y,z (0,0,0)")
    world.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), help="(1732 1000)")
    world.add_argument("--fov", type=float, nargs=2, metavar=("X", "Y"), help="degrees (120 90)")
    lyra = commands.add_parser("lyra", help="Lyra 2.0 scene.ply + camera.json")
    lyra.add_argument("--source", type=Path, required=True)
    lyra.add_argument("--world-scale", type=float, default=None, help="default: automatic")
    released = commands.add_parser("released", help="a released scene of the paper (data/3dgs)")
    released.add_argument("--source", type=Path, required=True)
    for command in (mipnerf, world, lyra, released):
        command.add_argument("--out", type=Path, required=True, help="package directory")
    for command in (mipnerf, world):
        command.add_argument("--clip", type=float, nargs=2, default=(DEFAULT_ZNEAR, DEFAULT_ZFAR))
    arguments = parser.parse_args(argv)
    setup_logging()
    if arguments.dataset == "mipnerf360":
        import_mipnerf360(
            arguments.ply,
            arguments.colmap,
            arguments.images,
            arguments.out,
            arguments.holdout,
            tuple(arguments.clip),
        )
    elif arguments.dataset == "world":
        axis_options = (arguments.forward, arguments.up, arguments.position, arguments.size)
        if arguments.camera is not None:
            if any(option is not None for option in (*axis_options, arguments.fov)):
                parser.error("--camera replaces --forward, --up, --position, --size and --fov")
            given = load_camera(arguments.camera)
            camera = Camera(given.width, given.height, given.K, given.c2w, *arguments.clip)
        else:
            position = np.array(
                [float(value) for value in (arguments.position or "0,0,0").split(",")]
            )
            camera = axis_camera(
                arguments.forward or "-x",
                arguments.up or "+z",
                position,
                tuple(arguments.size or (1732, 1000)),
                tuple(arguments.fov or (120.0, 90.0)),
                tuple(arguments.clip),
            )
        import_generated_world(arguments.ply, arguments.out, camera)
    elif arguments.dataset == "released":
        selected = import_released(arguments.source, arguments.out)
        logger.info("wrote %s (%s); next: ouroworld-reference-video", arguments.out, selected)
        return
    else:
        scale = import_lyra(
            arguments.source, arguments.out, LyraSettings(world_scale=arguments.world_scale)
        )
        logger.info("world scale %g", scale)
    logger.info("wrote %s; next: ouroworld-pivot", arguments.out)


def pivot_main(argv: list[str] | None = None) -> None:
    """Entry point of ``ouroworld-pivot``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pixel",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="skip the page and pick the pivot at this pixel",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    add_config_arguments(parser)
    arguments = parser.parse_args(argv)
    setup_logging()
    config = config_from_arguments(arguments)
    source = load_scene_source(Path(config.paths.scene_dir))
    renderer = load_static_renderer(source.gaussians_path, source.background)
    camera = source.import_camera
    view = renderer.view(camera)
    radius = config.reference.pivot_patch_radius

    def pick(x: float, y: float):  # noqa: ANN202
        return pick_pivot(view.depth, camera, (x, y), radius)

    def save(pivot) -> str:  # noqa: ANN001
        save_pivot(source.root, pivot)
        preview = source.root.parent / CANDIDATES_DIR / "pivot.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        save_rgb8(pivot_preview(view.image, camera, pivot.world), preview)
        return str(source.root)

    if arguments.pixel is not None:
        save(pick(*arguments.pixel))
    elif (
        serve_pivot_picker(
            view.image,
            pick,
            save,
            arguments.host,
            arguments.port,
            f"{config.dataset}/{config.scene}",
        )
        is None
    ):
        return
    logger.info("pivot saved; next: ouroworld-reference")


def reference_main(argv: list[str] | None = None) -> None:
    """Entry point of ``ouroworld-reference``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_arguments(parser)
    arguments = parser.parse_args(argv)
    setup_logging()
    config = config_from_arguments(arguments)
    reference = config.reference
    grid = reference.candidates
    settings = ReferenceSettings(
        selected=reference.selected,
        width=reference.width,
        height=reference.height,
        center_patch_fraction=reference.center_patch_fraction,
        candidates=CandidateSettings(
            grid.horizontal_degrees,
            grid.vertical_degrees,
            grid.horizontal_steps,
            grid.vertical_steps,
            tuple(grid.forward_fractions),
        ),
        thumbnail_width=grid.thumbnail_width,
        min_coverage=grid.min_coverage,
    )
    source = load_scene_source(Path(config.paths.scene_dir))
    renderer = load_static_renderer(source.gaussians_path, source.background)
    diagnostics = source.root.parent / CANDIDATES_DIR
    package = choose_reference(source, renderer, settings, diagnostics)
    logger.info(
        "reference %s -> %s; candidates in %s",
        reference.selected,
        package.reference_image_path,
        diagnostics / "candidates.png",
    )
