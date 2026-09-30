"""Argument handling shared by the command-line entry points."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from ouroworld.config.loader import load_config
from ouroworld.config.schema import Config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "default.yaml"


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--config`` files and trailing ``key=value`` overrides."""
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="YAML merged after configs/default.yaml; may be repeated",
    )
    parser.add_argument(
        "overrides", nargs="*", help="key=value overrides, e.g. train.iterations=100"
    )


def config_from_arguments(arguments: argparse.Namespace) -> Config:
    """Load ``configs/default.yaml``, the ``--config`` files and the overrides."""
    return load_config([DEFAULT_CONFIG, *arguments.config], arguments.overrides)


def setup_logging() -> None:
    """Log INFO and above to the console."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )


def load_dotenv(path: Path = REPOSITORY_ROOT / ".env") -> None:
    """Export ``KEY=VALUE`` lines of ``path``; variables already set are kept."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def require_env(name: str) -> str:
    """Return the environment variable ``name``, or exit with a message naming it."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set; export it or add it to {REPOSITORY_ROOT / '.env'}")
    return value
