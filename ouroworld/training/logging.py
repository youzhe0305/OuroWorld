"""Training metrics written as JSON lines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricsLog:
    """Append one JSON object per logged iteration to ``metrics.jsonl``."""

    def __init__(self, path: Path, resume: bool):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            self.path.write_text("", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        """Append ``record``."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
