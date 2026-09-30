"""Tables 2 and 3 from the per-metric result files.

Each metric run writes ``<output>/<group>/<view>/<metric>.json`` holding a
``videos`` list (per-video scores) or, for KVD, one set-level score. A table
cell is the mean over the scenes, KVD the pooled score.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MISSING = "–"


@dataclass(frozen=True)
class Column:
    """One table column: which metric file, which score, how to print it."""

    label: str
    metric: str
    key: str
    decimals: int
    static_only: bool = False


TABLE2 = (
    Column("Vividness Degree ↑", "vividness", "vividness", 4),
    Column("KVD ↓", "naturalness", "kvd", 2),
    Column("Motion-Aware Loop Fidelity ↑", "loop_seam", "malf", 4, static_only=True),
    Column("Seam SSIM ↑", "loop_seam", "seam_ssim", 4, static_only=True),
    Column("Aesthetic Quality ↑", "scene_quality", "aesthetic_quality", 4),
    Column("Overall Consist. ↑", "scene_quality", "overall_consistency", 4),
)
PERIOD = (
    Column("Seam SSIM ↑", "loop_seam", "seam_ssim", 4, static_only=True),
    Column("Motion-Aware Loop Fidelity ↑", "loop_seam", "malf", 4, static_only=True),
)
TABLE3 = (
    Column("Subject Consistency ↑", "scene_quality", "subject_consistency", 4),
    Column("Background Consistency ↑", "scene_quality", "background_consistency", 4),
    Column("VoL ↑", "sharpness", "vol", 2),
    Column("Vividness Degree ↑", "vividness", "vividness", 4),
)


def result_path(output_root: Path, group: str, view: str, metric: str) -> Path:
    """Where one metric of one group and view is stored."""
    return Path(output_root) / group / view / f"{metric}.json"


def write_result(path: Path, document: dict[str, Any]) -> None:
    """Write a metric result file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def cell(output_root: Path, group: str, view: str, column: Column) -> float | None:
    """The value of one cell, ``None`` if the metric was not run."""
    path = result_path(output_root, group, view, column.metric)
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if "videos" in document:
        return statistics.fmean(float(video[column.key]) for video in document["videos"])
    return float(document[column.key])


def build_table(
    output_root: Path, rows: list[tuple[str, str]], views: list[str], columns: tuple[Column, ...]
) -> list[dict[str, Any]]:
    """One record per (view, row): ``{"view", "method", <label>: value}``."""
    records = []
    for view_index, view in enumerate(views):
        for group, label in rows:
            record: dict[str, Any] = {"view": view, "method": label}
            for column in columns:
                skip = column.static_only and view_index > 0
                record[column.label] = None if skip else cell(output_root, group, view, column)
            records.append(record)
    return records


def markdown(records: list[dict[str, Any]], columns: tuple[Column, ...], title: str) -> str:
    """A Markdown table with the best value of each view in bold."""
    lines = [
        f"### {title}",
        "",
        "| Camera | Method | " + " | ".join(c.label for c in columns) + " |",
    ]
    lines.append("| --- | --- |" + " ---: |" * len(columns))
    for view in dict.fromkeys(record["view"] for record in records):
        subset = [record for record in records if record["view"] == view]
        best = {}
        for column in columns:
            values = [record[column.label] for record in subset if record[column.label] is not None]
            if values:
                best[column.label] = (min if "↓" in column.label else max)(values)
        for record in subset:
            cells = []
            for column in columns:
                value = record[column.label]
                text = MISSING if value is None else f"{value:.{column.decimals}f}"
                cells.append(
                    f"**{text}**" if value is not None and value == best[column.label] else text
                )
            lines.append(f"| {view} | {record['method']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_tables(
    output_root: Path,
    methods: list[tuple[str, str]],
    ablations: list[tuple[str, str]],
    period_ablation: tuple[str, str],
    views: list[str],
) -> Path:
    """Write ``tables.md`` and one CSV per table under ``output_root``.

    Besides Tables 2 and 3, the period ablation of §4.9 (Seam SSIM with a
    period-2T basis) is tabulated on the static camera.
    """
    output_root = Path(output_root)
    ours = [row for row in methods if row[0] == "ours"]
    parts = []
    for name, rows, columns, title in (
        ("table2", methods, TABLE2, "Table 2: comparison with the baselines"),
        ("table3", ablations + ours, TABLE3, "Table 3: ablation of the consistency components"),
        ("period", [period_ablation, *ours], PERIOD, "§4.9: period T vs 2T (static camera)"),
    ):
        records = build_table(output_root, rows, views[:1] if name == "period" else views, columns)
        with (output_root / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        parts.append(markdown(records, columns, title))
    path = output_root / "tables.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
