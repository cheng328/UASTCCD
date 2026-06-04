from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


VARIANTS = [
    "Full Mapper",
    "w/o L_sharp",
    "w/o L_balance",
    "w/o L_align",
    "w/o seed dictionary",
    "Random mapper",
]

VARIANT_SLUGS = {
    "Full Mapper": "full_mapper",
    "w/o L_sharp": "without_lsharp",
    "w/o L_balance": "without_lbalance",
    "w/o L_align": "without_lalign",
    "w/o seed dictionary": "without_seed_dict",
    "Random mapper": "random_mapper",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect mapper ablation results for Table 7.")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--metrics-dir", required=True, help="Directory containing per-variant metrics CSV/JSON files")
    parser.add_argument("--diagnostics-dir", required=True, help="Directory containing per-variant mapper diagnostics")
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _read_metric_row(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return _read_json(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return _aggregate_metric_rows(rows)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _aggregate_metric_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {}
    total_weight = 0.0
    weighted = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    counts = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0, "total": 0.0, "skipped": 0.0}
    for row in rows:
        weight = _to_float(row.get("total"), 0.0)
        if weight <= 0:
            weight = 1.0
        total_weight += weight
        for metric in weighted:
            weighted[metric] += _to_float(row.get(metric), 0.0) * weight
        for key in counts:
            counts[key] += _to_float(row.get(key), 0.0)
    out: Dict[str, object] = {metric: value / total_weight for metric, value in weighted.items()}
    out.update({key: int(value) for key, value in counts.items()})
    return out


def collect_rows(metrics_dir: Path, diagnostics_dir: Path) -> List[Dict[str, object]]:
    rows = []
    for variant in VARIANTS:
        slug = VARIANT_SLUGS[variant]
        metrics = _read_metric_row(metrics_dir / f"{slug}.csv") or _read_metric_row(metrics_dir / f"{slug}.json")
        diagnostics = _read_json(diagnostics_dir / slug / "mapper_diagnostics.json")
        rows.append(
            {
                "variant": variant,
                "precision": metrics.get("precision", ""),
                "recall": metrics.get("recall", ""),
                "f1": metrics.get("f1", ""),
                "active_categories": diagnostics.get("active_categories", metrics.get("active_categories", "")),
                "normalized_assignment_entropy": diagnostics.get(
                    "normalized_entropy", metrics.get("normalized_assignment_entropy", "")
                ),
                "seed_alignment_score": diagnostics.get("alignment_score", metrics.get("seed_alignment_score", "")),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    headers = [
        "variant",
        "precision",
        "recall",
        "f1",
        "active_categories",
        "normalized_assignment_entropy",
        "seed_alignment_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = collect_rows(Path(args.metrics_dir), Path(args.diagnostics_dir))
    write_csv(Path(args.output), rows)
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
