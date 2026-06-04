from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


DEFAULT_MODELS = [
    "gpt-oss-20b",
    "Llama4-16x17b",
    "Llama3.1-8b",
    "Llama3.1-70b",
    "Qwen3-Coder-30b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Evidence LLM robustness results for Table 6.")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--results-dir", required=True, help="Directory with per-evidence-model metrics")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    return parser.parse_args()


def _read_first_csv(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
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
    for row in rows:
        weight = _to_float(row.get("total"), 0.0)
        if weight <= 0:
            weight = 1.0
        total_weight += weight
        for metric in weighted:
            weighted[metric] += _to_float(row.get(metric), 0.0) * weight
    return {metric: value / total_weight for metric, value in weighted.items()}


def _slug(name: str) -> str:
    return name.lower().replace(".", "_").replace("-", "_").replace(":", "_")


def collect_rows(results_dir: Path, models: List[str]) -> List[Dict[str, object]]:
    rows = []
    for model in models:
        metrics = _read_first_csv(results_dir / _slug(model) / "metrics.csv")
        rows.append(
            {
                "evidence_llm": f"Evidence = {model}",
                "precision": metrics.get("precision", ""),
                "recall": metrics.get("recall", ""),
                "f1": metrics.get("f1", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["evidence_llm", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = collect_rows(Path(args.results_dir), models)
    write_csv(Path(args.output), rows)
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
