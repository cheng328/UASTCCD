from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_KS = [16, 32, 64, 96, 128]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect K sensitivity results for Figure 2.")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--runs-dir", required=True, help="Directory with K-specific run outputs")
    parser.add_argument("--k-values", default="16,32,64,96,128")
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


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
    f1 = 0.0
    for row in rows:
        weight = _to_float(row.get("total"), 0.0)
        if weight <= 0:
            weight = 1.0
        total_weight += weight
        f1 += _to_float(row.get("f1"), 0.0) * weight
    return {"f1": f1 / total_weight if total_weight else 0.0}


def collect_rows(runs_dir: Path, k_values: List[int]) -> List[Dict[str, object]]:
    rows = []
    for k in k_values:
        run_dir = runs_dir / f"k{k}"
        diagnostics = _read_json(run_dir / "mapper_diagnostics.json")
        metrics = _read_first_csv(run_dir / "metrics.csv")
        rows.append(
            {
                "K": k,
                "active_categories": diagnostics.get("active_categories", metrics.get("active_categories", "")),
                "f1": metrics.get("f1", diagnostics.get("f1", "")),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["K", "active_categories", "f1"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    k_values = [int(v.strip()) for v in args.k_values.split(",") if v.strip()]
    rows = collect_rows(Path(args.runs_dir), k_values)
    write_csv(Path(args.output), rows)
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
