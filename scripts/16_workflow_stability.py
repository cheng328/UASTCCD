from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


WORKFLOWS = [
    "Single-stage JSON",
    "Decoupled w/o Quality Guard",
    "Decoupled w/o Standard Keeper",
    "Full UAST-CCD",
]

WORKFLOW_SLUGS = {
    "Single-stage JSON": "single_stage_json",
    "Decoupled w/o Quality Guard": "decoupled_without_quality_guard",
    "Decoupled w/o Standard Keeper": "decoupled_without_standard_keeper",
    "Full UAST-CCD": "full_uast_ccd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect workflow stability results for Table 8.")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--logs-dir", required=True, help="Directory with per-workflow logs")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _rate(values: List[bool]) -> float:
    return sum(1 for v in values if v) / len(values) if values else 0.0


def _agreement(prediction_rows: Iterable[Dict[str, object]]) -> float:
    by_pair = defaultdict(list)
    for row in prediction_rows:
        by_pair[str(row.get("pair_id", ""))].append(row.get("pred_label"))
    if not by_pair:
        return 0.0
    agreed = 0
    for labels in by_pair.values():
        agreed += 1 if len(set(labels)) <= 1 else 0
    return agreed / len(by_pair)


def collect_rows(logs_dir: Path) -> List[Dict[str, object]]:
    rows = []
    for workflow in WORKFLOWS:
        slug = WORKFLOW_SLUGS[workflow]
        wf_dir = logs_dir / slug
        quality = list(iter_jsonl(wf_dir / "quality_guard_log.jsonl"))
        standard = list(iter_jsonl(wf_dir / "standard_keeper_log.jsonl"))
        predictions = list(iter_jsonl(wf_dir / "predictions_normalized.jsonl"))

        leakage_values = []
        for row in quality:
            errors = row.get("initial_errors") or row.get("final_errors") or []
            leakage_values.append(any("forbidden field" in str(e) for e in errors))

        rows.append(
            {
                "workflow": workflow,
                "valid_json": _rate([bool(row.get("initial_ok", row.get("final_ok"))) for row in quality]),
                "schema_pass": _rate([bool(row.get("final_ok")) for row in quality]),
                "leakage": _rate(leakage_values),
                "repair_success": _rate([bool(row.get("repair_success")) for row in quality]),
                "final_parse": _rate([bool(row.get("final_parse")) for row in standard]),
                "agreement": _agreement(predictions),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    headers = [
        "workflow",
        "valid_json",
        "schema_pass",
        "leakage",
        "repair_success",
        "final_parse",
        "agreement",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = collect_rows(Path(args.logs_dir))
    write_csv(Path(args.output), rows)
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
