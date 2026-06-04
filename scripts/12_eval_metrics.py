from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import compute_binary_metrics, group_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute binary clone detection metrics.")
    parser.add_argument("--input", required=True, help="Prediction JSONL/JSON path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--group-fields", default="dataset,language_pair,method")
    parser.add_argument("--gold-field", default="gold_label")
    parser.add_argument("--pred-field", default="pred_label")
    parser.add_argument("--default-method", default="UAST-CCD")
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
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


def iter_records(path: Path) -> Iterable[Dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        yield from iter_jsonl(path)
        return
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


def normalize_record(record: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    out = dict(record)
    out["gold_label"] = record.get(args.gold_field)
    out["pred_label"] = record.get(args.pred_field)
    out.setdefault("dataset", record.get("Dataset") or record.get("dataset") or "")
    if not out.get("language_pair"):
        lang_a = record.get("language_a") or record.get("Category1") or ""
        lang_b = record.get("language_b") or record.get("Category2") or ""
        out["language_pair"] = f"{lang_a}-{lang_b}" if lang_a or lang_b else ""
    out.setdefault("method", args.default_method)
    return out


def write_csv(path: Path, rows: List[Dict[str, object]], group_fields: List[str]) -> None:
    headers = [
        *group_fields,
        "total",
        "skipped",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "accuracy",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    group_fields = [f.strip() for f in args.group_fields.split(",") if f.strip()]

    records = []
    for idx, record in enumerate(iter_records(input_path)):
        if args.max_records and idx >= args.max_records:
            break
        records.append(normalize_record(record, args))

    rows = []
    for key, group in sorted(group_records(records, group_fields).items()):
        metrics = compute_binary_metrics(group)
        row = {field: key[i] for i, field in enumerate(group_fields)}
        row.update(metrics)
        rows.append(row)

    write_csv(output_path, rows, group_fields)
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
