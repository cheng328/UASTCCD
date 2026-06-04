from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.guard.standard_keeper import parse_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize prediction outputs.")
    parser.add_argument("--input", required=True, help="Input JSONL/JSON path")
    parser.add_argument("--output", required=True, help="Output JSONL/JSON path")
    parser.add_argument("--pred-field", default="prediction")
    parser.add_argument("--out-field", default="normalized_prediction")
    parser.add_argument("--label-field", default="label,Label,gold_label")
    parser.add_argument("--log", default="", help="Optional JSONL path for Standard Keeper results")
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


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def get_first(record: Dict[str, object], fields: str, default: object = None) -> object:
    for field in [f.strip() for f in fields.split(",") if f.strip()]:
        if field in record:
            return record.get(field)
    return default


def process_record(record: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    new_record = dict(record)
    pred = record.get(args.pred_field)
    result = parse_prediction(pred)
    gold = get_first(record, args.label_field)
    try:
        gold_label = int(gold)
    except (TypeError, ValueError):
        gold_label = None

    new_record[args.out_field] = result.normalized
    new_record[args.out_field + "_ok"] = result.ok
    new_record[args.out_field + "_errors"] = result.errors
    new_record["gold_label"] = gold_label
    new_record["pred_label"] = result.normalized["label"]
    new_record["confidence"] = result.normalized["confidence"]
    new_record["parse_status"] = "valid" if result.ok else "repaired"
    return new_record


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    log_rows = []

    if input_path.suffix.lower() == ".jsonl":
        rows = []
        for idx, record in enumerate(iter_jsonl(input_path)):
            if args.max_records and idx >= args.max_records:
                break
            out_record = process_record(record, args)
            rows.append(out_record)
            log_rows.append(
                {
                    "pair_id": out_record.get("pair_id", idx),
                    "final_parse": out_record.get(args.out_field + "_ok"),
                    "errors": out_record.get(args.out_field + "_errors"),
                    "pred_label": out_record.get("pred_label"),
                    "confidence": out_record.get("confidence"),
                }
            )
        write_jsonl(output_path, rows)
        if args.log:
            write_jsonl(Path(args.log), log_rows)
        return 0

    data = load_json(input_path)
    if isinstance(data, list):
        out = []
        for idx, record in enumerate(data):
            if args.max_records and idx >= args.max_records:
                break
            if isinstance(record, dict):
                out_record = process_record(record, args)
                out.append(out_record)
                log_rows.append(
                    {
                        "pair_id": out_record.get("pair_id", idx),
                        "final_parse": out_record.get(args.out_field + "_ok"),
                        "errors": out_record.get(args.out_field + "_errors"),
                        "pred_label": out_record.get("pred_label"),
                        "confidence": out_record.get("confidence"),
                    }
                )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        if args.log:
            write_jsonl(Path(args.log), log_rows)
        return 0

    raise SystemExit("Unsupported input JSON structure")


if __name__ == "__main__":
    raise SystemExit(main())

