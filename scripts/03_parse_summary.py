from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize parser success/failure by dataset and language.")
    parser.add_argument("--input", required=True, help="Parsed JSONL/JSON from 03_parse_ast.py")
    parser.add_argument("--output", required=True, help="Output CSV summary")
    parser.add_argument("--error-output", default="", help="Optional JSONL file with parse error records")
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def iter_records(path: Path) -> Iterator[Dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


def has_ast(record: Dict[str, object], field: str) -> bool:
    return isinstance(record.get(field), dict)


def side_key(record: Dict[str, object], side: str) -> Tuple[str, str, str]:
    dataset = str(record.get("dataset") or record.get("Dataset") or "")
    if side == "a":
        language = str(record.get("language_a") or record.get("Category1") or "")
    else:
        language = str(record.get("language_b") or record.get("Category2") or "")
    pair = str(record.get("language_pair") or record.get("PairType") or "")
    return dataset, language, pair


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "language",
        "language_pair",
        "side",
        "total",
        "parsed",
        "failed",
        "parse_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    counts: Dict[Tuple[str, str, str, str], Counter] = defaultdict(Counter)
    error_rows = []

    for idx, record in enumerate(iter_records(Path(args.input))):
        if args.max_records and idx >= args.max_records:
            break
        for side, ast_field in (("a", "ast1"), ("b", "ast2")):
            dataset, language, pair = side_key(record, side)
            key = (dataset, language, pair, side)
            counts[key]["total"] += 1
            if has_ast(record, ast_field):
                counts[key]["parsed"] += 1
            else:
                counts[key]["failed"] += 1
                errors = record.get("parse_errors")
                error_rows.append(
                    {
                        "pair_id": record.get("pair_id"),
                        "dataset": dataset,
                        "language": language,
                        "language_pair": pair,
                        "side": side,
                        "errors": errors,
                    }
                )

    rows = []
    for (dataset, language, pair, side), counter in sorted(counts.items()):
        total = int(counter["total"])
        parsed = int(counter["parsed"])
        failed = int(counter["failed"])
        rows.append(
            {
                "dataset": dataset,
                "language": language,
                "language_pair": pair,
                "side": side,
                "total": total,
                "parsed": parsed,
                "failed": failed,
                "parse_rate": parsed / total if total else 0.0,
            }
        )

    write_csv(Path(args.output), rows)
    if args.error_output:
        error_path = Path(args.error_output)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with error_path.open("w", encoding="utf-8") as f:
            for row in error_rows:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")

    print(json.dumps({"summary": args.output, "errors": args.error_output, "error_records": len(error_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
