from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect per-run metric CSV files into one metrics CSV.")
    parser.add_argument("--input-dir", required=True, help="Directory containing metric CSV files")
    parser.add_argument("--output", required=True, help="Merged output CSV")
    parser.add_argument("--pattern", default="*.csv")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "language_pair",
        "method",
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
        "source_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows: List[Dict[str, object]] = []
    for path in sorted(input_dir.rglob(args.pattern)):
        if path.resolve() == Path(args.output).resolve():
            continue
        for row in read_csv(path):
            row["source_file"] = str(path)
            rows.append(row)
    write_csv(Path(args.output), rows)
    print(f"Wrote: {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
