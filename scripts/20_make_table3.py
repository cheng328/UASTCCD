from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-dataset Table 3 statistics into one CSV.")
    parser.add_argument("--inputs", required=True, help="Comma-separated dataset=csv entries")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "stage",
        "node_types",
        "avg_nodes",
        "max_nodes",
        "avg_depth",
        "max_depth",
        "avg_tokens",
        "max_tokens",
        "node_reduction",
        "token_reduction",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_inputs(raw: str) -> List[tuple[str, Path]]:
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Expected dataset=path entry, got: {token}")
        dataset, path = token.split("=", 1)
        out.append((dataset.strip(), Path(path.strip())))
    return out


def main() -> int:
    args = parse_args()
    rows: List[Dict[str, object]] = []
    for dataset, path in parse_inputs(args.inputs):
        for row in read_csv(path):
            row = dict(row)
            row["dataset"] = dataset
            rows.append(row)
    write_csv(Path(args.output), rows)
    print(f"Wrote: {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
