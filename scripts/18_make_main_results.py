from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.aggregate_results import (
    CLCD_LANGUAGE_PAIRS,
    SINGLE_LANGUAGE_DATASETS,
    aggregate_clcd_table,
    aggregate_single_language_table,
    read_csv,
    write_csv,
)


METHODS = [
    "Base Prompt",
    "+ Task Prompt",
    "+ Raw AST",
    "+ UAST",
    "UAST-CCD",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Table 4/5 CSVs from metrics CSVs.")
    parser.add_argument("--kind", required=True, choices=["table4", "table5"])
    parser.add_argument("--metrics", required=True, help="Input metrics CSV from 12_eval_metrics.py")
    parser.add_argument("--output", required=True, help="Output paper table CSV")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_rows = read_csv(Path(args.metrics))

    if args.kind == "table4":
        rows = aggregate_clcd_table(metrics_rows)
        fieldnames = ["method", *CLCD_LANGUAGE_PAIRS, "macro_f1"]
    else:
        rows = aggregate_single_language_table(metrics_rows)
        fieldnames = ["method", *SINGLE_LANGUAGE_DATASETS, "macro_f1"]

    write_csv(Path(args.output), rows, fieldnames)
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
