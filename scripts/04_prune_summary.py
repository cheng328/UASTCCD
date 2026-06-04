from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.uast.stats import count_nodes_and_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize raw AST and pruned UAST sizes.")
    parser.add_argument("--input", required=True, help="Pruned JSONL/JSON from 04_prune_ast.py")
    parser.add_argument("--output", required=True, help="Output CSV summary")
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


def key_for(record: Dict[str, object]) -> Tuple[str, str]:
    dataset = str(record.get("dataset") or record.get("Dataset") or "")
    pair = str(record.get("language_pair") or record.get("PairType") or "")
    return dataset, pair


def add(values: Dict[str, List[int]], prefix: str, ast: object) -> None:
    nodes, depth, types = count_nodes_and_depth(ast)
    values[f"{prefix}_nodes"].append(nodes)
    values[f"{prefix}_depth"].append(depth)
    values[f"{prefix}_types"].append(len(types))


def avg(values: List[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "language_pair",
        "asts",
        "avg_raw_nodes",
        "avg_pruned_nodes",
        "node_reduction",
        "avg_raw_depth",
        "avg_pruned_depth",
        "depth_reduction",
        "avg_raw_types",
        "avg_pruned_types",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reduction(base: float, value: float) -> float:
    return 1.0 - value / base if base > 0 else 0.0


def main() -> int:
    args = parse_args()
    grouped: Dict[Tuple[str, str], Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for idx, record in enumerate(iter_records(Path(args.input))):
        if args.max_records and idx >= args.max_records:
            break
        key = key_for(record)
        values = grouped[key]
        for raw_field, pruned_field in (("ast1", "pruned_ast1"), ("ast2", "pruned_ast2")):
            raw_ast = record.get(raw_field)
            pruned_ast = record.get(pruned_field)
            if not isinstance(raw_ast, dict) or not isinstance(pruned_ast, dict):
                continue
            add(values, "raw", raw_ast)
            add(values, "pruned", pruned_ast)

    rows = []
    for (dataset, pair), values in sorted(grouped.items()):
        avg_raw_nodes = avg(values["raw_nodes"])
        avg_pruned_nodes = avg(values["pruned_nodes"])
        avg_raw_depth = avg(values["raw_depth"])
        avg_pruned_depth = avg(values["pruned_depth"])
        rows.append(
            {
                "dataset": dataset,
                "language_pair": pair,
                "asts": len(values["raw_nodes"]),
                "avg_raw_nodes": avg_raw_nodes,
                "avg_pruned_nodes": avg_pruned_nodes,
                "node_reduction": reduction(avg_raw_nodes, avg_pruned_nodes),
                "avg_raw_depth": avg_raw_depth,
                "avg_pruned_depth": avg_pruned_depth,
                "depth_reduction": reduction(avg_raw_depth, avg_pruned_depth),
                "avg_raw_types": avg(values["raw_types"]),
                "avg_pruned_types": avg(values["pruned_types"]),
            }
        )

    write_csv(Path(args.output), rows)
    print(json.dumps({"output": args.output, "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
