from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.uast.stats import (
    count_nodes_and_depth,
    count_suast_nodes_and_depth,
    count_tokens,
    compute_stats,
    get_token_counter,
    merge_types,
    serialize_ast,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Table 3 statistics for AST/UAST/S-UAST.")
    parser.add_argument("--input", required=True, help="Input JSONL/JSON path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--raw-keys", default="ast1,ast2", help="Raw AST field names")
    parser.add_argument("--pruned-keys", default="pruned_ast1,pruned_ast2", help="Pruned AST field names")
    parser.add_argument("--suast-keys", default="suast1,suast2", help="S-UAST field names")
    parser.add_argument("--token-encoding", default="", help="tiktoken encoding name (optional)")
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


def iter_records(path: Path) -> Iterable[Dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        yield from iter_jsonl(path)
        return
    data = load_json(path)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        for key in ("data", "records", "items"):
            if isinstance(data.get(key), list):
                for item in data[key]:
                    if isinstance(item, dict):
                        yield item
                return


def parse_keys(raw: str) -> List[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


def collect_stats(
    records: Iterable[Dict[str, object]],
    ast_keys: List[str],
    is_suast: bool,
    encoder,
    max_records: int,
):
    node_counts: List[int] = []
    depths: List[int] = []
    token_counts: List[int] = []
    type_sets = []
    total_records = 0

    for record in records:
        total_records += 1
        if max_records and total_records > max_records:
            break
        for key in ast_keys:
            val = record.get(key)
            if val is None:
                continue
            if is_suast:
                text = val if isinstance(val, str) else ""
                nodes, depth, types = count_suast_nodes_and_depth(text)
                node_counts.append(nodes)
                depths.append(depth)
                type_sets.append(types)
                token_counts.append(count_tokens(text, encoder))
                continue
            if not isinstance(val, dict):
                continue
            nodes, depth, types = count_nodes_and_depth(val)
            node_counts.append(nodes)
            depths.append(depth)
            type_sets.append(types)
            token_counts.append(count_tokens(serialize_ast(val), encoder))

    max_nodes, avg_nodes, _ = compute_stats(node_counts)
    max_depth, avg_depth, _ = compute_stats(depths)
    max_tokens, avg_tokens, _ = compute_stats(token_counts)
    node_types = merge_types(type_sets)

    return {
        "node_types": len(node_types),
        "avg_nodes": avg_nodes,
        "max_nodes": max_nodes,
        "avg_depth": avg_depth,
        "max_depth": max_depth,
        "avg_tokens": avg_tokens,
        "max_tokens": max_tokens,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    headers = [
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def reduction_ratio(base: float, value: float) -> float:
    if base <= 0:
        return 0.0
    return 1.0 - (value / base)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    encoder = get_token_counter(args.token_encoding) if args.token_encoding else None

    raw_keys = parse_keys(args.raw_keys)
    pruned_keys = parse_keys(args.pruned_keys)
    suast_keys = parse_keys(args.suast_keys)

    records = list(iter_records(input_path))

    raw_stats = collect_stats(records, raw_keys, is_suast=False, encoder=encoder, max_records=args.max_records)
    pruned_stats = collect_stats(records, pruned_keys, is_suast=False, encoder=encoder, max_records=args.max_records)
    suast_stats = collect_stats(records, suast_keys, is_suast=True, encoder=encoder, max_records=args.max_records)

    rows = []
    rows.append(
        {
            "stage": "raw_ast",
            **raw_stats,
            "node_reduction": 0.0,
            "token_reduction": 0.0,
        }
    )

    rows.append(
        {
            "stage": "pruned_uast",
            **pruned_stats,
            "node_reduction": reduction_ratio(raw_stats["avg_nodes"], pruned_stats["avg_nodes"]),
            "token_reduction": reduction_ratio(raw_stats["avg_tokens"], pruned_stats["avg_tokens"]),
        }
    )

    rows.append(
        {
            "stage": "suast",
            **suast_stats,
            "node_reduction": reduction_ratio(raw_stats["avg_nodes"], suast_stats["avg_nodes"]),
            "token_reduction": reduction_ratio(raw_stats["avg_tokens"], suast_stats["avg_tokens"]),
        }
    )

    write_csv(output_path, rows)
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

