from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.uast.pruning import prune_ast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune raw ASTs and write pruned ASTs (JSONL/JSON)."
    )
    parser.add_argument("--input", required=True, help="Input JSONL/JSON path")
    parser.add_argument("--output", required=True, help="Output JSONL/JSON path")
    parser.add_argument("--ast1-field", default="ast1", help="AST1 field name")
    parser.add_argument("--ast2-field", default="ast2", help="AST2 field name")
    parser.add_argument("--out-ast1-field", default="pruned_ast1", help="Output field for pruned AST1")
    parser.add_argument("--out-ast2-field", default="pruned_ast2", help="Output field for pruned AST2")
    parser.add_argument("--granularity", default="function", help="file/function/block")
    parser.add_argument("--max-records", type=int, default=0, help="Process only first N records")
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


def process_record(record: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    new_record = dict(record)
    ast1 = record.get(args.ast1_field)
    ast2 = record.get(args.ast2_field)

    if isinstance(ast1, dict):
        new_record[args.out_ast1_field] = prune_ast(ast1, args.granularity)
    else:
        new_record[args.out_ast1_field] = None

    if isinstance(ast2, dict):
        new_record[args.out_ast2_field] = prune_ast(ast2, args.granularity)
    else:
        new_record[args.out_ast2_field] = None

    return new_record


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.suffix.lower() == ".jsonl":
        rows = []
        for idx, record in enumerate(iter_jsonl(input_path)):
            if args.max_records and idx >= args.max_records:
                break
            rows.append(process_record(record, args))
        write_jsonl(output_path, rows)
        return 0

    data = load_json(input_path)
    if isinstance(data, list):
        out = []
        for idx, record in enumerate(data):
            if args.max_records and idx >= args.max_records:
                break
            if isinstance(record, dict):
                out.append(process_record(record, args))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        return 0

    raise SystemExit("Unsupported input JSON structure")


if __name__ == "__main__":
    raise SystemExit(main())

