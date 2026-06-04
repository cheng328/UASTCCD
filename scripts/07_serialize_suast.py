from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.uast.apply_mapping import apply_mapping_to_ast, load_mapping, load_tag_table
from src.uast.serializer import serialize_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serialize pruned ASTs into S-UAST strings using a mapping table."
    )
    parser.add_argument("--input", required=True, help="Input JSONL/JSON path")
    parser.add_argument("--output", required=True, help="Output JSONL/JSON path")
    parser.add_argument("--mapping", required=True, help="Path to mapping_table.json")
    parser.add_argument("--tag-table", default="", help="Optional category tag table JSON")
    parser.add_argument("--ast1-field", default="pruned_ast1")
    parser.add_argument("--ast2-field", default="pruned_ast2")
    parser.add_argument("--out1-field", default="suast1")
    parser.add_argument("--out2-field", default="suast2")
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


def process_record(
    record: Dict[str, object],
    mapping: Dict[str, int],
    tag_table: Dict[int, str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    new_record = dict(record)
    ast1 = record.get(args.ast1_field)
    ast2 = record.get(args.ast2_field)

    if isinstance(ast1, dict):
        mapped = apply_mapping_to_ast(ast1, mapping, tag_table)
        new_record[args.out1_field] = serialize_root(mapped)
    else:
        new_record[args.out1_field] = ""

    if isinstance(ast2, dict):
        mapped = apply_mapping_to_ast(ast2, mapping, tag_table)
        new_record[args.out2_field] = serialize_root(mapped)
    else:
        new_record[args.out2_field] = ""

    return new_record


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    mapping = load_mapping(Path(args.mapping))
    tag_table = load_tag_table(Path(args.tag_table)) if args.tag_table else {}

    if input_path.suffix.lower() == ".jsonl":
        rows = []
        for idx, record in enumerate(iter_jsonl(input_path)):
            if args.max_records and idx >= args.max_records:
                break
            rows.append(process_record(record, mapping, tag_table, args))
        write_jsonl(output_path, rows)
        return 0

    data = load_json(input_path)
    if isinstance(data, list):
        out = []
        for idx, record in enumerate(data):
            if args.max_records and idx >= args.max_records:
                break
            if isinstance(record, dict):
                out.append(process_record(record, mapping, tag_table, args))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        return 0

    raise SystemExit("Unsupported input JSON structure")


if __name__ == "__main__":
    raise SystemExit(main())

