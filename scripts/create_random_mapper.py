from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.mapper.node_vocab import NodeVocab
from src.mapper.tag_assignment import build_category_tag_table, write_tag_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a random mapper for the mapper ablation.")
    parser.add_argument("--vocab", required=True, help="Path to node_type_vocab.json")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-categories", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    with Path(args.vocab).open("r", encoding="utf-8") as f:
        vocab = NodeVocab.from_json(json.load(f))

    mapping_table = {
        node_type: random.randrange(args.num_categories)
        for node_type in vocab.id_to_node
    }
    active_categories = sorted(set(mapping_table.values()))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "node_type_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab.to_json(), f, ensure_ascii=False, indent=2)
    with (output_dir / "mapping_table.json").open("w", encoding="utf-8") as f:
        json.dump(mapping_table, f, ensure_ascii=False, indent=2)
    with (output_dir / "active_categories.json").open("w", encoding="utf-8") as f:
        json.dump({"active_categories": active_categories}, f, ensure_ascii=False, indent=2)

    tag_table = build_category_tag_table(mapping_table)
    write_tag_table(output_dir / "category_tag_table.json", tag_table)

    diagnostics = {
        "K": args.num_categories,
        "node_types": len(vocab.id_to_node),
        "active_categories": len(active_categories),
        "normalized_entropy": 0.0,
        "alignment_score": 0.0,
        "variant": "random_mapper",
    }
    with (output_dir / "mapper_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print(json.dumps({"output_dir": str(output_dir), **diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
