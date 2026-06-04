from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch

from .node_vocab import NodeVocab
from .soft_mapper import SoftMapper, SoftMapperConfig
from .tag_assignment import build_category_tag_table, write_tag_table


def discretize(mapper_path: Path, vocab_path: Path, output_dir: Path, num_categories: int) -> None:
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = NodeVocab.from_json(json.load(f))

    mapper = SoftMapper(SoftMapperConfig(num_node_types=len(vocab.id_to_node), num_categories=num_categories))
    state = torch.load(mapper_path, map_location="cpu")
    mapper.load_state_dict(state["mapper"])
    mapper.eval()

    with torch.no_grad():
        distributions = mapper.all_distributions()
        assignments = torch.argmax(distributions, dim=-1).cpu().tolist()

    mapping_table = {node: int(assignments[i]) for i, node in enumerate(vocab.id_to_node)}

    active_categories = sorted({int(x) for x in assignments})

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mapping_table.json").open("w", encoding="utf-8") as f:
        json.dump(mapping_table, f, ensure_ascii=False, indent=2)

    with (output_dir / "active_categories.json").open("w", encoding="utf-8") as f:
        json.dump({"active_categories": active_categories}, f, ensure_ascii=False, indent=2)

    tag_table = build_category_tag_table(mapping_table)
    write_tag_table(output_dir / "category_tag_table.json", tag_table)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Discretize a trained UAST mapper.")
    parser.add_argument("--mapper", required=True, help="Path to mapper.pt")
    parser.add_argument("--vocab", required=True, help="Path to node_type_vocab.json")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-categories", type=int, default=64)
    args = parser.parse_args()

    discretize(Path(args.mapper), Path(args.vocab), Path(args.output_dir), args.num_categories)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

