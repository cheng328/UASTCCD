from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from src.mapper.tag_assignment import build_category_tag_table, node_type_to_tag

from .tree_schema import ASTNode, iter_children


DEFAULT_TAG_PREFIX = "STRUCT_"


def load_mapping(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def load_tag_table(path: Optional[Path]) -> Dict[int, str]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[int, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                out[int(k)] = str(v)
            except (ValueError, TypeError):
                continue
    return out


def normalize_leaf_tag(node_type: str) -> str:
    return node_type_to_tag(node_type)


def map_node_tag(
    node_type: str, mapping: Dict[str, int], tag_table: Dict[int, str], is_leaf: bool
) -> str:
    if is_leaf:
        leaf_tag = normalize_leaf_tag(node_type)
        if leaf_tag.startswith("LIT_") or leaf_tag.startswith("OP_") or leaf_tag in {"ID", "TYPE"}:
            return leaf_tag
    cat = mapping.get(node_type, -1)
    if cat in tag_table:
        return tag_table[cat]
    if cat >= 0:
        return f"{DEFAULT_TAG_PREFIX}{cat}"
    return f"{DEFAULT_TAG_PREFIX}UNK"


def apply_mapping_to_ast(
    ast: Dict[str, object], mapping: Dict[str, int], tag_table: Dict[int, str]
) -> Dict[str, object]:
    root = ast.get("root") if isinstance(ast.get("root"), dict) else ast
    if not isinstance(root, dict):
        return ast
    if not tag_table:
        tag_table = build_category_tag_table(mapping)

    def remap(node: ASTNode) -> ASTNode:
        node_type = str(node.get("type", ""))
        children = list(iter_children(node))
        mapped = {
            "type": node_type,
            "tag": map_node_tag(node_type, mapping, tag_table, is_leaf=not children),
        }
        if children:
            mapped["children"] = [remap(child) for child in children]
        return mapped

    return remap(root)

