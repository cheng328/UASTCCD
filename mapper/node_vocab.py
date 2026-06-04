from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class NodeVocab:
    node_to_id: Dict[str, int]
    id_to_node: List[str]

    @classmethod
    def build(cls, node_types: Iterable[str]) -> "NodeVocab":
        unique = sorted(set(node_types))
        node_to_id = {t: i for i, t in enumerate(unique)}
        return cls(node_to_id=node_to_id, id_to_node=unique)

    def to_json(self) -> Dict[str, object]:
        return {"node_to_id": self.node_to_id, "id_to_node": self.id_to_node}

    @classmethod
    def from_json(cls, data: Dict[str, object]) -> "NodeVocab":
        node_to_id = {str(k): int(v) for k, v in data.get("node_to_id", {}).items()}
        id_to_node = [str(x) for x in data.get("id_to_node", [])]
        return cls(node_to_id=node_to_id, id_to_node=id_to_node)


def iter_jsonl(path: Path) -> Iterable[dict]:
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


def iter_records(path: Path) -> Iterable[dict]:
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


def collect_node_types(ast: object) -> List[str]:
    if not isinstance(ast, dict):
        return []
    root = ast.get("root") if isinstance(ast.get("root"), dict) else ast
    if not isinstance(root, dict):
        return []

    stack = [root]
    out: List[str] = []
    while stack:
        node = stack.pop()
        node_type = node.get("type")
        if isinstance(node_type, str):
            out.append(node_type)
        children = node.get("children")
        if isinstance(children, list):
            for child in reversed(children):
                if isinstance(child, dict):
                    stack.append(child)
    return out


def build_vocab_from_dataset(
    path: Path, ast_keys: Tuple[str, str], max_records: int = 0
) -> NodeVocab:
    node_types: List[str] = []
    for idx, record in enumerate(iter_records(path)):
        if max_records and idx >= max_records:
            break
        for key in ast_keys:
            node_types.extend(collect_node_types(record.get(key)))
    return NodeVocab.build(node_types)

