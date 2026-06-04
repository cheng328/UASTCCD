from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple


try:
    import tiktoken
except Exception:
    tiktoken = None


def count_nodes_and_depth(ast: object) -> Tuple[int, int, Set[str]]:
    if not isinstance(ast, dict):
        return 0, 0, set()
    root = ast.get("root") if isinstance(ast.get("root"), dict) else ast
    if not isinstance(root, dict):
        return 0, 0, set()

    node_types: Set[str] = set()
    node_count = 0
    max_depth = 0
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        if not isinstance(node, dict):
            continue
        node_count += 1
        max_depth = max(max_depth, depth)
        node_type = node.get("type")
        if isinstance(node_type, str):
            node_types.add(node_type)
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    stack.append((child, depth + 1))
    return node_count, max_depth, node_types


def count_suast_nodes_and_depth(text: str) -> Tuple[int, int, Set[str]]:
    if not isinstance(text, str) or not text:
        return 0, 0, set()

    tags: Set[str] = set()
    node_count = 0
    max_depth = 0
    depth = 0
    token_re = re.compile(r"[A-Z_][A-Z0-9_]*")

    i = 0
    while i < len(text):
        match = token_re.match(text, i)
        if match:
            token = match.group(0)
            next_char_idx = match.end()
            while next_char_idx < len(text) and text[next_char_idx].isspace():
                next_char_idx += 1
            if next_char_idx < len(text) and text[next_char_idx] in "({":
                depth += 1
                max_depth = max(max_depth, depth)
                node_count += 1
                tags.add(token)
            elif token not in {"COND", "THEN", "ELSE", "INIT", "STEP", "BODY"}:
                node_count += 1
                max_depth = max(max_depth, depth + 1)
                tags.add(token)
            i = match.end()
            continue

        ch = text[i]
        if ch in ")}" and depth > 0:
            depth -= 1
        i += 1

    return node_count, max_depth, tags


def serialize_ast(ast: object) -> str:
    return json.dumps(ast, ensure_ascii=False, separators=(",", ":"))


def get_token_counter(encoding_name: str = "cl100k_base"):
    if tiktoken is None:
        return None
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoder=None) -> int:
    if not text:
        return 0
    if encoder is None:
        return len(re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE))
    return len(encoder.encode(text))


def compute_stats(values: List[int]) -> Tuple[int, float, int]:
    if not values:
        return 0, 0.0, 0
    return max(values), sum(values) / len(values), len(values)


def merge_types(type_sets: Iterable[Set[str]]) -> Set[str]:
    out: Set[str] = set()
    for s in type_sets:
        out.update(s)
    return out

