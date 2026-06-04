from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping


OPERATOR_TAGS = {
    "+": "OP_+",
    "-": "OP_-",
    "*": "OP_*",
    "/": "OP_/",
    "%": "OP_%",
    "=": "OP_=",
    "+=": "OP_+=",
    "-=": "OP_-=",
    "*=": "OP_*=",
    "/=": "OP_/=",
    "==": "OP_==",
    "!=": "OP_!=",
    "<": "OP_<",
    ">": "OP_>",
    "<=": "OP_<=",
    ">=": "OP_>=",
    "&&": "OP_&&",
    "||": "OP_||",
    "&": "OP_&",
    "|": "OP_|",
    "^": "OP_^",
    "!": "OP_!",
    "~": "OP_~",
    "++": "OP_++",
    "--": "OP_--",
}

DIRECT_TAGS = {
    "identifier": "ID",
    "type_identifier": "TYPE",
    "property_identifier": "ID",
    "field_identifier": "ID",
    "integer": "LIT_NUM",
    "decimal_integer_literal": "LIT_NUM",
    "integer_literal": "LIT_NUM",
    "float": "LIT_NUM",
    "float_literal": "LIT_NUM",
    "true": "LIT_BOOL",
    "false": "LIT_BOOL",
    "string": "LIT_STR",
    "string_literal": "LIT_STR",
    "character_literal": "LIT_STR",
    "return": "RETURN",
    "break": "BREAK",
    "continue": "CONTINUE",
}

PATTERN_TAGS = [
    (re.compile(r"(for|while|loop|do)_?statement|^for$|^while$"), "LOOP"),
    (re.compile(r"(if|switch|conditional|else)_?"), "COND"),
    (re.compile(r"(call|invocation|argument_list|object_creation)"), "CALL"),
    (re.compile(r"(assign|declarator|declaration)"), "ASSIGN"),
    (re.compile(r"(return)_?statement"), "RETURN"),
    (re.compile(r"(lambda|function|method)_?"), "FUNC"),
    (re.compile(r"(class|struct|interface|enum)_?"), "DECL"),
    (re.compile(r"(array_access|subscript|index)"), "INDEX"),
    (re.compile(r"(field_access|member|attribute)"), "MEMBER"),
    (re.compile(r"(binary|unary|update|expression|cast|parenthesized)"), "EXPR"),
    (re.compile(r"(block|body|compound)"), "BLOCK"),
    (re.compile(r"(parameter|parameters)"), "PARAM"),
    (re.compile(r"(type|primitive|integral|boolean|void)"), "TYPE"),
]


def node_type_to_tag(node_type: str) -> str:
    node_type = str(node_type)
    if node_type in OPERATOR_TAGS:
        return OPERATOR_TAGS[node_type]
    if node_type in DIRECT_TAGS:
        return DIRECT_TAGS[node_type]

    normalized = node_type.lower()
    for pattern, tag in PATTERN_TAGS:
        if pattern.search(normalized):
            return tag
    return ""


def build_category_tag_table(
    mapping_table: Mapping[str, int], majority_threshold: float = 0.5
) -> Dict[int, str]:
    votes = defaultdict(Counter)
    for node_type, category in mapping_table.items():
        tag = node_type_to_tag(node_type)
        if not tag:
            continue
        votes[int(category)][tag] += 1

    out: Dict[int, str] = {}
    for category, counter in votes.items():
        total = sum(counter.values())
        tag, count = counter.most_common(1)[0]
        if total > 0 and count / total >= majority_threshold:
            out[int(category)] = tag
        else:
            out[int(category)] = f"STRUCT_{category}"
    return out


def load_mapping(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def write_tag_table(path: Path, tag_table: Mapping[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(tag_table.items())}, f, ensure_ascii=False, indent=2)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Assign readable tags to learned UAST categories.")
    parser.add_argument("--mapping", required=True, help="Path to mapping_table.json")
    parser.add_argument("--output", required=True, help="Path to write category_tag_table.json")
    parser.add_argument("--majority-threshold", type=float, default=0.5)
    args = parser.parse_args()

    mapping = load_mapping(Path(args.mapping))
    table = build_category_tag_table(mapping, args.majority_threshold)
    write_tag_table(Path(args.output), table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
