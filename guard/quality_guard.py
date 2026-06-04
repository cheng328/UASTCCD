from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


REQUIRED_FIELDS = [
    "structural_cues",
    "anchor_alignments",
    "control_flow_cues",
    "data_dependency_cues",
]

FORBIDDEN_FIELDS = {
    "label",
    "confidence",
    "score",
    "similarity",
    "is_clone",
    "prediction",
    "rationale",
}

NON_TAG_WORDS = {
    "A",
    "B",
    "JSON",
    "S",
    "UAST",
    "AST",
    "ID1",
    "ID2",
}

STRUCTURAL_ROLE_TAGS = {
    "SEL",
    "LOOP",
    "CALL",
    "BLOCK",
    "COND",
    "THEN",
    "ELSE",
    "INIT",
    "STEP",
    "BODY",
    "ARGS",
    "CALLEE",
    "EMPTY",
    "EXPR",
    "VALUE",
}


@dataclass
class GuardResult:
    ok: bool
    errors: List[str]


@dataclass
class GuardConfig:
    allowed_tags: Optional[Set[str]] = None


def _extract_tag_tokens(obj: object) -> Set[str]:
    if isinstance(obj, str):
        return {t for t in re.findall(r"[A-Z_][A-Z0-9_]*", obj) if t not in NON_TAG_WORDS}
    if isinstance(obj, list):
        out: Set[str] = set()
        for item in obj:
            out.update(_extract_tag_tokens(item))
        return out
    if isinstance(obj, dict):
        out: Set[str] = set()
        for val in obj.values():
            out.update(_extract_tag_tokens(val))
        return out
    return set()


def validate_report(report: Dict[str, object], config: GuardConfig) -> GuardResult:
    errors: List[str] = []

    for field in REQUIRED_FIELDS:
        if field not in report:
            errors.append(f"missing field: {field}")
        elif not isinstance(report[field], list):
            errors.append(f"field not list: {field}")

    for forbidden in FORBIDDEN_FIELDS:
        if forbidden in report:
            errors.append(f"forbidden field: {forbidden}")

    if config.allowed_tags:
        allowed_tags = set(config.allowed_tags) | STRUCTURAL_ROLE_TAGS
        tags = _extract_tag_tokens(report)
        unknown = sorted({t for t in tags if t not in allowed_tags})
        if unknown:
            errors.append(f"unknown tags: {','.join(unknown[:20])}")

    return GuardResult(ok=not errors, errors=errors)


def repair_report(report: Dict[str, object]) -> Dict[str, object]:
    fixed = dict(report)
    for key in list(fixed.keys()):
        if key in FORBIDDEN_FIELDS:
            fixed.pop(key, None)
    for field in REQUIRED_FIELDS:
        if field not in fixed or not isinstance(fixed[field], list):
            fixed[field] = []
    for key in list(fixed.keys()):
        if key not in REQUIRED_FIELDS:
            fixed.pop(key, None)
    return fixed


def load_allowed_tags(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        tags = data.get("tags") or data.get("allowed_tags")
        if isinstance(tags, list):
            return {str(t) for t in tags} | STRUCTURAL_ROLE_TAGS
        values = [str(v) for v in data.values() if isinstance(v, str)]
        if values:
            return set(values) | STRUCTURAL_ROLE_TAGS
    if isinstance(data, list):
        return {str(t) for t in data} | STRUCTURAL_ROLE_TAGS
    return None

