from __future__ import annotations

import json
from typing import Dict


def build_evaluation_prompt(payload: Dict[str, object]) -> str:
    meta = payload.get("meta") or {}
    variant = str(meta.get("variant", "uast_ccd"))
    parts = [
        "You are the Evaluation LLM for binary source code clone detection.",
        "Return exactly one JSON object with fields label, confidence, rationale.",
        "label must be 0 or 1. confidence must be a number in [0, 1].",
    ]
    if variant == "base_prompt":
        parts.append("Decide using only the two original code fragments.")
    elif variant == "task_prompt":
        parts.append("Decide using the task definition and the two original code fragments.")
    elif variant == "raw_ast":
        parts.append("Decide using the original code and the raw AST structures.")
    elif variant == "uast":
        parts.append("Decide using the original code and the S-UAST structures.")
    else:
        parts.append("Decide using the original code, S-UAST structures, and validated structural evidence.")

    parts.extend(
        [
            "",
            f"Meta: {json.dumps(meta, ensure_ascii=False)}",
            "",
            f"Code A:\n{payload.get('code_a', '')}",
            "",
            f"Code B:\n{payload.get('code_b', '')}",
        ]
    )
    if payload.get("raw_ast_a") or payload.get("raw_ast_b"):
        parts.extend(["", f"Raw AST A:\n{payload.get('raw_ast_a', '')}", "", f"Raw AST B:\n{payload.get('raw_ast_b', '')}"])
    if payload.get("suast_a") or payload.get("suast_b"):
        parts.extend(["", f"S-UAST A:\n{payload.get('suast_a', '')}", "", f"S-UAST B:\n{payload.get('suast_b', '')}"])
    if payload.get("evidence_report"):
        parts.extend(["", f"Evidence report:\n{json.dumps(payload.get('evidence_report') or {}, ensure_ascii=False)}"])
    return "\n".join(parts)
