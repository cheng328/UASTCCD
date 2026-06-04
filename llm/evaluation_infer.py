from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Set

from .backend import ChatConfig, call_chat_json
from .evaluation_prompt import build_evaluation_prompt


@dataclass
class EvaluationConfig:
    mode: str = "heuristic"
    endpoint: str = "http://localhost:11434/api/chat"
    model: str = "gpt-oss:20b"
    timeout: float | None = None
    stream: bool = False


def _tag_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Z_][A-Z0-9_+<>=!&|/%^-]*", text or "")


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def heuristic_predict(payload: Dict[str, object]) -> Dict[str, object]:
    tags_a = set(_tag_tokens(str(payload.get("suast_a", ""))))
    tags_b = set(_tag_tokens(str(payload.get("suast_b", ""))))
    shared = tags_a & tags_b
    score = _jaccard(tags_a, tags_b)

    evidence = payload.get("evidence_report") or {}
    evidence_items = 0
    if isinstance(evidence, dict):
        for value in evidence.values():
            if isinstance(value, list):
                evidence_items += len(value)

    if evidence_items:
        score = min(1.0, score + 0.15)
    label = 1 if score >= 0.35 else 0
    rationale = f"shared structural tags: {','.join(sorted(shared)[:6])}" if shared else "limited shared structure"
    return {"label": label, "confidence": round(score, 4), "rationale": rationale}


def predict_clone(payload: Dict[str, object], config: EvaluationConfig) -> Dict[str, object]:
    if config.mode == "heuristic":
        return heuristic_predict(payload)
    if config.mode == "ollama":
        prompt = build_evaluation_prompt(payload)
        return call_chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict binary clone detector. Return only JSON with "
                        "label, confidence, and rationale."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            config=ChatConfig(
                endpoint=config.endpoint,
                model=config.model,
                timeout=config.timeout,
                stream=config.stream,
                format_json=True,
            ),
        )
    raise ValueError(f"Unsupported evaluation mode: {config.mode}")


def build_payload_from_record(record: Dict[str, object], variant: str = "uast_ccd") -> Dict[str, object]:
    code_a = record.get("source_code_a") or record.get("Code1") or record.get("code_a") or ""
    code_b = record.get("source_code_b") or record.get("Code2") or record.get("code_b") or ""
    raw_ast_a = record.get("ast1") if variant == "raw_ast" else None
    raw_ast_b = record.get("ast2") if variant == "raw_ast" else None
    suast_a = record.get("suast1") or record.get("suast_a") or ""
    suast_b = record.get("suast2") or record.get("suast_b") or ""
    evidence = record.get("evidence_report") if variant == "uast_ccd" else {}

    if variant in {"base_prompt", "task_prompt", "raw_ast"}:
        suast_a = ""
        suast_b = ""
        evidence = {}
    elif variant == "uast":
        evidence = {}

    return {
        "meta": {
            "pair_id": record.get("pair_id") or f"{record.get('ID1', '')}-{record.get('ID2', '')}",
            "dataset": record.get("dataset") or record.get("Dataset") or "",
            "language_a": record.get("language_a") or record.get("Category1") or "",
            "language_b": record.get("language_b") or record.get("Category2") or "",
            "variant": variant,
        },
        "code_a": code_a,
        "code_b": code_b,
        "raw_ast_a": json.dumps(raw_ast_a, ensure_ascii=False, separators=(",", ":")) if raw_ast_a else "",
        "raw_ast_b": json.dumps(raw_ast_b, ensure_ascii=False, separators=(",", ":")) if raw_ast_b else "",
        "suast_a": suast_a,
        "suast_b": suast_b,
        "evidence_report": evidence,
    }
