from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from .evidence_prompt import build_evidence_prompt


@dataclass
class EvidenceConfig:
    mode: str = "heuristic"
    endpoint: str = "http://localhost:11434/api/chat"
    model: str = "qwen3:30b"
    timeout: float | None = None
    stream: bool = False


def _tokenize_tags(suast: str) -> List[str]:
    tokens = re.findall(r"[A-Z_][A-Z0-9_]*", suast)
    return tokens


def _top_shared_tags(suast_a: str, suast_b: str, k: int = 8) -> List[str]:
    a = _tokenize_tags(suast_a)
    b = _tokenize_tags(suast_b)
    shared = []
    seen: Set[str] = set()
    for t in a:
        if t in seen:
            continue
        if t in b:
            shared.append(t)
            seen.add(t)
        if len(shared) >= k:
            break
    return shared


def generate_evidence(payload: Dict[str, object], config: EvidenceConfig) -> Dict[str, object]:
    if config.mode == "ollama":
        from .backend import ChatConfig, call_chat_json

        prompt = build_evidence_prompt(payload)
        report = call_chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structural evidence for code clone detection. "
                        "Return only JSON. Do not output label, confidence, score, "
                        "similarity, is_clone, prediction, or rationale."
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
        return report

    if config.mode != "heuristic":
        raise ValueError(f"Unsupported evidence generation mode: {config.mode}")

    suast_a = str(payload.get("suast_a", ""))
    suast_b = str(payload.get("suast_b", ""))
    shared = _top_shared_tags(suast_a, suast_b)

    report = {
        "structural_cues": [f"shared_tags:{','.join(shared)}"] if shared else [],
        "anchor_alignments": shared,
        "control_flow_cues": [t for t in shared if t in {"LOOP", "COND", "BRANCH"}],
        "data_dependency_cues": [t for t in shared if t in {"ASSIGN", "RETURN", "CALL"}],
    }
    return report


def build_payload(
    language_a: str,
    language_b: str,
    granularity: str,
    suast_a: str,
    suast_b: str,
    meta: Dict[str, object],
    allowed_tags: object = None,
) -> Dict[str, object]:
    payload = {
        "language_a": language_a,
        "language_b": language_b,
        "granularity": granularity,
        "suast_a": suast_a,
        "suast_b": suast_b,
        "meta_information": meta,
    }
    if allowed_tags is not None:
        payload["allowed_tags"] = allowed_tags
    return payload


def serialize_report(report: Dict[str, object]) -> str:
    return json.dumps(report, ensure_ascii=False)


def get_prompt(payload: Dict[str, object]) -> str:
    return build_evidence_prompt(payload)

