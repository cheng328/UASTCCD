from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.llm.backend import extract_json_object


@dataclass
class StandardResult:
    ok: bool
    normalized: Dict[str, object]
    errors: List[str]


def normalize_output(output: Dict[str, object]) -> StandardResult:
    errors: List[str] = []

    label = output.get("label")
    if isinstance(label, bool):
        label = int(label)
    if label not in (0, 1):
        errors.append("label must be 0 or 1")
        label = 0

    confidence = output.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = float(label)
        errors.append("confidence not numeric")

    confidence = min(max(confidence, 0.0), 1.0)

    rationale = output.get("rationale")
    if not isinstance(rationale, str):
        rationale = str(rationale) if rationale is not None else ""
        errors.append("rationale not string")

    normalized = {"label": int(label), "confidence": confidence, "rationale": rationale}
    return StandardResult(ok=not errors, normalized=normalized, errors=errors)


def parse_prediction(value: object) -> StandardResult:
    if isinstance(value, dict):
        return normalize_output(value)
    if isinstance(value, str):
        try:
            obj = json.loads(value)
        except json.JSONDecodeError:
            obj = extract_json_object(value)
        if isinstance(obj, dict):
            result = normalize_output(obj)
            if result.ok:
                return result
            return StandardResult(
                ok=False,
                normalized=result.normalized,
                errors=["parsed json with normalization errors", *result.errors],
            )
        return StandardResult(
            ok=False,
            normalized={"label": 0, "confidence": 0.0, "rationale": ""},
            errors=["prediction is not parseable JSON"],
        )
    return StandardResult(
        ok=False,
        normalized={"label": 0, "confidence": 0.0, "rationale": ""},
        errors=["prediction is neither dict nor string"],
    )


def load_json_text(text: str) -> Dict[str, object]:
    obj = extract_json_object(text)
    if obj is not None:
        return obj
    return json.loads(text)

