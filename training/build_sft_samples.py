from __future__ import annotations

from typing import Dict


INSTRUCTION = "Determine whether the two code fragments form a clone pair."


def _first(record: Dict[str, object], *fields: str, default: object = "") -> object:
    for field in fields:
        value = record.get(field)
        if value is not None:
            return value
    return default


def _label(record: Dict[str, object]) -> int:
    value = _first(record, "label", "Label", "gold_label", default=0)
    try:
        label = int(value)
    except (TypeError, ValueError):
        label = 0
    return 1 if label == 1 else 0


def build_sft_sample(record: Dict[str, object]) -> Dict[str, object]:
    label = _label(record)
    language_a = _first(record, "language_a", "Category1")
    language_b = _first(record, "language_b", "Category2")
    pair_id = _first(record, "pair_id", default=f"{_first(record, 'ID1')}-{_first(record, 'ID2')}")

    return {
        "instruction": INSTRUCTION,
        "input": {
            "meta": {
                "pair_id": pair_id,
                "dataset": _first(record, "dataset", "Dataset"),
                "language_a": language_a,
                "language_b": language_b,
                "language_pair": record.get("language_pair") or f"{language_a}-{language_b}",
                "granularity": _first(record, "granularity", default="function"),
            },
            "code_a": _first(record, "source_code_a", "Code1", "code_a"),
            "code_b": _first(record, "source_code_b", "Code2", "code_b"),
            "suast_a": _first(record, "suast1", "suast_a"),
            "suast_b": _first(record, "suast2", "suast_b"),
            "evidence_report": record.get("evidence_report") or {},
        },
        "output": {
            "label": label,
            "confidence": float(label),
            "rationale": "Gold binary clone label from the supervised dataset.",
        },
    }
