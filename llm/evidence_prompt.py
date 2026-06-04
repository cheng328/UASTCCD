from __future__ import annotations

from typing import Dict


def build_evidence_prompt(payload: Dict[str, object]) -> str:
    language_a = payload.get("language_a", "")
    language_b = payload.get("language_b", "")
    granularity = payload.get("granularity", "")
    suast_a = payload.get("suast_a", "")
    suast_b = payload.get("suast_b", "")
    allowed_tags = payload.get("allowed_tags")
    tag_line = ""
    if isinstance(allowed_tags, (list, tuple, set)) and allowed_tags:
        tag_line = "Allowed UAST tags: " + ", ".join(sorted(str(t) for t in allowed_tags)) + "\n"

    return (
        "You are an evidence extractor for code clone detection.\n"
        "Return exactly one JSON object with these four list fields only: "
        "structural_cues, anchor_alignments, control_flow_cues, data_dependency_cues.\n"
        "Do not include label, confidence, score, similarity, is_clone, prediction, or rationale.\n"
        "Do not decide whether the pair is a clone. Only report structural evidence.\n"
        f"{tag_line}"
        f"Language A: {language_a}\n"
        f"Language B: {language_b}\n"
        f"Granularity: {granularity}\n"
        f"S-UAST A: {suast_a}\n"
        f"S-UAST B: {suast_b}\n"
    )

