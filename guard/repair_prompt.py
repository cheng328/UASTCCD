from __future__ import annotations

from typing import Dict


def build_repair_prompt(report_text: str, error_messages: list[str]) -> str:
    errors = "\n".join(f"- {e}" for e in error_messages)
    return (
        "Your previous JSON report is invalid. Fix it and return only JSON.\n"
        "Errors:\n"
        f"{errors}\n\n"
        "Report:\n"
        f"{report_text}\n"
    )

