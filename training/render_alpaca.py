from __future__ import annotations

import json
from typing import Dict


def render_alpaca_record(sample: Dict[str, object]) -> Dict[str, str]:
    return {
        "instruction": str(sample.get("instruction", "")),
        "input": json.dumps(sample.get("input") or {}, ensure_ascii=False),
        "output": json.dumps(sample.get("output") or {}, ensure_ascii=False),
    }
