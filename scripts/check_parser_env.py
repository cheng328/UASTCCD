from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser.language_registry import LANGUAGE_MODULES, get_parser


def main() -> int:
    results = {}
    ok = True
    for language, module_name in sorted(LANGUAGE_MODULES.items()):
        try:
            get_parser(language)
            results[language] = {"ok": True, "module": module_name}
        except Exception as exc:
            ok = False
            results[language] = {"ok": False, "module": module_name, "error": str(exc)}

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
