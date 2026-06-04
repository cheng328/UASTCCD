from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Evidence LLM Quality Guard logs.")
    parser.add_argument("--input", required=True, help="Quality log JSONL from 09_generate_evidence.py")
    parser.add_argument("--output", required=True, help="Output CSV summary")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "model",
        "total",
        "initial_ok",
        "initial_fail",
        "repair_attempted",
        "repair_success",
        "final_ok",
        "final_fail",
        "final_ok_rate",
        "top_errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    groups: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    errors: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    for row in iter_jsonl(Path(args.input)):
        key = (str(row.get("mode", "")), str(row.get("model", "")))
        groups[key]["total"] += 1
        for field in ("initial_ok", "repair_attempted", "repair_success", "final_ok"):
            if row.get(field) is True:
                groups[key][field] += 1
        for err in row.get("final_errors") or row.get("initial_errors") or []:
            errors[key][str(err)] += 1

    rows = []
    for (mode, model), counter in sorted(groups.items()):
        total = int(counter["total"])
        initial_ok = int(counter["initial_ok"])
        final_ok = int(counter["final_ok"])
        top_errors = "; ".join(f"{err}:{count}" for err, count in errors[(mode, model)].most_common(5))
        rows.append(
            {
                "mode": mode,
                "model": model,
                "total": total,
                "initial_ok": initial_ok,
                "initial_fail": total - initial_ok,
                "repair_attempted": int(counter["repair_attempted"]),
                "repair_success": int(counter["repair_success"]),
                "final_ok": final_ok,
                "final_fail": total - final_ok,
                "final_ok_rate": final_ok / total if total else 0.0,
                "top_errors": top_errors,
            }
        )

    write_csv(Path(args.output), rows)
    print(json.dumps({"output": args.output, "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
