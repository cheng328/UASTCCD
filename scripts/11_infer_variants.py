from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.llm.evaluation_infer import (
    EvaluationConfig,
    build_payload_from_record,
    predict_clone,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Evaluation LLM inference variants.")
    parser.add_argument("--input", required=True, help="Input JSONL/JSON with code, S-UAST, evidence")
    parser.add_argument("--output", required=True, help="Output JSONL/JSON path")
    parser.add_argument(
        "--variant",
        default="uast_ccd",
        choices=["base_prompt", "task_prompt", "raw_ast", "uast", "uast_ccd"],
    )
    parser.add_argument("--mode", default="heuristic", choices=["heuristic", "ollama"])
    parser.add_argument("--endpoint", default="http://localhost:11434/api/chat")
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--method", default="", help="Method name written to output")
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def iter_records(path: Path) -> Iterable[Dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        yield from iter_jsonl(path)
        return
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def process_record(record: Dict[str, object], args: argparse.Namespace, config: EvaluationConfig) -> Dict[str, object]:
    payload = build_payload_from_record(record, args.variant)
    prediction = predict_clone(payload, config)

    pair_id = record.get("pair_id") or f"{record.get('ID1', '')}-{record.get('ID2', '')}"
    language_a = record.get("language_a") or record.get("Category1") or ""
    language_b = record.get("language_b") or record.get("Category2") or ""
    method = args.method or {
        "base_prompt": "Base Prompt",
        "task_prompt": "+ Task Prompt",
        "raw_ast": "+ Raw AST",
        "uast": "+ UAST",
        "uast_ccd": "UAST-CCD",
    }[args.variant]

    out = dict(record)
    out["pair_id"] = pair_id
    out["prediction"] = prediction
    out["method"] = method
    out["language_pair"] = record.get("language_pair") or f"{language_a}-{language_b}"
    out["dataset"] = record.get("dataset") or record.get("Dataset") or ""
    return out


def main() -> int:
    args = parse_args()
    config = EvaluationConfig(
        mode=args.mode,
        endpoint=args.endpoint,
        model=args.model,
        timeout=args.timeout,
        stream=args.stream,
    )

    rows = []
    for idx, record in enumerate(iter_records(Path(args.input))):
        if args.max_records and idx >= args.max_records:
            break
        rows.append(process_record(record, args, config))

    write_jsonl(Path(args.output), rows)
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
