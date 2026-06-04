from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.guard.quality_guard import GuardConfig, repair_report, validate_report, load_allowed_tags
from src.guard.repair_prompt import build_repair_prompt
from src.llm.evidence_generator import EvidenceConfig, build_payload, generate_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evidence reports with quality guard.")
    parser.add_argument("--input", required=True, help="Input JSONL/JSON path")
    parser.add_argument("--output", required=True, help="Output JSONL/JSON path")
    parser.add_argument("--suast1-field", default="suast1")
    parser.add_argument("--suast2-field", default="suast2")
    parser.add_argument("--lang1-field", default="language_a,Category1")
    parser.add_argument("--lang2-field", default="language_b,Category2")
    parser.add_argument("--granularity", default="function")
    parser.add_argument("--out-field", default="evidence_report")
    parser.add_argument("--allowed-tags", default="", help="Optional JSON list or dict with tags")
    parser.add_argument("--mode", default="heuristic", choices=["heuristic", "ollama"])
    parser.add_argument("--endpoint", default="http://localhost:11434/api/chat")
    parser.add_argument("--model", default="qwen3:30b")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--repair-with-llm", action="store_true", help="Use one constrained LLM repair before local repair")
    parser.add_argument("--quality-log", default="", help="Optional JSONL path for guard results")
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


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def get_first(record: Dict[str, object], fields: str, default: object = "") -> object:
    for field in [f.strip() for f in fields.split(",") if f.strip()]:
        value = record.get(field)
        if value is not None:
            return value
    return default


def process_record(record: Dict[str, object], args: argparse.Namespace, guard: GuardConfig) -> Dict[str, object]:
    suast_a = str(record.get(args.suast1_field, ""))
    suast_b = str(record.get(args.suast2_field, ""))
    lang_a = str(get_first(record, args.lang1_field, ""))
    lang_b = str(get_first(record, args.lang2_field, ""))

    allowed_tag_list = sorted(guard.allowed_tags) if guard.allowed_tags else None
    payload = build_payload(
        lang_a,
        lang_b,
        args.granularity,
        suast_a,
        suast_b,
        {},
        allowed_tags=allowed_tag_list,
    )
    report = generate_evidence(
        payload,
        EvidenceConfig(
            mode=args.mode,
            endpoint=args.endpoint,
            model=args.model,
            timeout=args.timeout,
            stream=args.stream,
        ),
    )

    initial_result = validate_report(report, guard)
    repair_attempted = False
    if not initial_result.ok:
        repair_attempted = True
        if args.repair_with_llm and args.mode == "ollama":
            try:
                from src.llm.backend import ChatConfig, call_chat_json

                report = call_chat_json(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Repair an evidence JSON object. Return only JSON with the four required list fields. "
                                "Do not include decision fields."
                            ),
                        },
                        {
                            "role": "user",
                            "content": build_repair_prompt(
                                json.dumps(report, ensure_ascii=False),
                                initial_result.errors,
                            ),
                        },
                    ],
                    config=ChatConfig(
                        endpoint=args.endpoint,
                        model=args.model,
                        timeout=args.timeout,
                        stream=args.stream,
                        format_json=True,
                    ),
                )
            except Exception:
                report = repair_report(report)
        else:
            report = repair_report(report)
        final_result = validate_report(report, guard)
    else:
        final_result = initial_result

    new_record = dict(record)
    new_record[args.out_field] = report
    new_record[args.out_field + "_ok"] = final_result.ok
    new_record[args.out_field + "_errors"] = final_result.errors
    new_record[args.out_field + "_guard"] = {
        "initial_ok": initial_result.ok,
        "initial_errors": initial_result.errors,
        "repair_attempted": repair_attempted,
        "repair_success": repair_attempted and final_result.ok,
        "final_ok": final_result.ok,
        "final_errors": final_result.errors,
        "mode": args.mode,
        "model": args.model if args.mode != "heuristic" else "",
    }
    return new_record


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    allowed_tags = load_allowed_tags(args.allowed_tags) if args.allowed_tags else None
    guard = GuardConfig(allowed_tags=allowed_tags)
    quality_rows = []

    if input_path.suffix.lower() == ".jsonl":
        rows = []
        for idx, record in enumerate(iter_jsonl(input_path)):
            if args.max_records and idx >= args.max_records:
                break
            out_record = process_record(record, args, guard)
            rows.append(out_record)
            quality_rows.append(
                {
                    "pair_id": out_record.get("pair_id", idx),
                    **out_record.get(args.out_field + "_guard", {}),
                }
            )
        write_jsonl(output_path, rows)
        if args.quality_log:
            write_jsonl(Path(args.quality_log), quality_rows)
        return 0

    data = load_json(input_path)
    if isinstance(data, list):
        out = []
        for idx, record in enumerate(data):
            if args.max_records and idx >= args.max_records:
                break
            if isinstance(record, dict):
                out_record = process_record(record, args, guard)
                out.append(out_record)
                quality_rows.append(
                    {
                        "pair_id": out_record.get("pair_id", idx),
                        **out_record.get(args.out_field + "_guard", {}),
                    }
                )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        if args.quality_log:
            write_jsonl(Path(args.quality_log), quality_rows)
        return 0

    raise SystemExit("Unsupported input JSON structure")


if __name__ == "__main__":
    raise SystemExit(main())

