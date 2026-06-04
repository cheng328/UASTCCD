from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.build_sft_samples import build_sft_sample
from src.training.render_alpaca import render_alpaca_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Evaluation LLM SFT samples.")
    parser.add_argument("--input", required=True, help="Input JSONL/JSON with code, S-UAST, evidence, labels")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--format", default="full", choices=["full", "alpaca"])
    parser.add_argument("--require-valid-evidence", action="store_true")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--dataset-name",
        default="uast_ccd_eval_sft",
        help="Dataset name to register in LLaMA-Factory dataset_info.json",
    )
    parser.add_argument(
        "--llamafactory-dataset-info",
        default="",
        help="Optional path to write/update a LLaMA-Factory dataset_info.json entry",
    )
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


def write_llamafactory_dataset_info(path: Path, dataset_name: str, data_file: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, dict):
            existing = {}
    else:
        existing = {}

    existing[dataset_name] = {
        "file_name": data_file.name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    rows = []
    skipped = 0
    for idx, record in enumerate(iter_records(Path(args.input))):
        if args.max_records and idx >= args.max_records:
            break
        if args.require_valid_evidence and record.get("evidence_report_ok") is False:
            skipped += 1
            continue
        sample = build_sft_sample(record)
        if args.format == "alpaca":
            sample = render_alpaca_record(sample)
        rows.append(sample)

    output_path = Path(args.output)
    write_jsonl(output_path, rows)
    dataset_info_path = ""
    if args.llamafactory_dataset_info:
        write_llamafactory_dataset_info(Path(args.llamafactory_dataset_info), args.dataset_name, output_path)
        dataset_info_path = args.llamafactory_dataset_info
    print(
        json.dumps(
            {
                "written": len(rows),
                "skipped": skipped,
                "output": args.output,
                "dataset_name": args.dataset_name if dataset_info_path else "",
                "llamafactory_dataset_info": dataset_info_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
