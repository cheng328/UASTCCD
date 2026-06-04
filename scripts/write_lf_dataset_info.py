from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write/update a LLaMA-Factory dataset_info.json entry.")
    parser.add_argument("--output", required=True, help="dataset_info.json path")
    parser.add_argument("--dataset-name", default="uast_ccd_eval_sft")
    parser.add_argument("--file-name", default="uast_ccd_eval_sft.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        with output.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    else:
        data = {}

    data[args.dataset_name] = {
        "file_name": args.file_name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(output), "dataset_name": args.dataset_name, "file_name": args.file_name}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
