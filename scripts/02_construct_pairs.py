import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import CodeItem
from src.data.pair_builder import build_pairs
from src.data.dataset_stats import summarize_pairs


def parse_args():
    p = argparse.ArgumentParser(description="Construct pairs from split items")
    p.add_argument("--input", required=True, help="Path to *_items.jsonl")
    p.add_argument("--output", required=True, help="Output pairs JSONL")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos-neg-ratio", type=float, default=1.0)
    p.add_argument("--pair-types", default=None, help="Comma-separated pairs like java-python,java-c++")
    p.add_argument("--allow-cross-language", action="store_true", help="Allow cross-language pairing")
    p.add_argument("--summary", default=None, help="Optional output summary JSON path")
    p.add_argument("--allow-integrity-errors", action="store_true", help="Do not fail on invalid pair integrity checks")
    p.add_argument("--require-ratio", action="store_true", help="Fail if the requested negative/positive ratio is not reached")
    return p.parse_args()


def load_items(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            items.append(CodeItem(**obj))
    return items


def main():
    args = parse_args()
    items = load_items(Path(args.input))

    pair_types = None
    if args.pair_types:
        pair_types = []
        for token in args.pair_types.split(","):
            a, b = token.split("-")
            pair_types.append((a, b))

    pairs = build_pairs(
        items,
        pair_types=pair_types,
        pos_neg_ratio=args.pos_neg_ratio,
        seed=args.seed,
        allow_cross_language=args.allow_cross_language,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    pair_summary = summarize_pairs(pairs)
    actual_ratio = float(pair_summary.get("negative_positive_ratio", 0.0))
    ratio_tolerance = 0.05
    ratio_ok = (
        pair_summary.get("labels", {}).get(1, 0) == 0
        or abs(actual_ratio - args.pos_neg_ratio) <= ratio_tolerance
    )
    summary = {
        "input": str(Path(args.input)),
        "output": str(out_path),
        "items": len(items),
        "requested_negative_positive_ratio": args.pos_neg_ratio,
        "ratio_tolerance": ratio_tolerance,
        "ratio_ok": ratio_ok,
        **pair_summary,
    }
    summary_path = Path(args.summary) if args.summary else out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    if not args.allow_integrity_errors and not summary["pair_integrity_ok"]:
        raise SystemExit("Pair integrity checks failed")
    if args.require_ratio and not summary["ratio_ok"]:
        raise SystemExit("Requested negative/positive ratio was not reached")


if __name__ == "__main__":
    main()

