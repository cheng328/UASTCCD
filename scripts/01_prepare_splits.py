import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_items_from_pairs
from src.data.split import split_by_group
from src.data.dataset_stats import summarize_items, summarize_splits


def parse_args():
    p = argparse.ArgumentParser(description="Prepare dataset splits by group")
    p.add_argument("--input", required=True, help="Path to JSON/JSONL pair file")
    p.add_argument("--dataset", required=True, help="Dataset name")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--format", default="jsonl", choices=["json", "jsonl"], help="Input format")
    p.add_argument("--granularity", default="function", help="Granularity label")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio")
    p.add_argument("--valid-ratio", type=float, default=0.1, help="Valid ratio")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio")
    p.add_argument("--group-from-path-regex", default=None, help="Regex to extract group id from file path")

    p.add_argument("--id1-field", default="ID1")
    p.add_argument("--id2-field", default="ID2")
    p.add_argument("--file1-field", default="file1")
    p.add_argument("--file2-field", default="file2")
    p.add_argument("--code1-field", default="Code1")
    p.add_argument("--code2-field", default="Code2")
    p.add_argument("--lang1-field", default="Category1")
    p.add_argument("--lang2-field", default="Category2")
    p.add_argument("--group-field", default="Task")
    p.add_argument("--group1-field", default=None)
    p.add_argument("--group2-field", default=None)
    p.add_argument("--summary", default=None, help="Optional output summary JSON path")
    p.add_argument("--allow-leakage", action="store_true", help="Do not fail when group leakage is detected")
    return p.parse_args()


def main():
    args = parse_args()
    field_map = {
        "id1": args.id1_field,
        "id2": args.id2_field,
        "file1": args.file1_field,
        "file2": args.file2_field,
        "code1": args.code1_field,
        "code2": args.code2_field,
        "lang1": args.lang1_field,
        "lang2": args.lang2_field,
        "group": args.group_field,
    }
    if args.group1_field:
        field_map["group1"] = args.group1_field
    if args.group2_field:
        field_map["group2"] = args.group2_field

    items = load_items_from_pairs(
        args.input,
        dataset=args.dataset,
        granularity=args.granularity,
        field_map=field_map,
        group_from_path_regex=args.group_from_path_regex,
    )

    splits = split_by_group(
        items,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_items in splits.items():
        out_path = out_dir / f"{args.dataset}_{split_name}_items.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in split_items:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    summary = {
        "dataset": args.dataset,
        "input": str(Path(args.input)),
        "overall": summarize_items(items),
        "splits": summarize_splits(splits),
    }
    summary_path = Path(args.summary) if args.summary else out_dir / f"{args.dataset}_split_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    if not args.allow_leakage and not summary["splits"]["leakage_ok"]:
        raise SystemExit("Group leakage detected across splits")


if __name__ == "__main__":
    main()

