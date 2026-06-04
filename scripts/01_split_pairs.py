from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split normalized pair JSONL without rebuilding pairs.")
    parser.add_argument("--input", required=True, help="Normalized pair JSONL/JSON")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--group-fields",
        default="",
        help="Comma-separated fields used as a pair-level split group. Empty means each pair can be split independently.",
    )
    parser.add_argument(
        "--stratify-fields",
        default="label,language_pair",
        help="Comma-separated fields used for stratified splitting.",
    )
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def iter_records(path: Path) -> Iterator[Dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
        return
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


def split_keys(keys: Sequence[Tuple[str, ...]], ratios: Tuple[float, float, float], seed: int) -> Dict[str, set]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    keys = list(keys)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    return {
        "train": set(keys[:n_train]),
        "valid": set(keys[n_train : n_train + n_valid]),
        "test": set(keys[n_train + n_valid :]),
    }


def split_keys_stratified(
    keys_by_stratum: Dict[Tuple[str, ...], Sequence[Tuple[str, ...]]],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, set]:
    out = {"train": set(), "valid": set(), "test": set()}
    for idx, keys in enumerate(keys_by_stratum.values()):
        sub = split_keys(list(keys), ratios, seed + idx)
        for split_name, split_keys_set in sub.items():
            out[split_name].update(split_keys_set)
    return out


def pair_group(record: Dict[str, object], fields: List[str], fallback_idx: int) -> Tuple[str, ...]:
    if not fields:
        return (str(fallback_idx),)
    values = []
    for field in fields:
        value = record.get(field)
        values.append(str(value if value is not None else ""))
    return tuple(values)


def stratum_key(record: Dict[str, object], fields: List[str]) -> Tuple[str, ...]:
    values = []
    for field in fields:
        value = record.get(field)
        if value is None and field == "label":
            value = record.get("Label")
        if value is None and field == "language_pair":
            value = record.get("PairType")
        values.append(str(value if value is not None else ""))
    return tuple(values)


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "records": len(rows),
        "labels": dict(Counter(row.get("label", row.get("Label")) for row in rows)),
        "language_pairs": dict(Counter(row.get("language_pair", row.get("PairType", "")) for row in rows)),
        "datasets": dict(Counter(row.get("dataset", "") for row in rows)),
    }


def main() -> int:
    args = parse_args()
    group_fields = [f.strip() for f in args.group_fields.split(",") if f.strip()]
    stratify_fields = [f.strip() for f in args.stratify_fields.split(",") if f.strip()]

    rows = []
    for idx, row in enumerate(iter_records(Path(args.input))):
        if args.max_records and idx >= args.max_records:
            break
        rows.append(row)

    by_group: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    group_to_stratum: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
    for idx, row in enumerate(rows):
        key = pair_group(row, group_fields, idx)
        by_group[key].append(row)
        group_to_stratum.setdefault(key, stratum_key(row, stratify_fields))

    keys_by_stratum: Dict[Tuple[str, ...], List[Tuple[str, ...]]] = defaultdict(list)
    for key, stratum in group_to_stratum.items():
        keys_by_stratum[stratum].append(key)

    split_group_sets = split_keys_stratified(
        keys_by_stratum,
        (args.train_ratio, args.valid_ratio, args.test_ratio),
        args.seed,
    )
    splits = {"train": [], "valid": [], "test": []}
    for split_name, key_set in split_group_sets.items():
        for key in key_set:
            splits[split_name].extend(by_group[key])

    out_dir = Path(args.out_dir)
    for split_name, split_rows in splits.items():
        write_jsonl(out_dir / f"{args.dataset}_{split_name}_pairs.jsonl", split_rows)

    summary = {
        "dataset": args.dataset,
        "input": args.input,
        "group_fields": group_fields,
        "stratify_fields": stratify_fields,
        "overall": summarize(rows),
        "splits": {name: summarize(split_rows) for name, split_rows in splits.items()},
    }
    summary_path = out_dir / f"{args.dataset}_pair_split_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
