from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple


PRESETS = {
    "clcd": {
        "id1": "ID1",
        "id2": "ID2",
        "code1": "Code1",
        "code2": "Code2",
        "lang1": "Category1",
        "lang2": "Category2",
        "label": "Label",
        "group1": "Task1",
        "group2": "Task2",
    },
    "googlejam4": {
        "id1": "file1",
        "id2": "file2",
        "code1": "code1",
        "code2": "code2",
        "label": "label",
        "group": "index1",
        "ctype": "ctype",
        "language_a": "java",
        "language_b": "java",
    },
    "gcj": {
        "alias": "googlejam4",
    },
    "ojclone": {
        "id1": "file1",
        "id2": "file2",
        "code1": "code1",
        "code2": "code2",
        "label": "label",
        "group": "index1",
        "ctype": "ctype",
        "language_a": "c++",
        "language_b": "c++",
    },
    "bigclonebench": {
        "id1": "file1",
        "id2": "file2",
        "code1": "code1",
        "code2": "code2",
        "label": "label",
        "group": "index1",
        "ctype": "ctype",
        "language_a": "java",
        "language_b": "java",
    },
    "bcb": {
        "alias": "bigclonebench",
    },
}


LANG_NORMALIZE = {
    "py": "python",
    "python": "python",
    "java": "java",
    "c": "c",
    "cpp": "c++",
    "c++": "c++",
    "cc": "c++",
    "cxx": "c++",
    "cs": "c#",
    "c#": "c#",
}


def _preset(name: str) -> Dict[str, str]:
    key = name.lower()
    config = dict(PRESETS.get(key, {}))
    alias = config.get("alias")
    if alias:
        config = dict(PRESETS[alias])
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize raw clone-pair datasets into UAST-CCD pair JSONL.")
    parser.add_argument("--input", required=True, help="Raw JSON/JSONL pair file")
    parser.add_argument("--output", required=True, help="Output normalized JSONL")
    parser.add_argument("--dataset", required=True, help="Dataset name or preset: clcd, googlejam4/gcj, ojclone, bcb")
    parser.add_argument("--summary", default="", help="Optional summary JSON path")
    parser.add_argument("--granularity", default="function")
    parser.add_argument("--max-records", type=int, default=0)

    parser.add_argument("--id1-field", default="")
    parser.add_argument("--id2-field", default="")
    parser.add_argument("--code1-field", default="")
    parser.add_argument("--code2-field", default="")
    parser.add_argument("--lang1-field", default="")
    parser.add_argument("--lang2-field", default="")
    parser.add_argument("--label-field", default="")
    parser.add_argument("--group1-field", default="")
    parser.add_argument("--group2-field", default="")
    parser.add_argument("--group-field", default="")
    parser.add_argument("--ctype-field", default="")
    parser.add_argument("--language-a", default="")
    parser.add_argument("--language-b", default="")
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
    elif isinstance(data, dict):
        rows = data.get("data")
        if isinstance(rows, list):
            for item in rows:
                if isinstance(item, dict):
                    yield item
        else:
            yield data


def _get(record: Dict[str, object], field: Optional[str], default: object = "") -> object:
    if not field:
        return default
    value = record.get(field)
    return default if value is None else value


def normalize_label(value: object) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        text = str(value).strip().lower()
        return 1 if text in {"true", "clone", "positive", "pos", "yes"} else 0
    return 1 if numeric == 1 else 0


def normalize_language(value: object) -> str:
    text = str(value).strip().lower()
    return LANG_NORMALIZE.get(text, text)


def ctype_groups(ctype: object, fallback: object) -> Tuple[str, str]:
    text = str(ctype or "").strip()
    if "|" in text:
        left, right = text.split("|", 1)
        return left.strip(), right.strip()
    fallback_text = str(fallback or "").strip()
    return fallback_text, fallback_text


def normalize_record(
    record: Dict[str, object],
    idx: int,
    args: argparse.Namespace,
    config: Dict[str, str],
) -> Dict[str, object]:
    id1_field = args.id1_field or config.get("id1", "")
    id2_field = args.id2_field or config.get("id2", "")
    code1_field = args.code1_field or config.get("code1", "")
    code2_field = args.code2_field or config.get("code2", "")
    lang1_field = args.lang1_field or config.get("lang1", "")
    lang2_field = args.lang2_field or config.get("lang2", "")
    label_field = args.label_field or config.get("label", "")
    group1_field = args.group1_field or config.get("group1", "")
    group2_field = args.group2_field or config.get("group2", "")
    group_field = args.group_field or config.get("group", "")
    ctype_field = args.ctype_field or config.get("ctype", "")

    code_id_a = str(_get(record, id1_field, f"{args.dataset}_{idx}_a"))
    code_id_b = str(_get(record, id2_field, f"{args.dataset}_{idx}_b"))
    source_code_a = str(_get(record, code1_field, ""))
    source_code_b = str(_get(record, code2_field, ""))

    lang_a = args.language_a or config.get("language_a") or _get(record, lang1_field, "")
    lang_b = args.language_b or config.get("language_b") or _get(record, lang2_field, "")
    lang_a = normalize_language(lang_a)
    lang_b = normalize_language(lang_b)

    group_fallback = _get(record, group_field, "")
    ctype = _get(record, ctype_field, "")
    inferred_group_a, inferred_group_b = ctype_groups(ctype, group_fallback)
    group_a = str(_get(record, group1_field, inferred_group_a))
    group_b = str(_get(record, group2_field, inferred_group_b))
    if not group_a:
        group_a = code_id_a
    if not group_b:
        group_b = code_id_b

    label = normalize_label(_get(record, label_field, 0))
    pair_type = f"{lang_a}-{lang_b}"
    dataset = args.dataset.lower()

    out = {
        "pair_id": f"{dataset}-{idx}",
        "dataset": dataset,
        "code_id_a": code_id_a,
        "code_id_b": code_id_b,
        "language_a": lang_a,
        "language_b": lang_b,
        "language_pair": pair_type,
        "granularity": args.granularity,
        "group_a": group_a,
        "group_b": group_b,
        "source_code_a": source_code_a,
        "source_code_b": source_code_b,
        "label": label,
        "clone_type": str(ctype),
        "raw_label": _get(record, label_field, ""),
        "raw_index": _get(record, group_field, ""),
        "ID1": code_id_a,
        "ID2": code_id_b,
        "Category1": lang_a,
        "Category2": lang_b,
        "Code1": source_code_a,
        "Code2": source_code_b,
        "Task1": group_a,
        "Task2": group_b,
        "Task": group_a if group_a == group_b else "",
        "PairType": pair_type,
        "Label": label,
    }
    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def main() -> int:
    args = parse_args()
    config = _preset(args.dataset)
    rows = []
    for idx, record in enumerate(iter_records(Path(args.input)), start=1):
        if args.max_records and idx > args.max_records:
            break
        rows.append(normalize_record(record, idx, args, config))

    output_path = Path(args.output)
    write_jsonl(output_path, rows)

    summary = {
        "dataset": args.dataset.lower(),
        "input": args.input,
        "output": args.output,
        "records": len(rows),
        "labels": dict(Counter(row["label"] for row in rows)),
        "language_pairs": dict(Counter(row["language_pair"] for row in rows)),
        "empty_code_a": sum(not row["source_code_a"] for row in rows),
        "empty_code_b": sum(not row["source_code_b"] for row in rows),
    }
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
