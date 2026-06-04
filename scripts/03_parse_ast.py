from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.parser.tree_sitter_parser import ParseOptions, TreeSitterParser
from src.parser.language_registry import normalize_language


DEFAULTS = {
    "code1": "source_code_a,Code1,code_a,code1",
    "code2": "source_code_b,Code2,code_b,code2",
    "lang1": "language_a,Category1,lang_a,lang1",
    "lang2": "language_b,Category2,lang_b,lang2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse code pairs into raw Tree-sitter ASTs (JSONL/JSON)."
    )
    parser.add_argument("--input", required=True, help="Input JSONL/JSON path")
    parser.add_argument("--output", required=True, help="Output JSONL/JSON path")
    parser.add_argument("--code1-field", default=DEFAULTS["code1"], help="Field name(s) for code1, comma-separated fallback order")
    parser.add_argument("--code2-field", default=DEFAULTS["code2"], help="Field name(s) for code2, comma-separated fallback order")
    parser.add_argument("--lang1-field", default=DEFAULTS["lang1"], help="Field name(s) for code1 language, comma-separated fallback order")
    parser.add_argument("--lang2-field", default=DEFAULTS["lang2"], help="Field name(s) for code2 language, comma-separated fallback order")
    parser.add_argument("--ast1-field", default="ast1", help="Output field name for AST1")
    parser.add_argument("--ast2-field", default="ast2", help="Output field name for AST2")
    parser.add_argument("--default-language", default="python", help="Fallback language")
    parser.add_argument("--include-text", action="store_true", help="Include node text")
    parser.add_argument("--max-records", type=int, default=0, help="Process only first N records")
    return parser.parse_args()


def get_first(record: Dict[str, object], fields: str, default: object = "") -> object:
    for field in [f.strip() for f in fields.split(",") if f.strip()]:
        value = record.get(field)
        if value is not None:
            return value
    return default


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


def process_record(
    record: Dict[str, object],
    args: argparse.Namespace,
    parser_cache: Dict[str, TreeSitterParser],
) -> Dict[str, object]:
    code1 = get_first(record, args.code1_field, "")
    code2 = get_first(record, args.code2_field, "")
    lang1 = normalize_language(str(get_first(record, args.lang1_field, args.default_language)))
    lang2 = normalize_language(str(get_first(record, args.lang2_field, args.default_language)))

    new_record = dict(record)
    parse_errors = []

    if isinstance(code1, str) and lang1:
        try:
            parser = parser_cache.setdefault(
                lang1, TreeSitterParser(lang1, ParseOptions(include_text=args.include_text))
            )
            new_record[args.ast1_field] = parser.parse_code(code1)
        except Exception as exc:
            new_record[args.ast1_field] = None
            parse_errors.append({"side": "a", "language": lang1, "error": str(exc)})
    else:
        new_record[args.ast1_field] = None
        parse_errors.append({"side": "a", "language": lang1, "error": "missing code or language"})

    if isinstance(code2, str) and lang2:
        try:
            parser = parser_cache.setdefault(
                lang2, TreeSitterParser(lang2, ParseOptions(include_text=args.include_text))
            )
            new_record[args.ast2_field] = parser.parse_code(code2)
        except Exception as exc:
            new_record[args.ast2_field] = None
            parse_errors.append({"side": "b", "language": lang2, "error": str(exc)})
    else:
        new_record[args.ast2_field] = None
        parse_errors.append({"side": "b", "language": lang2, "error": "missing code or language"})

    new_record["parse_ok"] = not parse_errors
    if parse_errors:
        new_record["parse_errors"] = parse_errors

    return new_record


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    parser_cache: Dict[str, TreeSitterParser] = {}

    if input_path.suffix.lower() == ".jsonl":
        rows = []
        for idx, record in enumerate(iter_jsonl(input_path)):
            if args.max_records and idx >= args.max_records:
                break
            rows.append(process_record(record, args, parser_cache))
        write_jsonl(output_path, rows)
        return 0

    data = load_json(input_path)
    if isinstance(data, list):
        out = []
        for idx, record in enumerate(data):
            if args.max_records and idx >= args.max_records:
                break
            if isinstance(record, dict):
                out.append(process_record(record, args, parser_cache))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        return 0

    raise SystemExit("Unsupported input JSON structure")


if __name__ == "__main__":
    raise SystemExit(main())

