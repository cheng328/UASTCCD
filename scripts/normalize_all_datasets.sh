#!/usr/bin/env bash
set -euo pipefail

RAW_DIR=${RAW_DIR:-src/data}
OUT_DIR=${OUT_DIR:-data/normalized}

mkdir -p "${OUT_DIR}"

python scripts/00_normalize_pairs.py \
  --dataset clcd \
  --input "${RAW_DIR}/clcd_pairs.jsonl" \
  --output "${OUT_DIR}/clcd_pairs.jsonl"

python scripts/00_normalize_pairs.py \
  --dataset googlejam4 \
  --input "${RAW_DIR}/googlejam4_pairs.json" \
  --output "${OUT_DIR}/googlejam4_pairs.jsonl"

python scripts/00_normalize_pairs.py \
  --dataset ojclone \
  --input "${RAW_DIR}/ojclone_pairs.json" \
  --output "${OUT_DIR}/ojclone_pairs.jsonl"

python scripts/00_normalize_pairs.py \
  --dataset bcb \
  --input "${RAW_DIR}/bcb_pairs.json" \
  --output "${OUT_DIR}/bcb_pairs.jsonl"

echo "Normalized datasets written to ${OUT_DIR}"
