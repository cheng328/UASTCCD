#!/usr/bin/env bash
set -euo pipefail

DATASETS=${DATASETS:-"clcd googlejam4 ojclone bcb"}
SPLITS=${SPLITS:-"train valid test"}
PAIRS_DIR=${PAIRS_DIR:-data/processed}
OUT_DIR=${OUT_DIR:-data/processed}
GRANULARITY=${GRANULARITY:-function}
MAX_RECORDS=${MAX_RECORDS:-0}

max_args=()
if [[ "${MAX_RECORDS}" != "0" ]]; then
  max_args=(--max-records "${MAX_RECORDS}")
fi

for dataset in ${DATASETS}; do
  for split in ${SPLITS}; do
    input="${PAIRS_DIR}/${dataset}/pairs/${dataset}_${split}_pairs.jsonl"
    ast_out="${OUT_DIR}/${dataset}/ast/${dataset}_${split}_ast.jsonl"
    pruned_out="${OUT_DIR}/${dataset}/uast/${dataset}_${split}_pruned.jsonl"
    parse_summary="${OUT_DIR}/${dataset}/ast/${dataset}_${split}_parse_summary.csv"
    parse_errors="${OUT_DIR}/${dataset}/ast/${dataset}_${split}_parse_errors.jsonl"
    prune_summary="${OUT_DIR}/${dataset}/uast/${dataset}_${split}_prune_summary.csv"

    python scripts/03_parse_ast.py \
      --input "${input}" \
      --output "${ast_out}" \
      "${max_args[@]}"

    python scripts/03_parse_summary.py \
      --input "${ast_out}" \
      --output "${parse_summary}" \
      --error-output "${parse_errors}"

    python scripts/04_prune_ast.py \
      --input "${ast_out}" \
      --output "${pruned_out}" \
      --granularity "${GRANULARITY}" \
      "${max_args[@]}"

    python scripts/04_prune_summary.py \
      --input "${pruned_out}" \
      --output "${prune_summary}"
  done
done

echo "AST and pruned UAST files written under ${OUT_DIR}/<dataset>/{ast,uast}"
