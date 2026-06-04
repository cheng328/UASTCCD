#!/usr/bin/env bash
set -euo pipefail

DATASETS=${DATASETS:-"clcd googlejam4 ojclone bcb"}
IN_DIR=${IN_DIR:-data/processed}
OUT_DIR=${OUT_DIR:-data/processed}
MAPPER_DIR=${MAPPER_DIR:?Set MAPPER_DIR to a directory containing mapping_table.json and category_tag_table.json}
SPLITS=${SPLITS:-"train valid test"}
MAX_RECORDS=${MAX_RECORDS:-0}

max_args=()
if [[ "${MAX_RECORDS}" != "0" ]]; then
  max_args=(--max-records "${MAX_RECORDS}")
fi

for dataset in ${DATASETS}; do
  for split in ${SPLITS}; do
    python scripts/07_serialize_suast.py \
      --input "${IN_DIR}/${dataset}/uast/${dataset}_${split}_pruned.jsonl" \
      --output "${OUT_DIR}/${dataset}/suast/${dataset}_${split}_suast.jsonl" \
      --mapping "${MAPPER_DIR}/mapping_table.json" \
      --tag-table "${MAPPER_DIR}/category_tag_table.json" \
      "${max_args[@]}"
  done
done

echo "S-UAST files written under ${OUT_DIR}/<dataset>/suast"
