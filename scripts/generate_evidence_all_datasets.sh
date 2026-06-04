#!/usr/bin/env bash
set -euo pipefail

DATASETS=${DATASETS:-"clcd googlejam4 ojclone bcb"}
IN_DIR=${IN_DIR:-data/processed}
OUT_DIR=${OUT_DIR:-data/processed}
MAPPER_DIR=${MAPPER_DIR:?Set MAPPER_DIR to a mapper directory containing category_tag_table.json}
SPLITS=${SPLITS:-"train valid test"}
MODE=${MODE:-ollama}
ENDPOINT=${ENDPOINT:-http://localhost:11434/api/chat}
MODEL=${MODEL:?Set MODEL to the pre-LoRA Evidence LLM model}
TIMEOUT=${TIMEOUT:-}
MAX_RECORDS=${MAX_RECORDS:-0}
REPAIR_WITH_LLM=${REPAIR_WITH_LLM:-1}

common_args=(--mode "${MODE}" --endpoint "${ENDPOINT}" --model "${MODEL}")
if [[ -n "${TIMEOUT}" ]]; then
  common_args+=(--timeout "${TIMEOUT}")
fi
if [[ "${MAX_RECORDS}" != "0" ]]; then
  common_args+=(--max-records "${MAX_RECORDS}")
fi
if [[ "${REPAIR_WITH_LLM}" == "1" ]]; then
  common_args+=(--repair-with-llm)
fi

for dataset in ${DATASETS}; do
  for split in ${SPLITS}; do
    evidence_out="${OUT_DIR}/${dataset}/evidence/${dataset}_${split}_evidence.jsonl"
    quality_log="${OUT_DIR}/${dataset}/evidence/${dataset}_${split}_quality_guard.jsonl"
    quality_summary="${OUT_DIR}/${dataset}/evidence/${dataset}_${split}_quality_summary.csv"

    python scripts/09_generate_evidence.py \
      --input "${IN_DIR}/${dataset}/suast/${dataset}_${split}_suast.jsonl" \
      --output "${evidence_out}" \
      --allowed-tags "${MAPPER_DIR}/category_tag_table.json" \
      --quality-log "${quality_log}" \
      "${common_args[@]}"

    python scripts/09_evidence_quality_summary.py \
      --input "${quality_log}" \
      --output "${quality_summary}"
  done
done

echo "Evidence files and Quality Guard summaries written under ${OUT_DIR}/<dataset>/evidence"
