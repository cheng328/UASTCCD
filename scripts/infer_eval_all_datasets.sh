#!/usr/bin/env bash
set -euo pipefail

DATASETS=${DATASETS:-"clcd googlejam4 ojclone bcb"}
IN_DIR=${IN_DIR:-data/processed}
OUT_DIR=${OUT_DIR:-data/results}
SPLIT=${SPLIT:-test}
VARIANTS=${VARIANTS:-"base_prompt task_prompt raw_ast uast uast_ccd"}
MODE=${MODE:-ollama}
ENDPOINT=${ENDPOINT:-http://localhost:11434/api/chat}
MODEL=${MODEL:?Set MODEL to the LoRA-adapted Evaluation LLM model}
TIMEOUT=${TIMEOUT:-}
MAX_RECORDS=${MAX_RECORDS:-0}

common_args=(--mode "${MODE}" --endpoint "${ENDPOINT}" --model "${MODEL}")
if [[ -n "${TIMEOUT}" ]]; then
  common_args+=(--timeout "${TIMEOUT}")
fi
if [[ "${MAX_RECORDS}" != "0" ]]; then
  common_args+=(--max-records "${MAX_RECORDS}")
fi

for dataset in ${DATASETS}; do
  for variant in ${VARIANTS}; do
    raw="${OUT_DIR}/predictions/${dataset}_${SPLIT}_${variant}_raw.jsonl"
    norm="${OUT_DIR}/predictions/${dataset}_${SPLIT}_${variant}_norm.jsonl"
    keeper_log="${OUT_DIR}/predictions/${dataset}_${SPLIT}_${variant}_standard_keeper.jsonl"
    metrics="${OUT_DIR}/metrics/${dataset}_${SPLIT}_${variant}.csv"

    python scripts/11_infer_variants.py \
      --input "${IN_DIR}/${dataset}/evidence/${dataset}_${SPLIT}_evidence.jsonl" \
      --output "${raw}" \
      --variant "${variant}" \
      "${common_args[@]}"

    python scripts/10_apply_standard_keeper.py \
      --input "${raw}" \
      --output "${norm}" \
      --log "${keeper_log}"

    python scripts/12_eval_metrics.py \
      --input "${norm}" \
      --output "${metrics}"
  done
done

python scripts/19_collect_metrics.py \
  --input-dir "${OUT_DIR}/metrics" \
  --output "${OUT_DIR}/metrics/all_metrics.csv"

echo "Prediction, Standard Keeper, and metric files written under ${OUT_DIR}"
