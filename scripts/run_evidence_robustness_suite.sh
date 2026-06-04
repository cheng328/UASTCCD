#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-clcd}
SPLIT=${SPLIT:-test}
IN_DIR=${IN_DIR:-data/processed}
OUT_BASE=${OUT_BASE:-data/analysis/evidence_llm}
MAPPER_DIR=${MAPPER_DIR:?Set MAPPER_DIR to mapper directory containing category_tag_table.json}
EVIDENCE_MODELS=${EVIDENCE_MODELS:?Set EVIDENCE_MODELS to quoted model names separated by spaces}
EVAL_MODEL=${EVAL_MODEL:?Set EVAL_MODEL to the LoRA-adapted Evaluation LLM model}
ENDPOINT=${ENDPOINT:-http://localhost:11434/api/chat}
MODE=${MODE:-ollama}
TIMEOUT=${TIMEOUT:-}
MAX_RECORDS=${MAX_RECORDS:-0}
REPAIR_WITH_LLM=${REPAIR_WITH_LLM:-1}

slug() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[.:-]+/_/g; s/[^a-z0-9_]+/_/g; s/^_+|_+$//g'
}

common_llm_args=(--mode "${MODE}" --endpoint "${ENDPOINT}")
if [[ -n "${TIMEOUT}" ]]; then
  common_llm_args+=(--timeout "${TIMEOUT}")
fi
max_args=()
if [[ "${MAX_RECORDS}" != "0" ]]; then
  max_args=(--max-records "${MAX_RECORDS}")
fi

for evidence_model in ${EVIDENCE_MODELS}; do
  model_slug=$(slug "${evidence_model}")
  run_dir="${OUT_BASE}/${model_slug}"
  evidence="${run_dir}/${DATASET}_${SPLIT}_evidence.jsonl"
  quality_log="${run_dir}/quality_guard_log.jsonl"
  raw_pred="${run_dir}/${DATASET}_${SPLIT}_uast_ccd_raw.jsonl"
  norm_pred="${run_dir}/${DATASET}_${SPLIT}_uast_ccd_norm.jsonl"
  keeper_log="${run_dir}/standard_keeper_log.jsonl"

  evidence_args=("${common_llm_args[@]}" --model "${evidence_model}")
  if [[ "${REPAIR_WITH_LLM}" == "1" ]]; then
    evidence_args+=(--repair-with-llm)
  fi
  python scripts/09_generate_evidence.py \
    --input "${IN_DIR}/${DATASET}/suast/${DATASET}_${SPLIT}_suast.jsonl" \
    --output "${evidence}" \
    --allowed-tags "${MAPPER_DIR}/category_tag_table.json" \
    --quality-log "${quality_log}" \
    "${evidence_args[@]}" \
    "${max_args[@]}"

  python scripts/11_infer_variants.py \
    --input "${evidence}" \
    --output "${raw_pred}" \
    --variant uast_ccd \
    --method "UAST-CCD" \
    "${common_llm_args[@]}" \
    --model "${EVAL_MODEL}" \
    "${max_args[@]}"

  python scripts/10_apply_standard_keeper.py \
    --input "${raw_pred}" \
    --output "${norm_pred}" \
    --log "${keeper_log}"

  python scripts/12_eval_metrics.py \
    --input "${norm_pred}" \
    --output "${run_dir}/metrics.csv"
done

python scripts/15_evidence_llm_robustness.py \
  --output "${OUT_BASE}/table6_evidence_llm_robustness.csv" \
  --results-dir "${OUT_BASE}" \
  --models "$(echo "${EVIDENCE_MODELS}" | tr ' ' ',')"

echo "Evidence robustness outputs written under ${OUT_BASE}"
