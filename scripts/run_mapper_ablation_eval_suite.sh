#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-clcd}
SPLIT=${SPLIT:-test}
IN_DIR=${IN_DIR:-data/processed}
OUT_BASE=${OUT_BASE:-data/analysis/mapper_ablation}
DIAG_DIR=${DIAG_DIR:-${OUT_BASE}/diagnostics}
METRICS_DIR=${METRICS_DIR:-${OUT_BASE}/metrics}
ENDPOINT=${ENDPOINT:-http://localhost:11434/api/chat}
EVIDENCE_MODEL=${EVIDENCE_MODEL:?Set EVIDENCE_MODEL to the pre-LoRA Evidence LLM model}
EVAL_MODEL=${EVAL_MODEL:?Set EVAL_MODEL to the LoRA-adapted Evaluation LLM model}
MODE=${MODE:-ollama}
TIMEOUT=${TIMEOUT:-}
MAX_RECORDS=${MAX_RECORDS:-0}
REPAIR_WITH_LLM=${REPAIR_WITH_LLM:-1}

VARIANTS=${VARIANTS:-"full_mapper without_lsharp without_lbalance without_lalign without_seed_dict random_mapper"}

common_llm_args=(--mode "${MODE}" --endpoint "${ENDPOINT}")
if [[ -n "${TIMEOUT}" ]]; then
  common_llm_args+=(--timeout "${TIMEOUT}")
fi
max_args=()
if [[ "${MAX_RECORDS}" != "0" ]]; then
  max_args=(--max-records "${MAX_RECORDS}")
fi

mkdir -p "${METRICS_DIR}"

for variant in ${VARIANTS}; do
  mapper_dir="${DIAG_DIR}/${variant}"
  run_dir="${OUT_BASE}/runs/${variant}"
  suast="${run_dir}/${DATASET}_${SPLIT}_suast.jsonl"
  evidence="${run_dir}/${DATASET}_${SPLIT}_evidence.jsonl"
  quality_log="${run_dir}/${DATASET}_${SPLIT}_quality_guard.jsonl"
  raw_pred="${run_dir}/${DATASET}_${SPLIT}_uast_ccd_raw.jsonl"
  norm_pred="${run_dir}/${DATASET}_${SPLIT}_uast_ccd_norm.jsonl"
  keeper_log="${run_dir}/${DATASET}_${SPLIT}_standard_keeper.jsonl"

  python scripts/07_serialize_suast.py \
    --input "${IN_DIR}/${DATASET}/uast/${DATASET}_${SPLIT}_pruned.jsonl" \
    --output "${suast}" \
    --mapping "${mapper_dir}/mapping_table.json" \
    --tag-table "${mapper_dir}/category_tag_table.json" \
    "${max_args[@]}"

  evidence_args=("${common_llm_args[@]}" --model "${EVIDENCE_MODEL}")
  if [[ "${REPAIR_WITH_LLM}" == "1" ]]; then
    evidence_args+=(--repair-with-llm)
  fi
  python scripts/09_generate_evidence.py \
    --input "${suast}" \
    --output "${evidence}" \
    --allowed-tags "${mapper_dir}/category_tag_table.json" \
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
    --output "${METRICS_DIR}/${variant}.csv"
done

python scripts/13_mapper_ablation.py \
  --output "${OUT_BASE}/table7_mapper_ablation.csv" \
  --metrics-dir "${METRICS_DIR}" \
  --diagnostics-dir "${DIAG_DIR}"

echo "Mapper ablation closed-loop outputs written under ${OUT_BASE}"
