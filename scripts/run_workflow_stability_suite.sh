#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-clcd}
SPLIT=${SPLIT:-test}
IN_DIR=${IN_DIR:-data/processed}
OUT_BASE=${OUT_BASE:-data/analysis/workflow_stability}
MAPPER_DIR=${MAPPER_DIR:?Set MAPPER_DIR to mapper directory containing category_tag_table.json}
EVIDENCE_MODEL=${EVIDENCE_MODEL:?Set EVIDENCE_MODEL to the pre-LoRA Evidence LLM model}
EVAL_MODEL=${EVAL_MODEL:?Set EVAL_MODEL to the LoRA-adapted Evaluation LLM model}
ENDPOINT=${ENDPOINT:-http://localhost:11434/api/chat}
MODE=${MODE:-ollama}
TIMEOUT=${TIMEOUT:-}
MAX_RECORDS=${MAX_RECORDS:-0}
REPEATS=${REPEATS:-3}

common_llm_args=(--mode "${MODE}" --endpoint "${ENDPOINT}")
if [[ -n "${TIMEOUT}" ]]; then
  common_llm_args+=(--timeout "${TIMEOUT}")
fi
max_args=()
if [[ "${MAX_RECORDS}" != "0" ]]; then
  max_args=(--max-records "${MAX_RECORDS}")
fi

run_full_once() {
  local wf_dir=$1
  local repair=$2
  local keeper=$3
  local rep=$4
  local evidence="${wf_dir}/evidence_${rep}.jsonl"
  local quality="${wf_dir}/quality_guard_log_${rep}.jsonl"
  local raw="${wf_dir}/predictions_raw_${rep}.jsonl"
  local norm="${wf_dir}/predictions_norm_${rep}.jsonl"
  local keeper_log="${wf_dir}/standard_keeper_log_${rep}.jsonl"

  evidence_args=("${common_llm_args[@]}" --model "${EVIDENCE_MODEL}")
  if [[ "${repair}" == "1" ]]; then
    evidence_args+=(--repair-with-llm)
  fi
  python scripts/09_generate_evidence.py \
    --input "${IN_DIR}/${DATASET}/suast/${DATASET}_${SPLIT}_suast.jsonl" \
    --output "${evidence}" \
    --allowed-tags "${MAPPER_DIR}/category_tag_table.json" \
    --quality-log "${quality}" \
    "${evidence_args[@]}" \
    "${max_args[@]}"

  python scripts/11_infer_variants.py \
    --input "${evidence}" \
    --output "${raw}" \
    --variant uast_ccd \
    --method "UAST-CCD" \
    "${common_llm_args[@]}" \
    --model "${EVAL_MODEL}" \
    "${max_args[@]}"

  if [[ "${keeper}" == "1" ]]; then
    python scripts/10_apply_standard_keeper.py \
      --input "${raw}" \
      --output "${norm}" \
      --log "${keeper_log}"
  else
    cp "${raw}" "${norm}"
    : > "${keeper_log}"
  fi
}

for rep in $(seq 1 "${REPEATS}"); do
  run_full_once "${OUT_BASE}/decoupled_without_quality_guard" 0 1 "${rep}"
  run_full_once "${OUT_BASE}/decoupled_without_standard_keeper" 1 0 "${rep}"
  run_full_once "${OUT_BASE}/full_uast_ccd" 1 1 "${rep}"
done

# Single-stage JSON is approximated by Evaluation LLM without evidence/guard stages.
single_dir="${OUT_BASE}/single_stage_json"
mkdir -p "${single_dir}"
for rep in $(seq 1 "${REPEATS}"); do
  raw="${single_dir}/predictions_raw_${rep}.jsonl"
  norm="${single_dir}/predictions_norm_${rep}.jsonl"
  keeper_log="${single_dir}/standard_keeper_log_${rep}.jsonl"
  python scripts/11_infer_variants.py \
    --input "${IN_DIR}/${DATASET}/suast/${DATASET}_${SPLIT}_suast.jsonl" \
    --output "${raw}" \
    --variant uast \
    --method "Single-stage JSON" \
    "${common_llm_args[@]}" \
    --model "${EVAL_MODEL}" \
    "${max_args[@]}"
  python scripts/10_apply_standard_keeper.py \
    --input "${raw}" \
    --output "${norm}" \
    --log "${keeper_log}"
  : > "${single_dir}/quality_guard_log_${rep}.jsonl"
done

for wf_dir in "${OUT_BASE}"/*; do
  [[ -d "${wf_dir}" ]] || continue
  cat "${wf_dir}"/quality_guard_log_*.jsonl > "${wf_dir}/quality_guard_log.jsonl" 2>/dev/null || true
  cat "${wf_dir}"/standard_keeper_log_*.jsonl > "${wf_dir}/standard_keeper_log.jsonl" 2>/dev/null || true
  cat "${wf_dir}"/predictions_norm_*.jsonl > "${wf_dir}/predictions_normalized.jsonl" 2>/dev/null || true
done

python scripts/16_workflow_stability.py \
  --output "${OUT_BASE}/table8_workflow_stability.csv" \
  --logs-dir "${OUT_BASE}"

echo "Workflow stability outputs written under ${OUT_BASE}"
