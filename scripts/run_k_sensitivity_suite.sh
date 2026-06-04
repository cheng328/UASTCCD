#!/usr/bin/env bash
set -euo pipefail

INPUT=${INPUT:?Set INPUT to the pruned AST pair JSONL}
OUT_BASE=${OUT_BASE:-data/analysis/k_sensitivity}
SEED_MAP=${SEED_MAP:-}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-64}
K_VALUES=${K_VALUES:-"16 32 64 96 128"}
TEMP_START=${TEMP_START:-1.0}
TEMP_END=${TEMP_END:-0.1}

if [[ -n "${SEED_MAP}" ]]; then
  SEED_ARGS=(--seed-map "${SEED_MAP}")
else
  SEED_ARGS=()
fi

for k in ${K_VALUES}; do
  out_dir="${OUT_BASE}/k${k}"
  python scripts/05_train_mapper.py \
    --input "${INPUT}" \
    --output-dir "${out_dir}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-categories "${k}" \
    --temperature-start "${TEMP_START}" \
    --temperature-end "${TEMP_END}" \
    "${SEED_ARGS[@]}"

  python scripts/06_discretize_mapper.py \
    --mapper "${out_dir}/mapper.pt" \
    --vocab "${out_dir}/node_type_vocab.json" \
    --output-dir "${out_dir}" \
    --num-categories "${k}"
done

echo "K sensitivity mapper artifacts written under ${OUT_BASE}"
