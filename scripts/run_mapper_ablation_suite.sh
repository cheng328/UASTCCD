#!/usr/bin/env bash
set -euo pipefail

INPUT=${INPUT:?Set INPUT to the pruned AST pair JSONL}
OUT_BASE=${OUT_BASE:-data/analysis/mapper_ablation}
SEED_MAP=${SEED_MAP:-}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-64}
K=${K:-64}
TEMP_START=${TEMP_START:-1.0}
TEMP_END=${TEMP_END:-0.1}

train_variant() {
  local slug=$1
  shift
  local out_dir="${OUT_BASE}/diagnostics/${slug}"

  python scripts/05_train_mapper.py \
    --input "${INPUT}" \
    --output-dir "${out_dir}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-categories "${K}" \
    --temperature-start "${TEMP_START}" \
    --temperature-end "${TEMP_END}" \
    "$@"

  python scripts/06_discretize_mapper.py \
    --mapper "${out_dir}/mapper.pt" \
    --vocab "${out_dir}/node_type_vocab.json" \
    --output-dir "${out_dir}" \
    --num-categories "${K}"
}

if [[ -n "${SEED_MAP}" ]]; then
  SEED_ARGS=(--seed-map "${SEED_MAP}")
else
  SEED_ARGS=()
fi

train_variant full_mapper "${SEED_ARGS[@]}"
train_variant without_lsharp --lambda-sharp 0 "${SEED_ARGS[@]}"
train_variant without_lbalance --lambda-balance 0 "${SEED_ARGS[@]}"
train_variant without_lalign --lambda-align 0 "${SEED_ARGS[@]}"
train_variant without_seed_dict

python scripts/create_random_mapper.py \
  --vocab "${OUT_BASE}/diagnostics/full_mapper/node_type_vocab.json" \
  --output-dir "${OUT_BASE}/diagnostics/random_mapper" \
  --num-categories "${K}"

echo "Mapper ablation mapper artifacts written under ${OUT_BASE}/diagnostics"
