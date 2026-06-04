#!/usr/bin/env bash
set -euo pipefail

DATASETS=${DATASETS:-"clcd googlejam4 ojclone bcb"}
IN_DIR=${IN_DIR:-data/processed}
OUT_DIR=${OUT_DIR:-data/llamafactory}
SPLIT=${SPLIT:-train}
DATASET_NAME=${DATASET_NAME:-uast_ccd_eval_sft}
FORMAT=${FORMAT:-alpaca}
MAX_RECORDS=${MAX_RECORDS:-0}

mkdir -p "${OUT_DIR}"
merged="${OUT_DIR}/${DATASET_NAME}.jsonl"
: > "${merged}"

max_args=()
if [[ "${MAX_RECORDS}" != "0" ]]; then
  max_args=(--max-records "${MAX_RECORDS}")
fi

for dataset in ${DATASETS}; do
  tmp="${OUT_DIR}/${DATASET_NAME}_${dataset}_${SPLIT}.jsonl"
  python scripts/09_build_sft_data.py \
    --input "${IN_DIR}/${dataset}/evidence/${dataset}_${SPLIT}_evidence.jsonl" \
    --output "${tmp}" \
    --format "${FORMAT}" \
    --require-valid-evidence \
    "${max_args[@]}"
  cat "${tmp}" >> "${merged}"
done

python scripts/write_lf_dataset_info.py \
  --output "${OUT_DIR}/dataset_info.json" \
  --dataset-name "${DATASET_NAME}" \
  --file-name "${DATASET_NAME}.jsonl"

echo "Merged LLaMA-Factory SFT data: ${merged}"
echo "Dataset name: ${DATASET_NAME}"
echo "Dataset info: ${OUT_DIR}/dataset_info.json"
