#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:-data/sft_samples/evaluation_lora_alpaca.jsonl}"
OUTPUT_DIR="${2:-checkpoints/evaluation_lora}"
BASE_MODEL="${BASE_MODEL:-gpt-oss-20b}"

echo "Training Evaluation LLM LoRA adapter"
echo "  base model: ${BASE_MODEL}"
echo "  data:       ${DATA_PATH}"
echo "  output:     ${OUTPUT_DIR}"

# This repository keeps the training launcher as a server-side hook because
# model paths, GPUs, and the exact LLaMA-Factory/TRL installation are cluster
# specific. Recommended settings from the paper draft:
#   LoRA rank: 16
#   LoRA alpha/scaling: 32
#   epochs: 3
#   learning rate: 2e-4
#   effective batch size: 16
#   base model frozen; train LoRA adapter only
#
# Example LLaMA-Factory command shape:
# llamafactory-cli train \
#   --stage sft \
#   --model_name_or_path "${BASE_MODEL}" \
#   --dataset_dir "$(dirname "${DATA_PATH}")" \
#   --dataset "$(basename "${DATA_PATH}")" \
#   --template default \
#   --finetuning_type lora \
#   --lora_rank 16 \
#   --lora_alpha 32 \
#   --learning_rate 2e-4 \
#   --num_train_epochs 3 \
#   --per_device_train_batch_size 1 \
#   --gradient_accumulation_steps 16 \
#   --output_dir "${OUTPUT_DIR}"

echo "Edit scripts/10_train_evaluation_lora.sh with the server's training command before running full LoRA training."
