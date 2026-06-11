#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}"
REF_MODEL_NAME_OR_PATH="${REF_MODEL_NAME_OR_PATH:-${MODEL_NAME_OR_PATH}}"
TRAIN_FILE="${TRAIN_FILE:-../TrajRL/dataset/trex_renlg/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-../TrajRL/dataset/trex_renlg/dev.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage2_search_grpo/$(date +%Y%m%d_%H%M%S)}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8090}"
SEARCH_URL="${SEARCH_URL:-http://localhost:${RETRIEVER_PORT}}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"

ACCELERATE_ARGS=(
  --num_processes "${NUM_PROCESSES}"
  --mixed_precision "${MIXED_PRECISION}"
)

if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  ACCELERATE_ARGS+=(--use_deepspeed --deepspeed_config_file "${DEEPSPEED_CONFIG}")
fi

accelerate launch "${ACCELERATE_ARGS[@]}" \
  train_search_policy_grpo.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --ref_model_name_or_path "${REF_MODEL_NAME_OR_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --search_url "${SEARCH_URL}" \
  "$@"
