#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"

# DeepSpeed NVTX instrumentation can crash with the installed nvtx package in this environment.
# The Python entrypoint also disables it defensively.
DEEPSPEED_ENABLE_NVTX="${DEEPSPEED_ENABLE_NVTX:-0}"
export DEEPSPEED_ENABLE_NVTX

# Reduce peak memory for long Qwen2.5 sequences and keep allocations less fragmented.
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}"
REF_MODEL_NAME_OR_PATH="${REF_MODEL_NAME_OR_PATH:-${MODEL_NAME_OR_PATH}}"
TRAIN_FILE="${TRAIN_FILE:-data/trex-grpo-train.jsonl}"
EVAL_FILE="${EVAL_FILE:-data/trex-dev.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage2_search_grpo/$(date +%Y%m%d_%H%M%S)}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8090}"
SEARCH_URL="${SEARCH_URL:-http://localhost:${RETRIEVER_PORT}}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
MAX_TURN_NEW_TOKENS="${MAX_TURN_NEW_TOKENS:-128}"
FORCE_FINAL_NEW_TOKENS="${FORCE_FINAL_NEW_TOKENS:-48}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

ACCELERATE_ARGS=(
  --num_processes "${NUM_PROCESSES}"
  --mixed_precision "${MIXED_PRECISION}"
)

TRAIN_ARGS=(
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --logprob_micro_batch_size "${LOGPROB_MICRO_BATCH_SIZE}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --max_turn_new_tokens "${MAX_TURN_NEW_TOKENS}"
  --force_final_new_tokens "${FORCE_FINAL_NEW_TOKENS}"
)

if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  TRAIN_ARGS+=(--gradient_checkpointing)
fi

if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  ACCELERATE_ARGS+=(--use_deepspeed --deepspeed_config_file "${DEEPSPEED_CONFIG}")
fi

accelerate launch "${ACCELERATE_ARGS[@]}" \
  train_search_policy_grpo.py \
  "${TRAIN_ARGS[@]}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --ref_model_name_or_path "${REF_MODEL_NAME_OR_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --search_url "${SEARCH_URL}" \
  "$@"
