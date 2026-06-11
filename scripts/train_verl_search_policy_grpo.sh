#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}"

count_gpu_devices() {
  local devices="$1"
  if [[ -z "${devices}" ]]; then
    echo ""
    return
  fi
  local normalized="${devices// /}"
  if [[ -z "${normalized}" ]]; then
    echo ""
    return
  fi
  local old_ifs="${IFS}"
  IFS=,
  read -r -a parts <<< "${normalized}"
  IFS="${old_ifs}"
  echo "${#parts[@]}"
}

usage() {
  cat <<'EOF'
Usage: scripts/train_verl_search_policy_grpo.sh [options]

Common options:
  --gpu-devices 0,1,2,3       Set CUDA_VISIBLE_DEVICES and infer GPU count.
  --num-gpus 4                Set trainer.n_gpus_per_node.
  --rollout-tp 1              Set vLLM rollout tensor parallel size.
  --train-files PATH          VERL train parquet path.
  --test-files PATH           VERL validation parquet path.
  --actor-model-path PATH     Base actor model.
  --lora-adapter-path PATH    Trainable LoRA adapter to continue with.
  --save-path PATH            Output directory.
  --search-url URL            SearchPolicyGRPO BGE retriever URL.
EOF
}

TRAIN_FILES="${TRAIN_FILES:-outputs/verl_data/search_policy/train.parquet}"
TEST_FILES="${TEST_FILES:-outputs/verl_data/search_policy/val.parquet}"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
SAVE_PATH="${SAVE_PATH:-outputs/verl_search_policy_grpo/$(date +%Y%m%d_%H%M%S)}"
SEARCH_URL="${SEARCH_URL:-http://localhost:8090}"
PROJECT_NAME="${PROJECT_NAME:-SearchPolicyGRPO}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-verl-search-policy-grpo}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.55}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-128}"
ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}"
GPU_DEVICES="${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
NUM_GPUS="${NUM_GPUS:-}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-}"
NNODES="${NNODES:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
LR="${LR:-1e-6}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-False}"
DISABLE_TORCHDYNAMO="${DISABLE_TORCHDYNAMO:-1}"
ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-50}"
ACTOR_FORWARD_DTYPE="${ACTOR_FORWARD_DTYPE:-bf16}"
FSDP_PARAM_DTYPE="${FSDP_PARAM_DTYPE:-bf16}"
FSDP_REDUCE_DTYPE="${FSDP_REDUCE_DTYPE:-fp32}"
FSDP_BUFFER_DTYPE="${FSDP_BUFFER_DTYPE:-fp32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-768}"
MAX_TURNS="${MAX_TURNS:-4}"
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm_with_tool}"
REWARD_MANAGER="${REWARD_MANAGER:-naive}"
LOGGER="${LOGGER:-console,wandb}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-/workspace/sft}"
TOP_N="${TOP_N:-5}"
MAX_RESULT_CHARS="${MAX_RESULT_CHARS:-2000}"
RESULT_SUMMARY_MODE="${RESULT_SUMMARY_MODE:-llm}"
RESULT_SUMMARY_CHARS="${RESULT_SUMMARY_CHARS:-480}"
RESULT_SUMMARY_NEW_TOKENS="${RESULT_SUMMARY_NEW_TOKENS:-128}"
SEARCH_MAX_WORKERS="${SEARCH_MAX_WORKERS:-32}"
GPU_PARALLEL_OVERRIDE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-devices)
      GPU_DEVICES="$2"
      GPU_PARALLEL_OVERRIDE=1
      shift 2
      ;;
    --num-gpus|--n-gpus-per-node)
      NUM_GPUS="$2"
      GPU_PARALLEL_OVERRIDE=1
      shift 2
      ;;
    --rollout-tp)
      ROLLOUT_TP="$2"
      shift 2
      ;;
    --train-files)
      TRAIN_FILES="$2"
      shift 2
      ;;
    --test-files)
      TEST_FILES="$2"
      shift 2
      ;;
    --actor-model-path)
      ACTOR_MODEL_PATH="$2"
      shift 2
      ;;
    --save-path)
      SAVE_PATH="$2"
      shift 2
      ;;
    --lora-adapter-path)
      LORA_ADAPTER_PATH="$2"
      shift 2
      ;;
    --search-url)
      SEARCH_URL="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument '$1'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "${GPU_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
fi
if [[ -z "${NUM_GPUS}" ]]; then
  NUM_GPUS="$(count_gpu_devices "${GPU_DEVICES}")"
fi
if [[ "${GPU_PARALLEL_OVERRIDE}" == "1" || -z "${N_GPUS_PER_NODE}" ]]; then
  N_GPUS_PER_NODE="${NUM_GPUS:-1}"
fi

MAX_MODEL_LEN="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
case "${ROLLOUT_ENABLE_CHUNKED_PREFILL}" in
  1|true|True|TRUE|yes|Yes|YES|on|On|ON)
    if (( ROLLOUT_MAX_NUM_BATCHED_TOKENS < MAX_MODEL_LEN )); then
      echo "[train_verl] ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} is smaller than max_model_len=${MAX_MODEL_LEN}; increasing to ${MAX_MODEL_LEN}"
      ROLLOUT_MAX_NUM_BATCHED_TOKENS="${MAX_MODEL_LEN}"
    fi
    ;;
esac

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${DISABLE_TORCHDYNAMO}" in
  1|true|True|TRUE|yes|Yes|YES|on|On|ON)
    export TORCHDYNAMO_DISABLE=1
    ;;
  *)
    unset TORCHDYNAMO_DISABLE
    ;;
esac

if [[ -n "${LORA_ADAPTER_PATH}" && "${LORA_ADAPTER_PATH}" != "None" && "${LORA_ADAPTER_PATH}" != "null" && ! -f "${LORA_ADAPTER_PATH}/adapter_config.json" ]]; then
  echo "[train_verl] LoRA adapter_config.json not found under LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH}" >&2
  exit 1
fi

mkdir -p "${SAVE_PATH}/rollout" "${SAVE_PATH}/checkpoint"

if [[ -n "${WANDB_API_KEY}" && "${WANDB_API_KEY}" != "None" ]]; then
  wandb login --relogin "${WANDB_API_KEY}"
  export WANDB_DIR="${SAVE_PATH}"
fi

echo "[train_verl] PYTHONPATH=${PYTHONPATH}"
echo "[train_verl] train=${TRAIN_FILES}"
echo "[train_verl] val=${TEST_FILES}"
echo "[train_verl] search_url=${SEARCH_URL}"
echo "[train_verl] actor_model=${ACTOR_MODEL_PATH}"
echo "[train_verl] lora_adapter=${LORA_ADAPTER_PATH}"
echo "[train_verl] rollout=${ROLLOUT_NAME} n=${ROLLOUT_N} turns=${MAX_TURNS} gpu_util=${ROLLOUT_GPU_UTIL}"
echo "[train_verl] save=${SAVE_PATH}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${TEST_FILES}" \
  data.prompt_key=prompt \
  data.reward_fn_key=data_source \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.use_re_call=True \
  data.prompt_template_name=search_policy_template_sys \
  data.search_url="${SEARCH_URL}" \
  data.filter_overlong_prompts=True \
  actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}" \
  +actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.use_torch_compile="${USE_TORCH_COMPILE}" \
  +actor_rollout_ref.actor.forward_dtype="${ACTOR_FORWARD_DTYPE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype="${FSDP_PARAM_DTYPE}" \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype="${FSDP_REDUCE_DTYPE}" \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype="${FSDP_BUFFER_DTYPE}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$((4 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
  actor_rollout_ref.rollout.dtype="${ROLLOUT_DTYPE}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
  actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_UTIL}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.enable_chunked_prefill="${ROLLOUT_ENABLE_CHUNKED_PREFILL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.max_turns="${MAX_TURNS}" \
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
  actor_rollout_ref.rollout.free_cache_engine=False \
  +actor_rollout_ref.rollout.search_url="${SEARCH_URL}" \
  +actor_rollout_ref.rollout.search_max_workers="${SEARCH_MAX_WORKERS}" \
  +actor_rollout_ref.rollout.top_n="${TOP_N}" \
  +actor_rollout_ref.rollout.max_result_chars="${MAX_RESULT_CHARS}" \
  +actor_rollout_ref.rollout.result_summary_mode="${RESULT_SUMMARY_MODE}" \
  +actor_rollout_ref.rollout.result_summary_chars="${RESULT_SUMMARY_CHARS}" \
  +actor_rollout_ref.rollout.result_summary_new_tokens="${RESULT_SUMMARY_NEW_TOKENS}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$((4 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  +actor_rollout_ref.ref.fsdp_config.model_dtype="${FSDP_PARAM_DTYPE}" \
  +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype="${FSDP_PARAM_DTYPE}" \
  +actor_rollout_ref.ref.fsdp_config.mixed_precision.reduce_dtype="${FSDP_REDUCE_DTYPE}" \
  +actor_rollout_ref.ref.fsdp_config.mixed_precision.buffer_dtype="${FSDP_BUFFER_DTYPE}" \
  reward_model.reward_manager="${REWARD_MANAGER}" \
  custom_reward_function.path="${ROOT_DIR}/verl_search_policy/reward.py" \
  custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger="[${LOGGER}]" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${SAVE_PATH}" \
  trainer.rollout_save_path="${SAVE_PATH}/rollout" \
  hydra.run.dir="${SAVE_PATH}/checkpoint/outputs" \
  2>&1 | tee "${SAVE_PATH}/checkpoint/run.log"

