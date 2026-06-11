#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src:${PYTHONPATH:-}"

TRAIN_FILE="${TRAIN_FILE:-data/trex_grpo_train_6000_no_unseen_pred.jsonl}"
VAL_FILE="${VAL_FILE:-}"
SEARCH_URL="${SEARCH_URL:-http://localhost:8090}"
OUT_DIR="${OUT_DIR:-outputs/verl_data/search_policy}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-${OUT_DIR}/train.parquet}"
VAL_OUTPUT="${VAL_OUTPUT:-${OUT_DIR}/val.parquet}"
VAL_SIZE="${VAL_SIZE:-256}"
MAX_TRAIN_RECORDS="${MAX_TRAIN_RECORDS:--1}"
MAX_VAL_RECORDS="${MAX_VAL_RECORDS:--1}"
SHUFFLE="${SHUFFLE:-1}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/prepare_verl_data.sh

Environment variables:
  TRAIN_FILE              Input train JSON/JSONL.
  VAL_FILE                Optional validation JSON/JSONL.
  SEARCH_URL              Retriever URL stored in extra_info.
  OUT_DIR                 Output directory.
  TRAIN_OUTPUT            Output train parquet.
  VAL_OUTPUT              Output validation parquet.
  VAL_SIZE                Validation split size when VAL_FILE is empty.
  MAX_TRAIN_RECORDS       Limit train rows, -1 for all.
  MAX_VAL_RECORDS         Limit validation rows, -1 for all.
  SHUFFLE                 1 to shuffle train rows before split.
EOF
  exit 0
fi

ARGS=(
  --train-file "${TRAIN_FILE}"
  --train-output "${TRAIN_OUTPUT}"
  --val-output "${VAL_OUTPUT}"
  --search-url "${SEARCH_URL}"
  --val-size "${VAL_SIZE}"
  --max-train-records "${MAX_TRAIN_RECORDS}"
  --max-val-records "${MAX_VAL_RECORDS}"
)

if [[ -n "${VAL_FILE}" ]]; then
  ARGS+=(--val-file "${VAL_FILE}")
fi
if [[ "${SHUFFLE}" == "1" ]]; then
  ARGS+=(--shuffle)
fi

python -m verl_search_policy.prepare_verl_data "${ARGS[@]}"
