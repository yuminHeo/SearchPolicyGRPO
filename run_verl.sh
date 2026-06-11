#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${ROOT_DIR}"

INDEX_DIR="${INDEX_DIR:-data/indexes}"
CORPUS="${CORPUS:-${INDEX_DIR}/corpus.jsonl}"
EMBEDDING_CACHE="${EMBEDDING_CACHE:-${INDEX_DIR}/trex_renlg_bge.npy}"
INDEX_CACHE="${INDEX_CACHE:-${INDEX_DIR}/trex_renlg_bge_ivf4096.faiss}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8090}"
SEARCH_URL="${SEARCH_URL:-http://localhost:${RETRIEVER_PORT}}"
AUTO_START_RETRIEVER="${AUTO_START_RETRIEVER:-1}"
AUTO_PREPARE_DATA="${AUTO_PREPARE_DATA:-1}"
RETRIEVER_LOG="${RETRIEVER_LOG:-outputs/retriever/bge_${RETRIEVER_PORT}.log}"

retriever_ready() {
  local url="$1"
  python - "$url" <<'PY'
import sys
import requests

url = sys.argv[1].rstrip("/") + "/health"
try:
    response = requests.get(url, timeout=3)
    sys.exit(0 if response.status_code == 200 else 1)
except Exception:
    sys.exit(1)
PY
}

wait_for_retriever() {
  local url="$1"
  local max_wait="${2:-600}"
  local pid="${3:-}"
  local log_file="${4:-}"
  local waited=0
  until retriever_ready "$url"
  do
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      echo "[run_verl] retriever process exited before becoming ready (pid=${pid})" >&2
      if [[ -n "${log_file}" && -f "${log_file}" ]]; then
        tail -n 80 "${log_file}" >&2 || true
      fi
      return 1
    fi
    if (( waited >= max_wait )); then
      echo "[run_verl] retriever did not become ready within ${max_wait}s: ${url}" >&2
      if [[ -n "${log_file}" && -f "${log_file}" ]]; then
        tail -n 80 "${log_file}" >&2 || true
      fi
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
    echo "[run_verl] waiting for retriever ${url} (${waited}s)"
  done
}

RETRIEVER_PID=""
if [[ "${AUTO_START_RETRIEVER}" == "1" ]]; then
  if retriever_ready "${SEARCH_URL}"; then
    echo "[run_verl] existing retriever is ready at ${SEARCH_URL}"
  else
    mkdir -p "$(dirname "${RETRIEVER_LOG}")"
    echo "[run_verl] starting BGE retriever at ${SEARCH_URL}"
    INDEX_DIR="${INDEX_DIR}" \
    CORPUS="${CORPUS}" \
    EMBEDDING_CACHE="${EMBEDDING_CACHE}" \
    INDEX_CACHE="${INDEX_CACHE}" \
    RETRIEVER_PORT="${RETRIEVER_PORT}" \
    scripts/launch_bge_retriever.sh >"${RETRIEVER_LOG}" 2>&1 &
    RETRIEVER_PID="$!"
    trap 'if [[ -n "${RETRIEVER_PID}" ]]; then kill "${RETRIEVER_PID}" 2>/dev/null || true; fi' EXIT
    wait_for_retriever "${SEARCH_URL}" "${RETRIEVER_READY_TIMEOUT:-900}" "${RETRIEVER_PID}" "${RETRIEVER_LOG}"
  fi
else
  echo "[run_verl] using existing retriever at ${SEARCH_URL}"
fi

if [[ "${AUTO_PREPARE_DATA}" == "1" ]]; then
  SEARCH_URL="${SEARCH_URL}" scripts/prepare_verl_data.sh
fi

SEARCH_URL="${SEARCH_URL}" scripts/train_verl_search_policy_grpo.sh "$@"

