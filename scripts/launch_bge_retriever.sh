#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TRAJRL_DIR="${TRAJRL_DIR:-/data/YM/ExpCodes/TrajRL}"
INDEX_DIR="${INDEX_DIR:-${TRAJRL_DIR}/outputs/indexes}"

CORPUS="${CORPUS:-${TRAJRL_DIR}/dataset/trex_renlg/corpus.jsonl}"
MODEL_NAME="${MODEL_NAME:-${BGE_MODEL_NAME:-BAAI/bge-base-en-v1.5}}"
EMBEDDING_CACHE="${EMBEDDING_CACHE:-${INDEX_DIR}/trex_renlg_bge.npy}"
INDEX_CACHE="${INDEX_CACHE:-${INDEX_DIR}/trex_renlg_bge_ivf4096.faiss}"
OFFSET_CACHE="${OFFSET_CACHE:-${ROOT_DIR}/outputs/retriever/$(basename "${CORPUS}").offsets.npy}"
PORT="${PORT:-${RETRIEVER_PORT:-8090}}"

# Keep FAISS on CPU by default because GRPO usually occupies training GPUs.
DEVICE="${DEVICE:-${RETRIEVER_DEVICE:-cuda:3}}"
FAISS_DEVICE="${FAISS_DEVICE:-${RETRIEVER_FAISS_DEVICE:-cpu}}"
BATCH_SIZE="${BATCH_SIZE:-128}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-128}"
QUERY_BATCH_WAIT_MS="${QUERY_BATCH_WAIT_MS:-20}"
FAISS_NLIST="${FAISS_NLIST:-4096}"
FAISS_NPROBE="${FAISS_NPROBE:-32}"

if [[ ! -f "${CORPUS}" ]]; then
  echo "[launch_bge_retriever] missing corpus: ${CORPUS}" >&2
  exit 1
fi
if [[ ! -f "${INDEX_CACHE}" ]]; then
  echo "[launch_bge_retriever] missing index cache: ${INDEX_CACHE}" >&2
  exit 1
fi
if [[ -n "${EMBEDDING_CACHE}" && ! -f "${EMBEDDING_CACHE}" ]]; then
  echo "[launch_bge_retriever] warning: embedding cache not found; existing FAISS index will still be used: ${EMBEDDING_CACHE}" >&2
fi

echo "[launch_bge_retriever] root_dir=${ROOT_DIR}"
echo "[launch_bge_retriever] corpus=${CORPUS}"
echo "[launch_bge_retriever] index_dir=${INDEX_DIR}"
echo "[launch_bge_retriever] embedding_cache=${EMBEDDING_CACHE}"
echo "[launch_bge_retriever] index_cache=${INDEX_CACHE}"
echo "[launch_bge_retriever] offset_cache=${OFFSET_CACHE}"
echo "[launch_bge_retriever] model=${MODEL_NAME}"
echo "[launch_bge_retriever] device=${DEVICE}"
echo "[launch_bge_retriever] faiss_device=${FAISS_DEVICE}"
echo "[launch_bge_retriever] port=${PORT}"

cd "${ROOT_DIR}"
exec python bge_retriever_server.py \
  --corpus "${CORPUS}" \
  --model_name "${MODEL_NAME}" \
  --embedding_cache "${EMBEDDING_CACHE}" \
  --index_cache "${INDEX_CACHE}" \
  --offset_cache "${OFFSET_CACHE}" \
  --device "${DEVICE}" \
  --faiss_device "${FAISS_DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --query_batch_size "${QUERY_BATCH_SIZE}" \
  --query_batch_wait_ms "${QUERY_BATCH_WAIT_MS}" \
  --faiss_nprobe "${FAISS_NPROBE}" \
  --port "${PORT}"
