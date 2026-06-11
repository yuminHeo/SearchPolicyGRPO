#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request


class JsonlCorpus:
    def __init__(self, path: str | Path, offset_cache: str | Path = "") -> None:
        self.path = Path(path)
        self.offset_cache = Path(offset_cache) if offset_cache else None
        self.offsets = self._load_or_build_offsets()
        self._file = self.path.open("rb")
        self._lock = threading.Lock()

    def _load_or_build_offsets(self) -> np.ndarray:
        if self.offset_cache and self.offset_cache.exists():
            payload = np.load(self.offset_cache, allow_pickle=False)
            if payload.ndim == 1 and payload.size:
                print(f"[bge_retriever] loaded offset_cache={self.offset_cache} rows={payload.size}", flush=True)
                return payload.astype(np.int64, copy=False)

        print(f"[bge_retriever] building corpus offsets path={self.path}", flush=True)
        offsets: list[int] = []
        offset = 0
        started = time.monotonic()
        with self.path.open("rb") as f:
            for line_no, line in enumerate(f, start=1):
                if line.strip():
                    offsets.append(offset)
                offset += len(line)
                if line_no % 1_000_000 == 0:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    print(
                        f"[bge_retriever] offsets rows={len(offsets)} bytes={offset} rate={offset / elapsed / 1e6:.2f}MB/s",
                        flush=True,
                    )
        array = np.asarray(offsets, dtype=np.int64)
        if self.offset_cache:
            self.offset_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.offset_cache, array)
            print(f"[bge_retriever] saved offset_cache={self.offset_cache}", flush=True)
        print(f"[bge_retriever] corpus rows={array.size}", flush=True)
        return array

    def __len__(self) -> int:
        return int(self.offsets.size)

    def get(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            return {"id": str(index), "title": "", "contents": "", "metadata": {"missing": True}}
        with self._lock:
            self._file.seek(int(self.offsets[index]))
            raw = self._file.readline()
        row = json.loads(raw.decode("utf-8"))
        metadata = dict(row.get("metadata") or {})
        return {
            "id": str(row.get("id", index)),
            "title": str(row.get("title") or ""),
            "contents": str(row.get("contents") or row.get("text") or ""),
            "metadata": metadata,
        }


class BGERetriever:
    backend_name = "bge"

    def __init__(
        self,
        corpus: str,
        model_name: str,
        index_cache: str,
        embedding_cache: str = "",
        offset_cache: str = "",
        device: str = "cuda:0",
        faiss_device: str = "cpu",
        batch_size: int = 128,
        faiss_nprobe: int = 32,
        query_batch_size: int = 128,
        query_batch_wait_ms: int = 20,
    ) -> None:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "BGE retrieval requires faiss and sentence-transformers. "
                "Install faiss-cpu/faiss-gpu and sentence-transformers in this environment."
            ) from exc

        self.faiss = faiss
        self.model_name = model_name
        self.index_cache = str(index_cache)
        self.embedding_cache = str(embedding_cache or "")
        self.device = device
        self.faiss_device = faiss_device
        self.batch_size = max(1, int(batch_size))
        self.faiss_nprobe = max(1, int(faiss_nprobe))
        self.query_batch_size = max(1, int(query_batch_size))
        self.query_batch_wait_s = max(0.0, float(query_batch_wait_ms) / 1000.0)
        self.corpus = JsonlCorpus(corpus, offset_cache=offset_cache)
        self.model = SentenceTransformer(model_name, device=device)
        self.index = self._load_index(index_cache)
        self._lock = threading.Lock()
        self._request_queue: queue.Queue[tuple[str, int, Future]] = queue.Queue()
        self._batch_worker = threading.Thread(target=self._batch_loop, name="bge-query-batcher", daemon=True)
        self._batch_worker.start()
        print(
            f"[bge_retriever] query batching enabled batch_size={self.query_batch_size} "
            f"wait_ms={self.query_batch_wait_s * 1000:.1f}",
            flush=True,
        )

        if self.index.ntotal != len(self.corpus):
            print(
                f"[bge_retriever] WARNING index.ntotal={self.index.ntotal} corpus_rows={len(self.corpus)}",
                flush=True,
            )

    def _load_index(self, index_cache: str):
        print(f"[bge_retriever] loading faiss index={index_cache}", flush=True)
        index = self.faiss.read_index(str(index_cache))
        self._set_nprobe(index)
        index = self._maybe_move_to_gpu(index)
        self._set_nprobe(index)
        print(
            f"[bge_retriever] loaded index ntotal={index.ntotal} d={index.d} faiss_device={self.faiss_device}",
            flush=True,
        )
        return index

    def _set_nprobe(self, index: Any) -> None:
        if hasattr(index, "nprobe"):
            index.nprobe = min(self.faiss_nprobe, max(1, int(getattr(index, "nlist", self.faiss_nprobe))))
            return
        try:
            self.faiss.ParameterSpace().set_index_parameter(index, "nprobe", self.faiss_nprobe)
        except Exception:
            pass

    def _maybe_move_to_gpu(self, index: Any) -> Any:
        device_text = (self.faiss_device or "cpu").strip().lower()
        if device_text in {"", "cpu", "none"}:
            return index
        if not hasattr(self.faiss, "get_num_gpus") or self.faiss.get_num_gpus() <= 0:
            raise RuntimeError("FAISS GPU was requested but faiss reports no visible GPU.")
        if not hasattr(self.faiss, "StandardGpuResources"):
            raise RuntimeError("Installed faiss module has no GPU support. Use faiss-gpu or set FAISS_DEVICE=cpu.")
        gpu_id = 0
        if device_text.startswith("cuda:"):
            gpu_id = int(device_text.split(":", 1)[1])
        elif device_text.startswith("gpu:"):
            gpu_id = int(device_text.split(":", 1)[1].split(",", 1)[0])
        res = self.faiss.StandardGpuResources()
        self._gpu_resource = res
        print(f"[bge_retriever] moving FAISS index to gpu_id={gpu_id}", flush=True)
        try:
            return self.faiss.index_cpu_to_gpu(res, gpu_id, index)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to move FAISS index to {self.faiss_device}. "
                "Set RETRIEVER_FAISS_DEVICE=cpu to use CPU FAISS, or install/use faiss-gpu with enough free VRAM."
            ) from exc

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _materialize_results(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
        top_n: int,
    ) -> tuple[list[dict[str, Any]], list[float]]:
        docs: list[dict[str, Any]] = []
        out_scores: list[float] = []
        for idx, score in zip(indices[:top_n].tolist(), scores[:top_n].tolist()):
            if idx < 0:
                continue
            docs.append(self.corpus.get(int(idx)))
            out_scores.append(float(score))
        return docs, out_scores

    def _search_batch(self, requests: list[tuple[str, int, Future]]) -> None:
        live_requests = [(query, top_n, future) for query, top_n, future in requests if not future.cancelled()]
        if not live_requests:
            return
        queries = [query for query, _, _ in live_requests]
        max_top_n = max(top_n for _, top_n, _ in live_requests)
        try:
            with self._lock:
                query_embeddings = self.encode_queries(queries)
                scores, indices = self.index.search(query_embeddings, int(max_top_n))
            for row, (_, top_n, future) in enumerate(live_requests):
                if not future.cancelled():
                    future.set_result(self._materialize_results(indices[row], scores[row], top_n))
        except Exception as exc:
            for _, _, future in live_requests:
                if not future.cancelled():
                    future.set_exception(exc)

    def _batch_loop(self) -> None:
        while True:
            first = self._request_queue.get()
            requests = [first]
            deadline = time.monotonic() + self.query_batch_wait_s
            while len(requests) < self.query_batch_size:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    requests.append(self._request_queue.get(timeout=timeout))
                except queue.Empty:
                    break
            self._search_batch(requests)

    def search(self, query: str, top_n: int = 5) -> tuple[list[dict[str, Any]], list[float]]:
        query = str(query or "").strip()
        if not query:
            return [], []
        future: Future = Future()
        self._request_queue.put((query, int(top_n), future))
        return future.result()


RETRIEVER: BGERetriever | None = None
SERVER_CONFIG: dict[str, Any] = {}


def create_app():
    import asyncio

    app = FastAPI()

    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "on"}

    def _coerce_top_n(value: Any) -> int:
        try:
            top_n = int(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="top_n must be an integer") from exc
        if top_n < 1:
            raise HTTPException(status_code=422, detail="top_n must be >= 1")
        return top_n

    async def _parse_search_request(request: Request) -> tuple[str, int, bool]:
        body: dict[str, Any] = {}
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            parsed = await request.json()
            if parsed is not None and not isinstance(parsed, dict):
                raise HTTPException(status_code=422, detail="JSON body must be an object")
            body = parsed or {}
        query = body.get("query", request.query_params.get("query", ""))
        top_n = body.get("top_n", request.query_params.get("top_n", 5))
        return_score = body.get("return_score", request.query_params.get("return_score", False))
        return str(query), _coerce_top_n(top_n), _coerce_bool(return_score)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        if RETRIEVER is None:
            return {"status": "starting", "documents": 0, "backend": None}
        return {
            "status": "healthy",
            "documents": len(RETRIEVER.corpus),
            "backend": RETRIEVER.backend_name,
            "config": SERVER_CONFIG,
        }

    @app.post("/search")
    async def search_endpoint(request: Request):
        if RETRIEVER is None:
            raise HTTPException(status_code=503, detail="retriever is not initialized")
        query, top_n, return_score = await _parse_search_request(request)
        if not query.strip():
            raise HTTPException(status_code=400, detail="query is empty")
        docs, scores = await asyncio.to_thread(RETRIEVER.search, query, top_n=top_n)
        if return_score:
            return docs, scores
        return docs

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone BGE FAISS retriever server.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model_name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--embedding_cache", default="")
    parser.add_argument("--index_cache", required=True)
    parser.add_argument("--offset_cache", default="outputs/retriever/corpus_offsets.npy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--faiss_device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--query_batch_size", type=int, default=128)
    parser.add_argument("--query_batch_wait_ms", type=int, default=20)
    parser.add_argument("--faiss_nprobe", type=int, default=32)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="0.0.0.0")
    return parser.parse_args()


def main() -> None:
    global RETRIEVER, SERVER_CONFIG
    args = parse_args()
    SERVER_CONFIG = vars(args)
    RETRIEVER = BGERetriever(
        corpus=args.corpus,
        model_name=args.model_name,
        embedding_cache=args.embedding_cache,
        index_cache=args.index_cache,
        offset_cache=args.offset_cache,
        device=args.device,
        faiss_device=args.faiss_device,
        batch_size=args.batch_size,
        faiss_nprobe=args.faiss_nprobe,
        query_batch_size=args.query_batch_size,
        query_batch_wait_ms=args.query_batch_wait_ms,
    )
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
