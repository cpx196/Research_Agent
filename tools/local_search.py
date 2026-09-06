"""Tool wrapper around the persisted local paper retriever."""

from __future__ import annotations

import os
import json
from pathlib import Path
import threading
import time
from typing import Any

from rag.embedding import EmbeddingModel, EmbeddingModelError, HashingEmbeddingModel
from rag.retriever import Retriever
from rag.vector_store import VectorStore, VectorStoreError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "vector_db"
_RETRIEVER_CACHE: dict[tuple[str, str, str, str], Retriever] = {}
_RETRIEVER_LOCK = threading.Lock()


def _limit(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be an integer") from exc
    return max(1, min(result, 10))


def _index_dir() -> Path:
    return Path(os.getenv("RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR))).expanduser()


def _get_retriever() -> Retriever:
    index_dir = _index_dir()
    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.json"
    config_path = index_dir / "config.json"
    config: dict[str, Any] = {}
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
    backend = str(config.get("backend") or os.getenv("RAG_EMBEDDING_BACKEND", "sentence-transformers"))
    model_name = str(config.get("model_name") or os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    device = str(config.get("device") or os.getenv("RAG_EMBEDDING_DEVICE", "cpu"))
    cache_key = (str(index_path), str(metadata_path), f"{backend}:{model_name}", device)
    with _RETRIEVER_LOCK:
        if cache_key not in _RETRIEVER_CACHE:
            store = VectorStore(index_path, metadata_path).load()
            if backend in {"hashing", "local-hash", "offline"} or model_name.startswith("local-hash"):
                model = HashingEmbeddingModel()
            else:
                model = EmbeddingModel(model_name=model_name, device=device)
            _RETRIEVER_CACHE[cache_key] = Retriever(model, store)
        return _RETRIEVER_CACHE[cache_key]


def clear_retriever_cache() -> None:
    """Clear the process-local model/index cache (useful for tests and rebuilds)."""

    with _RETRIEVER_LOCK:
        _RETRIEVER_CACHE.clear()


def warmup_local_rag(probe_query: str = "JEPA representation learning") -> dict[str, Any]:
    """Load BGE + FAISS and execute one tiny query outside the request path."""

    started = time.perf_counter()
    index_dir = _index_dir()
    if not (index_dir / "index.faiss").is_file() or not (index_dir / "metadata.json").is_file():
        return {
            "ready": False,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": "Local vector database not found.",
        }
    try:
        retriever = _get_retriever()
        retriever.search(probe_query, top_k=1)
        return {
            "ready": True,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": "",
        }
    except Exception as exc:
        return {
            "ready": False,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def local_search(query: str, top_k: int = 5) -> str:
    """Search locally indexed paper passages and retain source/page citations."""

    if not isinstance(query, str) or not query.strip():
        return "ToolError: query must be a non-empty string"
    try:
        limit = _limit(top_k)
    except ValueError as exc:
        return f"ToolError: {exc}"

    index_dir = _index_dir()
    if not (index_dir / "index.faiss").is_file() or not (index_dir / "metadata.json").is_file():
        return (
            "ToolError: Local vector database not found. "
            "Run python -m rag.ingest first."
        )

    try:
        retriever = _get_retriever()
        results = retriever.search(query.strip(), top_k=limit)
    except FileNotFoundError:
        return (
            "ToolError: Local vector database not found. "
            "Run python -m rag.ingest first."
        )
    except (EmbeddingModelError, VectorStoreError) as exc:
        return f"ToolError: Local retrieval failed: {exc}"
    except Exception as exc:  # keep any backend failure inside the Observation
        return f"ToolError: Local retrieval failed: {type(exc).__name__}: {exc}"

    if not results:
        return "No relevant local paper passages found."
    blocks = ["[RAG] Query embedding generated", "[RAG] Top-K retrieval complete"]
    for index, result in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Result {index}]",
                    f"Source: {result.get('source', 'unknown')}",
                    f"Page: {result.get('page', 'unknown')}",
                    f"Score: {float(result.get('score', 0.0)):.4f}",
                    f"Text: {str(result.get('text', '')).strip()}",
                ]
            )
        )
    return "\n\n".join(blocks)


local_search_schema = {
    "type": "function",
    "function": {
        "name": "local_search",
        "description": (
            "Search the local academic paper knowledge base. Use this for questions "
            "that should be answered from the imported local papers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Question or search query for the local paper library.",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of passages to retrieve, capped at 10.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
