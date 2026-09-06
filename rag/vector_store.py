"""Persistent FAISS inner-product index and aligned JSON metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - exercised in a missing-dependency environment
    faiss = None  # type: ignore[assignment]


class VectorStoreError(RuntimeError):
    """Raised for invalid or unavailable vector stores."""


class VectorStore:
    """FAISS index where row N corresponds exactly to metadata[N]."""

    def __init__(self, index_path: str | Path, metadata_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index: Any | None = None
        self.metadata: list[dict[str, Any]] = []

    @property
    def exists(self) -> bool:
        return self.index_path.is_file() and self.metadata_path.is_file()

    @property
    def dimension(self) -> int:
        if self.index is None:
            raise VectorStoreError("Vector index is not loaded.")
        return int(self.index.d)

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        metadata: Sequence[dict[str, Any]],
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> "VectorStore":
        if faiss is None:
            raise VectorStoreError("faiss-cpu is not installed; install the V2 requirements first.")
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
            raise VectorStoreError(f"Embeddings must be a non-empty 2-D array, got {vectors.shape}.")
        if len(metadata) != vectors.shape[0]:
            raise VectorStoreError(
                f"Metadata/vector count mismatch: {len(metadata)} metadata entries for {vectors.shape[0]} vectors."
            )
        vectors = np.ascontiguousarray(vectors)
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        store = cls(index_path, metadata_path)
        store.index = index
        store.metadata = [dict(item) for item in metadata]
        store.save()
        return store

    def save(self) -> None:
        if faiss is None:
            raise VectorStoreError("faiss-cpu is not installed; install the V2 requirements first.")
        if self.index is None:
            raise VectorStoreError("Cannot save an unloaded vector index.")
        if self.index.ntotal != len(self.metadata):
            raise VectorStoreError("FAISS index and metadata are not aligned.")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, ensure_ascii=False, indent=2)

    def load(self) -> "VectorStore":
        if faiss is None:
            raise VectorStoreError("faiss-cpu is not installed; install the V2 requirements first.")
        if not self.exists:
            raise FileNotFoundError(f"Vector store not found: {self.index_path.parent}")
        try:
            index = faiss.read_index(str(self.index_path))
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise VectorStoreError(f"Could not load local vector database: {exc}") from exc
        if not isinstance(metadata, list) or not all(isinstance(item, dict) for item in metadata):
            raise VectorStoreError("metadata.json must contain a JSON list of objects.")
        if index.ntotal != len(metadata):
            raise VectorStoreError(
                f"FAISS index has {index.ntotal} vectors but metadata has {len(metadata)} entries."
            )
        self.index = index
        self.metadata = metadata
        return self

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[float, dict[str, Any]]]:
        if self.index is None:
            raise VectorStoreError("Vector index is not loaded.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[0] != 1 or query.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Query vector shape {query.shape} does not match index dimension {self.dimension}."
            )
        query = np.ascontiguousarray(query)
        faiss.normalize_L2(query)
        distances, ids = self.index.search(query, min(top_k, self.index.ntotal))
        results: list[tuple[float, dict[str, Any]]] = []
        for score, item_id in zip(distances[0], ids[0]):
            if int(item_id) < 0:
                continue
            results.append((float(score), dict(self.metadata[int(item_id)])))
        return results

    def file_size_bytes(self) -> int:
        return self.index_path.stat().st_size if self.index_path.exists() else 0
