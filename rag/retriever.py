"""Semantic retriever over the local FAISS store."""

from __future__ import annotations

from typing import Any

from .embedding import EmbeddingModel
from .vector_store import VectorStore


class Retriever:
    """Embed only the incoming query, then search the persisted index."""

    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_vector = self.embedding_model.encode_query(query)
        matches = self.vector_store.search(query_vector, top_k=top_k)
        return [{"score": score, **metadata} for score, metadata in matches]
