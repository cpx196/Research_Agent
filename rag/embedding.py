"""Sentence-transformers embedding adapter with a CPU-friendly default."""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - exercised in a missing-dependency environment
    SentenceTransformer = None  # type: ignore[assignment,misc]

try:
    from sklearn.feature_extraction.text import HashingVectorizer
except ImportError:  # pragma: no cover - requirements install scikit-learn transitively
    HashingVectorizer = None  # type: ignore[assignment,misc]


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingModelError(RuntimeError):
    """Raised when the embedding model cannot be loaded or used."""


class HashingEmbeddingModel:
    """Offline fallback using deterministic character n-gram vectors.

    This is not a learned semantic model, but it preserves the essential V2
    contract when a model download is unavailable: documents and queries are
    mapped to the same normalized vector space and FAISS can search it.
    """

    model_name = "local-hash-char-3-5"

    def __init__(self, n_features: int = 2048) -> None:
        if HashingVectorizer is None:
            raise EmbeddingModelError("scikit-learn is required for the local hashing fallback.")
        self.vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )
        self.batch_size = 0
        self.device = "cpu"
        self.dimension = n_features

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        try:
            vectors = self.vectorizer.transform(list(texts)).toarray()
        except Exception as exc:
            raise EmbeddingModelError(f"Local hashing embedding failed: {exc}") from exc
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise EmbeddingModelError("Query must be a non-empty string.")
        return self._encode([query.strip()])


class EmbeddingModel:
    """Encode documents and queries in one normalized vector space."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 32,
        device: str | None = None,
        query_instruction: str | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
        self.batch_size = max(1, int(batch_size))
        # Explicit CPU is intentional: this project targets macOS machines
        # without a discrete CUDA GPU. Sentence-transformers can still use
        # Apple's MPS when explicitly requested by the caller.
        self.device = device or os.getenv("RAG_EMBEDDING_DEVICE", "cpu")
        self.query_instruction = (
            DEFAULT_QUERY_INSTRUCTION
            if query_instruction is None and os.getenv("RAG_QUERY_INSTRUCTION") is None
            else os.getenv("RAG_QUERY_INSTRUCTION", "") if query_instruction is None else query_instruction
        )
        if model is not None:
            self.model = model
        else:
            if SentenceTransformer is None:
                raise EmbeddingModelError(
                    "sentence-transformers is not installed; install the V2 requirements first."
                )
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                raise EmbeddingModelError(
                    f"Could not load embedding model {self.model_name!r}: {exc}"
                ) from exc

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dimension = self.dimension
            return np.empty((0, dimension), dtype=np.float32)
        try:
            encoded = self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except TypeError:
            # Keeps the adapter usable with small fake models in unit tests.
            encoded = self.model.encode(list(texts))
        except Exception as exc:
            raise EmbeddingModelError(f"Embedding generation failed: {exc}") from exc
        vectors = np.asarray(encoded, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise EmbeddingModelError(
                f"Embedding model returned unexpected shape {vectors.shape}; expected ({len(texts)}, dimension)."
            )
        # Normalize again for fake/custom backends and for cosine-as-inner
        # product retrieval. Zero vectors are left unchanged.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        return np.ascontiguousarray(vectors, dtype=np.float32)

    @property
    def dimension(self) -> int:
        dimension = getattr(self.model, "get_sentence_embedding_dimension", lambda: None)()
        if dimension:
            return int(dimension)
        probe = self._encode_without_dimension_probe("probe")
        return int(probe.shape[-1])

    def _encode_without_dimension_probe(self, text: str) -> np.ndarray:
        try:
            encoded = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        except TypeError:
            encoded = self.model.encode([text])
        vectors = np.asarray(encoded, dtype=np.float32)
        return vectors.reshape(1, -1)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Batch-encode document chunks."""

        return self._encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one query and return a ``(1, dimension)`` array."""

        if not isinstance(query, str) or not query.strip():
            raise EmbeddingModelError("Query must be a non-empty string.")
        return self._encode([f"{self.query_instruction}{query.strip()}"])
