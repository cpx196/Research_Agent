import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag.retriever import Retriever
from rag.vector_store import VectorStore


class FakeEmbeddingModel:
    def encode_query(self, query: str) -> np.ndarray:
        del query
        return np.array([[1.0, 0.0]], dtype=np.float32)


class RetrieverTests(unittest.TestCase):
    def test_search_returns_aligned_metadata_and_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = [
                {"chunk_id": 0, "source": "first.pdf", "page": 2, "text": "first"},
                {"chunk_id": 1, "source": "second.pdf", "page": 7, "text": "second"},
            ]
            store = VectorStore.build(
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                metadata,
                root / "index.faiss",
                root / "metadata.json",
            )
            loaded = VectorStore(root / "index.faiss", root / "metadata.json").load()
            results = Retriever(FakeEmbeddingModel(), loaded).search("find first", top_k=1)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["source"], "first.pdf")
            self.assertEqual(results[0]["page"], 2)
            self.assertEqual(results[0]["text"], "first")
            self.assertAlmostEqual(results[0]["score"], 1.0, places=5)
            self.assertEqual(store.index.ntotal, len(metadata))


if __name__ == "__main__":
    unittest.main()
