import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import TOOL_REGISTRY, TOOL_SCHEMAS
from tools.local_search import local_search, warmup_local_rag


class FakeRetriever:
    def search(self, query: str, top_k: int):
        self.query = query
        self.top_k = top_k
        return [{"score": 0.9, "source": "jepa.pdf", "page": 4, "text": "latent prediction"}]


class LocalSearchTests(unittest.TestCase):
    def test_missing_database_is_a_tool_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"RAG_INDEX_DIR": directory}, clear=False
        ):
            result = local_search("what is JEPA?")
        self.assertTrue(result.startswith("ToolError: Local vector database not found."))

    def test_formats_source_page_and_rag_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"RAG_INDEX_DIR": directory}, clear=False
        ), patch("tools.local_search._get_retriever", return_value=FakeRetriever()), patch(
            "tools.local_search._RETRIEVER_CACHE", {}
        ):
            Path(directory, "index.faiss").touch()
            Path(directory, "metadata.json").write_text("[]", encoding="utf-8")
            result = local_search("what is JEPA?", top_k=3)
        self.assertIn("[RAG] Query embedding generated", result)
        self.assertIn("Source: jepa.pdf", result)
        self.assertIn("Page: 4", result)
        self.assertIn("Text: latent prediction", result)

    def test_registry_exposes_local_search_schema(self) -> None:
        self.assertIn("local_search", TOOL_REGISTRY)
        self.assertTrue(any(schema["function"]["name"] == "local_search" for schema in TOOL_SCHEMAS))

    def test_warmup_loads_index_and_runs_probe_query(self) -> None:
        retriever = FakeRetriever()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"RAG_INDEX_DIR": directory}, clear=False
        ), patch("tools.local_search._get_retriever", return_value=retriever):
            Path(directory, "index.faiss").touch()
            Path(directory, "metadata.json").write_text("[]", encoding="utf-8")
            result = warmup_local_rag("JEPA warmup")
        self.assertTrue(result["ready"])
        self.assertEqual(retriever.query, "JEPA warmup")
        self.assertEqual(retriever.top_k, 1)


if __name__ == "__main__":
    unittest.main()
