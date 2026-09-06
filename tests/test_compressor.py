import unittest

from agent.context.compressor import ToolResultCompressor


class FakeExtractor:
    def chat(self, messages, tools=None):
        del messages, tools
        return {
            "choices": [
                {
                    "message": {
                        "content": "Source: jepa.pdf\nPage: 7\nClaim: latent prediction supports the task.",
                    }
                }
            ]
        }


class CompressorTests(unittest.TestCase):
    def test_compression_preserves_local_citations(self) -> None:
        long_text = "important evidence " * 500
        raw = (
            "[RAG] Query embedding generated\n"
            "[RAG] Top-K retrieval complete\n\n"
            "[Result 1]\nSource: jepa.pdf\nPage: 7\nScore: 0.8800\n"
            f"Text: {long_text}"
        )
        compressor = ToolResultCompressor(threshold_tokens=20, max_chars=700, max_evidence_chars=300)
        result = compressor.compress_tool_result(raw, tool_name="local_search")
        self.assertTrue(result.compressed)
        self.assertLess(result.final_tokens, result.raw_tokens)
        self.assertIn("Source: jepa.pdf", result.text)
        self.assertIn("Page: 7", result.text)

    def test_short_result_is_not_changed(self) -> None:
        result = ToolResultCompressor(threshold_tokens=100).compress_tool_result("Title: JEPA")
        self.assertFalse(result.compressed)
        self.assertEqual(result.text, "Title: JEPA")

    def test_optional_level_two_extractor_preserves_citation(self) -> None:
        raw = "[Result 1]\nSource: jepa.pdf\nPage: 7\nText: " + ("long fact " * 100)
        compressor = ToolResultCompressor(
            threshold_tokens=5,
            extractor_llm=FakeExtractor(),
            llm_extractor_enabled=True,
        )
        result = compressor.compress_tool_result(raw, tool_name="local_search", query="JEPA objective")
        self.assertIn("Source: jepa.pdf", result.text)
        self.assertIn("Page: 7", result.text)


if __name__ == "__main__":
    unittest.main()
