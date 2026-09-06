import unittest

from agent.context.selector import EvidenceSelector


class EvidenceSelectorTests(unittest.TestCase):
    def test_deduplicates_source_page_and_selects_top_k(self) -> None:
        selector = EvidenceSelector()
        evidence = [
            {"source": "jepa.pdf", "page": 4, "content": "latent prediction", "score": 0.8},
            {"source": "jepa.pdf", "page": 4, "content": "latent prediction", "score": 0.7},
            {"source": "dino.pdf", "page": 8, "content": "dense visual features", "score": 0.9},
        ]
        result = selector.select_with_stats(evidence, "latent prediction", top_k=1)
        self.assertEqual(result.available_count, 3)
        self.assertEqual(result.deduplicated_count, 1)
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0]["source"], "jepa.pdf")

    def test_select_method_returns_plain_evidence_list(self) -> None:
        selector = EvidenceSelector()
        selected = selector.select(
            [{"source": "paper.pdf", "page": 1, "content": "robot action"}],
            "robot action",
            top_k=5,
        )
        self.assertIsInstance(selected, list)
        self.assertEqual(selected[0]["source"], "paper.pdf")


if __name__ == "__main__":
    unittest.main()
