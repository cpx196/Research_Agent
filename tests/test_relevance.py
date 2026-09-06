import unittest

from agent.graph.nodes.relevance import assess_evidence_relevance


class RelevanceTests(unittest.TestCase):
    def test_rejects_keyword_drift_from_chinese_search_filler(self) -> None:
        passed, score, _ = assess_evidence_relevance(
            "查找并推荐 JEPA 相关核心论文",
            "JEPA",
            [{
                "source_type": "web_search",
                "title": "找字的笔顺",
                "content": "找字共有七画，部首是扌。",
                "source": "https://example.test/hanzi",
            }],
        )
        self.assertFalse(passed)
        self.assertLess(score, 0.25)

    def test_accepts_topic_aligned_paper_evidence(self) -> None:
        passed, score, _ = assess_evidence_relevance(
            "查找并推荐 JEPA 相关核心论文",
            "JEPA",
            [{
                "source_type": "paper_search",
                "title": "A Path Towards Autonomous Machine Intelligence",
                "content": "Joint Embedding Predictive Architecture (JEPA) learns in representation space.",
                "source": "https://arxiv.org/abs/2202.05861",
            }],
        )
        self.assertTrue(passed)
        self.assertGreaterEqual(score, 0.25)


if __name__ == "__main__":
    unittest.main()
