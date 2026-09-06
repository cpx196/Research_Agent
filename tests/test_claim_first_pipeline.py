import unittest

from agent.graph import LangGraphResearchAgent
from agent.graph.nodes.prewrite import assess_evidence_quality
from agent.llm import DemoLLM
from tools.local_search import local_search_schema


def local_search(query: str, top_k: int = 5) -> str:
    del top_k
    return (
        "[Result 1]\nSource: representation.pdf\nPage: 8\nScore: 0.9400\n"
        f"Text: The video encoder predicts temporally evolving visual representations for {query}."
    )


class ClaimFirstPipelineTests(unittest.TestCase):
    def test_quality_gate_distinguishes_passages_from_links_and_references(self) -> None:
        role, _, passed, _ = assess_evidence_quality({
            "source_type": "local_search",
            "source": "paper.pdf",
            "page": 4,
            "content": "The encoder predicts latent representations across time and captures motion dynamics.",
        })
        self.assertEqual(role, "primary")
        self.assertTrue(passed)

        role, _, passed, _ = assess_evidence_quality({
            "source_type": "web_search",
            "source": "https://example.org",
            "content": "A search result snippet that points at a possibly relevant public source.",
        })
        self.assertEqual(role, "discovery")
        self.assertFalse(passed)

        role, _, passed, _ = assess_evidence_quality({
            "source_type": "local_search",
            "source": "paper.pdf",
            "page": 65,
            "content": "References [1] Example. arXiv: 2401.00001. [2] Example. doi:10.1000/test.",
        })
        self.assertEqual(role, "rejected")
        self.assertFalse(passed)

    def test_claim_contract_is_built_before_writer_and_survives_to_audit(self) -> None:
        agent = LangGraphResearchAgent(
            DemoLLM(),
            tools={"local_search": local_search},
            tool_schemas=[local_search_schema],
            verifier_mode="active",
            verbose=False,
        )
        agent.run("根据本地论文比较视频表征并给出机器人控制建议。")

        self.assertTrue(agent.last_state["qualified_evidence"])
        self.assertTrue(agent.last_state["planned_claims"])
        trace = agent.last_state["graph_trace"]
        quality = trace.index("[Node] EvidenceQualityGate")
        planner = trace.index("[Node] ClaimPlanner")
        writer = trace.index("[Node] Writer")
        extractor = trace.index("[Node] ClaimExtractor")
        self.assertLess(quality, planner)
        self.assertLess(planner, writer)
        self.assertLess(writer, extractor)
        self.assertTrue(all("evidence_ids" in claim for claim in agent.last_state["planned_claims"]))


if __name__ == "__main__":
    unittest.main()
