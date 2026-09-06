import json
import unittest

from agent.graph import LangGraphResearchAgent
from agent.graph.nodes.claims import align_claims, enrich_evidence, extract_claims_deterministic
from agent.graph.nodes.repair import make_repair_node
from agent.graph.context import WorkflowContext
from agent.graph.router import HybridRouter
from agent.llm import DemoLLM
from tools import TOOL_REGISTRY, TOOL_SCHEMAS


def response(content: str):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class LocalRepairLLM:
    def chat(self, messages, tools=None):
        del messages, tools
        return response(json.dumps({
            "replacements": [{
                "original": "V-JEPA 2 在所有机器人控制任务中全面优于 DINOv3。",
                "replacement": "V-JEPA 2 在需要动态预测的控制任务中具有更匹配的归纳偏置。",
                "action": "QUALIFY",
            }]
        }, ensure_ascii=False))


class ClaimVerificationTests(unittest.TestCase):
    def test_claim_alignment_marks_strong_comparison_as_partial_without_direct_comparable_evidence(self):
        claims = extract_claims_deterministic("V-JEPA 2 在机器人控制上全面优于 DINOv3。")
        evidence = [enrich_evidence({
            "source_type": "local_search",
            "source": "vjepa2.pdf",
            "page": 4,
            "content": "V-JEPA 2 learns predictive video representations for planning and robot control.",
            "score": 0.95,
        })]
        aligned, relations = align_claims(claims, evidence)
        self.assertFalse(relations)
        self.assertEqual(aligned[0]["status"], "UNSUPPORTED")
        self.assertEqual(aligned[0]["risk"], "high")

    def test_claim_extractor_keeps_filenames_versions_and_skips_markdown_tables(self):
        draft = (
            "DINOv3 略高于 V-JEPA 2.1（`vjepa2_1.pdf, p.17`）。\n\n"
            "| 模型 | 特点 |\n|---|---|\n| DINOv3 | dense |\n"
            "因此，在静态感知任务中 DINOv3 更适合。"
        )
        claims = extract_claims_deterministic(draft)
        texts = [item["text"] for item in claims]
        self.assertTrue(any("V-JEPA 2.1" in item and "vjepa2_1.pdf" in item for item in texts))
        self.assertFalse(any(item.startswith("|") for item in texts))
        self.assertTrue(any(item["claim_type"] == "recommendation" for item in claims))

    def test_localized_repair_preserves_unaffected_text(self):
        original_claim = "V-JEPA 2 在所有机器人控制任务中全面优于 DINOv3。"
        draft = f"前文保持不变。\n{original_claim}\n后文和引用保持不变。[1]"
        context = WorkflowContext(
            llm=LocalRepairLLM(), tools={}, tool_schemas=[], max_iterations=2
        )
        node = make_repair_node(context)
        result = node({
            "draft": draft,
            "claims": [{
                "claim_id": "C1", "text": original_claim, "status": "UNSUPPORTED",
                "verification_question": "是否有统一基准？",
            }],
            "verifier_feedback": [{
                "claim": original_claim,
                "problem": "没有统一控制基准上的直接证据。",
                "suggested_action": "targeted_rewrite",
            }],
            "verification_evidence": [],
            "accepted_evidence": [{"source": "vjepa2.pdf", "content": "video prediction"}],
            "verifier_mode": "active",
        })
        self.assertTrue(result["draft"].startswith("前文保持不变。"))
        self.assertTrue(result["draft"].endswith("后文和引用保持不变。[1]"))
        self.assertNotIn(original_claim, result["draft"])
        self.assertTrue(result["claim_audits"][0]["changed"])

    def test_flagship_query_routes_to_all_research_sources(self):
        query = (
            "根据本地论文与公开资料，比较 V-JEPA 2、DINOv3 和普通 MAE Encoder，"
            "分析哪个更有利于机器人控制。"
        )
        decision = HybridRouter(DemoLLM(), list(TOOL_REGISTRY)).classify(query)
        self.assertEqual(decision.route, "research")
        self.assertEqual(decision.intent, "research")
        self.assertIn("local_search", decision.allowed_tools)
        self.assertIn("paper_search", decision.allowed_tools)
        self.assertIn("web_search", decision.allowed_tools)
        self.assertNotEqual(decision.intent, "calculation")

    def test_writer_only_baseline_has_no_verifier_node(self):
        agent = LangGraphResearchAgent(DemoLLM(), verifier_mode="none", verbose=False)
        nodes = set(agent.graph.get_graph().nodes)
        self.assertNotIn("verifier", nodes)
        self.assertNotIn("claim_extractor", nodes)


if __name__ == "__main__":
    unittest.main()
