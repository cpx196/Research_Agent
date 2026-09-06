import unittest

from agent.context import ContextConfig, ContextManager
from agent.graph.nodes.evidence import parse_tool_evidence


class ContextManagerTests(unittest.TestCase):
    def test_node_specific_context_excludes_raw_messages(self) -> None:
        manager = ContextManager(ContextConfig())
        state = {
            "query": "compare JEPA and DINO",
            "plan": [{"id": 1, "task": "JEPA objective", "tool": "local_search", "status": "pending"}],
            "current_step": 0,
            "open_questions": ["DINO robotics evidence"],
            "verification_feedback": "missing benchmark",
            "evidence": [],
            "messages": [{"role": "tool", "content": "RAW PRIVATE TRACE SHOULD NOT APPEAR"}],
        }
        built = manager.build_researcher_context(state)
        self.assertIn("Current Query", built.text)
        self.assertIn("DINO robotics evidence", built.text)
        self.assertNotIn("RAW PRIVATE TRACE SHOULD NOT APPEAR", built.text)

    def test_budget_keeps_query_and_reports_tokens(self) -> None:
        manager = ContextManager(
            ContextConfig(
                researcher_budget=120,
                researcher_top_k=2,
            )
        )
        state = {
            "query": "JEPA",
            "plan": [{"id": 1, "task": "JEPA objective", "tool": "local_search", "status": "pending"}],
            "current_step": 0,
            "open_questions": ["a very long open question " * 20],
            "verification_feedback": "feedback",
            "evidence": [],
        }
        built = manager.build_researcher_context(state)
        self.assertIn("Current Query", built.text)
        self.assertGreaterEqual(built.budget.raw_tokens, built.budget.final_tokens)
        self.assertLessEqual(built.budget.final_tokens, built.budget.budget)

    def test_level_two_citation_block_becomes_structured_evidence(self) -> None:
        records = parse_tool_evidence(
            "local_search",
            "Source: jepa.pdf\nPage: 7\nClaim: latent prediction",
            "JEPA objective",
            "call-1",
        )
        self.assertEqual(records[0]["source"], "jepa.pdf")
        self.assertEqual(records[0]["page"], 7)
        self.assertEqual(records[0]["content"], "latent prediction")

    def test_writer_context_preserves_session_language_and_history(self) -> None:
        manager = ContextManager(ContextConfig())
        built = manager.build_writer_context({
            "query": "Who is Danking?",
            "conversation_history": [
                {"role": "user", "content": "后续都用中文回答我"},
                {"role": "assistant", "content": "好的。"},
            ],
            "session_preferences": {"response_language": "zh-CN"},
            "plan": [],
            "evidence": [],
        })
        self.assertIn("后续都用中文回答我", built.text)
        self.assertIn("response_language: Simplified Chinese", built.text)


if __name__ == "__main__":
    unittest.main()
