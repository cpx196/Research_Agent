import unittest
import json
from pathlib import Path
import tempfile
import threading
from http.server import ThreadingHTTPServer

import requests

from agent.graph import LangGraphResearchAgent
from agent.llm import DemoLLM
from tools.local_search import local_search_schema
from tools.web_search import web_search_schema
from web_server import Handler, ResearchWebApp


def fake_local_search(query: str, top_k: int = 5) -> str:
    del top_k
    return (
        "[Result 1]\n"
        "Source: verified-jepa.pdf\n"
        "Page: 4\n"
        "Score: 0.9500\n"
        f"Text: Evidence for {query}."
    )


def fake_web_search(query: str, max_results: int = 5) -> str:
    del max_results
    return f"Title: Example\nURL: https://example.test/result\nSnippet: Web evidence for {query}."


class UncitedStreamingLLM(DemoLLM):
    def chat_stream(self, messages, tools=None):
        del messages, tools
        yield "这是一个没有来源标注的流式草稿。"


def rendered_answer(events) -> str:
    text = ""
    for event in events:
        if event["event"] == "answer_reset":
            text = ""
        elif event["event"] == "answer_chunk":
            text += str(event.get("text", ""))
    return text


class WebStreamingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_logs = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary_logs.cleanup()

    def app(self, agent, mode="graph_context") -> ResearchWebApp:
        return ResearchWebApp(agent, mode, True, log_dir=self.temporary_logs.name)

    def test_stream_reconciles_writer_fallback_with_final_draft(self) -> None:
        agent = LangGraphResearchAgent(
            UncitedStreamingLLM(),
            tools={"local_search": fake_local_search},
            tool_schemas=[local_search_schema],
            context_enabled=True,
            verbose=False,
        )
        app = self.app(agent)
        events = list(app.stream_chat("根据本地论文解释 JEPA。"))
        self.assertEqual(rendered_answer(events), agent.last_state["draft"])
        self.assertTrue(any(event["event"] == "answer_reset" for event in events))
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(events[-1]["answer"], agent.last_state["draft"])

    def test_trace_events_are_emitted_during_graph_stream(self) -> None:
        agent = LangGraphResearchAgent(DemoLLM(), verbose=False)
        app = self.app(agent)
        events = list(app.stream_chat("你好"))
        trace_lines = [event["line"] for event in events if event["event"] == "trace"]
        self.assertIn("[Node] Router", trace_lines)
        self.assertIn("[Node] DirectAnswer", trace_lines)
        self.assertIn("[Graph] END", trace_lines)
        self.assertEqual(rendered_answer(events), agent.last_state["draft"])

    def test_claim_audit_is_emitted_before_done(self) -> None:
        agent = LangGraphResearchAgent(
            DemoLLM(),
            tools={"local_search": fake_local_search},
            tool_schemas=[local_search_schema],
            verifier_mode="active",
            verbose=False,
        )
        events = list(self.app(agent).stream_chat("根据本地论文分析 V-JEPA 2。"))
        audit_indexes = [index for index, event in enumerate(events) if event["event"] == "claim_audit"]
        self.assertTrue(audit_indexes)
        self.assertLess(audit_indexes[0], len(events) - 1)
        self.assertEqual(events[-1]["event"], "done")
        self.assertTrue(any(event.get("claims") for event in events if event["event"] == "claim_audit"))
        self.assertTrue(any(
            event.get("planned_claims") for event in events if event["event"] == "claim_audit"
        ))
        final_audit = events[-1]["claim_audit"]
        self.assertTrue(final_audit["planned_claims"])
        self.assertIn("evidence_quality_checks", final_audit)
        self.assertTrue(any(
            event.get("stage") == "researching" for event in events if event["event"] == "claim_audit"
        ))

    def test_http_sse_closes_after_done_event(self) -> None:
        agent = LangGraphResearchAgent(DemoLLM(), verbose=False)
        Handler.app = self.app(agent)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            response = requests.post(
                f"http://127.0.0.1:{port}/api/chat/stream",
                json={"query": "你好"},
                stream=True,
                timeout=5,
            )
            event_name = ""
            event_names: list[str] = []
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:") and event_name:
                    json.loads(line[5:].strip())
                    event_names.append(event_name)
                    event_name = ""
            self.assertEqual(response.status_code, 200)
            self.assertIn("answer_chunk", event_names)
            self.assertEqual(event_names[-1], "done")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_complete_request_is_saved_as_json(self) -> None:
        agent = LangGraphResearchAgent(DemoLLM(), verbose=False)
        app = self.app(agent)
        events = list(app.stream_chat("你好"))
        done = events[-1]
        path = Path(done["log_file"])
        self.assertTrue(path.is_file())
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["schema_version"], 4)
        self.assertEqual(record["request"]["query"], "你好")
        self.assertEqual(record["response"]["answer"], agent.last_state["draft"])
        self.assertTrue(record["execution"]["trace"])
        self.assertIn("messages", record["execution"]["state"])
        self.assertIn("tool_process", record["execution"])
        self.assertEqual(record["execution"]["verification"]["mode"], "simple")
        self.assertIn("potential_errors", record["execution"]["verification"])
        self.assertIn("questions", record["execution"]["verification"])

    def test_browser_session_preserves_language_and_routes_person_lookup_to_web(self) -> None:
        agent = LangGraphResearchAgent(
            DemoLLM(),
            tools={"web_search": fake_web_search},
            tool_schemas=[web_search_schema],
            context_enabled=True,
            verbose=False,
        )
        app = self.app(agent)
        session_id = "browser-session-test"
        list(app.stream_chat("后续都用中文回答我", session_id=session_id))
        events = list(app.stream_chat("danking是谁", session_id=session_id))
        self.assertEqual(agent.last_state["session_preferences"]["response_language"], "zh-CN")
        self.assertEqual(agent.last_state["conversation_history"][0]["content"], "后续都用中文回答我")
        self.assertEqual(agent.last_state["route_decision"]["route"], "react")
        self.assertTrue(agent.last_state["evidence"])
        done = events[-1]
        self.assertTrue(any("\u4e00" <= character <= "\u9fff" for character in done["answer"]))
        self.assertEqual(agent.last_state["evidence"][0]["url"], "https://example.test/result")
        record = json.loads(Path(done["log_file"]).read_text(encoding="utf-8"))
        self.assertEqual(record["session_id"], session_id)
        self.assertTrue(record["request"]["conversation_history"])
        self.assertEqual(record["request"]["session_preferences"]["response_language"], "zh-CN")


if __name__ == "__main__":
    unittest.main()
