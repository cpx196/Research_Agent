"""Local browser UI for the Research Agent.

This deliberately uses Python's standard library so the macOS CPU setup needs
no additional web framework or frontend build step.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime
import json
import queue
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from agent.agent import ResearchAgent
from agent.graph import LangGraphResearchAgent, MCPResearchAgent
from agent.llm import DemoLLM, LLMClient, LLMClientError
from agent.graph.nodes.claims import extract_claims_deterministic
from tools.local_search import warmup_local_rag


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DEFAULT_LOG_DIR = ROOT / "logs" / "web_runs"


def start_local_rag_warmup() -> threading.Thread:
    """Warm BGE/FAISS in the background without delaying HTTP startup."""

    def worker() -> None:
        result = warmup_local_rag()
        if result["ready"]:
            print(f"[Warmup] Local RAG ready in {result['elapsed_seconds']:.3f}s")
        else:
            print(f"[Warmup] Local RAG unavailable: {result['error']}")

    thread = threading.Thread(target=worker, name="local-rag-warmup", daemon=True)
    thread.start()
    return thread


def _json_safe(value: Any) -> Any:
    """Convert graph/message models to JSON without leaking runtime objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


class ResearchWebApp:
    def __init__(
        self,
        agent: Any,
        mode: str,
        demo: bool,
        log_dir: str | Path = DEFAULT_LOG_DIR,
    ) -> None:
        self.agent = agent
        self.mode = mode
        self.demo = demo
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.lock = threading.Lock()
        self.sessions: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _session_id(value: Any) -> str:
        candidate = str(value or "default").strip()[:128]
        return candidate if candidate and all(character.isalnum() or character in "-_" for character in candidate) else "default"

    def _session_context(self, session_id: str, query: str) -> tuple[list[dict[str, str]], dict[str, str]]:
        session = self.sessions.setdefault(session_id, {"history": [], "preferences": {}})
        preferences = dict(session["preferences"])
        normalized = query.lower()
        if (
            "中文" in query
            and any(token in query for token in ("用", "说", "回答", "回复"))
            and not any(token in query for token in ("不要中文", "别用中文", "不用中文"))
        ):
            preferences["response_language"] = "zh-CN"
        if any(token in query for token in ("用英文", "英文回答", "英语回答", "改成英文")) or any(
            token in normalized for token in ("reply in english", "answer in english", "respond in english")
        ):
            preferences["response_language"] = "en"
        session["preferences"] = preferences
        return list(session["history"][-12:]), preferences

    def _record_turn(self, session_id: str, query: str, answer: str) -> None:
        session = self.sessions.setdefault(session_id, {"history": [], "preferences": {}})
        session["history"].extend((
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ))
        session["history"] = session["history"][-24:]

    def _agent_state(self) -> dict[str, Any]:
        state = getattr(self.agent, "last_state", None)
        if isinstance(state, Mapping):
            return dict(state)
        messages = getattr(self.agent, "last_messages", None)
        return {"messages": list(messages or [])}

    def _tool_process(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        process: list[dict[str, Any]] = []
        for message in state.get("messages", []) or []:
            calls = getattr(message, "tool_calls", None)
            if calls is None and isinstance(message, Mapping):
                calls = message.get("tool_calls")
            for call in calls or []:
                process.append({"type": "tool_call", **_json_safe(call)})
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id is None and isinstance(message, Mapping):
                tool_call_id = message.get("tool_call_id")
            if tool_call_id:
                content = getattr(message, "content", None)
                if content is None and isinstance(message, Mapping):
                    content = message.get("content")
                process.append({
                    "type": "tool_result",
                    "tool_call_id": str(tool_call_id),
                    "content": _json_safe(content),
                })
        return process

    def _claim_audit_payload(self) -> dict[str, Any]:
        state = self._agent_state()
        claims = list(state.get("claims", []) or [])
        audits = list(state.get("claim_audits", []) or [])
        counts: dict[str, int] = {}
        for claim in claims:
            status = str(claim.get("status", "UNSUPPORTED"))
            counts[status] = counts.get(status, 0) + 1
        return {
            "planned_claims": _json_safe(state.get("planned_claims", [])),
            "evidence_quality_checks": _json_safe(state.get("evidence_quality_checks", [])),
            "claims": _json_safe(claims),
            "relations": _json_safe(state.get("claim_evidence", [])),
            "repairs": _json_safe(audits),
            "status_counts": counts,
        }

    def _save_run(
        self,
        *,
        request_id: str,
        started_at: datetime,
        query: str,
        session_id: str,
        conversation_history: list[dict[str, str]],
        session_preferences: Mapping[str, str],
        answer: str,
        trace: list[str],
        stats: Mapping[str, Any],
        error: str = "",
    ) -> tuple[str, str]:
        completed_at = datetime.now().astimezone()
        state = self._agent_state()
        record = {
            "schema_version": 4,
            "request_id": request_id,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "mode": self.mode,
            "demo": self.demo,
            "session_id": session_id,
            "request": {
                "query": query,
                "conversation_history": _json_safe(conversation_history),
                "session_preferences": _json_safe(session_preferences),
            },
            "response": {"answer": answer, "error": error or None},
            "execution": {
                "trace": list(trace),
                "stats": _json_safe(stats),
                "query_resolution": {
                    "original_query": _json_safe(state.get("original_query", query)),
                    "standalone_query": _json_safe(state.get("standalone_query", query)),
                    "topic": _json_safe(state.get("resolved_topic", "")),
                    "intent": _json_safe(state.get("resolved_intent", "general")),
                    "search_queries": _json_safe(state.get("search_queries", {})),
                },
                "relevance_checks": _json_safe(state.get("relevance_checks", [])),
                "verification": {
                    "mode": _json_safe(state.get("verifier_mode", "simple")),
                    "potential_errors": _json_safe(state.get("potential_errors", [])),
                    "questions": _json_safe(state.get("verification_questions", [])),
                    "evidence": _json_safe(state.get("verification_evidence", [])),
                    "feedback": _json_safe(state.get("verifier_feedback", [])),
                    "iteration": _json_safe(state.get("verification_iteration", 0)),
                    "details": _json_safe(state.get("verification_details", {})),
                    "claim_audit": {
                        "planned_claims": _json_safe(state.get("planned_claims", [])),
                        "evidence_quality_checks": _json_safe(
                            state.get("evidence_quality_checks", [])
                        ),
                        "claims": _json_safe(state.get("claims", [])),
                        "relations": _json_safe(state.get("claim_evidence", [])),
                        "repairs": _json_safe(state.get("claim_audits", [])),
                    },
                },
                "tool_process": self._tool_process(state),
                "state": _json_safe(state),
            },
        }
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = started_at.strftime("%Y%m%dT%H%M%S_%f%z")
            path = self.log_dir / f"{timestamp}_{request_id}.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
            try:
                display_path = str(path.relative_to(ROOT))
            except ValueError:
                display_path = str(path)
            return display_path, ""
        except OSError as exc:
            return "", f"{type(exc).__name__}: {exc}"

    def chat(self, query: str, session_id: str = "default") -> dict[str, Any]:
        session_id = self._session_id(session_id)
        request_id = uuid4().hex
        started_at = datetime.now().astimezone()
        started = time.perf_counter()
        with self.lock:
            history, preferences = self._session_context(session_id, query)
            trace_start = len(self.agent.trace.lines)
            answer = self.agent.run(
                query,
                conversation_history=history,
                session_preferences=preferences,
            )
            self._record_turn(session_id, query, answer)
            trace = list(self.agent.trace.lines[trace_start:])
            stats = dict(getattr(self.agent, "last_run_stats", {}))
        final_stats = {
            **stats,
            "request_latency_seconds": round(time.perf_counter() - started, 3),
        }
        log_file, log_error = self._save_run(
            request_id=request_id,
            started_at=started_at,
            query=query,
            session_id=session_id,
            conversation_history=history,
            session_preferences=preferences,
            answer=answer,
            trace=trace,
            stats=final_stats,
        )
        return {
            "request_id": request_id,
            "session_id": session_id,
            "answer": answer,
            "trace": trace,
            "stats": final_stats,
            "mode": self.mode,
            "log_file": log_file,
            "log_error": log_error,
            "claim_audit": self._claim_audit_payload(),
        }

    def stream_chat(self, query: str, session_id: str = "default"):
        """Yield live Trace events followed by chunked answer events."""
        session_id = self._session_id(session_id)
        request_id = uuid4().hex
        started_at = datetime.now().astimezone()
        started = time.perf_counter()
        events: queue.Queue[dict[str, Any]] = queue.Queue()
        streamed = False
        streamed_parts: list[str] = []
        research_streaming = False
        last_provisional_length = 0
        history: list[dict[str, str]] = []
        preferences: dict[str, str] = {}

        def emit(line: str) -> None:
            nonlocal research_streaming
            events.put({"event": "trace", "line": line})
            if line == "Route: research":
                research_streaming = True
                events.put({
                    "event": "claim_audit",
                    "claims": [],
                    "relations": [],
                    "repairs": [],
                    "stage": "researching",
                })

        def emit_token(token: str) -> None:
            nonlocal streamed, last_provisional_length
            streamed = True
            text = str(token)
            streamed_parts.append(text)
            events.put({"event": "answer_chunk", "text": text})
            current = "".join(streamed_parts)
            sentence_boundary = any(mark in text for mark in ("。", "！", "？", ". ", "! ", "? ", "\n"))
            if research_streaming and sentence_boundary and len(current) - last_provisional_length >= 120:
                provisional = [dict(item) for item in extract_claims_deterministic(current)]
                if provisional:
                    for claim in provisional:
                        claim["status"] = "PENDING"
                    last_provisional_length = len(current)
                    events.put({
                        "event": "claim_audit",
                        "claims": provisional,
                        "relations": [],
                        "repairs": [],
                        "stage": "writer_stream",
                    })

        def emit_claim_audit(payload: Mapping[str, Any]) -> None:
            events.put({"event": "claim_audit", **_json_safe(dict(payload))})

        def worker() -> None:
            nonlocal history, preferences
            try:
                with self.lock:
                    history, preferences = self._session_context(session_id, query)
                    events.put({"event": "answer_start"})
                    old_callback = getattr(self.agent.trace, "on_emit", None)
                    old_token_callback = getattr(self.agent, "stream_callback", None)
                    old_live_callback = getattr(self.agent, "live_trace_callback", None)
                    old_claim_callback = getattr(self.agent, "live_claim_callback", None)
                    if hasattr(self.agent, "live_trace_callback"):
                        self.agent.live_trace_callback = emit
                        self.agent.live_claim_callback = emit_claim_audit
                        self.agent.trace.on_emit = None
                    else:
                        self.agent.trace.on_emit = emit
                    self.agent.stream_callback = emit_token
                    trace_start = len(self.agent.trace.lines)
                    try:
                        answer = self.agent.run(
                            query,
                            conversation_history=history,
                            session_preferences=preferences,
                        )
                    finally:
                        self.agent.trace.on_emit = old_callback
                        self.agent.stream_callback = old_token_callback
                        if hasattr(self.agent, "live_trace_callback"):
                            self.agent.live_trace_callback = old_live_callback
                            self.agent.live_claim_callback = old_claim_callback
                    trace = list(self.agent.trace.lines[trace_start:])
                    stats = dict(getattr(self.agent, "last_run_stats", {}))
                    self._record_turn(session_id, query, answer)
                if not trace:
                    events.put({"event": "trace", "line": "[Graph] completed"})
                final_stats = {
                    **stats,
                    "streaming": True,
                    "request_latency_seconds": round(time.perf_counter() - started, 3),
                }
                log_file, log_error = self._save_run(
                    request_id=request_id,
                    started_at=started_at,
                    query=query,
                    session_id=session_id,
                    conversation_history=history,
                    session_preferences=preferences,
                    answer=answer,
                    trace=trace,
                    stats=final_stats,
                )
                streamed_answer = "".join(streamed_parts).strip()
                if streamed and streamed_answer != answer.strip():
                    events.put({"event": "answer_reset", "reason": "final_draft_reconciled"})
                    for offset in range(0, len(answer), 12):
                        events.put({"event": "answer_chunk", "text": answer[offset:offset + 12]})
                elif not streamed:
                    for offset in range(0, len(answer), 12):
                        events.put({"event": "answer_chunk", "text": answer[offset:offset + 12]})
                events.put({
                    "event": "done",
                    "request_id": request_id,
                    "session_id": session_id,
                    "mode": self.mode,
                    "answer": answer,
                    "stats": final_stats,
                    "log_file": log_file,
                    "log_error": log_error,
                    "claim_audit": self._claim_audit_payload(),
                })
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                log_file, log_error = self._save_run(
                    request_id=request_id,
                    started_at=started_at,
                    query=query,
                    session_id=session_id,
                    conversation_history=history,
                    session_preferences=preferences,
                    answer="",
                    trace=[],
                    stats={"request_latency_seconds": round(time.perf_counter() - started, 3)},
                    error=error,
                )
                events.put({
                    "event": "error",
                    "error": error,
                    "request_id": request_id,
                    "log_file": log_file,
                    "log_error": log_error,
                })
            finally:
                events.put({"event": "close"})

        threading.Thread(target=worker, daemon=True).start()
        while True:
            try:
                event = events.get(timeout=0.25)
            except queue.Empty:
                yield {"event": "ping"}
                continue
            if event["event"] == "close":
                break
            yield event

    def health(self) -> dict[str, Any]:
        stats = dict(getattr(self.agent, "last_run_stats", {}))
        result: dict[str, Any] = {
            "ok": True,
            "mode": self.mode,
            "demo": self.demo,
            "platform": "macOS CPU",
            "rag": "BGE/FAISS local retrieval preserved",
            "last_run": stats,
            "log_dir": str(self.log_dir),
            "active_sessions": len(self.sessions),
        }
        if isinstance(self.agent, MCPResearchAgent):
            discovered = [
                name for name, adapter in self.agent.registry.adapters.items()
                if getattr(getattr(adapter, "spec", None), "source", "") == "MCP"
            ]
            result["mcp"] = {
                "server": self.agent.mcp_config.name,
                "transport": self.agent.mcp_config.transport,
                "tools": self.agent.last_run_stats.get("mcp_tools", discovered),
                "discovery_error": self.agent.discovery_error,
            }
        return result


class Handler(BaseHTTPRequestHandler):
    app: ResearchWebApp

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            # Browsers may cancel an in-flight SSE request; that is normal.
            pass

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        files = {
            "/": (WEB_ROOT / "index.html", "text/html; charset=utf-8"),
            "/index.html": (WEB_ROOT / "index.html", "text/html; charset=utf-8"),
            "/app.js": (WEB_ROOT / "app.js", "text/javascript; charset=utf-8"),
            "/styles.css": (WEB_ROOT / "styles.css", "text/css; charset=utf-8"),
        }
        if path == "/api/health":
            self._send_json(self.app.health())
        elif path in files:
            self._send_file(*files[path])
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path == "/api/chat/stream":
            self._stream_chat()
            return
        if path != "/api/chat":
            self.send_error(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("request body is empty or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            query = str(payload.get("query", "")).strip()
            session_id = self.app._session_id(payload.get("session_id"))
            if not query:
                raise ValueError("query is empty")
            self._send_json(self.app.chat(query, session_id=session_id))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # the browser receives a controlled error
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _stream_chat(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("request body is empty or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            query = str(payload.get("query", "")).strip()
            session_id = self.app._session_id(payload.get("session_id"))
            if not query:
                raise ValueError("query is empty")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for event in self.app.stream_chat(query, session_id=session_id):
                if event["event"] == "ping":
                    self.wfile.write(b": keep-alive\n\n")
                else:
                    event_name = str(event.get("event", "message"))
                    data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                    self.wfile.write(f"event: {event_name}\ndata: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True

    def log_message(self, format: str, *args: Any) -> None:
        # Keep request logs compact and separate from MCP stdio protocol logs.
        print(f"[Web] {self.address_string()} - {format % args}")


def build_agent(
    mode: str,
    demo: bool,
    verifier_mode: str = "active",
    max_verification_tool_calls: int = 2,
) -> Any:
    if demo:
        llm: Any = DemoLLM()
    else:
        try:
            llm = LLMClient.from_env()
        except LLMClientError as exc:
            raise SystemExit(f"Configuration error: {exc}. Use --demo or configure .env.") from exc
    if mode == "legacy":
        return ResearchAgent(llm, verbose=False)
    if mode == "graph_mcp":
        return MCPResearchAgent(
            llm,
            verifier_mode=verifier_mode,
            max_verification_tool_calls=max_verification_tool_calls,
            verbose=False,
        )
    return LangGraphResearchAgent(
        llm,
        max_iterations=2,
        verifier_mode=verifier_mode,
        max_verification_tool_calls=max_verification_tool_calls,
        context_enabled=mode == "graph_context",
        verbose=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research Agent browser UI")
    parser.add_argument(
        "--mode",
        choices=("legacy", "graph_baseline", "graph_context", "graph_mcp", "graph"),
        default="graph_mcp",
        help="Agent workflow exposed to the browser (default: graph_mcp)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--demo", action="store_true", help="Use deterministic local DemoLLM")
    parser.add_argument(
        "--verifier",
        choices=("none", "simple", "active"),
        default="active",
        help="Verifier workflow (default: active)",
    )
    parser.add_argument(
        "--max-verification-tool-calls",
        type=int,
        choices=range(1, 5),
        default=2,
        metavar="1-4",
        help="Tool-call budget for Active Verifier (default: 2)",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Directory for one complete JSON log per request",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    app = ResearchWebApp(
        build_agent(
            args.mode,
            args.demo,
            args.verifier,
            args.max_verification_tool_calls,
        ),
        args.mode,
        args.demo,
        log_dir=args.log_dir,
    )
    Handler.app = app
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Research Agent browser UI: http://{args.host}:{args.port}")
    print(f"Mode: {args.mode} | Verifier: {args.verifier} | Demo: {args.demo} | Ctrl-C to stop")
    if not args.demo and args.mode in {"graph", "graph_baseline", "graph_context", "graph_mcp"}:
        start_local_rag_warmup()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
