"""Direct-answer node for queries that need no tools."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from ..context import WorkflowContext
from ..state import ResearchState
from .common import message_to_openai, response_to_ai_message, trace


def _language_instruction(state: ResearchState) -> str:
    language = str(state.get("session_preferences", {}).get("response_language", ""))
    if language == "zh-CN":
        return "Respond in Simplified Chinese."
    if language == "en":
        return "Respond in English."
    return ""


def make_direct_node(context: WorkflowContext):
    def direct_node(state: ResearchState) -> dict[str, Any]:
        instruction = (
            "You are the direct-answer node. Answer the current user query concisely. "
            "Do not call tools, do not invent current facts, and follow the session constraints."
        )
        language = _language_instruction(state)
        if language:
            instruction += " " + language
        messages = [
            {"role": "system", "content": instruction},
            *[message_to_openai(message) for message in state.get("messages", [])],
        ]
        draft = ""
        ai_message: AIMessage | None = None
        try:
            if context.stream_callback is not None and hasattr(context.llm, "chat_stream"):
                streamed_parts: list[str] = []
                for token in context.llm.chat_stream(messages=messages, tools=None):
                    token = str(token)
                    if token:
                        streamed_parts.append(token)
                        context.stream_callback(token)
                draft = "".join(streamed_parts).strip()
                if draft:
                    ai_message = AIMessage(content=draft)
            else:
                ai_message = response_to_ai_message(context.llm.chat(messages=messages, tools=None))
                if ai_message is not None:
                    draft = str(ai_message.content or "").strip()
        except Exception as exc:
            draft = f"AgentError: {type(exc).__name__}: {exc}"
        if not draft:
            draft = "AgentError: LLM returned an empty direct answer."
            ai_message = AIMessage(content=draft)
        return {
            "draft": draft,
            "messages": [ai_message or AIMessage(content=draft)],
            "verification_passed": True,
            "verification_feedback": "Skipped by Router: direct answer.",
            "graph_trace": trace(
                "[Node] DirectAnswer",
                "No planning or tools required.",
                "[State Update] draft=current direct answer",
                "[Edge] DirectAnswer -> END",
            ),
        }

    return direct_node
