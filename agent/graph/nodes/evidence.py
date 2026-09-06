"""Convert ToolNode messages into explicit Evidence records."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import ToolMessage

from ..context import WorkflowContext
from ..state import Evidence, ResearchState
from .common import trace
from .claims import enrich_evidence


_LOCAL_RESULT = re.compile(
    r"\[Result\s+(?P<number>\d+)\]\s*\n"
    r"Source:\s*(?P<source>.*?)\s*\n"
    r"Page:\s*(?P<page>.*?)\s*\n"
    r"Score:\s*(?P<score>[-+]?\d*\.?\d+)\s*\n"
    r"Text:\s*(?P<text>.*?)(?=\n\n\[Result\s+\d+\]|\Z)",
    re.DOTALL,
)


def _latest_tool_message(state: ResearchState) -> ToolMessage | None:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage):
            return message
    return None


def parse_tool_evidence(tool_name: str, content: str, query: str, call_id: str) -> list[Evidence]:
    if content.startswith("ToolError:") or "\n[Observation]\nToolError:" in content:
        return [
            {
                "source_type": tool_name,
                "source": "tool-error",
                "query": query,
                "content": content,
                "tool_call_id": call_id,
                "error": True,
            }
        ]

    if tool_name == "local_search":
        parsed: list[Evidence] = []
        for match in _LOCAL_RESULT.finditer(content):
            try:
                page: int | str = int(match.group("page"))
            except ValueError:
                page = match.group("page").strip()
            parsed.append(
                {
                    "source_type": "local_search",
                    "source": match.group("source").strip(),
                    "page": page,
                    "query": query,
                    "content": match.group("text").strip(),
                    "score": float(match.group("score")),
                    "tool_call_id": call_id,
                }
            )
        if parsed:
            return parsed
        # Level-2 LLM extraction may return a concise citation block instead
        # of the original [Result N] envelope.  Keep its source/page anchors
        # rather than collapsing it into an uncited generic record.
        fields = {}
        for line in content.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip().lower()] = value.strip()
        source = fields.get("source", "")
        page = fields.get("page", "")
        extracted_text = fields.get("text") or fields.get("claim") or fields.get("evidence") or content
        if source and page:
            try:
                page_value: int | str = int(page)
            except ValueError:
                page_value = page
            item: Evidence = {
                "source_type": "local_search",
                "source": source,
                "page": page_value,
                "query": query,
                "content": extracted_text,
                "tool_call_id": call_id,
            }
            if "score" in fields:
                try:
                    item["score"] = float(fields["score"])
                except ValueError:
                    pass
            return [item]

    fields = {"title": "", "url": "", "snippet": "", "abstract": ""}
    if tool_name == "web_search":
        blocks = content.split("\n\n")
        result: list[Evidence] = []
        for block in blocks:
            values = dict(fields)
            for line in block.splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip().lower() in values:
                    values[key.strip().lower()] = value.strip()
            if any(values.values()):
                result.append(
                    {
                        "source_type": tool_name,
                        "source": values["url"] or values["title"] or tool_name,
                        "title": values["title"],
                        "url": values["url"],
                        "query": query,
                        "content": values["snippet"],
                        "tool_call_id": call_id,
                    }
                )
        if result:
            return result

    if tool_name == "paper_search":
        result = []
        for block in content.split("\n\n"):
            values = {"title": "", "authors": "", "published": "", "arxiv": "", "abstract": ""}
            for line in block.splitlines():
                key, separator, value = line.partition(":")
                normalized_key = key.strip().lower()
                if separator and normalized_key in values:
                    values[normalized_key] = value.strip()
            if values["title"] or values["abstract"]:
                result.append(
                    {
                        "source_type": "paper_search",
                        "source": values["arxiv"] or values["title"] or "paper_search",
                        "title": values["title"],
                        "url": values["arxiv"],
                        "query": query,
                        "content": values["abstract"] or values["title"],
                        "authors": [
                            author.strip() for author in values["authors"].split(",") if author.strip()
                        ],
                        "published_at": values["published"],
                        "tool_call_id": call_id,
                    }
                )
        if result:
            return result

    return [
        {
            "source_type": tool_name,
            "source": tool_name,
            "query": query,
            "content": content,
            "tool_call_id": call_id,
        }
    ]


def make_evidence_node(context: WorkflowContext):
    def evidence_node(state: ResearchState) -> dict[str, Any]:
        message = _latest_tool_message(state)
        compression = None
        current_step = int(state.get("current_step", 0))
        plan = list(state.get("plan", []))
        evidence_query = (
            str(plan[current_step].get("task", ""))
            if plan and current_step < len(plan)
            else str(state.get("standalone_query") or state.get("query", ""))
        )
        if message is None:
            evidence: list[Evidence] = []
            tool_name = "unknown"
            content = "No ToolNode result was produced."
            call_id = ""
        else:
            tool_name = str(message.name or "tool")
            content = str(message.content)
            call_id = str(message.tool_call_id or "")
            if context.context_manager is not None:
                compression = context.context_manager.compress_tool_result(
                    content,
                    tool_name=tool_name,
                    query=evidence_query,
                )
                content = compression.text
            evidence = [
                enrich_evidence(item)
                for item in parse_tool_evidence(tool_name, content, evidence_query, call_id)
            ]

        if plan and current_step < len(plan):
            updated_plan = [dict(step) for step in plan]
            updated_plan[current_step]["status"] = "completed"
            open_questions = list(state.get("open_questions", []))
            completed_question = str(updated_plan[current_step].get("task", ""))
            if completed_question in open_questions:
                open_questions.remove(completed_question)
        else:
            updated_plan = plan
            open_questions = list(state.get("open_questions", []))
        lines = trace(
            "[Node] Evidence",
            f"Tool: {tool_name}",
            f"Retrieval Query: {evidence_query}",
            f"Evidence records added: {len(evidence)}",
        )
        if compression is not None:
            lines[1:1] = [
                "[Context Manager] Tool Result Compression",
                f"Raw Context Tokens: {compression.raw_tokens}",
                f"Final Context Tokens: {compression.final_tokens}",
                f"Compressed: {compression.compressed}",
                f"Compression Ratio: {compression.ratio:.1%}",
            ]
        for item in evidence[:3]:
            lines.append(
                f"Source: {item.get('source', 'unknown')} | Page: {item.get('page', 'n/a')}"
            )
        lines.extend(
            (
                "[State Update] evidence += normalized ToolNode result",
                "[Edge] Evidence -> RelevanceGate",
            )
        )
        updates: dict[str, Any] = {
            "evidence": evidence,
            "plan": updated_plan,
            "open_questions": open_questions,
            "current_step": current_step + 1 if plan else current_step,
            "graph_trace": lines,
        }
        if compression is not None:
            updates["context_stats"] = context.context_manager.record_compression(
                state, tool=tool_name, result=compression
            )
        return updates

    return evidence_node
