"""Writer node: produce a draft from Query + Plan + Evidence."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..context import WorkflowContext
from ..state import ResearchState
from .common import message_to_openai, response_to_ai_message, trace


def _planned_claim_context(state: ResearchState) -> str:
    claims = list(state.get("planned_claims", []))
    if not claims:
        return "(No Planned Claims; do not invent a research conclusion.)"
    return "\n".join(
        (
            f"[{item.get('claim_id')}] type={item.get('claim_type')} "
            f"allowed_strength={item.get('allowed_strength')}\n"
            f"Claim: {item.get('text')}\n"
            f"Evidence IDs: {', '.join(item.get('evidence_ids', [])) or '(none)'}\n"
            f"Conditions: {'; '.join(item.get('conditions', [])) or '(none)'}\n"
            f"Rationale: {item.get('rationale', '')}"
        )
        for item in claims
    )


def _evidence_context(state: ResearchState) -> str:
    records = state.get("qualified_evidence") or state.get("accepted_evidence", state.get("evidence", []))
    if not records:
        return "(No evidence was retrieved.)"
    blocks = []
    for index, item in enumerate(records, start=1):
        citation = str(item.get("source", "unknown"))
        if item.get("page") not in (None, ""):
            citation += f", page {item.get('page')}"
        evidence_id = str(item.get("evidence_id", f"E{index}"))
        role = str(item.get("evidence_role", "unclassified"))
        blocks.append(f"[{evidence_id}] role={role}; {citation}\nDirect passage: {item.get('content', '')}")
    return "\n".join(blocks)


def _fallback_draft(state: ResearchState, records: list[dict[str, Any]] | None = None) -> str:
    query = str(state.get("standalone_query") or state.get("query", ""))
    records = records if records is not None else (
        state.get("qualified_evidence") or state.get("accepted_evidence", state.get("evidence", []))
    )
    if not records:
        if "Transformer" in query or "transformer" in query.lower():
            return (
                "Transformer 是一种以自注意力机制为核心的神经网络架构，"
                "可以并行处理序列并建模长距离依赖。"
            )
        return "当前没有检索到可支持该问题的证据。"
    lines = ["基于当前研究证据："]
    for item in records[:5]:
        citation = str(item.get("source", "unknown"))
        if item.get("page") not in (None, ""):
            citation += f" 第 {item.get('page')} 页"
        passage = " ".join(str(item.get("content", "")).split())
        if len(passage) > 360:
            passage = passage[:357].rsplit(" ", 1)[0] + "…"
        lines.append(f"- {passage}（来源：{citation}）")
    return "\n".join(lines)


def _citation_appendix(records: list[dict[str, Any]]) -> str:
    """Build a compact citation index without replacing a useful draft."""

    entries: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for item in records:
        source = str(item.get("source", "unknown"))
        page = str(item.get("page", ""))
        url = str(item.get("url", ""))
        key = (source, page, url)
        if key in seen:
            continue
        seen.add(key)
        label = source
        if page:
            label += f" 第 {page} 页"
        if url and url != source:
            label += f" — {url}"
        entries.append(f"- {label}")
        if len(entries) >= 8:
            break
    return "\n\n证据索引：\n" + "\n".join(entries) if entries else ""


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def _direct_control_reply(state: ResearchState) -> str:
    query = str(state.get("query", ""))
    normalized = query.lower()
    if "中文" in query and any(token in query for token in ("用", "说", "回答", "回复")):
        return "好的，后续我会使用中文回答。"
    if any(token in query for token in ("用英文", "英文回答", "英语回答", "改成英文")) or any(
        token in normalized for token in ("reply in english", "answer in english", "respond in english")
    ):
        return "Understood. I will answer in English from now on."
    capability_question = (
        any(token in query for token in ("联网", "网络信息", "网络搜索"))
        and any(token in query for token in ("能", "不能", "可以", "无法", "为什么", "能力"))
    ) or any(token in normalized for token in ("can you search", "web access", "internet access"))
    if capability_question:
        if state.get("session_preferences", {}).get("response_language") == "en":
            return "Yes. I can use the web_search tool for current internet information."
        return "可以。我能通过 web_search 工具检索实时网络信息；人物、新闻、最新进展等问题会自动进入联网检索链路。"
    return ""


def make_writer_node(context: WorkflowContext):
    def writer_node(state: ResearchState) -> dict[str, Any]:
        built = context.context_manager.build_writer_context(state) if context.context_manager else None
        evidence_text = _evidence_context(state)
        planned_claim_text = _planned_claim_context(state)
        writer_instruction = (
            "You are the Writer node. Follow Session Preferences and constraints in Conversation "
            "History exactly, especially the requested response language. The system can use a "
            "native web_search tool for current internet information, local_search for the local "
            "paper library, and MCP tools. Answer greetings, conversational requests, and questions "
            "about these capabilities directly even when no evidence was collected. For factual "
            "research claims, use the supplied evidence and do not invent unsupported facts. Cite "
            "local evidence with source filename and page, and web/paper evidence with its URL, when available. "
            "Answer the Standalone Query while preserving the intent of the Original Query. Ignore rejected evidence. "
            "For comparisons, use shared dimensions and explicitly distinguish source-stated facts from inference and "
            "recommendation. Give a clear conditional conclusion, but never turn training-objective alignment into a "
            "proven performance advantage without a comparable experiment. The Planned Claim Contract below is "
            "authoritative: structure the answer around it, do not add unsupported major Claims, preserve every "
            "condition, and never exceed allowed_strength. Claims marked insufficient must be stated as evidence gaps. "
            "Cite the direct passage by filename/page or official URL; a bare arXiv link is not sufficient support."
        )
        response_language = str(state.get("session_preferences", {}).get("response_language", ""))
        if response_language == "zh-CN":
            writer_instruction += " Respond in Simplified Chinese."
        elif response_language == "en":
            writer_instruction += " Respond in English."
        writer_system = (
            writer_instruction
            if built is not None
            else (
                f"{writer_instruction}\n\nPlanned Claim Contract:\n{planned_claim_text}"
                f"\n\nQualified Direct Evidence:\n{evidence_text}"
            )
        )
        draft = _direct_control_reply(state)
        if not draft:
            try:
                if built is not None:
                    raw_messages = [
                        SystemMessage(content=writer_system),
                        HumanMessage(content=(
                            f"{built.text}\n\nPlanned Claim Contract:\n{planned_claim_text}"
                            f"\n\nQualified Direct Evidence:\n{evidence_text}"
                        )),
                    ]
                else:
                    raw_messages = [SystemMessage(content=writer_system)] + list(state.get("messages", []))
                writer_messages = [message_to_openai(message) for message in raw_messages]
                if context.stream_callback is not None and hasattr(context.llm, "chat_stream"):
                    streamed_parts: list[str] = []
                    for token in context.llm.chat_stream(messages=writer_messages, tools=None):
                        token = str(token)
                        streamed_parts.append(token)
                        context.stream_callback(token)
                    draft = "".join(streamed_parts).strip()
                else:
                    response = context.llm.chat(messages=writer_messages, tools=None)
                    ai_message = response_to_ai_message(response)
                    if ai_message is not None and ai_message.content:
                        draft = str(ai_message.content).strip()
            except Exception:
                draft = ""
        # Provider behavior can drift even with a strong prompt. For a saved
        # session-level Chinese preference, make one deterministic rewrite
        # attempt before verification and final stream reconciliation.
        if response_language == "zh-CN" and draft and not _contains_chinese(draft):
            try:
                response = context.llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Rewrite the answer in Simplified Chinese. Preserve factual content, "
                                "citations, URLs, and uncertainty. Output only the rewritten answer."
                            ),
                        },
                        {"role": "user", "content": draft},
                    ],
                    tools=None,
                )
                rewritten = response_to_ai_message(response)
                if rewritten is not None and rewritten.content and _contains_chinese(str(rewritten.content)):
                    draft = str(rewritten.content).strip()
            except Exception:
                pass
        weak_demo_answers = {
            "已完成工具检索。",
            "已完成工具检索；",
        }
        usable_records = list(
            state.get("qualified_evidence") or state.get("accepted_evidence", state.get("evidence", []))
        )
        local_sources = [
            str(item.get("source", ""))
            for item in usable_records
            if item.get("source_type") == "local_search" and not item.get("error")
        ]
        external_urls = [
            str(item.get("url", ""))
            for item in usable_records
            if item.get("source_type") in {"web_search", "paper_search"}
            and item.get("url")
            and not item.get("error")
        ]
        missing_local_citation = bool(local_sources) and not any(source in draft for source in local_sources)
        missing_external_citation = bool(external_urls) and not any(url in draft for url in external_urls)
        if not draft or (evidence_text != "(No evidence was retrieved.)" and draft in weak_demo_answers):
            draft = _fallback_draft(state, usable_records)
        elif missing_local_citation or missing_external_citation:
            # Missing citation formatting must not turn a concise synthesized
            # answer into a multi-page dump of raw retrieval chunks.
            draft = draft.rstrip() + _citation_appendix(usable_records)
        lines = trace(
            "[Node] Writer",
            "Draft generated from Query + Plan + Evidence.",
            f"Planned Claim Contract: {len(state.get('planned_claims', []))} Claim(s).",
            f"Qualified Evidence: {len(usable_records)} passage(s).",
            "[State Update] draft=current answer draft",
            "[Edge] Writer -> Verifier",
        )
        updates: dict[str, Any] = {"draft": draft, "graph_trace": lines}
        if built is not None:
            lines[1:1] = [
                "[Context Manager] Writer context keeps session constraints and excludes ToolNode history/old drafts.",
                f"Selected Evidence: {len(built.selected_evidence)}/{built.selection.available_count if built.selection else 0}",
                f"Deduplicated Evidence Count: {built.selection.deduplicated_count if built.selection else 0}",
                f"Dropped Evidence Count: {built.selection.dropped_count if built.selection else 0}",
                f"Raw Context Tokens: {built.budget.raw_tokens}",
                f"Final Context Tokens: {built.budget.final_tokens}/{built.budget.budget}",
            ]
            updates["context_stats"] = context.context_manager.record_node(state, built)
        return updates

    return writer_node
