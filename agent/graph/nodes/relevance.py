"""Low-cost evidence relevance gate with one controlled retrieval retry."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import ToolMessage

from ..context import WorkflowContext
from ..state import Evidence, PlanStep, ResearchState
from .common import trace


_LATIN_TERM = re.compile(r"[a-zA-Z][a-zA-Z0-9._-]{1,}")
_CJK_CHUNK = re.compile(r"[\u3400-\u9fff]{2,}")
_STOPWORDS = {
    "about", "answer", "core", "find", "for", "from", "information", "latest",
    "look", "paper", "papers", "please", "recommend", "related", "research",
    "search", "show", "some", "the", "this", "what", "which", "who", "with",
}
_GENERIC_CJK = (
    "查找", "推荐", "相关", "核心", "论文", "文献", "回答", "看看", "一些", "信息",
    "内容", "介绍", "分析", "比较", "最新", "官方", "本地", "资料",
)
_TOPIC_EXPANSIONS = {
    "jepa": "joint embedding predictive architecture",
    "v-jepa": "video joint embedding predictive architecture",
    "v-jepa 2": "video joint embedding predictive architecture world model",
    "dino": "self supervised visual representation",
    "dinov3": "self supervised visual representation",
    "patch policy": "robot learning",
    "dank1ng": "danking",
}


def _normalized(text: str) -> str:
    return " ".join(str(text).lower().split())


def _terms(text: str) -> set[str]:
    normalized = _normalized(text)
    terms = {
        token.lower()
        for token in _LATIN_TERM.findall(normalized)
        if token.lower() not in _STOPWORDS
    }
    for chunk in _CJK_CHUNK.findall(normalized):
        cleaned = chunk
        for phrase in _GENERIC_CJK:
            cleaned = cleaned.replace(phrase, " ")
        for segment in cleaned.split():
            if len(segment) <= 4:
                terms.add(segment)
            else:
                terms.update(segment[index:index + 2] for index in range(len(segment) - 1))
    return {term for term in terms if term}


def _evidence_text(item: Evidence) -> str:
    return " ".join(
        str(item.get(key, ""))
        for key in ("title", "content", "source", "url")
    )


def assess_evidence_relevance(
    query: str,
    topic: str,
    evidence: list[Evidence],
) -> tuple[bool, float, str]:
    """Return a transparent relevance decision without another LLM call."""

    usable = [item for item in evidence if not item.get("error")]
    if not usable:
        return False, 0.0, "No usable evidence was returned."

    query_terms = _terms(query)
    topic_terms = _terms(topic)
    expanded = _TOPIC_EXPANSIONS.get(_normalized(topic), "")
    topic_terms |= _terms(expanded)
    compact_topic = re.sub(r"\W+", "", _normalized(topic))
    best = 0.0

    for item in usable:
        text = _evidence_text(item)
        normalized_text = _normalized(text)
        compact_text = re.sub(r"\W+", "", normalized_text)
        evidence_terms = _terms(text)
        query_overlap = len(query_terms & evidence_terms) / max(len(query_terms), 1)
        topic_overlap = len(topic_terms & evidence_terms) / max(len(topic_terms), 1) if topic_terms else 0.0
        exact_topic = bool(compact_topic and compact_topic in compact_text)
        raw_score = item.get("score")
        semantic_hint = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
        score = max(query_overlap, topic_overlap * 0.8, 0.9 if exact_topic else 0.0)
        # Existing local BGE score is a useful hint, but never enough by
        # itself to pass unrelated content.
        if score > 0:
            score = min(1.0, score + min(max(semantic_hint, 0.0), 1.0) * 0.1)
        best = max(best, score)

    passed = best >= 0.25
    if passed:
        return True, round(best, 4), "Evidence matches the resolved query/topic."
    return False, round(best, 4), "Evidence does not match the resolved query/topic."


def _latest_tool_message(state: ResearchState) -> ToolMessage | None:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage):
            return message
    return None


def _retry_tool(state: ResearchState, current_tool: str) -> str:
    decision = dict(state.get("route_decision", {}))
    for name in decision.get("fallback_tools", []) or []:
        candidate = str(name)
        if candidate and candidate != current_tool:
            return candidate
    return current_tool


def _retry_query(state: ResearchState, tool: str) -> str:
    decision = dict(state.get("route_decision", {}))
    candidate = str(dict(decision.get("search_queries", {})).get(tool, "")).strip()
    topic = str(state.get("resolved_topic", "")).strip()
    standalone = str(state.get("standalone_query") or state.get("query", "")).strip()
    if candidate:
        return candidate
    if tool == "paper_search":
        return f"{topic} research papers".strip()
    if tool == "web_search":
        return f"{topic or standalone} official authoritative sources".strip()
    return f"{topic} {standalone}".strip()


def make_relevance_node(context: WorkflowContext, *, react_path: bool = False):
    del context  # Reserved for an optional LLM grader on ambiguous scores.

    def relevance_node(state: ResearchState) -> dict[str, Any]:
        message = _latest_tool_message(state)
        call_id = str(message.tool_call_id or "") if message is not None else ""
        tool_name = str(message.name or "tool") if message is not None else "unknown"
        batch = [
            dict(item)
            for item in state.get("evidence", [])
            if str(item.get("tool_call_id", "")) == call_id
        ]
        query = str(state.get("standalone_query") or state.get("query", ""))
        topic = str(state.get("resolved_topic", ""))
        passed, score, reason = assess_evidence_relevance(query, topic, batch)
        retry_count = int(state.get("retrieval_retry_count", 0))
        max_retries = int(state.get("max_retrieval_retries", 1))
        check = {
            "tool": tool_name,
            "query": query,
            "topic": topic,
            "score": score,
            "passed": passed,
            "reason": reason,
            "retry_count": retry_count,
            "evidence_count": len(batch),
            "tool_call_id": call_id,
        }
        updates: dict[str, Any] = {
            "relevance_checks": [check],
            "relevance_passed": passed,
            "relevance_feedback": reason,
            "retry_query": "",
            "retry_tool": "",
        }
        lines = trace(
            "[Node] RelevanceGate",
            f"Tool: {tool_name}",
            f"Resolved Query: {query}",
            f"Topic: {topic or '(none)'}",
            f"Score: {score:.4f}",
            f"Passed: {passed}",
            f"Reason: {reason}",
        )

        if passed:
            updates["accepted_evidence"] = batch
        elif not react_path and int(state.get("current_step", 0)) < len(state.get("plan", [])):
            # Multi-source plans (for example local_search + paper_search)
            # already contain the next independent source. Do not insert a
            # duplicate fallback or spend the retry budget.
            next_step = state.get("plan", [])[int(state.get("current_step", 0))]
            lines.extend((
                "Controlled Retry: not needed; another planned source remains.",
                f"Next Planned Tool: {next_step.get('tool', 'unknown')}",
            ))
        elif not react_path and state.get("accepted_evidence"):
            # One source in a deliberate multi-source plan may fail while a
            # previous source already passed. Repeating either source adds no
            # value and can produce duplicate tool calls.
            lines.append("Controlled Retry: not needed; another planned source already passed.")
        elif retry_count < max_retries:
            retry_tool = _retry_tool(state, tool_name)
            retry_query = _retry_query(state, retry_tool)
            updates["retrieval_retry_count"] = retry_count + 1
            updates["retry_query"] = retry_query
            updates["retry_tool"] = retry_tool
            lines.extend((
                f"Controlled Retry: {retry_count + 1}/{max_retries}",
                f"Retry Tool: {retry_tool}",
                f"Retry Query: {retry_query}",
            ))
            if not react_path:
                plan = [dict(step) for step in state.get("plan", [])]
                insertion = int(state.get("current_step", 0))
                retry_step: PlanStep = {
                    "id": insertion + 1,
                    "task": retry_query,
                    "tool": retry_tool,
                    "status": "pending",
                    "retry": True,
                }
                plan.insert(insertion, retry_step)
                for index, step in enumerate(plan, start=1):
                    step["id"] = index
                updates["plan"] = plan
                open_questions = list(state.get("open_questions", []))
                if retry_query not in open_questions:
                    open_questions.append(retry_query)
                updates["open_questions"] = open_questions
        else:
            lines.append(f"Retry Budget Exhausted: {retry_count}/{max_retries}")

        if react_path:
            next_edge = "ReAct"
        else:
            plan = list(updates.get("plan", state.get("plan", [])))
            next_edge = "Researcher" if int(state.get("current_step", 0)) < len(plan) else "Writer"
        lines.append(f"[Conditional Edge] RelevanceGate -> {next_edge}")
        updates["graph_trace"] = lines
        return updates

    return relevance_node
