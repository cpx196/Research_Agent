"""Deterministic answer verifier with explicit quality dimensions."""

from __future__ import annotations

import re
from typing import Any

from ..context import WorkflowContext
from ..state import ResearchState
from .common import trace


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def _topic_aligned(topic: str, draft: str, evidence: list[dict[str, Any]]) -> bool:
    if not topic:
        return True
    compact_topic = re.sub(r"\W+", "", topic.lower())
    compact_draft = re.sub(r"\W+", "", draft.lower())
    if compact_topic and compact_topic in compact_draft:
        return True
    # A canonical name can differ from a retrieved alias (DANK1NG/Danking,
    # JEPA/full expansion), so accepted evidence may establish the alignment.
    evidence_text = " ".join(
        str(item.get(key, ""))
        for item in evidence
        for key in ("title", "content", "source")
    ).lower()
    topic_tokens = set(re.findall(r"[a-zA-Z0-9_-]{2,}|[\u3400-\u9fff]{2,}", topic.lower()))
    return bool(topic_tokens and any(token in draft.lower() and token in evidence_text for token in topic_tokens))


def _citations_present(draft: str, evidence: list[dict[str, Any]]) -> bool:
    if not evidence:
        return True
    anchors: list[str] = []
    for item in evidence:
        if item.get("url"):
            anchors.append(str(item["url"]))
        elif item.get("source") and item.get("source") not in {"web_search", "paper_search", "local_search"}:
            anchors.append(str(item["source"]))
    return not anchors or any(anchor in draft for anchor in anchors)


def make_verifier_node(context: WorkflowContext):
    def verifier_node(state: ResearchState) -> dict[str, Any]:
        draft = str(state.get("draft", "")).strip()
        plan = list(state.get("plan", []))
        evidence = [dict(item) for item in state.get("accepted_evidence", state.get("evidence", []))]
        checks = list(state.get("relevance_checks", []))
        iteration = int(state.get("iteration", 0)) + 1
        max_iterations = int(state.get("max_iterations", context.max_iterations))
        topic = str(state.get("resolved_topic", ""))
        response_language = str(state.get("session_preferences", {}).get("response_language", ""))
        built = context.context_manager.build_verifier_context(state) if context.context_manager else None

        details: dict[str, bool] = {
            "draft_present": bool(draft),
            "evidence_available": not plan or bool(evidence),
            "evidence_relevant": not plan or bool(evidence) and (not checks or any(bool(item.get("passed")) for item in checks)),
            "intent_aligned": _topic_aligned(topic, draft, evidence),
            "citations_present": _citations_present(draft, evidence),
            "language_correct": response_language != "zh-CN" or _contains_chinese(draft),
        }

        if context.verifier_override is not None:
            passed, feedback = context.verifier_override(state)
        elif iteration <= context.forced_verifier_failures:
            passed = False
            feedback = "Forced test failure: repair the answer using accepted evidence."
        else:
            failed = [name for name, value in details.items() if not value]
            passed = not failed
            feedback = (
                "All verification dimensions passed."
                if passed
                else "Failed verification dimensions: " + ", ".join(failed)
            )

        updates: dict[str, Any] = {
            "verification_passed": passed,
            "verification_feedback": feedback,
            "verification_details": details,
            "iteration": iteration,
        }
        next_edge = "END" if passed or iteration >= max_iterations else "AnswerRepair"
        lines = trace(
            "[Node] Verifier",
            f"Intent Aligned: {details['intent_aligned']}",
            f"Evidence Relevant: {details['evidence_relevant']}",
            f"Citations Present: {details['citations_present']}",
            f"Language Correct: {details['language_correct']}",
            f"Passed: {passed}",
            f"Feedback: {feedback}",
            f"[State Update] iteration={iteration}, verification_passed={passed}",
            f"[Conditional Edge] Verifier -> {next_edge}",
        )
        if built is not None:
            lines[1:1] = [
                "[Context Manager] Verifier context contains resolved intent + Draft + accepted Evidence.",
                f"Selected Evidence: {len(built.selected_evidence)}/{built.selection.available_count if built.selection else 0}",
                f"Deduplicated Evidence Count: {built.selection.deduplicated_count if built.selection else 0}",
                f"Dropped Evidence Count: {built.selection.dropped_count if built.selection else 0}",
                f"Raw Context Tokens: {built.budget.raw_tokens}",
                f"Final Context Tokens: {built.budget.final_tokens}/{built.budget.budget}",
            ]
            updates["context_stats"] = context.context_manager.record_node(state, built)
        if not passed and iteration >= max_iterations:
            if not evidence:
                updates["draft"] = "当前没有检索到与问题足够相关、可验证的证据，因此暂时无法给出可靠答案。"
            else:
                updates["draft"] = (
                    f"{draft}\n\n[Verifier] 已达到最大修复轮次（{iteration}），"
                    "以上回答可能仍有未通过的质量检查。"
                ).strip()
        updates["graph_trace"] = lines
        return updates

    return verifier_node
