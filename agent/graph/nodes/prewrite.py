"""Pre-writing Evidence quality gate and Claim planning nodes."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..context import WorkflowContext
from ..state import Evidence, PlannedClaim, ResearchState
from .claims import enrich_evidence
from .common import extract_response_message, trace


_REFERENCE_MARKERS = re.compile(
    r"(\breferences\b|\bbibliography\b|arxiv:|doi:|proceedings of|conference on)",
    re.IGNORECASE,
)


def _json_response(context: WorkflowContext, prompt: str) -> Mapping[str, Any] | None:
    try:
        response = context.llm.chat(
            messages=[
                {"role": "system", "content": "You plan evidence-grounded Claims. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            tools=None,
        )
        content = str(extract_response_message(response).get("content", "") or "").strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I | re.S).strip()
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end >= start:
            parsed = json.loads(content[start:end + 1])
            return parsed if isinstance(parsed, Mapping) else None
    except Exception:
        return None
    return None


def _is_reference_chunk(content: str) -> bool:
    markers = len(_REFERENCE_MARKERS.findall(content))
    numbered = len(re.findall(r"\[\d{1,4}\]", content))
    urls = len(re.findall(r"https?://", content))
    return "references" in content[:180].lower() or markers >= 2 or (numbered >= 2 and urls >= 1)


def assess_evidence_quality(item: Evidence) -> tuple[str, float, bool, str]:
    """Classify whether a retrieval result is usable as Claim support."""

    if item.get("error"):
        return "rejected", 0.0, False, "Tool returned an error."
    content = str(item.get("content", "")).strip()
    source_type = str(item.get("source_type", ""))
    if len(content) < 60:
        return "rejected", 0.1, False, "Evidence content is too short to support a Claim."
    if source_type == "local_search":
        if _is_reference_chunk(content):
            return "rejected", 0.15, False, "Local chunk appears to be bibliography/reference material."
        return "primary", 0.9, True, "Page-addressable local PDF passage."
    if source_type == "paper_search":
        return "secondary", 0.6, True, "Paper abstract is usable only for contribution-level, conditional Claims."
    if source_type == "web_search":
        return "discovery", 0.25, False, "Search-result snippet discovers a source but is not direct support."
    if source_type == "calculator":
        return "primary", 0.98, True, "Deterministic calculator result."
    return "secondary", 0.5, True, "Unclassified source; use with limited Claim strength."


def make_evidence_quality_node(context: WorkflowContext):
    del context

    def evidence_quality(state: ResearchState) -> dict[str, Any]:
        qualified: list[Evidence] = []
        checks: list[dict[str, object]] = []
        seen: set[str] = set()
        for raw in state.get("accepted_evidence", state.get("evidence", [])):
            item = enrich_evidence(dict(raw))
            evidence_id = str(item.get("evidence_id", ""))
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            role, score, passed, reason = assess_evidence_quality(item)
            item["evidence_role"] = role  # type: ignore[typeddict-item]
            item["quality_score"] = score
            item["quality_passed"] = passed
            item["quality_reason"] = reason
            checks.append({
                "evidence_id": evidence_id,
                "source": str(item.get("source", "")),
                "page": item.get("page", ""),
                "role": role,
                "score": score,
                "passed": passed,
                "reason": reason,
            })
            if passed:
                qualified.append(item)
        return {
            "qualified_evidence": qualified,
            "evidence_quality_checks": checks,
            "graph_trace": trace(
                "[Node] EvidenceQualityGate",
                f"Accepted Input: {len(seen)}",
                f"Qualified Support: {len(qualified)}",
                f"Rejected/Discovery Only: {len(seen) - len(qualified)}",
                *[
                    f"{item['evidence_id']} [{item['role']}/{item['score']}]: {item['reason']}"
                    for item in checks
                ],
                "[Edge] EvidenceQualityGate -> ClaimPlanner",
            ),
        }

    return evidence_quality


def _fallback_plan(query: str, evidence: Sequence[Evidence]) -> list[PlannedClaim]:
    claims: list[PlannedClaim] = []
    comparison = any(token in query.lower() for token in ("比较", "哪个更好", "compare", "better"))
    if comparison:
        claims.append({
            "claim_id": "PC1",
            "text": "现有材料没有给出这些模型在同一机器人控制任务、数据和训练预算下的直接比较，因此不能宣称某一模型全面最优；结论必须按任务条件给出。",
            "claim_type": "recommendation",
            "importance": 1.0,
            "allowed_strength": "conditional",
            "evidence_ids": [str(item.get("evidence_id", "")) for item in evidence[:5]],
            "conditions": ["区分静态感知、时序理解和机器人控制需求"],
            "rationale": "Cross-paper results do not establish a controlled head-to-head comparison.",
        })
    query_terms = {
        token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", query)
    } | {"representation", "encoder", "predict", "temporal", "control", "robot", "jepa", "dino", "mae"}
    for item in evidence[:5]:
        content = " ".join(str(item.get("content", "")).split())
        sentences = [
            value.strip() for value in re.split(r"(?<=[。！？])|(?<=[.!?])\s+", content)
            if 30 <= len(value.strip()) <= 360
        ]
        sentence = max(
            sentences,
            key=lambda value: sum(term in value.lower() for term in query_terms),
            default="",
        )
        if len(sentence) < 20:
            continue
        role = str(item.get("evidence_role", "secondary"))
        claims.append({
            "claim_id": f"PC{len(claims) + 1}",
            "text": sentence[:300],
            "claim_type": "fact",
            "importance": 0.65,
            "allowed_strength": "direct" if role == "primary" else "conditional",
            "evidence_ids": [str(item.get("evidence_id", ""))],
            "conditions": [],
            "rationale": "Fallback Claim derived from one qualified passage.",
        })
    if not claims:
        claims.append({
            "claim_id": "PC1",
            "text": f"当前合格证据不足，不能对“{query[:180]}”下事实性结论。",
            "claim_type": "inference",
            "importance": 1.0,
            "allowed_strength": "insufficient",
            "evidence_ids": [],
            "conditions": ["需要补充可核查的正文段落，而不是仅提供链接或搜索摘要"],
            "rationale": "EvidenceQualityGate found no passage that can directly support a Claim.",
        })
    return claims[:10]


def make_claim_planner_node(context: WorkflowContext):
    def claim_planner(state: ResearchState) -> dict[str, Any]:
        query = str(state.get("standalone_query") or state.get("query", ""))
        evidence = list(state.get("qualified_evidence", []))
        evidence_map = {str(item.get("evidence_id", "")): item for item in evidence}
        prompt = (
            "Create the Claim contract that the Writer must follow. Return JSON "
            "{\"planned_claims\":[{\"text\":\"...\",\"claim_type\":"
            "\"fact|comparison|inference|recommendation\",\"importance\":0.0,"
            "\"allowed_strength\":\"direct|conditional|insufficient\","
            "\"evidence_ids\":[\"E-...\"],\"conditions\":[\"...\"],\"rationale\":\"...\"}]}. "
            "Use 4-10 Claims. Every evidence_id must exist below. A paper abstract can support only contribution-level "
            "or conditional Claims. A best/greater-than Claim requires directly comparable primary evidence; otherwise "
            "plan a conditional recommendation. Explicitly answer the user's decision question.\n\n"
            f"Query:\n{query}\n\nQualified Evidence:\n" + json.dumps([
                {
                    "evidence_id": evidence_id,
                    "role": item.get("evidence_role"),
                    "quality_score": item.get("quality_score"),
                    "source": item.get("source"),
                    "page": item.get("page"),
                    "url": item.get("url"),
                    "quote": str(item.get("content", ""))[:900],
                }
                for evidence_id, item in list(evidence_map.items())[:12]
            ], ensure_ascii=False)
        )
        parsed = (
            _json_response(context, prompt)
            if os.getenv("RESEARCH_AGENT_LLM_CLAIM_PLANNING", "1").strip().lower()
            in {"1", "true", "yes"}
            else None
        )
        planned: list[PlannedClaim] = []
        valid_types = {"fact", "comparison", "inference", "recommendation"}
        valid_strengths = {"direct", "conditional", "insufficient"}
        for raw in (parsed or {}).get("planned_claims", []) or []:
            if not isinstance(raw, Mapping):
                continue
            text = str(raw.get("text", "")).strip()
            claim_type = str(raw.get("claim_type", "fact")).lower()
            strength = str(raw.get("allowed_strength", "conditional")).lower()
            ids = [str(value) for value in raw.get("evidence_ids", []) if str(value) in evidence_map]
            if not text or claim_type not in valid_types or strength not in valid_strengths:
                continue
            if not ids:
                strength = "insufficient"
            if strength == "direct" and any(
                evidence_map[evidence_id].get("evidence_role") != "primary" for evidence_id in ids
            ):
                strength = "conditional"
            planned.append({
                "claim_id": f"PC{len(planned) + 1}",
                "text": text[:500],
                "claim_type": claim_type,  # type: ignore[typeddict-item]
                "importance": max(0.0, min(float(raw.get("importance", 0.7)), 1.0)),
                "allowed_strength": strength,  # type: ignore[typeddict-item]
                "evidence_ids": ids,
                "conditions": [str(value) for value in raw.get("conditions", []) if str(value).strip()][:5],
                "rationale": str(raw.get("rationale", ""))[:500],
            })
        if not planned:
            planned = _fallback_plan(query, evidence)
        return {
            "planned_claims": planned[:10],
            "graph_trace": trace(
                "[Node] ClaimPlanner",
                f"Planned Claims: {len(planned[:10])}",
                *[
                    f"{item['claim_id']} [{item['claim_type']}/{item['allowed_strength']}]: {item['text']} | Evidence: {', '.join(item['evidence_ids']) or '(none)'}"
                    for item in planned[:10]
                ],
                "[Edge] ClaimPlanner -> Writer",
            ),
        }

    return claim_planner
