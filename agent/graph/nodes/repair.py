"""Bounded, Claim-level repair that preserves unaffected answer spans."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from ..context import WorkflowContext
from ..state import ClaimAudit, ResearchState
from .common import extract_response_message, trace


def _format_evidence(state: ResearchState) -> str:
    records = list(state.get("accepted_evidence", state.get("evidence", [])))
    records.extend(state.get("verification_evidence", []))
    if not records:
        return "(No accepted evidence.)"
    return "\n".join(
        f"[{item.get('evidence_id', index)}] {item.get('source', 'unknown')}: {item.get('content', '')}"
        for index, item in enumerate(records, start=1)
    )


def _json_object(response: Any) -> Mapping[str, Any] | None:
    content = str(extract_response_message(response).get("content", "") or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I | re.S).strip()
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _fallback_repair(original: str) -> str:
    bare = original.rstrip("。.!！")
    return f"{bare}；但当前证据不足以支持这一结论作为无条件判断。"


def _apply_exact_replacements(draft: str, replacements: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Apply only exact Claim substitutions; all unrelated bytes remain unchanged."""

    applicable: list[tuple[int, str, str, str]] = []
    for item in replacements:
        original = str(item.get("original", ""))
        replacement = str(item.get("replacement", ""))
        action = str(item.get("action", "QUALIFY")).upper()
        start = draft.find(original)
        if start >= 0 and original and replacement != original and (replacement or action == "REMOVE"):
            applicable.append((start, original, replacement, action))
    result = draft
    applied: list[dict[str, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, original, replacement, action in sorted(applicable, reverse=True):
        end = start + len(original)
        if any(not (end <= left or start >= right) for left, right in occupied):
            continue
        result = result[:start] + replacement + result[end:]
        occupied.append((start, end))
        applied.append({"original": original, "replacement": replacement, "action": action})
    return result, list(reversed(applied))


def make_repair_node(context: WorkflowContext):
    def repair_node(state: ResearchState) -> dict[str, Any]:
        draft = str(state.get("draft", "")).strip()
        feedback_items = [dict(item) for item in state.get("verifier_feedback", [])]
        evidence = _format_evidence(state)
        prompt = (
            "Repair only the failed Claims. Return JSON only: "
            "{\"replacements\":[{\"original\":\"exact substring from draft\","
            "\"replacement\":\"localized evidence-grounded replacement\","
            "\"action\":\"STRENGTHEN|QUALIFY|REPLACE|REMOVE\"}]}. "
            "Never rewrite unaffected text. Prefer QUALIFY when direct comparable evidence is missing. "
            "Preserve citations and the requested response language.\n\n"
            f"Feedback:\n{json.dumps(feedback_items, ensure_ascii=False)}\n\n"
            f"Draft:\n{draft}\n\nEvidence:\n{evidence}"
        )
        parsed = None
        try:
            parsed = _json_object(context.llm.chat(
                messages=[
                    {"role": "system", "content": "You perform localized Claim repair. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
            ))
        except Exception:
            parsed = None

        replacements: list[dict[str, str]] = []
        for item in (parsed or {}).get("replacements", []) or []:
            if not isinstance(item, Mapping):
                continue
            action = str(item.get("action", "QUALIFY")).upper()
            if action not in {"STRENGTHEN", "QUALIFY", "REPLACE", "REMOVE"}:
                action = "QUALIFY"
            original = str(item.get("original", ""))
            replacement = str(item.get("replacement", ""))
            if action == "REMOVE" and original and not replacement:
                replacement = ""
            replacements.append({"original": original, "replacement": replacement, "action": action})

        if not replacements:
            for item in feedback_items:
                original = str(item.get("claim", "")).strip()
                if original and original in draft:
                    replacements.append({
                        "original": original,
                        "replacement": _fallback_repair(original),
                        "action": "QUALIFY",
                    })

        repaired, applied = _apply_exact_replacements(draft, replacements)
        claims = [dict(item) for item in state.get("claims", [])]
        audits: list[ClaimAudit] = []
        for feedback in feedback_items:
            original = str(feedback.get("claim", ""))
            claim = next((item for item in claims if item.get("text") == original), None)
            applied_item = next((item for item in applied if item["original"] == original), None)
            claim_id = str(claim.get("claim_id", "")) if claim else ""
            final_claim = applied_item["replacement"] if applied_item else original
            action = applied_item["action"] if applied_item else ""
            if claim is not None:
                claim["repair_action"] = action
                claim["repaired_text"] = final_claim if applied_item else ""
            audits.append({
                "claim_id": claim_id,
                "original_claim": original,
                "final_claim": final_claim,
                "original_status": str(claim.get("status", "UNSUPPORTED")) if claim else "UNSUPPORTED",
                "final_status": "PARTIALLY_SUPPORTED" if action == "QUALIFY" else "SUPPORTED" if action else "UNSUPPORTED",
                "problem": str(feedback.get("problem", "Evidence remains insufficient.")),
                "verification_question": str(claim.get("verification_question", "")) if claim else "",
                "new_evidence_ids": [
                    str(item.get("evidence_id", ""))
                    for item in state.get("verification_evidence", [])
                    if not claim_id or item.get("claim_id") == claim_id
                ],
                "repair_action": action,
                "changed": bool(applied_item),
            })

        next_node = "ClaimExtractor" if state.get("verifier_mode") == "active" else "Verifier"
        return {
            "draft": repaired,
            "claims": claims,
            "claim_audits": audits,
            "potential_errors": [],
            "verification_questions": [],
            "graph_trace": trace(
                "[Node] AnswerRepair",
                f"Target Claims: {len(feedback_items)}",
                f"Localized Replacements: {len(applied)}",
                *[
                    f"Repair Diff [{item['action']}]: - {item['original']} | + {item['replacement']}"
                    for item in applied
                ],
                "[State Update] draft=locally repaired answer; claim_audits += repair records",
                f"[Edge] AnswerRepair -> {next_node}",
            ),
        }

    return repair_node
