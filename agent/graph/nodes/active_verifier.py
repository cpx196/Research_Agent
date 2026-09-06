"""Lightweight active verification workflow for evidence-grounded answers."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from ..context import WorkflowContext
from ..state import Claim, Evidence, PotentialError, ResearchState, VerifierFeedback
from .common import extract_response_message, trace
from .claims import align_claims, enrich_evidence
from .evidence import parse_tool_evidence
from .relevance import assess_evidence_relevance
from .repair import _apply_exact_replacements, _fallback_repair


_HIGH_RISK_CLAIM = re.compile(
    r"(完全|全部|已经|已|首次|唯一|保证|必然|开源|checkpoint|released|fully|always|never)",
    re.IGNORECASE,
)
_RESEARCH_TOOLS = ("web_search", "paper_search", "local_search")


def _json_response(context: WorkflowContext, system: str, prompt: str) -> Mapping[str, Any] | None:
    try:
        response = context.llm.chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tools=None,
        )
        message = extract_response_message(response)
        content = str(message.get("content", "") or "").strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I | re.S).strip()
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end >= start:
            parsed = json.loads(content[start:end + 1])
            return parsed if isinstance(parsed, Mapping) else None
    except (ValueError, TypeError, KeyError):
        return None
    except Exception:
        return None
    return None


def _format_evidence(records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        return "(none)"
    return "\n".join(
        f"[{index}] {item.get('source', 'unknown')}: {item.get('content', '')}"
        for index, item in enumerate(records, start=1)
    )


def _claims(draft: str) -> list[str]:
    result: list[str] = []
    for raw in re.split(r"(?<=[。！？.!?])\s+|\n+", draft):
        claim = raw.strip().lstrip("-*0123456789.、 ")
        if len(claim) >= 10 and claim not in result:
            result.append(claim[:500])
    return result[:12]


def _anchors(records: Sequence[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    for item in records:
        anchor = str(item.get("url") or item.get("source") or "").strip()
        if anchor and anchor not in {"web_search", "paper_search", "local_search"}:
            result.append(anchor)
    return result


def _claim_terms(text: str) -> set[str]:
    terms = {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)}
    for chunk in re.findall(r"[\u3400-\u9fff]{2,}", text):
        if len(chunk) <= 4:
            terms.add(chunk)
        else:
            terms.update(chunk[index:index + 2] for index in range(len(chunk) - 1))
    return terms


def _base_errors(state: ResearchState) -> list[PotentialError]:
    draft = str(state.get("draft", "")).strip()
    evidence = list(state.get("accepted_evidence", []))
    plan = list(state.get("plan", []))
    errors: list[PotentialError] = []
    if not draft:
        return [{"claim": "(empty answer)", "reason": "Writer produced no answer."}]
    if plan and not evidence:
        return [{"claim": draft[:500], "reason": "No accepted evidence supports the research answer."}]

    claims = list(state.get("claims", []))
    if claims:
        for item in claims:
            status = str(item.get("status", "UNSUPPORTED"))
            risk = str(item.get("risk", "low"))
            if status == "UNSUPPORTED" and risk in {"medium", "high"}:
                errors.append({
                    "claim": str(item.get("text", "")),
                    "reason": "This important Claim has no aligned supporting evidence.",
                    "claim_id": str(item.get("claim_id", "")),
                    "status": status,
                })
            elif status == "PARTIALLY_SUPPORTED" and risk == "high":
                errors.append({
                    "claim": str(item.get("text", "")),
                    "reason": "This high-risk Claim is only partially supported and needs qualification or direct comparable evidence.",
                    "claim_id": str(item.get("claim_id", "")),
                    "status": status,
                })
            if not item.get("plan_consistent", True) and risk in {"medium", "high"}:
                errors.append({
                    "claim": str(item.get("text", "")),
                    "reason": "Writer introduced an important Claim outside the pre-writing Claim contract.",
                    "claim_id": str(item.get("claim_id", "")),
                    "status": status,
                })
        unique: list[PotentialError] = []
        seen_claims: set[str] = set()
        for error in errors:
            claim_id = str(error.get("claim_id") or error.get("claim", ""))
            if claim_id and claim_id not in seen_claims:
                seen_claims.add(claim_id)
                unique.append(error)
        return unique[:4]

    anchors = _anchors(evidence)
    if anchors and not any(anchor in draft for anchor in anchors):
        errors.append({
            "claim": draft[:500],
            "reason": "The answer contains factual claims but does not cite the available evidence anchor.",
        })

    evidence_text = _format_evidence(evidence).lower()
    evidence_terms = _claim_terms(evidence_text)
    for claim in _claims(draft):
        if not _HIGH_RISK_CLAIM.search(claim):
            continue
        if any(anchor in claim for anchor in anchors):
            continue
        terms = _claim_terms(claim) - {"已经", "完全", "全部", "开源"}
        overlap = len(terms & evidence_terms)
        if terms and overlap / len(terms) < 0.35:
            errors.append({
                "claim": claim,
                "reason": "This strong factual claim is not clearly supported by the accepted evidence.",
            })
    return errors[:4]


def make_error_detector_node(context: WorkflowContext):
    def error_detector(state: ResearchState) -> dict[str, Any]:
        deterministic = _base_errors(state)
        prompt = (
            "Locate only factual claims that deserve further verification. Do not judge the whole answer. "
            "Return JSON: {\"potential_errors\":[{\"claim\":\"...\",\"reason\":\"...\"}]}. "
            "Use an empty list when the evidence clearly supports the answer.\n\n"
            f"User Query: {state.get('standalone_query') or state.get('query', '')}\n"
            f"Final Answer:\n{state.get('draft', '')}\n\n"
            f"Evidence:\n{_format_evidence(list(state.get('accepted_evidence', [])))}"
        )
        # ClaimExtractor + ClaimEvidenceAligner already performed the semantic
        # audit. Avoid a redundant full-answer LLM call on the focused path;
        # retain the legacy detector only for states without structured Claims.
        parsed = (
            None
            if state.get("claims")
            else _json_response(context, "You are an evidence error detector. Output JSON only.", prompt)
        )
        llm_errors: list[PotentialError] = []
        for item in (parsed or {}).get("potential_errors", []) or []:
            if not isinstance(item, Mapping):
                continue
            claim, reason = str(item.get("claim", "")).strip(), str(item.get("reason", "")).strip()
            if claim and reason:
                llm_errors.append({"claim": claim[:500], "reason": reason[:500]})
        errors = deterministic or llm_errors[:4]
        claims = [dict(item) for item in state.get("claims", [])]
        for error in errors:
            if error.get("claim_id"):
                continue
            matching = next(
                (item for item in claims if str(item.get("text", "")) == str(error.get("claim", ""))),
                None,
            )
            if matching is not None:
                error["claim_id"] = str(matching.get("claim_id", ""))
        return {
            "potential_errors": errors,
            "verification_questions": [],
            "verifier_feedback": [],
            "graph_trace": trace(
                "[Node] VerificationErrorDetector",
                f"Potential Errors: {len(errors)}",
                *[f"Claim: {item['claim']} | Reason: {item['reason']}" for item in errors],
                "[Edge] VerificationErrorDetector -> VerificationQuestionGenerator",
            ),
        }

    return error_detector


def make_verification_question_node(context: WorkflowContext):
    def question_generator(state: ResearchState) -> dict[str, Any]:
        errors = list(state.get("potential_errors", []))
        if not errors:
            questions: list[str] = []
        else:
            questions = [
                f"核查该结论是否有直接、可比的官方或论文证据支持：{item.get('claim', '')}"
                for item in errors[:2]
            ]
            if os.getenv("RESEARCH_AGENT_LLM_VERIFICATION_QUESTIONS", "0").strip().lower() in {
                "1", "true", "yes",
            }:
                prompt = (
                    "Convert each potential error into a concise independently searchable verification question. "
                    "Return JSON: {\"verification_questions\":[\"...\"]}. Generate at most two questions.\n\n"
                    + json.dumps(errors, ensure_ascii=False)
                )
                parsed = _json_response(
                    context, "You generate targeted verification questions. Output JSON only.", prompt
                )
                raw_questions = (parsed or {}).get("verification_questions", []) or []
                generated = [str(item).strip() for item in raw_questions if str(item).strip()][:2]
                if generated:
                    questions = generated
        claims = [dict(item) for item in state.get("claims", [])]
        for index, error in enumerate(errors[:len(questions)]):
            claim_id = str(error.get("claim_id", ""))
            for claim in claims:
                if claim_id and claim.get("claim_id") == claim_id:
                    claim["verification_question"] = questions[index]
                    break
        return {
            "verification_questions": questions,
            "claims": claims,
            "graph_trace": trace(
                "[Node] VerificationQuestionGenerator",
                f"Questions: {len(questions)}",
                *[f"Verification Question: {question}" for question in questions],
                "[Edge] VerificationQuestionGenerator -> VerificationResearcher",
            ),
        }

    return question_generator


def _verification_tools(context: WorkflowContext) -> list[str]:
    names = list(context.tools)
    preferred = [name for name in _RESEARCH_TOOLS if name in names]
    github = [name for name in names if "github" in name.lower()]
    return preferred + [name for name in github if name not in preferred]


def _pick_tool(question: str, available: Sequence[str]) -> str:
    lowered = question.lower()
    if ("github" in lowered or "代码" in question or "仓库" in question) and any(
        "github" in name.lower() for name in available
    ):
        return next(name for name in available if "github" in name.lower())
    if any(token in lowered for token in ("paper", "论文", "arxiv")) and "paper_search" in available:
        return "paper_search"
    if any(token in question for token in ("本地", "UST")) and "local_search" in available:
        return "local_search"
    if "web_search" in available:
        return "web_search"
    return available[0] if available else ""


def _tool_args(context: WorkflowContext, name: str, question: str) -> dict[str, Any]:
    schema = next(
        (item.get("function", {}) for item in context.tool_schemas if item.get("function", {}).get("name") == name),
        {},
    )
    properties = schema.get("parameters", {}).get("properties", {}) if isinstance(schema, Mapping) else {}
    args: dict[str, Any] = {}
    for field in ("query", "q", "search_query"):
        if field in properties:
            args[field] = question
            break
    if "max_results" in properties:
        args["max_results"] = 3
    if "top_k" in properties:
        args["top_k"] = 3
    return args


def make_verification_researcher_node(context: WorkflowContext):
    available = _verification_tools(context)

    def verification_researcher(state: ResearchState) -> dict[str, Any]:
        questions = list(state.get("verification_questions", []))
        budget = min(
            int(state.get("max_verification_tool_calls", context.max_verification_tool_calls)),
            context.max_verification_tool_calls,
        )
        evidence: list[Evidence] = []
        messages: list[Any] = []
        lines = ["[Node] VerificationResearcher", f"Tool Budget: {budget}"]
        calls = 0
        errors = list(state.get("potential_errors", []))
        for question_index, question in enumerate(questions[:budget]):
            tool_name = _pick_tool(question, available)
            if not tool_name:
                lines.append(f"No verification tool available for: {question}")
                continue
            call_id = f"verification-{uuid.uuid4().hex[:12]}"
            args = _tool_args(context, tool_name, question)
            messages.append(AIMessage(content="", tool_calls=[{
                "name": tool_name, "args": args, "id": call_id, "type": "tool_call",
            }]))
            lines.extend((f"Verification Tool Call: {tool_name}", f"Arguments: {json.dumps(args, ensure_ascii=False)}"))
            try:
                content = str(context.tools[tool_name](**args))
            except Exception as exc:
                content = f"ToolError: {type(exc).__name__}: {exc}"
            messages.append(ToolMessage(content=content, tool_call_id=call_id, name=tool_name))
            claim_id = (
                str(errors[question_index].get("claim_id", ""))
                if question_index < len(errors)
                else ""
            )
            parsed = [
                enrich_evidence(item, claim_id=claim_id)
                for item in parse_tool_evidence(tool_name, content, question, call_id)
            ]
            passed, score, reason = assess_evidence_relevance(
                question, str(state.get("resolved_topic", "")), parsed
            )
            for item in parsed:
                item["score"] = max(float(item.get("score", 0.0) or 0.0), score)
            evidence.extend(parsed)
            calls += 1
            lines.extend((
                f"Verification Observation: {len(parsed)} evidence record(s)",
                f"Verification Relevance: {passed} ({score:.4f}) - {reason}",
            ))
        lines.extend((
            f"Verification Tool Calls Used: {calls}/{budget}",
            "[Edge] VerificationResearcher -> VerificationJudge",
        ))
        return {
            "messages": messages,
            "verification_evidence": evidence,
            "verification_tool_calls": calls,
            "graph_trace": lines,
        }

    return verification_researcher


def _feedback_from_errors(errors: Sequence[Mapping[str, Any]]) -> list[VerifierFeedback]:
    return [
        {
            "claim": str(item.get("claim", "")),
            "problem": str(item.get("reason", "Evidence remains insufficient.")),
            "suggested_action": "targeted_rewrite",
        }
        for item in errors
    ]


def make_verification_judge_node(context: WorkflowContext):
    def judge(state: ResearchState) -> dict[str, Any]:
        errors = list(state.get("potential_errors", []))
        questions = list(state.get("verification_questions", []))
        verification_evidence = list(state.get("verification_evidence", []))
        all_evidence = list(state.get("accepted_evidence", state.get("evidence", [])))
        all_evidence.extend(verification_evidence)
        aligned_claims, claim_relations = align_claims(list(state.get("claims", [])), all_evidence)
        iteration = int(state.get("verification_iteration", 0)) + 1
        max_iterations = int(state.get("max_verification_iterations", 2))

        if not errors:
            passed, feedback = True, []
        else:
            prompt = (
                "Judge only the listed potential errors using the verification evidence. Return JSON: "
                "{\"passed\":true|false,\"feedback\":[{\"claim\":\"...\",\"problem\":\"...\","
                "\"suggested_action\":\"targeted_rewrite\"}]}. Pass only if every flagged claim is now supported.\n\n"
                f"Final Answer:\n{state.get('draft', '')}\n\n"
                f"Potential Errors:\n{json.dumps(errors, ensure_ascii=False)}\n\n"
                f"Questions:\n{json.dumps(questions, ensure_ascii=False)}\n\n"
                f"Verification Evidence:\n{_format_evidence(verification_evidence)}"
            )
            parsed = _json_response(context, "You are the final verification judge. Output JSON only.", prompt)
            if parsed is not None and isinstance(parsed.get("passed"), bool):
                passed = bool(parsed["passed"])
                feedback = []
                for item in parsed.get("feedback", []) or []:
                    if isinstance(item, Mapping):
                        feedback.append({
                            "claim": str(item.get("claim", "")),
                            "problem": str(item.get("problem", "")),
                            "suggested_action": str(item.get("suggested_action", "targeted_rewrite")),
                        })
            else:
                # Without a parseable Judge result, do not treat retrieval as proof.
                passed = False
                feedback = _feedback_from_errors(errors)

        if context.verifier_override is not None:
            passed, override_feedback = context.verifier_override(state)
            feedback = [] if passed else [{
                "claim": "Verifier override",
                "problem": override_feedback,
                "suggested_action": "targeted_rewrite",
            }]
        elif iteration <= context.forced_verifier_failures:
            passed = False
            feedback = [{
                "claim": "Forced test claim",
                "problem": "Forced test failure.",
                "suggested_action": "targeted_rewrite",
            }]

        feedback_text = "All active verification checks passed." if passed else json.dumps(feedback, ensure_ascii=False)
        details = {
            "mode": "active",
            "potential_error_count": len(errors),
            "verification_question_count": len(questions),
            "verification_evidence_count": len(verification_evidence),
            "verification_tool_calls": int(state.get("verification_tool_calls", 0)),
            "judge_passed": passed,
        }
        next_edge = "END" if passed or iteration >= max_iterations else "AnswerRepair"
        updates: dict[str, Any] = {
            "verification_passed": passed,
            "verification_feedback": feedback_text,
            "verifier_feedback": feedback,
            "verification_details": details,
            "verification_iteration": iteration,
            "iteration": iteration,
            "claims": aligned_claims,
            "claim_evidence": claim_relations,
            "graph_trace": trace(
                "[Node] VerificationJudge",
                f"Passed: {passed}",
                f"Feedback Items: {len(feedback)}",
                f"[State Update] verification_iteration={iteration}, verification_passed={passed}",
                f"[Conditional Edge] VerificationJudge -> {next_edge}",
            ),
        }
        if not passed and iteration >= max_iterations:
            draft = str(state.get("draft", "")).strip()
            replacements = [
                {
                    "original": str(item.get("claim", "")),
                    "replacement": _fallback_repair(str(item.get("claim", ""))),
                    "action": "QUALIFY",
                }
                for item in feedback
                if str(item.get("claim", "")) in draft
            ]
            terminal_draft, applied = _apply_exact_replacements(draft, replacements)
            terminal_audits = []
            for item in applied:
                claim = next(
                    (entry for entry in aligned_claims if entry.get("text") == item["original"]),
                    {},
                )
                terminal_audits.append({
                    "claim_id": str(claim.get("claim_id", "")),
                    "original_claim": item["original"],
                    "final_claim": item["replacement"],
                    "original_status": str(claim.get("status", "UNSUPPORTED")),
                    "final_status": "PARTIALLY_SUPPORTED",
                    "problem": next(
                        (str(entry.get("problem", "")) for entry in feedback if entry.get("claim") == item["original"]),
                        "Evidence remains insufficient.",
                    ),
                    "verification_question": str(claim.get("verification_question", "")),
                    "new_evidence_ids": [
                        str(entry.get("evidence_id", "")) for entry in verification_evidence
                        if not claim.get("claim_id") or entry.get("claim_id") == claim.get("claim_id")
                    ],
                    "repair_action": item["action"],
                    "changed": True,
                })
            suffix = (
                f"\n\n[Verifier] 已达到最大主动验证轮次（{iteration}）；"
                f"已局部降级 {len(applied)} 条未充分验证的 Claim。"
            )
            updates["draft"] = (terminal_draft + suffix).strip()
            if terminal_audits:
                updates["claim_audits"] = terminal_audits
            updates["graph_trace"].extend(
                f"Terminal Repair Diff [QUALIFY]: - {item['original']} | + {item['replacement']}"
                for item in applied
            )
        return updates

    return judge
