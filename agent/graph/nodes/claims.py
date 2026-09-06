"""Claim extraction and deterministic Claim-to-Evidence alignment."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from ..context import WorkflowContext
from ..state import Claim, ClaimEvidence, Evidence, ResearchState
from .common import extract_response_message, trace


_STRONG = re.compile(
    r"(最好|最强|全面|优于|高于|低于|首次|唯一|保证|必然|全部|完全|开源|"
    r"best|better than|outperform|state.of.the.art|first|only|always|never|released)",
    re.IGNORECASE,
)
_INFERENCE = re.compile(
    r"(因此|因而|意味着|说明|更适合|更有利|可能|倾向|归纳偏置|"
    r"therefore|suggests|implies|more suitable|recommend)",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"(相比|相较|而|优于|高于|低于|更强|更弱|比较|versus|\bvs\.?\b|than)",
    re.IGNORECASE,
)
_RECOMMENDATION = re.compile(
    r"(建议|优先|应该选择|总体判断|更有利于|更适合|更适用|适用性|推荐|"
    r"recommend|should|prefer|suitable)",
    re.IGNORECASE,
)


def _json_response(context: WorkflowContext, system: str, prompt: str) -> Mapping[str, Any] | None:
    try:
        response = context.llm.chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
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


def _terms(text: str) -> set[str]:
    terms = {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)}
    for chunk in re.findall(r"[\u3400-\u9fff]{2,}", text):
        if len(chunk) <= 4:
            terms.add(chunk)
        else:
            terms.update(chunk[index:index + 2] for index in range(len(chunk) - 1))
    return terms - {
        "根据", "当前", "研究", "证据", "来源", "论文", "回答", "可以", "进行",
        "this", "that", "with", "from", "evidence", "source",
    }


def evidence_identity(item: Mapping[str, Any]) -> str:
    """Return a stable ID for equivalent evidence across verification rounds."""

    existing = str(item.get("evidence_id", "")).strip()
    if existing:
        return existing
    raw = "|".join(
        str(item.get(key, "")).strip()
        for key in ("source_type", "source", "page", "url", "content")
    )
    return f"E-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def enrich_evidence(item: Evidence, *, claim_id: str = "") -> Evidence:
    result: Evidence = dict(item)
    result["evidence_id"] = evidence_identity(result)
    if claim_id:
        result["claim_id"] = claim_id
    source_type = str(result.get("source_type", ""))
    result.setdefault(
        "source_authority",
        0.92 if source_type == "local_search" else 0.86 if source_type == "paper_search" else 0.65,
    )
    result.setdefault("retrieved_at", datetime.now(timezone.utc).isoformat())
    result.setdefault("quote", str(result.get("content", "")))
    return result


def _claim_type(text: str) -> str:
    if _RECOMMENDATION.search(text):
        return "recommendation"
    if _COMPARISON.search(text):
        return "comparison"
    if _INFERENCE.search(text):
        return "inference"
    return "fact"


def _claim_risk(text: str, claim_type: str) -> str:
    if _STRONG.search(text) or claim_type == "recommendation":
        return "high"
    if claim_type in {"comparison", "inference"} or re.search(r"\d", text):
        return "medium"
    return "low"


def _sentence_spans(draft: str) -> list[tuple[str, int, int, str]]:
    """Extract auditable prose spans while retaining their exact offsets."""

    spans: list[tuple[str, int, int, str]] = []
    section = "Answer"
    offset = 0
    for line in draft.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("#"):
            section = stripped.lstrip("# ") or section
            offset += len(line)
            continue
        if (
            stripped.startswith("|")
            or re.fullmatch(r"[-|: ]+", stripped)
            or stripped.startswith(("```", "[Verifier]"))
        ):
            offset += len(line)
            continue
        line_start = offset + len(line) - len(line.lstrip())
        cleaned = stripped.lstrip("-*0123456789.、 ")
        prefix = len(stripped) - len(cleaned)
        # ASCII periods only terminate a sentence before whitespace/end. This
        # keeps vjepa2.pdf, V-JEPA 2.1, p.3, and decimal metrics intact.
        for match in re.finditer(r".+?(?:[。！？]|[.!?](?=\s|$)|$)", cleaned):
            text = match.group(0).strip()
            plain = re.sub(r"[`*_#]", "", text).strip()
            semantic_plain = plain.lstrip(".。!！?？;；:： ")
            if (
                len(plain) < 10
                or plain.startswith(("http://", "https://", "来源：", "Source:"))
                or semantic_plain.startswith(("（来源", "(来源", "（Source", "(Source"))
                or not re.search(r"[A-Za-z\u3400-\u9fff]", plain)
            ):
                continue
            start = line_start + prefix + match.start() + len(match.group(0)) - len(match.group(0).lstrip())
            spans.append((text[:500], start, start + len(text), section))
        offset += len(line)
    return spans


def extract_claims_deterministic(draft: str) -> list[Claim]:
    """Extract provisional Claims without an LLM; safe for Writer streaming."""
    candidates = []
    for text, start, end, section in _sentence_spans(draft):
        claim_type = _claim_type(text)
        risk = _claim_risk(text, claim_type)
        importance = 0.92 if claim_type == "recommendation" else 0.84 if risk == "high" else 0.72 if risk == "medium" else 0.58
        candidates.append((importance, start, text, end, section, claim_type, risk))
    # Keep the most decision-relevant Claims rather than merely the first 15
    # sentences of a long research report, then restore answer order.
    selected = sorted(sorted(candidates, key=lambda item: (-item[0], item[1]))[:15], key=lambda item: item[1])
    result: list[Claim] = []
    for index, (importance, start, text, end, section, claim_type, risk) in enumerate(selected, start=1):
        result.append({
            "claim_id": f"C{index}",
            "text": text,
            "claim_type": claim_type,  # type: ignore[typeddict-item]
            "importance": importance,
            "risk": risk,  # type: ignore[typeddict-item]
            "status": "UNSUPPORTED",
            "answer_section": section,
            "answer_start": start,
            "answer_end": end,
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "verification_question": "",
            "repair_action": "",
            "repaired_text": "",
        })
    return result


def make_claim_extractor_node(context: WorkflowContext):
    def claim_extractor(state: ResearchState) -> dict[str, Any]:
        draft = str(state.get("draft", ""))
        fallback = extract_claims_deterministic(draft)
        prompt = (
            "Extract only independently verifiable, important claims from the answer. Return JSON "
            "{\"claims\":[{\"text\":\"exact answer substring\",\"claim_type\":\"fact|comparison|"
            "inference|recommendation\",\"importance\":0.0,\"risk\":\"low|medium|high\"}]}. "
            "Use 6-15 claims at most and copy each text exactly from the answer.\n\n"
            f"Answer:\n{draft}"
        )
        parsed = (
            _json_response(context, "You are a Claim Extractor. Output JSON only.", prompt)
            if os.getenv("RESEARCH_AGENT_LLM_CLAIM_EXTRACTION", "0").strip().lower()
            in {"1", "true", "yes"}
            else None
        )
        extracted: list[Claim] = []
        for raw in (parsed or {}).get("claims", []) or []:
            if not isinstance(raw, Mapping):
                continue
            text = str(raw.get("text", "")).strip()
            start = draft.find(text)
            if not text or start < 0 or len(text) < 10:
                continue
            claim_type = str(raw.get("claim_type", _claim_type(text))).lower()
            if claim_type not in {"fact", "comparison", "inference", "recommendation"}:
                claim_type = _claim_type(text)
            risk = str(raw.get("risk", _claim_risk(text, claim_type))).lower()
            if risk not in {"low", "medium", "high"}:
                risk = _claim_risk(text, claim_type)
            extracted.append({
                "claim_id": f"C{len(extracted) + 1}",
                "text": text[:500],
                "claim_type": claim_type,  # type: ignore[typeddict-item]
                "importance": max(0.0, min(float(raw.get("importance", 0.7)), 1.0)),
                "risk": risk,  # type: ignore[typeddict-item]
                "status": "UNSUPPORTED",
                "answer_section": "Answer",
                "answer_start": start,
                "answer_end": start + len(text),
                "supporting_evidence_ids": [],
                "contradicting_evidence_ids": [],
                "verification_question": "",
                "repair_action": "",
                "repaired_text": "",
            })
        claims = extracted[:15] if extracted else fallback
        return {
            "claims": claims,
            "claim_evidence": [],
            "original_draft": state.get("original_draft") or draft,
            "graph_trace": trace(
                "[Node] ClaimExtractor",
                f"Claims: {len(claims)}",
                *[f"{item['claim_id']} [{item['claim_type']}/{item['risk']}]: {item['text']}" for item in claims],
                "[Edge] ClaimExtractor -> ClaimEvidenceAligner",
            ),
        }

    return claim_extractor


def align_claims(
    claims: Sequence[Claim], evidence: Sequence[Evidence]
) -> tuple[list[Claim], list[ClaimEvidence]]:
    enriched = [enrich_evidence(dict(item)) for item in evidence if not item.get("error")]
    relations: list[ClaimEvidence] = []
    aligned: list[Claim] = []
    for claim in claims:
        current: Claim = dict(claim)
        claim_terms = _terms(str(current.get("text", "")))
        supporting: list[str] = []
        best = 0.0
        claim_relations: list[ClaimEvidence] = []
        for item in enriched:
            evidence_terms = _terms(" ".join(str(item.get(k, "")) for k in ("title", "content", "source")))
            overlap = len(claim_terms & evidence_terms) / max(len(claim_terms), 1)
            semantic = float(item.get("score", 0.0) or 0.0)
            entailment = min(1.0, overlap + (0.12 * semantic if overlap else 0.0))
            if entailment < 0.28:
                continue
            evidence_id = evidence_identity(item)
            claim_relations.append({
                "evidence_id": evidence_id,
                "claim_id": str(current.get("claim_id", "")),
                "stance": "support",
                "source": str(item.get("source", "")),
                "source_type": str(item.get("source_type", "")),
                "source_authority": float(item.get("source_authority", 0.0) or 0.0),
                "title": str(item.get("title", "")),
                "authors": list(item.get("authors", [])),
                "published_at": str(item.get("published_at", "")),
                "retrieved_at": str(item.get("retrieved_at", "")),
                "page": item.get("page", ""),
                "url": str(item.get("url", "")),
                "quote": str(item.get("quote") or item.get("content", "")),
                "context": str(item.get("context", "")),
                "retrieval_score": float(item.get("score", 0.0) or 0.0),
                "entailment_score": round(entailment, 4),
                "comparable_setting": bool(item.get("comparable_setting", False)),
                "worker_id": str(item.get("worker_id", "")),
                "evidence_role": str(item.get("evidence_role", "")),
                "quality_score": float(item.get("quality_score", 0.0) or 0.0),
            })
        # A Claim card should expose the strongest passages, not every chunk
        # with incidental vocabulary overlap.
        claim_relations.sort(
            key=lambda item: (
                float(item.get("entailment_score", 0.0) or 0.0),
                float(item.get("quality_score", 0.0) or 0.0),
                float(item.get("source_authority", 0.0) or 0.0),
            ),
            reverse=True,
        )
        selected_relations = claim_relations[:3]
        relations.extend(selected_relations)
        supporting = [str(item.get("evidence_id", "")) for item in selected_relations]
        best = max(
            (float(item.get("entailment_score", 0.0) or 0.0) for item in selected_relations),
            default=0.0,
        )
        current["supporting_evidence_ids"] = list(dict.fromkeys(supporting))
        current["contradicting_evidence_ids"] = []
        claim_type = str(current.get("claim_type", "fact"))
        risk = str(current.get("risk", "low"))
        if best >= 0.65:
            status = "INFERRED" if claim_type in {"inference", "recommendation"} else "SUPPORTED"
        elif best >= 0.35:
            status = "PARTIALLY_SUPPORTED"
        else:
            status = "UNSUPPORTED"
        # Strong comparative/recommendation claims require direct comparable evidence.
        if risk == "high" and claim_type in {"comparison", "recommendation"}:
            comparable = any(
                relation.get("claim_id") == current.get("claim_id")
                and relation.get("comparable_setting")
                for relation in relations
            )
            if status == "SUPPORTED" and not comparable:
                status = "PARTIALLY_SUPPORTED"
        current["status"] = status  # type: ignore[typeddict-item]
        aligned.append(current)
    return aligned, relations


def make_claim_aligner_node(context: WorkflowContext):
    def claim_aligner(state: ResearchState) -> dict[str, Any]:
        evidence = list(
            state.get("qualified_evidence")
            or state.get("accepted_evidence", state.get("evidence", []))
        )
        evidence.extend(state.get("verification_evidence", []))
        claims, relations = align_claims(list(state.get("claims", [])), evidence)
        evidence_by_id = {
            evidence_identity(item): enrich_evidence(dict(item))
            for item in evidence
            if not item.get("error")
        }
        planned = list(state.get("planned_claims", []))
        relation_keys = {
            (str(item.get("claim_id", "")), str(item.get("evidence_id", ""))) for item in relations
        }
        for claim in claims:
            claim_terms = _terms(str(claim.get("text", "")))
            aligned_ids = {
                str(value) for value in claim.get("supporting_evidence_ids", []) if str(value)
            }
            best_plan = None
            best_overlap = 0.0
            best_shared = False
            for planned_claim in planned:
                planned_terms = _terms(str(planned_claim.get("text", "")))
                overlap = len(claim_terms & planned_terms) / max(min(len(claim_terms), len(planned_terms)), 1)
                planned_ids = {str(value) for value in planned_claim.get("evidence_ids", [])}
                shared = bool(aligned_ids & planned_ids)
                if shared and not best_shared:
                    best_overlap, best_plan, best_shared = overlap, planned_claim, True
                elif shared == best_shared and overlap > best_overlap:
                    best_overlap, best_plan = overlap, planned_claim
            # Evidence identity is the strongest bridge between a pre-writing
            # contract and a differently worded sentence in the final answer.
            if best_plan is None or (not best_shared and best_overlap < 0.18):
                claim["planned_claim_id"] = ""
                claim["plan_consistent"] = str(claim.get("risk", "low")) == "low"
                continue
            claim["planned_claim_id"] = str(best_plan.get("claim_id", ""))
            claim["plan_consistent"] = True
            inherited_ids: list[str] = []
            for evidence_id in best_plan.get("evidence_ids", []):
                item = evidence_by_id.get(str(evidence_id))
                if item is None:
                    continue
                inherited_ids.append(str(evidence_id))
                key = (str(claim.get("claim_id", "")), str(evidence_id))
                if key in relation_keys:
                    continue
                relation_keys.add(key)
                relations.append({
                    "evidence_id": str(evidence_id),
                    "claim_id": str(claim.get("claim_id", "")),
                    "stance": "support",
                    "source": str(item.get("source", "")),
                    "source_type": str(item.get("source_type", "")),
                    "source_authority": float(item.get("source_authority", 0.0) or 0.0),
                    "title": str(item.get("title", "")),
                    "authors": list(item.get("authors", [])),
                    "published_at": str(item.get("published_at", "")),
                    "retrieved_at": str(item.get("retrieved_at", "")),
                    "page": item.get("page", ""),
                    "url": str(item.get("url", "")),
                    "quote": str(item.get("quote") or item.get("content", "")),
                    "context": str(item.get("context", "")),
                    "retrieval_score": float(item.get("score", 0.0) or 0.0),
                    "entailment_score": round(min(0.95, best_overlap + 0.2), 4),
                    "comparable_setting": bool(item.get("comparable_setting", False)),
                    "worker_id": str(item.get("worker_id", "")),
                    "evidence_role": str(item.get("evidence_role", "")),
                    "quality_score": float(item.get("quality_score", 0.0) or 0.0),
                })
            claim["supporting_evidence_ids"] = list(dict.fromkeys(
                list(claim.get("supporting_evidence_ids", [])) + inherited_ids
            ))
            strength = str(best_plan.get("allowed_strength", "conditional"))
            if inherited_ids and strength == "direct" and best_overlap >= 0.3:
                claim["status"] = "SUPPORTED"
            elif inherited_ids and strength == "conditional":
                claim["status"] = (
                    "INFERRED" if claim.get("claim_type") in {"inference", "recommendation"}
                    else "PARTIALLY_SUPPORTED"
                )
        prompt = (
            "Align each Claim only to evidence that semantically supports or contradicts it. Do not use mere "
            "keyword overlap as proof. Return JSON {\"alignments\":[{\"claim_id\":\"C1\","
            "\"evidence_id\":\"E-...\",\"stance\":\"support|contradict|neutral\","
            "\"entailment_score\":0.0,\"comparable_setting\":false}],"
            "\"claim_statuses\":[{\"claim_id\":\"C1\",\"status\":"
            "\"SUPPORTED|PARTIALLY_SUPPORTED|INFERRED|CONTRADICTED|UNSUPPORTED\"}]}. "
            "A comparative or best-model Claim is SUPPORTED only when the evidence is directly comparable.\n\n"
            f"Claims:\n{json.dumps(claims, ensure_ascii=False)}\n\n"
            "Evidence:\n" + json.dumps([
                {
                    "evidence_id": evidence_id,
                    "source": item.get("source"),
                    "page": item.get("page"),
                    "url": item.get("url"),
                    "content": str(item.get("content", ""))[:700],
                }
                for evidence_id, item in list(evidence_by_id.items())[:12]
            ], ensure_ascii=False)
        )
        semantic = (
            _json_response(context, "You are a Claim-Evidence Aligner. Output JSON only.", prompt)
            if os.getenv("RESEARCH_AGENT_LLM_CLAIM_ALIGNMENT", "0").strip().lower()
            in {"1", "true", "yes"}
            else None
        )
        if semantic is not None:
            claim_by_id = {str(item.get("claim_id", "")): item for item in claims}
            semantic_relations: list[ClaimEvidence] = []
            for raw in semantic.get("alignments", []) or []:
                if not isinstance(raw, Mapping):
                    continue
                claim_id = str(raw.get("claim_id", ""))
                evidence_id = str(raw.get("evidence_id", ""))
                stance = str(raw.get("stance", "neutral")).lower()
                if claim_id not in claim_by_id or evidence_id not in evidence_by_id:
                    continue
                if stance not in {"support", "contradict", "neutral"}:
                    stance = "neutral"
                item = evidence_by_id[evidence_id]
                semantic_relations.append({
                    "evidence_id": evidence_id,
                    "claim_id": claim_id,
                    "stance": stance,  # type: ignore[typeddict-item]
                    "source": str(item.get("source", "")),
                    "source_type": str(item.get("source_type", "")),
                    "source_authority": float(item.get("source_authority", 0.0) or 0.0),
                    "title": str(item.get("title", "")),
                    "authors": list(item.get("authors", [])),
                    "published_at": str(item.get("published_at", "")),
                    "retrieved_at": str(item.get("retrieved_at", "")),
                    "page": item.get("page", ""),
                    "url": str(item.get("url", "")),
                    "quote": str(item.get("quote") or item.get("content", "")),
                    "context": str(item.get("context", "")),
                    "retrieval_score": float(item.get("score", 0.0) or 0.0),
                    "entailment_score": max(0.0, min(float(raw.get("entailment_score", 0.0)), 1.0)),
                    "comparable_setting": bool(raw.get("comparable_setting", False)),
                    "worker_id": str(item.get("worker_id", "")),
                    "evidence_role": str(item.get("evidence_role", "")),
                    "quality_score": float(item.get("quality_score", 0.0) or 0.0),
                })
            if semantic_relations:
                relations = semantic_relations
                for claim in claims:
                    claim_id = str(claim.get("claim_id", ""))
                    claim["supporting_evidence_ids"] = [
                        item["evidence_id"] for item in relations
                        if item.get("claim_id") == claim_id and item.get("stance") == "support"
                    ]
                    claim["contradicting_evidence_ids"] = [
                        item["evidence_id"] for item in relations
                        if item.get("claim_id") == claim_id and item.get("stance") == "contradict"
                    ]
            valid_statuses = {
                "SUPPORTED", "PARTIALLY_SUPPORTED", "INFERRED", "CONTRADICTED", "UNSUPPORTED"
            }
            for raw in semantic.get("claim_statuses", []) or []:
                if not isinstance(raw, Mapping):
                    continue
                claim = claim_by_id.get(str(raw.get("claim_id", "")))
                status = str(raw.get("status", ""))
                if claim is None or status not in valid_statuses:
                    continue
                if claim.get("risk") == "high" and claim.get("claim_type") in {"comparison", "recommendation"}:
                    comparable = any(
                        item.get("claim_id") == claim.get("claim_id")
                        and item.get("stance") == "support"
                        and item.get("comparable_setting")
                        for item in relations
                    )
                    if status == "SUPPORTED" and not comparable:
                        status = "PARTIALLY_SUPPORTED"
                claim["status"] = status  # type: ignore[typeddict-item]
        counts: dict[str, int] = {}
        for item in claims:
            status = str(item.get("status", "UNSUPPORTED"))
            counts[status] = counts.get(status, 0) + 1
        return {
            "claims": claims,
            "claim_evidence": relations,
            "graph_trace": trace(
                "[Node] ClaimEvidenceAligner",
                f"Relations: {len(relations)}",
                f"Status Counts: {json.dumps(counts, ensure_ascii=False, sort_keys=True)}",
                "[Edge] ClaimEvidenceAligner -> VerificationErrorDetector",
            ),
        }

    return claim_aligner
