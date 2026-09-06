"""Shared state for the V3 LangGraph workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class PlanStep(TypedDict, total=False):
    """One structured research task produced by Planner."""

    id: int
    task: str
    tool: str
    status: Literal["pending", "completed"]
    retry: bool


class Evidence(TypedDict, total=False):
    """A normalized piece of evidence independent from chat messages."""

    source_type: str
    source: str
    page: int | str
    query: str
    content: str
    score: float
    title: str
    url: str
    tool_call_id: str
    error: bool
    evidence_id: str
    claim_id: str
    stance: Literal["support", "contradict", "neutral"]
    source_authority: float
    entailment_score: float
    comparable_setting: bool
    quote: str
    context: str
    published_at: str
    retrieved_at: str
    authors: list[str]
    worker_id: str
    evidence_role: Literal["primary", "secondary", "discovery", "rejected"]
    quality_score: float
    quality_reason: str
    quality_passed: bool
    section: str


class Claim(TypedDict, total=False):
    """One independently auditable conclusion in the generated answer."""

    claim_id: str
    text: str
    claim_type: Literal["fact", "comparison", "inference", "recommendation"]
    importance: float
    risk: Literal["low", "medium", "high"]
    status: Literal[
        "SUPPORTED", "PARTIALLY_SUPPORTED", "INFERRED", "CONTRADICTED", "UNSUPPORTED"
    ]
    answer_section: str
    answer_start: int
    answer_end: int
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    verification_question: str
    repair_action: Literal["", "STRENGTHEN", "QUALIFY", "REPLACE", "REMOVE"]
    repaired_text: str
    planned_claim_id: str
    plan_consistent: bool


class PlannedClaim(TypedDict, total=False):
    """A pre-writing conclusion contract grounded in qualified Evidence."""

    claim_id: str
    text: str
    claim_type: Literal["fact", "comparison", "inference", "recommendation"]
    importance: float
    allowed_strength: Literal["direct", "conditional", "insufficient"]
    evidence_ids: list[str]
    conditions: list[str]
    rationale: str


class ClaimEvidence(TypedDict, total=False):
    """A normalized Claim-to-Evidence relation used by the verifier."""

    evidence_id: str
    claim_id: str
    stance: Literal["support", "contradict", "neutral"]
    source: str
    source_type: str
    source_authority: float
    title: str
    authors: list[str]
    published_at: str
    retrieved_at: str
    page: int | str
    url: str
    quote: str
    context: str
    retrieval_score: float
    entailment_score: float
    comparable_setting: bool
    worker_id: str
    evidence_role: str
    quality_score: float


class ClaimAudit(TypedDict, total=False):
    """Immutable audit record for one targeted Claim repair."""

    claim_id: str
    original_claim: str
    final_claim: str
    original_status: str
    final_status: str
    problem: str
    verification_question: str
    new_evidence_ids: list[str]
    repair_action: str
    changed: bool


class RelevanceCheck(TypedDict, total=False):
    """Auditable decision made after one retrieval result."""

    tool: str
    query: str
    topic: str
    score: float
    passed: bool
    reason: str
    retry_count: int
    evidence_count: int
    tool_call_id: str


class PotentialError(TypedDict, total=False):
    """A claim that deserves targeted verification rather than a full rerun."""

    claim: str
    reason: str
    claim_id: str
    status: str


class VerifierFeedback(TypedDict, total=False):
    """Structured Judge feedback consumed by AnswerRepair."""

    claim: str
    problem: str
    suggested_action: str


class ResearchState(TypedDict, total=False):
    """State shared by every node in the V3 graph.

    ``messages`` is still kept for compatibility with the existing raw
    OpenAI-compatible LLM adapter.  ``evidence`` is deliberately separate so
    Writer and Verifier do not have to infer research facts from chat history.
    Reducers append messages/evidence/trace entries; scalar fields are updated
    by the node that owns them.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    original_query: str
    standalone_query: str
    resolved_topic: str
    resolved_intent: str
    search_queries: dict[str, str]
    conversation_history: list[dict[str, str]]
    session_preferences: dict[str, str]
    route_decision: dict[str, object]
    react_steps: int
    react_pending: bool
    plan: list[PlanStep]
    current_step: int
    evidence: Annotated[list[Evidence], operator.add]
    accepted_evidence: Annotated[list[Evidence], operator.add]
    qualified_evidence: list[Evidence]
    evidence_quality_checks: list[dict[str, object]]
    relevance_checks: Annotated[list[RelevanceCheck], operator.add]
    relevance_passed: bool
    relevance_feedback: str
    retrieval_retry_count: int
    max_retrieval_retries: int
    retry_query: str
    retry_tool: str
    open_questions: list[str]
    draft: str
    original_draft: str
    planned_claims: list[PlannedClaim]
    claims: list[Claim]
    claim_evidence: list[ClaimEvidence]
    claim_audits: Annotated[list[ClaimAudit], operator.add]
    verification_passed: bool
    verification_feedback: str
    verification_details: dict[str, object]
    potential_errors: list[PotentialError]
    verification_questions: list[str]
    verification_evidence: Annotated[list[Evidence], operator.add]
    verifier_feedback: list[VerifierFeedback]
    verification_iteration: int
    max_verification_iterations: int
    verification_tool_calls: int
    max_verification_tool_calls: int
    verifier_mode: Literal["none", "simple", "active"]
    iteration: int
    max_iterations: int
    context_stats: dict[str, object]
    graph_trace: Annotated[list[str], operator.add]
