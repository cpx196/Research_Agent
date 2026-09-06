"""Node-specific Context Manager for the V4 single-task workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .budget import ContextConfig, ContextSection, BudgetResult, fit_to_budget
from .compressor import CompressionResult, ToolResultCompressor
from .metrics import ContextMetrics, add_context_event, merge_context_stats
from .selector import EvidenceSelector, SelectionResult


@dataclass
class ContextBuild:
    node: str
    text: str
    budget: BudgetResult
    selected_evidence: list[dict[str, Any]]
    selection: SelectionResult | None = None

    @property
    def stats(self) -> dict[str, Any]:
        selection = self.selection
        return ContextMetrics.node_stats(
            node=self.node,
            raw_tokens=self.budget.raw_tokens,
            final_tokens=self.budget.final_tokens,
            budget=self.budget.budget,
            selected_evidence=len(self.selected_evidence),
            available_evidence=selection.available_count if selection else 0,
            deduplicated_evidence=selection.deduplicated_count if selection else 0,
            dropped_evidence=selection.dropped_count if selection else 0,
            dropped_sections=self.budget.dropped_sections + self.budget.truncated_sections,
        )


class ContextManager:
    """Build the minimum sufficient context for each V3/V4 node."""

    def __init__(
        self,
        config: ContextConfig | None = None,
        selector: EvidenceSelector | None = None,
        compressor: ToolResultCompressor | None = None,
        extractor_llm: Any | None = None,
    ) -> None:
        self.config = config or ContextConfig.from_env()
        self.selector = selector or EvidenceSelector(self.config.dedup_similarity_threshold)
        self.compressor = compressor or ToolResultCompressor(
            threshold_tokens=self.config.tool_result_compress_threshold,
            max_chars=self.config.max_tool_result_chars,
            max_evidence_chars=self.config.max_evidence_chars,
            extractor_llm=extractor_llm,
            llm_extractor_enabled=self.config.llm_extractor_enabled,
        )

    @staticmethod
    def _query(state: Mapping[str, Any]) -> str:
        return str(state.get("standalone_query") or state.get("query", "")).strip()

    @staticmethod
    def _original_query(state: Mapping[str, Any]) -> str:
        return str(state.get("original_query") or state.get("query", "")).strip()

    @staticmethod
    def _usable_evidence(state: Mapping[str, Any]) -> list[dict[str, Any]]:
        if "accepted_evidence" in state:
            return [dict(item) for item in state.get("accepted_evidence", [])]
        return [dict(item) for item in state.get("evidence", [])]

    @staticmethod
    def _format_evidence(evidence: Sequence[Mapping[str, Any]]) -> str:
        if not evidence:
            return "(No usable evidence has been collected.)"
        blocks: list[str] = []
        for index, item in enumerate(evidence, start=1):
            citation = str(item.get("source", "unknown"))
            if item.get("page") not in (None, ""):
                citation += f", page {item.get('page')}"
            if item.get("url"):
                citation += f", URL {item.get('url')}"
            blocks.append(f"[{index}] {citation}\n{str(item.get('content', '')).strip()}")
        return "\n\n".join(blocks)

    @staticmethod
    def _format_conversation(state: Mapping[str, Any]) -> str:
        history = list(state.get("conversation_history", []))
        if not history:
            return "(No earlier turns in this browser session.)"
        lines: list[str] = []
        for item in history[-12:]:
            role = str(item.get("role", "user")).upper()
            content = str(item.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) or "(No earlier turns in this browser session.)"

    @staticmethod
    def _format_preferences(state: Mapping[str, Any]) -> str:
        preferences = dict(state.get("session_preferences", {}))
        if not preferences:
            return "(No persistent session preferences.)"
        labels = {"zh-CN": "Simplified Chinese", "en": "English"}
        return "\n".join(
            f"- {key}: {labels.get(str(value), value)}"
            for key, value in preferences.items()
        )

    @staticmethod
    def _format_plan(state: Mapping[str, Any], current_only: bool = False) -> str:
        plan = list(state.get("plan", []))
        if current_only:
            index = int(state.get("current_step", 0))
            if 0 <= index < len(plan):
                step = plan[index]
                return f"Step {index + 1}/{len(plan)}: {step.get('task', '')} [{step.get('tool', '')}]"
            return "No pending plan step."
        if not plan:
            return "(No external research plan.)"
        return "\n".join(
            f"{step.get('id', index + 1)}. {step.get('task', '')} "
            f"[{step.get('tool', '')}; {step.get('status', 'pending')}]"
            for index, step in enumerate(plan)
        )

    def select_evidence(self, state: Mapping[str, Any], query: str, top_k: int) -> list[dict[str, Any]]:
        return self.selector.select(self._usable_evidence(state), query, top_k)

    def _selection(self, state: Mapping[str, Any], query: str, top_k: int) -> SelectionResult:
        return self.selector.select_with_stats(self._usable_evidence(state), query, top_k)

    def compress_tool_result(
        self,
        result: str,
        tool_name: str = "tool",
        query: str = "",
    ) -> CompressionResult:
        return self.compressor.compress_tool_result(result, tool_name=tool_name, query=query)

    def estimate_tokens(self, context: str) -> int:
        from .budget import estimate_tokens

        return estimate_tokens(context)

    def build_planner_context(self, state: Mapping[str, Any]) -> ContextBuild:
        sections = [
            ContextSection("Original Query", self._original_query(state), priority=0),
            ContextSection("Current Query (Standalone)", self._query(state), priority=0),
            ContextSection("Session Preferences", self._format_preferences(state), priority=0),
            ContextSection("Conversation History", self._format_conversation(state), priority=1),
            ContextSection("Router Decision", str(dict(state.get("route_decision", {}))) or "(none)", priority=0),
            ContextSection(
                "Task Constraints",
                "Create a structured research plan. Preserve user constraints from the conversation. "
                "Do not reuse prior Tool Results or old Drafts as research evidence.",
                priority=1,
            ),
        ]
        budget = fit_to_budget(sections, self.config.planner_budget)
        return ContextBuild("planner", budget.text, budget, [])

    def build_researcher_context(self, state: Mapping[str, Any]) -> ContextBuild:
        step_query = self._format_plan(state, current_only=True)
        selection = self._selection(state, step_query, self.config.researcher_top_k)
        sections = [
            ContextSection("Original Query", self._original_query(state), priority=0),
            ContextSection("Current Query (Standalone)", self._query(state), priority=0),
            ContextSection("Session Preferences", self._format_preferences(state), priority=0),
            ContextSection("Conversation History", self._format_conversation(state), priority=1),
            ContextSection("Router Decision", str(dict(state.get("route_decision", {}))) or "(none)", priority=1),
            ContextSection("Current Plan Step", step_query, priority=0),
            ContextSection(
                "Open Questions",
                "\n".join(f"- {item}" for item in state.get("open_questions", [])) or "(none)",
                priority=1,
            ),
            ContextSection(
                "Verifier Feedback",
                str(state.get("verification_feedback", "")) or "(none)",
                priority=1,
            ),
            ContextSection("Relevant Evidence", self._format_evidence(selection.evidence), priority=1),
        ]
        budget = fit_to_budget(sections, self.config.researcher_budget)
        return ContextBuild("researcher", budget.text, budget, selection.evidence, selection)

    def build_writer_context(self, state: Mapping[str, Any]) -> ContextBuild:
        selection = self._selection(state, self._query(state), self.config.writer_top_k)
        sections = [
            ContextSection("Original Query", self._original_query(state), priority=0),
            ContextSection("Current Query (Standalone)", self._query(state), priority=0),
            ContextSection("Session Preferences", self._format_preferences(state), priority=0),
            ContextSection("Conversation History", self._format_conversation(state), priority=1),
            ContextSection("Router Decision", str(dict(state.get("route_decision", {}))) or "(none)", priority=1),
            ContextSection("Research Plan", self._format_plan(state), priority=1),
            ContextSection(
                "Retrieval Relevance",
                str(list(state.get("relevance_checks", []))[-3:]) or "(none)",
                priority=1,
            ),
            ContextSection("Relevant Evidence", self._format_evidence(selection.evidence), priority=1),
        ]
        budget = fit_to_budget(sections, self.config.writer_budget)
        return ContextBuild("writer", budget.text, budget, selection.evidence, selection)

    def build_verifier_context(self, state: Mapping[str, Any]) -> ContextBuild:
        draft = str(state.get("draft", ""))
        selection = self._selection(state, self._query(state) + " " + draft, self.config.verifier_top_k)
        sections = [
            ContextSection("Original Query", self._original_query(state), priority=0),
            ContextSection("Current Query (Standalone)", self._query(state), priority=0),
            ContextSection("Session Preferences", self._format_preferences(state), priority=0),
            ContextSection("Conversation History", self._format_conversation(state), priority=1),
            ContextSection("Router Decision", str(dict(state.get("route_decision", {}))) or "(none)", priority=1),
            ContextSection("Draft", draft or "(empty draft)", priority=0),
            ContextSection(
                "Retrieval Relevance",
                str(list(state.get("relevance_checks", []))[-3:]) or "(none)",
                priority=0,
            ),
            ContextSection("Relevant Evidence", self._format_evidence(selection.evidence), priority=1),
            ContextSection(
                "Open Questions / Requirements",
                "\n".join(f"- {item}" for item in state.get("open_questions", [])) or "(none)",
                priority=1,
            ),
        ]
        budget = fit_to_budget(sections, self.config.verifier_budget)
        return ContextBuild("verifier", budget.text, budget, selection.evidence, selection)

    def record_node(self, state: Mapping[str, Any], context: ContextBuild) -> dict[str, Any]:
        return merge_context_stats(state, context.node, context.stats)

    def record_compression(
        self,
        state: Mapping[str, Any],
        *,
        tool: str,
        result: CompressionResult,
    ) -> dict[str, Any]:
        return add_context_event(
            state,
            "tool_results",
            {
                "tool": tool,
                "raw_tokens": result.raw_tokens,
                "final_tokens": result.final_tokens,
                "compressed": result.compressed,
                "compression_ratio": result.ratio,
                "dropped_fields": result.dropped_fields,
            },
        )
