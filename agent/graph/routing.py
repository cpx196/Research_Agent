"""Conditional edges for the V3 graph."""

from __future__ import annotations

from typing import Literal

from .state import ResearchState


def route_after_router(state: ResearchState) -> Literal["direct", "react", "research"]:
    route = str(state.get("route_decision", {}).get("route", "react"))
    if route in {"direct", "react", "research"}:
        return route  # type: ignore[return-value]
    return "react"


def route_after_react(state: ResearchState) -> Literal["tools", "end"]:
    return "tools" if state.get("react_pending", False) else "end"


def route_after_researcher(state: ResearchState) -> Literal["tools", "writer"]:
    if int(state.get("current_step", 0)) >= len(state.get("plan", [])):
        return "writer"
    return "tools"


def route_after_evidence(state: ResearchState) -> Literal["researcher", "writer"]:
    if int(state.get("current_step", 0)) >= len(state.get("plan", [])):
        return "writer"
    return "researcher"


def route_after_relevance(state: ResearchState) -> Literal["researcher", "writer"]:
    """Continue the plan, including any retry inserted by RelevanceGate."""

    if int(state.get("current_step", 0)) >= len(state.get("plan", [])):
        return "writer"
    return "researcher"


def route_after_verifier(state: ResearchState) -> Literal["end", "repair"]:
    if state.get("verification_passed", False):
        return "end"
    if int(state.get("iteration", 0)) >= int(state.get("max_iterations", 2)):
        return "end"
    return "repair"
