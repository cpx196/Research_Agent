"""LangGraph-based Research Agent V3.

The V0/V2 hand-written loop remains in :mod:`agent.agent`.  This package is
the additive V3 workflow implementation.
"""

from .graph import LangGraphResearchAgent, build_research_graph
from .mcp_agent import MCPResearchAgent
from .router import HybridRouter, RouterDecision, validate_router_output
from .state import Evidence, PlanStep, ResearchState

__all__ = [
    "Evidence",
    "LangGraphResearchAgent",
    "MCPResearchAgent",
    "HybridRouter",
    "RouterDecision",
    "validate_router_output",
    "PlanStep",
    "ResearchState",
    "build_research_graph",
]
