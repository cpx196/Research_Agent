"""Runtime dependencies shared by V3 node factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from agent.context.manager import ContextManager


VerifierOverride = Callable[[Mapping[str, Any]], tuple[bool, str]]


@dataclass(frozen=True)
class WorkflowContext:
    """Dependencies injected into graph nodes, keeping nodes testable."""

    llm: Any
    tools: Mapping[str, Callable[..., str]]
    tool_schemas: Sequence[Mapping[str, Any]]
    max_iterations: int
    verifier_mode: str = "simple"
    max_verification_tool_calls: int = 2
    context_manager: ContextManager | None = None
    verifier_override: VerifierOverride | None = None
    forced_verifier_failures: int = 0
    stream_callback: Callable[[str], None] | None = None
