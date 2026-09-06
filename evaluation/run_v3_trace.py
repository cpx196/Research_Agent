"""Generate V3 Graph traces and a Legacy-vs-Graph baseline comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.agent import ResearchAgent
from agent.graph import LangGraphResearchAgent
from agent.llm import DemoLLM


TRACE_QUERIES = [
    "什么是 Transformer？",
    "根据本地论文，V-JEPA 2-AC 是如何使用 action 的？",
    "比较 V-JEPA 2 和 DINOv3 在机器人控制中的潜在优势。",
]

COMPARISON_QUERIES = [
    "什么是 Transformer？",
    "计算 12 * 8",
    "根据本地论文，V-JEPA 2 的核心训练目标是什么？",
    "根据本地论文，DINOv3 的表示学习特点是什么？",
    "比较 V-JEPA 2 和 DINOv3 在机器人控制中的潜在优势。",
]


def _legacy_metrics(agent: ResearchAgent, answer: str) -> dict[str, Any]:
    messages = agent.last_messages
    return {
        "success": not answer.startswith("AgentError") and "max_steps" not in answer,
        "tool_calls": agent.last_run_stats.get("tool_calls", 0),
        "llm_calls": agent.last_run_stats.get("steps", 0),
        "retrieval_calls": sum(
            1 for message in messages if message.get("role") == "tool" and "[RAG]" in message.get("content", "")
        ),
        "total_steps": agent.last_run_stats.get("steps", 0),
        "latency": agent.last_run_stats.get("latency_seconds", 0),
    }


def _graph_metrics(agent: LangGraphResearchAgent, answer: str) -> dict[str, Any]:
    state = agent.last_state
    evidence = state.get("evidence", [])
    retrieval_ids = {
        str(item.get("tool_call_id"))
        for item in evidence
        if item.get("source_type") == "local_search"
    }
    return {
        "success": not answer.startswith("AgentError") and bool(state.get("verification_passed")),
        "tool_calls": agent.last_run_stats.get("tool_calls", 0),
        "llm_calls": agent.last_run_stats.get("llm_calls", 0),
        "retrieval_calls": len(retrieval_ids),
        "total_steps": agent.last_run_stats.get("graph_steps", 0),
        "latency": agent.last_run_stats.get("latency_seconds", 0),
    }


def main() -> int:
    trace_agent = LangGraphResearchAgent(DemoLLM(), verbose=False, max_iterations=2)
    traces: list[str] = []
    for query in TRACE_QUERIES:
        start = len(trace_agent.trace.lines)
        trace_agent.run(query)
        traces.append("\n".join(trace_agent.trace.lines[start:]))

    evaluation_dir = Path(__file__).resolve().parent
    (evaluation_dir / "traces_v3.md").write_text(
        "# V3 LangGraph Trace\n\n"
        "以下为 DemoLLM + LangGraph StateGraph 的完整执行轨迹。\n\n"
        + "\n\n".join(
            f"## Trace {index}\n\n```text\n{trace}\n```"
            for index, trace in enumerate(traces, start=1)
        )
        + "\n",
        encoding="utf-8",
    )

    rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for query in COMPARISON_QUERIES:
        legacy = ResearchAgent(DemoLLM(), verbose=False, max_steps=10)
        legacy_answer = legacy.run(query)
        graph = LangGraphResearchAgent(DemoLLM(), verbose=False, max_iterations=2)
        graph_answer = graph.run(query)
        rows.append((query, _legacy_metrics(legacy, legacy_answer), _graph_metrics(graph, graph_answer)))

    table = [
        "| Query | System | Success | Tool Calls | LLM Calls | Retrieval Calls | Total Steps | Latency |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for query, legacy, graph in rows:
        short_query = query.replace("|", "\\|")
        for label, metrics in (("Legacy ReAct", legacy), ("LangGraph Workflow", graph)):
            table.append(
                f"| {short_query} | {label} | {metrics['success']} | {metrics['tool_calls']} | "
                f"{metrics['llm_calls']} | {metrics['retrieval_calls']} | {metrics['total_steps']} | {metrics['latency']} |"
            )

    (evaluation_dir / "results_v3.md").write_text(
        "# V3 Evaluation\n\n"
        "## Graph 设计\n\n"
        "`START → Planner → Researcher → ToolNode → Evidence → Writer → Verifier`；"
        "Verifier 通过时到 `END`，失败且未达到上限时回到 `Researcher`。"
        "每次运行使用 `InMemorySaver` 保存 thread state snapshot。\n\n"
        "## Legacy vs Graph Baseline\n\n"
        + "\n".join(table)
        + "\n\n"
        "Latency 为本机 Demo 运行值，实际开销取决于 LLM、BGE 首次加载和本地 FAISS 查询；"
        "答案质量仍需人工评估，DemoLLM 只用于可重复 smoke test。\n\n"
        "## 当前存在的问题\n\n"
        "- Planner 目前使用可测试的确定性规则生成结构化计划，尚未让 LLM 自由生成 JSON Plan。\n"
        "- Writer/Verifier 的默认实现带有确定性兜底；接入真实 LLM 后答案质量仍取决于模型和提示词。\n"
        "- `InMemorySaver` 只提供进程内 checkpoint；需要跨进程恢复时应替换为 SQLite 或其他持久化 saver。\n"
        "- macOS CPU 上 BGE 首次加载会增加首个 local_search 的延迟。\n",
        encoding="utf-8",
    )
    print(evaluation_dir / "traces_v3.md")
    print(evaluation_dir / "results_v3.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
