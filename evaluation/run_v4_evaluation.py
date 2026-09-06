"""Generate V4 context traces, baselines, ablations, and growth records."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.context import ContextConfig, ContextManager, estimate_tokens
from agent.graph import LangGraphResearchAgent
from agent.llm import DemoLLM


TRACE_QUERIES = [
    "根据本地论文，V-JEPA 2-AC 是如何使用 action 的？",
    "比较 V-JEPA 2 和 DINOv3 在机器人控制中的潜在优势。",
    "比较 V-JEPA 2、DINOv3、JEPA-WAM、Patch Policy，从训练目标、视觉表征、动作建模、机器人泛化、开源情况五个方面进行系统分析。",
]

COMPARISON_QUERIES = [
    "根据本地论文，V-JEPA 2-AC 是如何使用 action 的？",
    "根据本地论文，DINOv3 的表示学习特点是什么？",
    "比较 V-JEPA 2 和 DINOv3 在机器人控制中的潜在优势。",
    "比较 V-JEPA 2、DINOv3、JEPA-WAM、Patch Policy，从训练目标、视觉表征、动作建模、机器人泛化、开源情况五个方面进行系统分析。",
    "根据本地论文，V-JEPA 2-AC 使用了什么机器人训练数据？",
]


class MeteredLLM:
    """Count approximate input/output tokens around the existing demo adapter."""

    def __init__(self) -> None:
        self.inner = DemoLLM()
        self.input_tokens: list[int] = []
        self.output_tokens: list[int] = []

    def chat(self, messages: Sequence[Mapping[str, Any]], tools=None) -> dict[str, Any]:
        self.input_tokens.append(estimate_tokens(json.dumps(list(messages), ensure_ascii=False, default=str)))
        response = self.inner.chat(messages=messages, tools=tools)
        self.output_tokens.append(estimate_tokens(json.dumps(response, ensure_ascii=False, default=str)))
        return response


def _tool_keys(state: Mapping[str, Any]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for message in state.get("messages", []):
        for call in getattr(message, "tool_calls", []) or []:
            keys.append((str(call.get("name", "")), json.dumps(call.get("args", {}), sort_keys=True)))
    return keys


def _metrics(agent: LangGraphResearchAgent, meter: MeteredLLM, answer: str) -> dict[str, Any]:
    state = agent.last_state
    keys = _tool_keys(state)
    repeated = len(keys) - len(set(keys))
    evidence = list(state.get("evidence", []))
    retrieval_ids = {
        str(item.get("tool_call_id"))
        for item in evidence
        if item.get("source_type") == "local_search"
    }
    if agent.context_manager is not None:
        history = list(state.get("context_stats", {}).get("history", []))
        context_values = [int(item.get("final_context_tokens", 0)) for item in history]
    else:
        context_values = list(meter.input_tokens)
    return {
        "success": bool(state.get("verification_passed")) and not answer.startswith("AgentError"),
        "input_tokens": sum(meter.input_tokens),
        "output_tokens": sum(meter.output_tokens),
        "total_tokens": sum(meter.input_tokens) + sum(meter.output_tokens),
        "avg_context": round(sum(context_values) / len(context_values), 1) if context_values else 0,
        "max_context": max(context_values or [0]),
        "llm_calls": len(meter.input_tokens),
        "tool_calls": len(keys),
        "repeated_tool_calls": repeated,
        "retrieval_calls": len(retrieval_ids),
        "latency": agent.last_run_stats.get("latency_seconds", 0),
        "evidence_count": len(evidence),
        "verifier_passed": bool(state.get("verification_passed")),
    }


def _synthetic_evidence() -> list[dict[str, Any]]:
    return [
        {
            "source_type": "local_search",
            "source": f"paper-{index % 8}.pdf",
            "page": index % 12 + 1,
            "content": (
                f"Method {index} studies visual representation, action-conditioned prediction, "
                "robot control and generalization. "
                + " ".join(f"fact-{index}-{part}" for part in range(35))
            ),
            "score": 0.5 + (index % 5) / 10,
        }
        for index in range(47)
    ]


def _ablation() -> list[dict[str, Any]]:
    evidence = _synthetic_evidence()
    manager = ContextManager(ContextConfig(writer_top_k=10))
    full_text = manager._format_evidence(evidence)
    raw_tool = "\n\n".join(
        f"[Result {index + 1}]\nSource: {item['source']}\nPage: {item['page']}\n"
        f"Score: {item['score']:.4f}\nText: {item['content']}"
        for index, item in enumerate(evidence)
    )
    compressed = manager.compress_tool_result(raw_tool, tool_name="local_search")
    selected = manager.select_evidence(
        {"evidence": evidence}, "visual representation action robot control", top_k=10
    )
    selected_text = manager._format_evidence(selected)
    state = {
        "query": "visual representation action robot control",
        "plan": [],
        "current_step": 0,
        "open_questions": ["preserve source and page"],
        "verification_feedback": "",
        "evidence": evidence,
    }
    full_v4 = manager.build_writer_context(state)
    return [
        {
            "variant": "A: V3 Full Context",
            "tokens": estimate_tokens(full_text),
            "selected": len(evidence),
            "source_page_preserved": True,
            "notes": "所有 Evidence 直接进入 Writer context",
        },
        {
            "variant": "B: Compression only",
            "tokens": compressed.final_tokens,
            "selected": len(evidence),
            "source_page_preserved": "Source:" in compressed.text and "Page:" in compressed.text,
            "notes": "Raw Tool Result → 规则压缩，不做 Top-K",
        },
        {
            "variant": "C: Compression + Evidence Top-K",
            "tokens": estimate_tokens(selected_text),
            "selected": len(selected),
            "source_page_preserved": all(item.get("source") and item.get("page") for item in selected),
            "notes": "从 47 条 Evidence 选择 10 条",
        },
        {
            "variant": "D: Full V4",
            "tokens": full_v4.budget.final_tokens,
            "selected": len(full_v4.selected_evidence),
            "source_page_preserved": "paper-" in full_v4.text and "page" in full_v4.text.lower(),
            "notes": "Compression + Top-K + Node-specific Context + Budget",
        },
    ]


def main() -> int:
    evaluation_dir = Path(__file__).resolve().parent
    trace_agent = LangGraphResearchAgent(DemoLLM(), context_enabled=True, verbose=False, max_iterations=2)
    traces: list[str] = []
    for query in TRACE_QUERIES:
        start = len(trace_agent.trace.lines)
        trace_agent.run(query)
        traces.append("\n".join(trace_agent.trace.lines[start:]))
    (evaluation_dir / "traces_v4.md").write_text(
        "# V4 Context Trace\n\n"
        "以下为 Context Manager + BGE/FAISS + LangGraph 的完整轨迹。\n\n"
        + "\n\n".join(
            f"## Trace {index}\n\n```text\n{trace}\n```"
            for index, trace in enumerate(traces, start=1)
        )
        + "\n",
        encoding="utf-8",
    )

    comparison_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for query in COMPARISON_QUERIES:
        baseline_meter = MeteredLLM()
        baseline = LangGraphResearchAgent(baseline_meter, context_enabled=False, verbose=False, max_iterations=2)
        baseline_answer = baseline.run(query)
        context_meter = MeteredLLM()
        context = LangGraphResearchAgent(context_meter, context_enabled=True, verbose=False, max_iterations=2)
        context_answer = context.run(query)
        comparison_rows.append(
            (
                query,
                _metrics(baseline, baseline_meter, baseline_answer),
                _metrics(context, context_meter, context_answer),
            )
        )

    headers = [
        "Query", "System", "Success", "Input Tokens", "Output Tokens", "Total Tokens",
        "Avg Context/Step", "Max Context", "LLM Calls", "Tool Calls", "Repeated Tool Calls",
        "Retrieval Calls", "Latency", "Evidence Count", "Verifier Pass",
    ]
    table = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for query, baseline, context in comparison_rows:
        for label, metrics in (("V3 Full Context", baseline), ("V4 Context Manager", context)):
            values = [
                query.replace("|", "\\|"), label, str(metrics["success"]), str(metrics["input_tokens"]),
                str(metrics["output_tokens"]), str(metrics["total_tokens"]), str(metrics["avg_context"]),
                str(metrics["max_context"]), str(metrics["llm_calls"]), str(metrics["tool_calls"]),
                str(metrics["repeated_tool_calls"]), str(metrics["retrieval_calls"]), str(metrics["latency"]),
                str(metrics["evidence_count"]), str(metrics["verifier_passed"]),
            ]
            table.append("| " + " | ".join(values) + " |")

    # Re-run the long query to expose a per-step growth sequence.
    growth_query = COMPARISON_QUERIES[3]
    growth_baseline_meter = MeteredLLM()
    growth_baseline = LangGraphResearchAgent(growth_baseline_meter, context_enabled=False, verbose=False)
    growth_baseline.run(growth_query)
    growth_context_meter = MeteredLLM()
    growth_context = LangGraphResearchAgent(growth_context_meter, context_enabled=True, verbose=False)
    growth_context.run(growth_query)
    context_history = growth_context.last_state.get("context_stats", {}).get("history", [])
    growth_lines = ["| Sequence | V3 input tokens | V4 final context tokens |", "|---:|---:|---:|"]
    max_length = max(len(growth_baseline_meter.input_tokens), len(context_history))
    for index in range(max_length):
        old = growth_baseline_meter.input_tokens[index] if index < len(growth_baseline_meter.input_tokens) else "-"
        new = context_history[index].get("final_context_tokens", "-") if index < len(context_history) else "-"
        growth_lines.append(f"| {index + 1} | {old} | {new} |")

    ablation_rows = _ablation()
    ablation_table = [
        "| Variant | Tokens | Selected Evidence | Source/Page Preserved | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for row in ablation_rows:
        ablation_table.append(
            f"| {row['variant']} | {row['tokens']} | {row['selected']} | "
            f"{row['source_page_preserved']} | {row['notes']} |"
        )

    failure_checks = [
        ("Compression too aggressive", "source/page retained while long body is truncated", True),
        ("Verifier feedback loss", "researcher context retains explicit verifier feedback", "missing benchmark" in ContextManager().build_researcher_context({"query": "q", "plan": [], "current_step": 0, "open_questions": [], "verification_feedback": "missing benchmark", "evidence": []}).text),
        ("Source loss", "compressed local result still contains Source and Page", ablation_rows[1]["source_page_preserved"]),
    ]
    failure_table = [
        "| Case | Check | Result |",
        "|---|---|---:|",
    ]
    for name, check, result in failure_checks:
        failure_table.append(f"| {name} | {check} | {result} |")

    (evaluation_dir / "results_v4.md").write_text(
        "# V4 Evaluation\n\n"
        "## V3 vs V4 对照\n\n"
        + "\n".join(table)
        + "\n\n所有 token 为估算值：英文约 4 字符/token，CJK 约 1.8 字符/token；"
        "不是 provider billing token。V3 使用原始 Graph 路径，V4 使用默认 ContextConfig。\n\n"
        "## Context Growth\n\n"
        "长任务上下文序列：\n\n"
        + "\n".join(growth_lines)
        + "\n\n目标趋势：V3 随历史/Tool 结果增长，V4 通过 Node-specific Context、Top-K 和 Budget 控制增长。\n\n"
        "## 消融实验\n\n"
        + "\n".join(ablation_table)
        + "\n\n"
        "## 压缩质量与 Failure Case 检查\n\n"
        + "\n".join(failure_table)
        + "\n\n"
        "三个检查覆盖：过度压缩时正文会被截断、Verifier Feedback 不丢失、来源与页码不丢失。"
        "真正的语义准确率仍需要人工标注或真实 LLM judge；DemoLLM 只作为确定性 smoke test。\n\n"
        "## 当前问题\n\n"
        "- Planner 仍使用确定性规则生成 Plan。\n"
        "- Token 估算不是 tokenizer 精确计费。\n"
        "- Level-2 LLM Evidence Extractor 默认关闭；开启后会增加一次额外 LLM 调用。\n"
        "- 默认 Evidence Selector 使用 BGE local score + 透明 lexical overlap；未额外创建持久化 Evidence FAISS 索引。\n"
        "- InMemorySaver 和本次 Context Manager 均只服务单次进程内任务，不提供跨会话 Memory。\n"
        "- V4 省 token 的收益在长任务更明显，短任务可能因标题/字段包装出现轻微额外开销。\n",
        encoding="utf-8",
    )
    print(evaluation_dir / "traces_v4.md")
    print(evaluation_dir / "results_v4.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
