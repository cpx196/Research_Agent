"""Generate reproducible local Agent traces after the index has been built."""

from __future__ import annotations

from pathlib import Path

from agent.agent import ResearchAgent
from agent.llm import DemoLLM


QUERIES = [
    "根据本地论文，V-JEPA 2-AC 是如何使用 action 的？",
    "根据本地论文，V-JEPA 2 的核心训练目标是什么？",
    "根据本地论文，V-JEPA 2-AC 使用了什么机器人训练数据？",
]


def main() -> int:
    agent = ResearchAgent(DemoLLM(), verbose=False, max_steps=2)
    traces: list[str] = []
    for query in QUERIES:
        start = len(agent.trace.lines)
        agent.run(query)
        traces.append("\n".join(agent.trace.lines[start:]))
    target = Path(__file__).resolve().parent / "traces.md"
    target.write_text(
        "# V2 Agent Trace\n\n"
        "以下为 DemoLLM + 已持久化 BGE/FAISS 索引的完整三条 Tool Loop Trace。\n\n"
        + "\n\n".join(f"## Trace {index}\n\n```text\n{trace}\n```" for index, trace in enumerate(traces, 1))
        + "\n",
        encoding="utf-8",
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
