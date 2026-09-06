# V3 Evaluation

## Graph 设计

`START → Planner → Researcher → ToolNode → Evidence → Writer → Verifier`；Verifier 通过时到 `END`，失败且未达到上限时回到 `Researcher`。每次运行使用 `InMemorySaver` 保存 thread state snapshot。

## Legacy vs Graph Baseline

| Query | System | Success | Tool Calls | LLM Calls | Retrieval Calls | Total Steps | Latency |
|---|---|---:|---:|---:|---:|---:|---:|
| 什么是 Transformer？ | Legacy ReAct | True | 0 | 1 | 0 | 1 | 0.0 |
| 什么是 Transformer？ | LangGraph Workflow | True | 0 | 2 | 0 | 4 | 0.002 |
| 计算 12 * 8 | Legacy ReAct | True | 1 | 2 | 0 | 2 | 0.0 |
| 计算 12 * 8 | LangGraph Workflow | True | 1 | 2 | 0 | 5 | 0.004 |
| 根据本地论文，V-JEPA 2 的核心训练目标是什么？ | Legacy ReAct | True | 1 | 2 | 1 | 2 | 0.02 |
| 根据本地论文，V-JEPA 2 的核心训练目标是什么？ | LangGraph Workflow | True | 1 | 2 | 1 | 5 | 0.026 |
| 根据本地论文，DINOv3 的表示学习特点是什么？ | Legacy ReAct | True | 1 | 2 | 1 | 2 | 0.022 |
| 根据本地论文，DINOv3 的表示学习特点是什么？ | LangGraph Workflow | True | 1 | 2 | 1 | 5 | 0.023 |
| 比较 V-JEPA 2 和 DINOv3 在机器人控制中的潜在优势。 | Legacy ReAct | True | 0 | 1 | 0 | 1 | 0.0 |
| 比较 V-JEPA 2 和 DINOv3 在机器人控制中的潜在优势。 | LangGraph Workflow | True | 3 | 4 | 3 | 9 | 0.067 |

Latency 为本机 Demo 运行值，实际开销取决于 LLM、BGE 首次加载和本地 FAISS 查询；答案质量仍需人工评估，DemoLLM 只用于可重复 smoke test。

## 当前存在的问题

- Planner 目前使用可测试的确定性规则生成结构化计划，尚未让 LLM 自由生成 JSON Plan。
- Writer/Verifier 的默认实现带有确定性兜底；接入真实 LLM 后答案质量仍取决于模型和提示词。
- `InMemorySaver` 只提供进程内 checkpoint；需要跨进程恢复时应替换为 SQLite 或其他持久化 saver。
- macOS CPU 上 BGE 首次加载会增加首个 local_search 的延迟。
