# Research Agent V3 完成报告

## 1. 最终目录树

```text
agent/
├── agent.py                 # 保留的 V0/V2 手写 Agent Loop
├── llm.py                   # OpenAI-compatible LLM 与 DemoLLM
└── graph/
    ├── state.py             # ResearchState / PlanStep / Evidence
    ├── graph.py             # StateGraph assembly、invoke、runner
    ├── routing.py           # Conditional Edge 路由
    ├── tools.py             # 旧 Tool Registry -> ToolNode adapter
    └── nodes/
        ├── planner.py
        ├── researcher.py
        ├── evidence.py
        ├── writer.py
        └── verifier.py
tools/                       # 原有 calculator/web/paper/local tools
rag/                         # 原有 BGE/FAISS RAG
evaluation/
├── run_v3_trace.py
├── traces_v3.md
├── results_v3.md
└── report_v3.md
tests/test_graph.py
main.py
```

## 2. 两个入口

```bash
.venv-v2/bin/python main.py --mode graph --demo
.venv-v2/bin/python main.py --mode legacy --demo
```

`graph` 是 V3 默认模式；`legacy` 保留并运行 V0/V2 手写循环。

## 3. ResearchState

`agent/graph/state.py` 中显式保存：

```text
messages, query, plan, current_step, evidence,
draft, verification_passed, verification_feedback,
iteration, max_iterations, graph_trace
```

`messages` 兼容原有 OpenAI-compatible LLM，`evidence` 使用独立 reducer 保存规范化证据。

## 4. Graph 结构

```text
START
  ↓
Planner → Researcher → ToolNode → Evidence
              ↑                      │
              └────── 仍有计划 ──────┘
                                     ↓
                                   Writer
                                     ↓
                                  Verifier
                                /         \
                             Pass         Fail
                              ↓             ↓
                             END       Researcher
```

## 5-10. Nodes 与路由

- `Planner`：用确定性规则生成结构化 `PlanStep` 列表，避免 Demo 环境依赖额外 LLM JSON 输出。
- `Researcher`：读取当前计划，向 LLM 请求一次工具调用；若 provider 返回文本或错误工具，则用计划中的工具和参数兜底。
- `ToolNode`：使用 LangGraph 官方 `ToolNode`，执行原有 `TOOL_REGISTRY` 中的 calculator、web_search、paper_search、local_search。
- `Evidence`：把 ToolNode 的 ToolMessage 解析为带 `source/page/score/content` 的 Evidence。
- `Writer`：只接收 Query、Plan、Evidence 生成 Draft，不主动搜索。
- `Verifier`：检查 Draft 和可用 Evidence；输出 Pass/Fail 与反馈。
- `route_after_verifier`：Pass 到 `END`，Fail 且未达到上限回到 `Researcher`。

## 11. Checkpoint

`LangGraphResearchAgent` 默认使用 `InMemorySaver`，每次运行带独立 `thread_id`，可通过：

```python
graph.get_state({"configurable": {"thread_id": thread_id}})
```

读取最终 State Snapshot。当前是进程内 checkpoint，跨进程持久化可以后续替换为 SQLite saver。

## 12. ToolNode 方案

原有 `tools/__init__.py` 中的 `TOOL_REGISTRY` 和 `execute_tool()` 没有删除。`agent/graph/tools.py` 只把原有 OpenAI-style schema 适配成 `StructuredTool`，再交给官方 `ToolNode` 执行。因此：

```text
Legacy: execute_tool(name, arguments)
V3:    Researcher AIMessage → LangGraph ToolNode → ToolMessage
```

两种模式共享同一批 Tool 和同一套 BGE/FAISS `local_search`。

## 13. Legacy Regression

旧版测试和 V3 测试一起运行，原 `ResearchAgent`、calculator、web/paper/local registry 均保留；CLI `--mode legacy` 已实测可启动并完成 calculator Tool Loop。

## 14. 测试结果

```text
Ran 20 tests in 0.077s
OK
```

覆盖 Graph compile/invoke、ToolNode、Evidence、Checkpoint、Verifier Fail 回路、最大迭代次数和 Legacy 回归。

## 15. Graph Trace

已生成 3 条完整轨迹：[`traces_v3.md`](traces_v3.md)。每条包含 Node、State Update、Tool Call、Evidence、Edge、Verifier Result 和 iteration。

## 16. Legacy vs Graph

5 条相同 Query 的对照结果见 [`results_v3.md`](results_v3.md)。Graph 的控制流更显式，但通常 Node/LLM/Tool 步骤更多；V3 当前目标是可控、可解释、可追踪，而不是立即降低延迟。

## 17. 当前问题

- Planner 目前是确定性规则，不是 LLM JSON Planner。
- Writer/Verifier 有确定性兜底，真实答案质量仍取决于配置的 LLM。
- Checkpoint 目前只在内存中保存。
- macOS CPU 上 BGE 首次加载会增加首个 `local_search` 延迟。

## V3 核心理解

LangGraph 主要替代的是 V0 的手写流程控制：`while`、`if tool_calls`、`continue/break` 和手动状态流转；它不替代 LLM、RAG、BGE、FAISS 或工具本身。State 是原有 `messages` 的扩展，Node 是普通 Python 函数，Edge 是流程连接，Conditional Edge 负责动态分支，Verifier Fail 回 Researcher 用于补证据，最大迭代次数用于防止死循环。

