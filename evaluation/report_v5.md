# Research Agent V5 完成报告

## 1. 最终目录树

```text
agent/graph/mcp_agent.py       # V5 MCP + LangGraph runner
mcp_layer/
├── client.py                  # official SDK stdio lifecycle wrapper
├── manager.py                 # connect/discovery/call/resource/prompt/close
├── adapters.py                # ToolSpec, NativeToolAdapter, MCPToolAdapter
├── registry.py                # UnifiedToolRegistry
├── config.py                  # portable stdio config and timeout
├── servers/research_server.py # calculator, paper_search, Resource, Prompt
└── skills/paper_research.py   # fixed multi-tool Skill
tests/test_mcp_*.py
tests/test_skill.py
evaluation/traces_v5.md
evaluation/results_v5.md
```

## 2. MCP Server

使用官方 Python MCP SDK `mcp 1.29.1` 的 `FastMCP`，本地 Research MCP Server 默认使用 stdio。Server 暴露 `calculator`、`paper_search` 两个 Tool，不向 stdout 写日志，避免污染 MCP 协议。

## 3. Client / Manager / 生命周期

`MCPClient` 封装 `ClientSession` 和 `stdio_client`；`MCPClientManager` 提供 async `connect`、`list_tools`、`call_tool`、`list_resources`、`read_resource`、`list_prompts`、`get_prompt`、`close`。连接失败、调用失败、超时、协议异常都会变成 `MCPError` 或错误结果，不直接击穿 Graph。

## 4. Discovery 与 Schema Adapter

MCP `Tool.inputSchema` 被转换为项目已有的 OpenAI-compatible function schema，供 Researcher LLM 和 LangGraph ToolNode 使用。启动时先 discovery，之后动态注册 MCP Tool。

## 5. Unified Tool Layer

`ToolSpec`、`ToolCallResult` 和 `ToolAdapter` 统一 Native/MCP 来源。`ToolCallResult` 记录 source、server、transport、latency、fallback、is_error，并可格式化为 Observation。

## 6. Native Preservation 与 Fallback

原 `tools/TOOL_REGISTRY`、`execute_tool`、V0/V2 Agent、V2 BGE/FAISS RAG 均保留。MCP calculator/paper_search 失败时分别降级到原生函数；缺少 MCP Server、错误参数、超时和 `isError` 都在这一边界处理。

## 7. Resource / Prompt

Resource：`research://manifest`，只读取项目 `papers/manifest.json`；Prompt：`paper_research(topic)`，返回固定的论文研究提示。没有任意文件路径读取接口。

## 8. Skill

`PaperResearchSkill` 将 `local_search` 与可选的 `paper_search` 组成固定流程，输出 topic、steps、evidence、status。它是 V5 定义的“多个工具 + 固定 workflow”的 Skill，而非 MCP 协议概念。

## 9. CLI 与 Graph

新增：

```bash
.venv-v2/bin/python main.py --mode graph_mcp --demo
```

V5 Host 使用 MCP discovery 后，将统一工具送入现有 LangGraph `ToolNode`，继续经过 Evidence、V4 Context Manager、Writer、Verifier。V0/V2、V3、V4 模式不变。

## 10. 最小验收

```text
Ran 38 tests
OK
```

新增测试覆盖真实 stdio discovery/call、Resource、Prompt、schema/trace adapter、MCP failure → Native fallback、Skill 和 graph_mcp calculator smoke。完整两条 V5 Trace 见 [`traces_v5.md`](traces_v5.md)，结果见 [`results_v5.md`](results_v5.md)。

## 11. 已知问题与最终评测 TODO

- 当前只实现 stdio；HTTP MCP transport 尚未接入。
- 为兼容当前同步 LangGraph ToolNode，V5 MCP Tool call 使用一次连接/调用/关闭的 loop-safe wrapper，真实生产环境可改为长连接 async graph 以降低启动延迟。
- Planner/Verifier 仍是 V3/V4 的规则实现，未改成 MCP 专用 LLM planner/judge。
- V5 仅做最小 smoke/regression；真实 LLM 下的 task success、citation correctness、fallback rate、延迟 P50/P95、并发和 server crash recovery 仍待最终评测。
- 继续使用 macOS CPU 与已有 BGE/FAISS RAG，不引入 CUDA 或大规模 benchmark。
