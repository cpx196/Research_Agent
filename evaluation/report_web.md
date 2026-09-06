# Research Agent Web UI 验收记录

## 验收范围

- 纯色、无纹理静态界面；
- `/api/health`、`/api/chat`、`/api/chat/stream`；
- Writer provider token streaming；
- LangGraph Node Trace 实时事件；
- MCP calculator；
- 本地 BGE/FAISS RAG；
- Writer fallback / Verifier 最终 draft 一致性；
- 空请求和浏览器断开连接；
- V0/V2、V3、V4、V5 回归。

## 本轮修复

1. 移除噪点元素和 repeating-gradient，页面背景改为纯色。
2. `LLMClient.chat_stream()` 使用 OpenAI-compatible `stream=true` 并解析 SSE delta。
3. LangGraph 使用 `graph.stream(..., stream_mode="updates")` 将 Node Trace 实时发送到页面。
4. 新增 `answer_reset`：流式草稿被 citation fallback、Verifier 或最终 draft 替换时，页面清空旧草稿并展示最终答案。
5. `done` 事件携带最终 answer，前端做最后一次一致性校验。
6. SSE 完成后使用 `Connection: close`；前端收到 `done` 后主动 cancel reader，修复按钮长期停留在 RUNNING 的问题。
7. 运行期间不再禁用输入框，用户可以提前输入下一条问题；发送按钮继续防止同一 Agent 并发执行。
8. MCP Trace 在实时 Graph Trace 中继续显示。
9. 浏览器取消 SSE 请求时静默处理 BrokenPipe / ConnectionReset。
10. 页面使用固定视口；Conversation 和 Execution Trace 各自独立滚动。
11. 每个请求保存独立 JSON，包含问答、Trace、Graph State、Messages、Tool Call/Result、Evidence 和 Verifier 状态。

## HTTP 端到端结果

| 场景 | 状态 | 最终显示与 Graph draft | Trace Node | 耗时 |
|---|---:|---:|---:|---:|
| 你好 | 200 | 一致 | 4 | 0.01s |
| MCP 计算 `3 + 4` | 200 | 一致 | 5 | 2.49s |
| 本地论文 JEPA / BGE-FAISS | 200 | 一致 | 5 | 10.83s |
| 空 Query | 400 | 正确错误响应 | - | - |

静态检查确认 HTML 不再包含 `noise` 元素，CSS 不包含 gradient。

## 自动化回归

```text
Ran 44 tests in 8.025s
OK
```

新增测试覆盖 provider SSE delta、Writer token callback、Graph live trace、fallback draft reconciliation，以及 HTTP SSE 在 `done` 后正常关闭。

## 说明

DemoLLM 使用确定性增量输出；真实 LLM 模式会直接转发 OpenAI-compatible provider 返回的 token delta。真实 provider 必须支持 Chat Completions `stream=true`。
