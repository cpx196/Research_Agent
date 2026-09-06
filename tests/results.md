# V0 Evaluation Results

测试命令：

```bash
python -m unittest discover -s tests -v
```

单元测试使用 Fake LLM，因此不依赖 LLM API Key；Q1-Q3 的 CLI 验证使用 `--demo`，并记录在 `logs/q1-q3.trace.txt`。

单元测试结果：7 tests，全部通过。

| Test | Outcome | Tool Calls | Steps | Latency (s) | Notes |
|---|---:|---:|---:|---:|---|
| Transformer（无 Tool） | Pass | 0 | 1 | 0.000 | 单元测试覆盖 |
| Q1 Calculator | Pass | 1 | 2 | 0.002 | `3947 * 8123` → `32061481` |
| Q2 Paper Search | Pass | 1 | 2 | 0.910 | V-JEPA 2，3 篇 |
| Q3 Multi-Step | Pass | 2 | 3 | 3.832 | Paper + Web + Summary route |
| Tool Error / max_steps | Pass | 2 | 2 | 0.000 | 单元测试覆盖 |

Q1-Q3 Demo 汇总：平均 Agent Steps `2.33`，Tool Calls `4`，Tool Errors `0`。Q2/Q3 的网络延迟来自本次运行，实际值会随公共 API 和网络变化。

## 统计口径

- `Task Success`：人工判断最终回答是否满足 Query。
- `Tool Call Count`：一次 Query 中 assistant 发出的 Tool Call 数。
- `Average Agent Steps`：从首次 LLM 请求到 Final Answer/硬停止的步数。
- `Tool Error Count`：作为 Observation 返回、以 `ToolError:` 开头的结果数。
- `Latency`：CLI 端到端耗时；公共搜索接口的延迟会随网络变化。
