# Research Agent V4 完成报告

## 1. 最终目录树

```text
agent/
├── agent.py                 # V0/V2 Legacy ReAct，未删除
├── llm.py
├── graph/                   # V3 Graph Baseline + V4 可选 Context Graph
│   ├── state.py
│   ├── graph.py
│   ├── routing.py
│   ├── tools.py
│   └── nodes/
└── context/                 # V4 单任务 Context Manager
    ├── manager.py
    ├── selector.py
    ├── compressor.py
    ├── budget.py
    └── metrics.py
tools/                       # 原有 tools 和 TOOL_REGISTRY
rag/                         # 原有 BGE/FAISS RAG
evaluation/
├── run_v4_evaluation.py
├── traces_v4.md
├── results_v4.md
└── report_v4.md
tests/
├── test_context_manager.py
├── test_evidence_selector.py
├── test_compressor.py
└── test_graph.py
```

## 2. 三种运行模式

```bash
.venv-v2/bin/python main.py --mode legacy
.venv-v2/bin/python main.py --mode graph_baseline
.venv-v2/bin/python main.py --mode graph_context
```

`--mode graph` 继续作为 V3 Baseline 别名；默认模式已切换为 `graph_context`。V0/V2 Legacy、V3 Graph 和 V2 RAG 均未删除。

## 3. Context Manager 结构

`ContextManager` 统一提供：

```text
build_planner_context
build_researcher_context
build_writer_context
build_verifier_context
select_evidence
compress_tool_result
estimate_tokens
```

`ResearchState.messages` 仍保存完整原始执行日志，但 V4 的 LLM Node 不再把完整历史自动传入。上下文统计保存到 `context_stats`，每次 Node 调用写入 `history`。

## 4. Node-specific Context

| Node | 保留内容 | 排除内容 |
|---|---|---|
| Planner | Query、任务约束 | Tool Result、Evidence、旧 Draft |
| Researcher | Query、当前 Plan Step、Open Questions、Verifier Feedback、相关 Evidence | 完整 messages、Raw Tool Result、旧 Draft |
| Writer | Query、Plan、Evidence Top-K | Tool 调用过程、失败日志、完整历史 |
| Verifier | Query、Draft、Evidence Top-K、Open Questions | 完整 Planner/Tool Trace |

## 5. Tool Result Compression

采用规则压缩作为第一层：

- local_search 保留 `Source / Page / Score / Text`；
- web_search 保留 `Title / URL / Snippet`；
- 删除 RAG 进度字段、重复空白和过长正文；
- `tool_result_compress_threshold` 超过阈值后才压缩；
- 压缩结果继续进入 Evidence Parser，不改变 source/page。

默认 smoke/test 模式不启用额外 LLM Extractor，避免在 macOS CPU 上为每个 Tool Result 增加一次模型调用；已实现可选 Level 2 LLM Evidence Extraction，超长结果可通过 `CONTEXT_LLM_EXTRACTOR=true` 启用，并且 local evidence 必须保留 Source/Page 才会接受抽取结果。

## 6. Evidence Selection / Dedup

`EvidenceSelector` 先按 `source + page` 合并，再按相似文本去重，最后按：

```text
local BGE score + lexical overlap
```

排序选择 Top-K。这样既复用了 V2 local_search 的语义分数，也能处理 web/paper evidence 没有向量分数的情况；没有新建持久化 Evidence FAISS 文件。

## 7. Context Budget

默认配置集中在 `ContextConfig`，也可以通过 `.env` 覆盖：

```text
Planner:   3000
Researcher: 6000
Writer:    8000
Verifier:  8000
```

优先级为 P0 Query/当前任务，P1 Feedback/Open Questions/高相关 Evidence，P2 辅助 Evidence，P3 历史低价值信息。超限顺序是低优先级删除、P1 截断，P0 不删除；最终 Trace 记录 Raw/Final Tokens、Dropped Sections 和 Budget。

Token 估算采用英文约 4 字符/token、CJK 约 1.8 字符/token，不等同于 provider billing token。

## 8. V3 Regression 与 V4 测试

```text
Ran 31 tests
OK
```

覆盖：

- Legacy calculator/tool loop；
- V3 Graph compile/invoke、ToolNode、Checkpoint、Verifier 回路；
- V4 Node-specific Context；
- Context Budget 和 P0 保留；
- Tool Result Compression；
- Evidence Top-K、source/page 保留、去重；
- 压缩后的 V4 Graph 端到端执行。

## 9. 对照、消融和 Trace

- 3 条完整 V4 Trace：[`traces_v4.md`](traces_v4.md)；
- 5 条相同 Query 的 V3/V4 指标：[`results_v4.md`](results_v4.md)；
- A/B/C/D 四组消融：V3 Full Context、Compression、Compression+Top-K、Full V4；
- 长任务 Context Growth 表；
- 3 个压缩质量/失败风险检查：过度截断、Verifier Feedback 丢失、Source/Page 丢失。

## 10. 结果解释

V4 的目标不是让所有短任务都更快，而是控制长任务的上下文增长。实测中 V4 的 `Max Context` 受 Node Budget 控制；短任务可能因上下文 section 标题产生少量额外 token。所有结果使用 DemoLLM 做可重复 smoke test，最终语义准确率仍需真实 LLM 和人工标注验证。

## 11. 当前已知问题

- Planner 目前是确定性规则，不是 LLM JSON Planner；
- Verifier 默认是规则检查，不是 LLM judge；
- Token 估算不是精确 tokenizer 统计；
- `InMemorySaver` 只支持进程内 checkpoint；
- 当前不做跨会话 Memory；
- V4 Selector 默认使用已有 BGE score + lexical overlap，未额外建立 Evidence 向量库；
- 过度压缩仍可能损失细节，必须结合 Task Success 一起调预算。

## 12. V5 建议

优先接入 tokenizer 精确计数、可选 LLM Evidence Extractor、SQLite Checkpoint，以及真实 LLM 下的自动 Accuracy/Judge 评估；暂不建议引入跨会话 Memory 或 Multi-Agent。
