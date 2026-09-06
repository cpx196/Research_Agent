# Evidence-First Research Agent

本项目当前聚焦两个核心能力：

1. **Claim-Level Active Verification**：把研究回答拆成可独立核查的 Claim，并建立 Claim–Evidence 对应关系；
2. **Targeted Claim Repair**：只对无证据、弱证据或冲突 Claim 定向补证和局部修补，不重新生成整篇答案。

旗舰问题比较 V-JEPA 2、DINOv3 与普通 MAE Encoder 的表征特点及其机器人控制适用性。系统会区分论文事实、跨论文比较、推断和推荐，并阻止把“训练目标更匹配”直接写成“控制性能已经全面更优”。完整方案见 [`Evidence_First_Research_Agent_Plan.md`](Evidence_First_Research_Agent_Plan.md)。

运行真实旗舰 Demo：

```bash
.venv-v2/bin/python main.py --mode graph_mcp --verifier active --flagship-demo
```

运行三个 Baseline（Writer Only、Whole-answer Verifier、Claim-level Targeted Repair）：

```bash
.venv-v2/bin/python -m evaluation.run_evidence_first_evaluation
```

运行 30 条人工评测种子集（共 90 次 Baseline Run，建议先用 `--limit` 小规模检查）：

```bash
.venv-v2/bin/python -m evaluation.run_evidence_first_evaluation \
  --cases evaluation/evidence_first_cases.json --limit 3
```

以下 V0–V5 架构仍作为兼容 Baseline 保留。

这是一个 Research Agent：V0/V2 的原生 Python Tool Calling Loop、V3 LangGraph Baseline、V4 Context Manager 均保留，V5 在 LangGraph Researcher 下加入官方 MCP Python SDK、统一工具层和 Skill。

```text
User Query → LLM → local_search → Query Embedding → FAISS → Observation → LLM
```

当前知识库来自 Zotero `UST` 分类中抽取的 10 篇论文，包含 JEPA 系列和 DINOv3。PDF 不会上传到 LLM；只有检索到的文本片段作为 Tool Observation/Evidence 返回。

## V3/V4 LangGraph Workflow

默认入口运行 V4 Context Graph；可切换四种模式：

```text
legacy        = V0/V2 手写 ReAct
graph_baseline = V3 Full Context Graph
graph_context  = V4 Context Manager Graph
graph_mcp      = V5 MCP + Context Graph
```

V4 的 Graph：

```text
START → Planner → Researcher → ToolNode → Evidence
                                      ↘ Researcher（仍有计划）
Evidence → Writer → Verifier ── Pass → END
                         └──── Fail → Researcher（最多 2 轮）
```

V3/V4 使用 `StateGraph`、`START/END`、Conditional Edge、官方 `ToolNode` 和 `InMemorySaver`。现有 `TOOL_REGISTRY` 和工具函数没有重写；LangGraph 的 ToolNode 只是将它们适配到图执行接口。V4 的 `ContextManager` 在每个 LLM Node 前生成 Node-specific Context，并维护 `open_questions`、`context_stats`、Tool Result Compression、Evidence Dedup、Evidence Top-K 和 Context Budget。

运行 V4 或 V5：

```bash
.venv-v2/bin/python main.py --mode graph_context --demo
.venv-v2/bin/python main.py --mode graph_mcp --demo
```

运行 V3 Baseline 和旧版：

```bash
.venv-v2/bin/python main.py --mode graph_baseline --demo
.venv-v2/bin/python main.py --mode legacy --demo
```

原来的 `--mode graph` 仍作为 V3 Baseline 别名保留。

## V5 MCP Layer

V5 将 Research Agent 作为 MCP Host：Host 内的 `MCPClientManager` 通过官方 Python MCP SDK 以 stdio 启动本地 `Research MCP Server`，发现 Tool schema，再把 MCP Tool 与原生 Python Tool 统一为 `ToolAdapter`。原有 `tools/TOOL_REGISTRY` 未改写；`calculator` 和 `paper_search` 优先走 MCP，MCP 启动、连接、超时、协议或 server error 时自动降级到同名 Native Tool。

```text
LangGraph Researcher → MCPClientManager → stdio → Research MCP Server
                    → UnifiedToolRegistry → ToolNode → Evidence
                    → V4 Context Manager → Writer / Verifier
```

MCP Server 提供受控 Resource `research://manifest` 和 Prompt `paper_research`。Resource 只读取项目 `papers/manifest.json`，没有任意路径读取接口。V5 Skill `PaperResearchSkill` 将 local_search 和可选 paper_search 组成固定、可审计的多工具流程。

运行 V5：

```bash
.venv-v2/bin/python main.py --mode graph_mcp --demo
```

直接启动 MCP Server（通常由 Host 自动启动）：

```bash
.venv-v2/bin/python mcp_layer/servers/research_server.py
```

启动浏览器界面：

```bash
.venv-v2/bin/python web_server.py --demo --mode graph_mcp
```

然后打开 <http://127.0.0.1:8000>。真实 LLM 模式去掉 `--demo`；也可以用 `--mode graph_context`、`--mode graph_baseline` 或 `--mode legacy` 查看旧工作流。界面会通过 `/api/chat/stream` 的 SSE 连接实时展示 Graph Trace 和 Writer 的增量回答，并显示延迟、Evidence 数量和 MCP 状态。对话区和 Trace 区使用独立滚动容器，页面保持固定视口。

默认启用 Active Verifier，它会执行 Claim 定位、验证问题生成、最多两次定向工具调用和 Judge。可使用 `--verifier simple` 切回旧基线，或用 `--max-verification-tool-calls 1-4` 调整单轮验证预算：

```bash
.venv-v2/bin/python web_server.py --mode graph_mcp --verifier active --max-verification-tool-calls 2
.venv-v2/bin/python main.py --mode graph_mcp --verifier simple
```

对于 JEPA、DINO、V-JEPA、World Model、Robot 和 Patch Policy 等本地论文库主题，Router 会把 `local_search` 作为首要来源，并将 `local_search + paper_search` 都写入执行计划；涉及最新或官方信息时再追加 `web_search`。真实 Web 服务启动后会在后台预热 BGE/FAISS，HTTP 服务不会等待模型冷启动完成。

每次请求默认保存为 `logs/web_runs/<timestamp>_<request-id>.json`，内容包括 Query、最终 Answer、完整 Trace、Graph State、Messages、Tool Call/Result、Evidence、Verifier 和统计信息。可自定义目录：

```bash
.venv-v2/bin/python web_server.py --mode graph_mcp --log-dir /path/to/logs
```

运行保留的 V0/V2 手写 Agent Loop：

```bash
.venv-v2/bin/python main.py --mode legacy --demo
```

Graph 实现位于 [`agent/graph/`](agent/graph/)，旧实现仍位于 [`agent/agent.py`](agent/agent.py)。

## macOS CPU 环境

推荐使用项目内独立 venv（本机为 Apple Silicon，默认强制 CPU，不依赖 CUDA）：

```bash
python -m venv .venv-v2
.venv-v2/bin/python -m pip install -r requirements.txt
.venv-v2/bin/python -m rag.import_zotero
.venv-v2/bin/python -m rag.ingest --device cpu --batch-size 32
```

也可以使用 Conda；本次 macOS 机器访问 `repo.anaconda.com` 超时，因此实际验证采用 `.venv-v2`，未修改 base 环境。

首次运行 `sentence-transformers` 会下载 `BAAI/bge-small-en-v1.5`；模型约 130 MB，文档向量只在 ingestion 时计算一次。若当前网络无法访问 Hugging Face，可设置 `RAG_EMBEDDING_MODEL` 为本机已有的 sentence-transformers 模型。

若 Hugging Face 下载通道不可用，可使用无需模型下载的字符 n-gram fallback 跑通完整 pipeline：

```bash
.venv-v2/bin/python -m rag.ingest --backend hashing --device cpu
```

该 fallback 不是学习到的语义模型，而是 deterministic local embedding；它仍保证 document/query 在同一向量空间，并在索引 `config.json` 中记录 backend。恢复模型网络后可用 `--backend sentence-transformers` 重建索引。

运行真实 LLM 的交互式 V4/V5 Agent：

```bash
cp .env.example .env
.venv-v2/bin/python main.py --mode graph_context
.venv-v2/bin/python main.py --mode graph_mcp
```

也可以显式指定模式：

```bash
.venv-v2/bin/python main.py --mode graph
.venv-v2/bin/python main.py --mode graph_context
.venv-v2/bin/python main.py --mode legacy
```

没有 LLM API Key 时，可先运行确定性的本地 Loop 验证 RAG Tool：

```bash
.venv-v2/bin/python main.py --demo
```

## 从 Zotero 导入论文

`rag.import_zotero` 读取 Zotero SQLite 的 `UST` collection 与 attachment 关系，只复制明确选中的 10 篇 PDF 到 `papers/`，并生成 `papers/manifest.json` 保存 Zotero key 和原始路径：

```bash
.venv-v2/bin/python -m rag.import_zotero \
  --zotero-dir /Users/ocbrain/Zotero \
  --output-dir papers \
  --count 10
```

导入的主题相关论文包括：V-JEPA 2、V-JEPA 2.1、图像 JEPA、VLA-JEPA、DINOv3、World Action Models、Fast-WAM、LaWAM、Patch Policy、Causal World Modeling。

## RAG Pipeline

```text
papers/*.pdf
  ↓ PyMuPDF（按页抽取）
文本页
  ↓ 字符近似切块
约 2000 characters / overlap 200 characters
  ↓ BAAI/bge-small-en-v1.5（batch，CPU）
归一化 embedding
  ↓ FAISS IndexFlatIP
data/vector_db/index.faiss + metadata.json
```

`metadata.json` 的第 N 项与 FAISS 第 N 个向量严格对应，并保留 `source`、一基页码 `page`、原文 `text`。当前 chunking 使用字符近似而不是 tokenizer token；这是 V2 任务书允许的基础实现。向量归一化后使用 Inner Product，等价于 Cosine Similarity。

离线阶段是 PDF 解析、切块、文档 embedding 和 FAISS 建库；在线查询只计算一个 query embedding，再执行 Top-K 检索，不会重新编码全部论文。

## 工具

- `tools/calculator.py`：安全基本算术。
- `tools/web_search.py`：互联网查询，适合当前消息或官方更新。
- `tools/paper_search.py`：arXiv 元数据查询。
- `tools/local_search.py`：本地论文语义检索，输出 source/page/score/text。

Agent 会优先使用 `local_search` 回答“根据本地论文/本地资料”的问题，使用 `web_search` 回答最新外部信息。

## V4 Context Manager

V4 不做跨会话 Memory，只管理单次任务上下文。原始 `messages` 继续保留作执行日志，但不同 Node 使用不同上下文：Planner 只看 Query/约束；Researcher 看当前 Step、Open Questions、Verifier Feedback 和相关 Evidence；Writer 看 Query、Plan 和 Evidence；Verifier 看 Query、Draft 和相关 Evidence。

配置位于 `.env.example`，默认预算为 Planner 3000、Researcher 6000、Writer 8000、Verifier 8000 个估算 tokens。估算使用 CJK 约 1.8 字符/token、其他文本约 4 字符/token，不代表账单 token。V4 Trace 会记录 Raw/Final Context Tokens、Selected Evidence、Dropped Evidence、Compression Ratio 和 Context Budget。

如果本地向量库不存在，工具会返回：

```text
ToolError: Local vector database not found. Run python -m rag.ingest first.
```

不会因此使 Agent 进程崩溃。

## 测试与验收记录

```bash
.venv-v2/bin/python -m unittest discover -s tests -v
```

`tests/` 覆盖 chunk overlap、FAISS/检索器 metadata 对齐、local_search 错误处理、Legacy Agent Tool Loop、V3 LangGraph compile/invoke、ToolNode、Evidence、Verifier 回路、Checkpoint、V4 Context Manager，以及 V5 MCP stdio discovery/call、Resource、Prompt、统一适配器、Native fallback、Skill 和 graph_mcp。Trace 会记录 Node、State Update、Tool Call、Evidence、Edge、Verifier Result、iteration、上下文统计和 MCP 协议字段。

V3 完成报告见 [`evaluation/report_v3.md`](evaluation/report_v3.md)，5 条相同 Query 的 Legacy/Graph 对照见 [`evaluation/results_v3.md`](evaluation/results_v3.md)。

V4 报告见 [`evaluation/report_v4.md`](evaluation/report_v4.md)，V3/V4 对照与消融记录见 [`evaluation/results_v4.md`](evaluation/results_v4.md)，完整 V4 Trace 见 [`evaluation/traces_v4.md`](evaluation/traces_v4.md)。

V5 最小验收报告见 [`evaluation/report_v5.md`](evaluation/report_v5.md)，MCP/Skill smoke 结果见 [`evaluation/results_v5.md`](evaluation/results_v5.md)，完整 V5 Trace 见 [`evaluation/traces_v5.md`](evaluation/traces_v5.md)。V5 不做大规模 benchmark；重新生成最小 Trace：

浏览器界面、SSE、MCP 与本地 RAG 的验收记录见 [`evaluation/report_web.md`](evaluation/report_web.md)。

```bash
.venv-v2/bin/python -m evaluation.run_v5_evaluation
```

重新生成三条完整 Trace：

```bash
.venv-v2/bin/python -m evaluation.run_trace
```

## 目录

```text
agent/                 Tool Calling Agent
agent/graph/           V3 LangGraph StateGraph、Nodes、Routing、ToolNode、V5 MCP runner
agent/context/         V4 Context Manager、Budget、Compression、Selector、Metrics
mcp_layer/             V5 MCP Client/Manager、统一 Tool Adapter、Server、Skill
tools/local_search.py  本地检索 Tool
rag/
  pdf_parser.py        PyMuPDF 按页解析
  chunker.py           字符近似 chunk + overlap
  embedding.py         sentence-transformers CPU adapter
  vector_store.py      FAISS + metadata 持久化
  retriever.py         Query embedding + Top-K
  ingest.py            离线建库命令
  import_zotero.py     Zotero UST 抽取命令
papers/                UST 抽取的 PDF（运行导入命令后生成）
data/vector_db/        FAISS index 与 metadata（运行 ingestion 后生成）
evaluation/            V2/V3/V4/V5 Query、Agent Trace 与对照记录
```
