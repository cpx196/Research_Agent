# Evidence-First Research Agent：项目包装与简历方案

## 1. 一句话定位

> **一个面向科研阅读与写作的 Claim-first Research Agent：先从文献证据中形成明确、可检验的观点，再逐条验证观点，并只修补证据不足的局部结论。**

项目不是普通的“论文搜索 + 摘要生成”工具，而是试图解决科研工作中更困难的一步：

> **如何让 LLM 不只是整理资料，而是敢于形成有边界的研究判断，并让每个判断都能回到原始证据。**

---

## 2. 项目要解决的真实痛点

### 2.1 LLM 很会总结，但不喜欢形成 Claim

在论文阅读、Related Work 调研和技术路线比较中，用户真正需要的通常不是更多摘要，而是明确回答：

- 不同方法最关键的区别是什么？
- 哪一种方案更适合当前任务？
- 文献中的结果能够支持多强的结论？
- 哪些是论文事实，哪些是跨论文推断？
- 如果必须做技术选择，应该选什么，为什么？

普通 LLM 面对这些问题时容易输出：

```text
各有优缺点
需要根据具体任务选择
仍需进一步研究
```

这种回答看似谨慎，却没有真正完成科研判断。它回避了核心 Claim，也没有说明判断成立的条件。

### 2.2 一旦 LLM 给出观点，人工求证非常费劲

另一种情况是 LLM 给出了一个听起来合理的结论，例如：

> V-JEPA 2 的表征比 DINOv3 更适合机器人控制。

用户随后必须手动完成一系列工作：

1. 判断这句话是论文原文、跨论文比较还是模型推断；
2. 找到具体是哪篇论文、哪一页、哪段原文支持它；
3. 检查不同论文是否使用了相同数据集、指标和实验条件；
4. 判断“训练目标更匹配”是否被夸大成“性能已经更好”；
5. 发现证据不足后，重新搜索并修改整段文字；
6. 确认修改没有破坏其他已经正确的内容。

在真实科研阅读和写作中，这种逐句求证往往比生成初稿更耗时间。

### 2.3 传统 Research Agent 的验证发生得太晚

常见工作流是：

```text
Search → Retrieve → Write → Verify
```

Verifier 只能在答案生成后判断“可能有问题”，却没有参与观点形成。结果通常是：

- Writer 已经写入了证据之外的重要结论；
- Claim 与 Evidence 没有稳定对应关系；
- 检查失败后整篇重新生成；
- 原本正确的内容也被改写；
- 增加 Tool Call、Token、延迟和新的幻觉风险。

本项目的核心判断是：

> **Claim 不应该只是写作后的检查对象，而应该成为写作前的生成约束。**

---

## 3. 核心方法

### 3.1 Claim-first Generation

系统首先检索本地论文和公开资料，然后通过 Evidence Quality Gate 判断材料能否作为直接支持：

```text
Local PDF / Paper / Public Source
              ↓
       Evidence Quality Gate
              ↓
          Claim Planner
              ↓
       Planned Claim Contract
              ↓
            Writer
```

Claim Planner 不直接写答案，而是生成一个结构化 Claim Contract：

```json
{
  "claim": "动态规划型机器人任务更适合优先考虑 V-JEPA 2",
  "claim_type": "recommendation",
  "allowed_strength": "conditional",
  "evidence_ids": ["E-12", "E-18"],
  "conditions": ["任务依赖视频历史与动态预测"],
  "rationale": "时序预测目标与动态控制需求更一致"
}
```

Writer 必须围绕 Planned Claims 组织答案：

- 不能新增没有证据的重要观点；
- 不能超过 `allowed_strength`；
- 必须保留结论成立的条件；
- 证据不足的 Claim 必须写成研究空白，而不是事实；
- 必须区分 Fact、Comparison、Inference 和 Recommendation。

项目的第一个核心 Claim：

> **将 Claim 规划放到 Writer 之前，可以让 LLM 给出更明确的研究判断，同时减少无支持结论。**

### 3.2 Claim-level Active Verification

生成答案后，系统将正文拆分成可独立核查的 Claim，并建立 Claim–Evidence Provenance：

```text
Output Claim
  ├─ Supporting Evidence
  ├─ Contradicting Evidence
  ├─ Source / Page / URL
  ├─ Direct Passage
  ├─ Evidence Quality
  └─ Support Status
```

每条 Claim 被标记为：

```text
SUPPORTED
PARTIALLY_SUPPORTED
INFERRED
CONTRADICTED
UNSUPPORTED
```

系统不把关键词重合作为证明，也不把 arXiv 链接本身当作证据：

- 本地 PDF 正文段落可以作为 Primary Evidence；
- 论文 Abstract 只能支持贡献级或条件性 Claim；
- 搜索结果摘要只用于发现来源；
- 参考文献页、纯链接和无正文材料不能直接支持 Claim；
- “更好”“最优”等比较 Claim 需要可比实验，否则必须降低结论强度。

### 3.3 Targeted Claim Repair

当系统发现某条 Claim 无证据、表述过强或存在冲突时，不重新执行完整研究任务：

```text
Risky Claim
    ↓
Verification Question
    ↓
Bounded Evidence Search
    ↓
Claim-level Judge
    ↓
Keep / Weaken / Rewrite / Remove
```

例如：

```text
原 Claim：
V-JEPA 2 已被证明在机器人控制上全面优于 DINOv3。

修补后：
V-JEPA 2 的时序预测目标与动态控制需求更匹配，
但现有材料不足以证明它在统一机器人控制基准上全面优于 DINOv3。
```

其他已经得到支持的段落保持不变。

项目的第二个核心 Claim：

> **Claim 级定向修补可以用更少的检索和改写范围处理证据问题，并保留原答案中已经正确的内容。**

---

## 4. 为什么这比普通 RAG 更有价值

普通 RAG 解决的是：

> 找到与问题相关的文本，并把它放入 LLM Context。

本项目进一步解决：

> 从相关文本中能够形成哪些观点、观点允许有多强，以及正文中的每项判断究竟由什么证据支持。

因此系统的输出不只是 Answer，而是：

```text
Answer
+ Planned Claims
+ Output Claims
+ Claim–Evidence Relations
+ Source Passages
+ Support Status
+ Repair Diff
```

对科研用户的价值是：

- 减少从论文摘要到研究判断的人工整理；
- 减少逐句查找出处的时间；
- 降低跨论文比较中的过度概括；
- 让技术选型结论带有明确条件；
- 让写作中的观点、证据和推断边界可追踪。

---

## 5. 旗舰 Demo

### Demo 问题

> 根据本地论文和公开资料，比较 V-JEPA 2、DINOv3 与普通 MAE Encoder 学到的表征特点，并判断哪一种更有利于机器人控制，为什么？

### 为什么选择这个问题

这个问题不是简单的信息抽取，而是同时要求：

- 阅读多篇论文；
- 建立统一比较维度；
- 区分静态空间表征与时序动态表征；
- 从视觉预训练目标推导机器人控制适用性；
- 给出明确选择；
- 避免把归纳偏置直接写成已验证性能优势。

它能够完整展示 Claim 形成、证据支持、条件判断和局部修补。

### 系统应当敢于给出的结论

> **如果机器人任务依赖视频历史、环境动态和交互结果预测，V-JEPA 2 的训练目标与控制需求更匹配；如果任务主要依赖单帧空间语义、目标定位和高质量冻结视觉特征，DINOv3 更实用；普通 MAE 可以作为稳定基线，但像素重建目标与动作相关动态之间的联系更间接。**

同时必须给出证据边界：

> **在没有统一机器人控制任务、数据规模和训练预算下的直接实验时，这是一项带条件的研究判断，而不是已经被证明的全面性能排名。**

这体现了项目希望实现的“敢下结论”：

```text
不是拒绝判断
也不是无条件断言
而是给出明确、可验证、有适用边界的 Claim
```

---

## 6. Demo 展示流程

1. 输入 V-JEPA 2、DINOv3 和 MAE 的机器人控制适用性问题。
2. 展示本地 PDF 与公开资料检索结果。
3. 展示 Evidence Quality Gate 如何拒绝参考文献页、搜索摘要和纯链接。
4. 在 Writer 输出前展示 Planned Claim Contract。
5. 展示 Writer 如何围绕核心 Claim 给出明确而有条件的判断。
6. 在 Claim Audit 面板查看每条 Claim 对应的文件、页码和原文片段。
7. 注入一条“V-JEPA 2 全面优于 DINOv3”的过强 Claim。
8. 展示 Active Verifier 只补证和修补这一条 Claim。
9. 展示修补前后 Diff、额外 Tool Call 和保留的原文比例。

完整演示控制在 3–5 分钟。

---

## 7. 项目评测

项目使用三组 Baseline：

1. **Writer Only**：检索后直接生成；
2. **Whole-answer Verifier**：生成后整体验证，失败时整篇重跑；
3. **Claim-first + Targeted Repair**：写作前 Claim Contract，写作后 Claim 级验证与局部修补。

评测集包含 30 条人工标注问题，覆盖：

- 论文事实；
- 多模型比较；
- 跨论文推断；
- 技术选择建议；
- 人为注入的过强或无支持 Claim。

核心指标：

| 指标 | 衡量内容 |
|---|---|
| Claim Evidence Coverage | 重要 Claim 是否具有直接证据 |
| Unsupported Claim Rate | 输出中无支持结论的比例 |
| Overclaim Detection Recall | 过强结论被发现的比例 |
| Claim–Evidence Alignment Accuracy | Claim 与证据是否真正相关 |
| Correct Content Preservation | 修补后保留正确原文的比例 |
| Rewrite Ratio | 被修改字符占原答案的比例 |
| Verification Tool Calls | 验证阶段增加的工具调用 |
| End-to-end Latency | 完整研究任务延迟 |

简历和 README 只使用最终真实实验结果，不预先填写提升比例。

---

## 8. 系统架构包装

```text
                         User Query
                              ↓
                         Hybrid Router
                    ┌─────────┼─────────┐
                 Direct      ReAct    Research
                                         ↓
                                      Planner
                                         ↓
                    Local PDF RAG / Paper / Web / MCP
                                         ↓
                                Evidence Quality Gate
                                         ↓
                                    Claim Planner
                                         ↓
                               Planned Claim Contract
                                         ↓
                                       Writer
                                         ↓
                      Claim Extractor → Evidence Aligner
                                         ↓
                                  Active Verifier
                              ┌──────────┴──────────┐
                            Pass             Targeted Repair
                              └──────────┬──────────┘
                                      Answer
                                         +
                         Claim–Evidence Provenance
```

技术栈：

- Python / LangGraph：状态化 Research Workflow；
- FAISS / BGE：本地论文语义检索；
- PyMuPDF：PDF 正文与页码提取；
- MCP：标准化外部工具接入与 Native fallback；
- SSE / Web UI：实时答案、Graph Trace 和 Claim Audit；
- Structured State / JSON Logs：保存 Evidence、Claim、Verification 与 Repair 过程。

技术栈服务于 Claim-first 和 Targeted Repair，不将“用了很多框架”作为项目价值本身。

---

## 9. 简历表述模板

### 中文版

**Evidence-First Research Agent｜个人项目**

- 针对 LLM 在科研阅读中倾向于总结资料、回避明确观点，以及生成观点后人工逐句求证成本高的问题，设计面向论文研究的 Claim-first Research Agent。
- 在 Writer 前加入 Evidence Quality Gate 与 Claim Planner，将来源、证据 ID、结论强度和适用条件组织为 Claim Contract，使 LLM 输出明确但有证据边界的研究判断。
- 实现 Claim–Evidence 对齐、主动验证和局部修补，对无支持、表达过强或存在冲突的 Claim 执行有限补充检索，只修改问题结论并保留其他正确内容。
- 基于 LangGraph、FAISS、BGE、PyMuPDF 与 MCP 实现本地论文 RAG、工具降级、SSE 实时 Claim Audit 和可复现运行日志；通过三组 Baseline 评估证据覆盖、过度断言检出、Tool Call 与改写范围。

### English Version

**Evidence-First Research Agent | Personal Project**

- Built a claim-first research agent to address two common LLM failure modes in scientific reading: avoiding falsifiable conclusions and making claims that require expensive manual source verification.
- Introduced a pre-writing Evidence Quality Gate and Claim Planner that encode sources, evidence IDs, allowed claim strength, and conditions into an evidence-backed Claim Contract.
- Implemented claim–evidence alignment, active verification, and targeted repair to retrieve additional evidence and revise only unsupported, overstated, or contradicted claims while preserving correct content.
- Built local-paper RAG and an auditable research workflow with LangGraph, FAISS, BGE, PyMuPDF, MCP fallback, SSE-based claim inspection, and structured experiment logs.

完成评测后，可增加一条带真实数字的结果：

> 在 30 条人工标注研究问题上，相比 Writer-only 与整篇重写 Baseline，将 Claim Evidence Coverage 提升至 `XX%`，同时将验证 Tool Call 或 Rewrite Ratio 降低 `XX%`。

---

## 10. 面试时的 90 秒讲法

> 我发现 LLM 在科研阅读中有两个相反的问题：第一，它很擅长总结论文，但面对“哪个方法更好、我应该选什么”时经常用“各有优劣”回避形成可检验观点；第二，它一旦给出观点，用户又需要逐句回到论文中找出处，检查这个观点究竟是原文事实、跨论文推断还是过度概括，这个求证过程非常费劲。我的项目把 Claim 从写作后的检查对象变成写作前的生成约束。系统先筛选能够作为直接支持的证据，再生成带证据 ID、允许强度和适用条件的 Claim Contract，Writer 必须围绕它写作。生成后，系统建立 Claim–Evidence 对应关系，对证据不足或表述过强的 Claim 发起有限补充检索并只修改这一处。旗舰 Demo 比较 V-JEPA 2、DINOv3 和 MAE 对机器人控制的适用性，既要求 Agent 给出明确选择，也要求它说明这个选择在哪些条件下成立、哪些部分还没有直接实验支持。

面试追问可以继续展开：

1. 为什么 Post-hoc Verifier 不能真正约束生成；
2. Claim Contract 如何让 LLM 敢于给出有边界的判断；
3. 如何区分论文事实、比较、推断和推荐；
4. 如何避免关键词重合被误判为证据支持；
5. Targeted Repair 如何保留其他正确内容；
6. 三组 Baseline 与人工评测如何设计。

---

## 11. 项目最终呈现

项目首页围绕一条清晰叙事展开：

```text
科研痛点：LLM 不愿形成明确 Claim，人工逐句求证成本高
                              ↓
方法一：Claim-first Generation
                              ↓
方法二：Targeted Claim Repair
                              ↓
旗舰 Demo：V-JEPA 2 vs DINOv3 vs MAE
                              ↓
证据：Claim–Evidence Provenance 与 Repair Diff
                              ↓
结果：三组 Baseline 的量化对比
```

最终展示材料包括：

- 一张 Claim-first 工作流架构图；
- 一张三组 Baseline 结果表；
- 一个 3–5 分钟旗舰 Demo；
- 一个过强 Claim 被定向修补的案例；
- 中英文项目介绍；
- 可复现的 Quick Start 和评测命令。

项目的记忆点不是“实现了一个功能很多的 Research Agent”，而是：

> **让 LLM 在科研阅读中敢于形成明确观点，同时让每个观点都可核查、可降级、可局部修补。**
