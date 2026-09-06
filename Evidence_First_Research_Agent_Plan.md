# Evidence-First Research Agent 方案

## 1. 项目定位

本项目不再以“覆盖尽可能多的 Agent 组件”为主要目标，而是聚焦于一个明确问题：

> Research Agent 如何对自己生成的关键结论进行细粒度证据审计，并以有限的额外检索成本，只修补存在问题的 Claim？

项目名称暂定为：

> **Evidence-First Research Agent：面向论文研究的 Claim 级证据验证与定向修补系统**

现有 Router、LangGraph、Context Manager、MCP、Tool Adapter、Skill、Web UI 等模块继续保留，但统一降级为支撑基础设施，不再作为项目的主要创新点。

项目只突出两个相互衔接的能力：

1. **Claim-Level Active Verification**：将回答拆解为可验证 Claim，判断每条 Claim 是否得到证据支持。
2. **Targeted Claim Repair**：只对证据不足、表述过强或存在冲突的 Claim 补证和改写，不重新执行完整研究流程。

---

## 2. 核心概念

### 2.1 Claim 不是“大胆断言”

本项目中的 Claim 是答案中可独立核查的事实性结论或分析判断。

系统目标不是让 Agent 无条件给出更强结论，而是让它：

- 证据充分时，给出明确结论；
- 证据部分充分时，说明结论成立的条件；
- 缺少直接证据时，区分“论文事实”和“系统推断”；
- 证据冲突时，明确展示分歧；
- 无法验证时，降低表述强度或删除 Claim。

### 2.2 Claim 的类型

建议将 Claim 分为四类：

| 类型 | 含义 | 示例 |
|---|---|---|
| Fact | 来源直接陈述的事实 | V-JEPA 2 使用视频进行自监督预训练 |
| Comparison | 多个对象之间的比较 | DINOv3 的密集空间特征比普通 MAE 更适合目标定位 |
| Inference | 根据事实推导出的判断 | 时序预测表征更适合需要动态建模的控制任务 |
| Recommendation | 面向用户问题的最终选择 | 动态规划型机器人任务优先考虑 V-JEPA 2 |

不同 Claim 使用不同验证标准：

- Fact 需要直接来源支持；
- Comparison 最好需要统一指标或可比实验；
- Inference 需要清晰的推理链和多个前提证据；
- Recommendation 必须声明任务条件、证据边界和不确定性。

### 2.3 支持状态

每条 Claim 最终只能进入以下状态之一：

```text
SUPPORTED             证据直接且充分支持
PARTIALLY_SUPPORTED   证据支持部分内容或仅支持特定条件
INFERRED              由已有证据合理推导，但不是来源直接结论
CONTRADICTED          存在直接反证或来源冲突
UNSUPPORTED           当前没有足够证据
```

---

## 3. 旗舰 Demo

### 3.1 Demo 问题

> 根据本地论文与公开资料，比较 V-JEPA 2、DINOv3 和普通 MAE Encoder 所学习视觉表征的特点。分析这些表征在空间感知、时序建模、物体交互和机器人控制中的适用性，并判断哪一种更有利于机器人控制，说明判断成立的条件和证据。

### 3.2 为什么选择这个 Demo

该问题同时包含：

- 三种视觉表征的多对象比较；
- 本地论文检索；
- 公开资料补充；
- 论文事实与系统推断的区分；
- 从表征特点到机器人控制价值的推理；
- 一个必须给出明确判断、但不能过度概括的问题；
- 一个非常适合展示 Claim 审计与局部修补的高风险结论。

### 3.3 预期比较维度

Agent 不应直接问“谁最好”，而应先建立统一比较维度：

| 维度 | 需要核查的问题 |
|---|---|
| 训练目标 | 模型通过什么自监督目标学习表征？ |
| 空间语义 | 是否能稳定表达物体、区域和全局语义？ |
| 密集特征 | Patch-level 特征是否适合定位、对应和交互区域识别？ |
| 时序建模 | 是否原生利用视频历史和时间变化？ |
| 动态预测 | 是否能表达未来状态或动作引起的变化？ |
| 冻结迁移 | 冻结 Encoder 后在下游任务中的可用性如何？ |
| 数据与算力 | 使用成本、模型规模和部署限制是什么？ |
| 控制适配 | 更适合感知型、反应型还是规划型机器人控制？ |

### 3.4 预期结论形式

最终答案不能简单输出“V-JEPA 2 最好”。推荐的结论形式是：

> 如果机器人任务主要依赖静态场景理解、目标定位和密集视觉感知，DINOv3 可能是更实用的通用视觉 Encoder；如果任务需要利用视频历史理解环境动态、预测交互结果或进行长时规划，V-JEPA 2 的训练目标与控制需求更加一致。普通 MAE 可以作为强视觉预训练基线，但像素重建目标与动作相关动态之间的联系更间接。

如果必须给出总体选择：

> 在需要动态理解和规划的机器人控制任务中，V-JEPA 2 具有更匹配的归纳偏置；在以当前帧感知为主、依赖高质量冻结视觉特征的任务中，DINOv3 可能更实用。除非存在统一控制基准上的直接对照，否则不能宣称任何一种表征在所有机器人控制任务中全面最优。

上述内容只是 Demo 的预期论证方向，不应作为系统预置答案。最终结论必须由实际检索证据生成。

---

## 4. 目标工作流

```text
User Query
    ↓
Research Scope Builder
    ↓
Evidence Retrieval
    ├── Local Paper Search
    ├── Paper Metadata Search
    └── Public / Official Web Search
    ↓
Evidence Normalization
    ↓
Initial Writer
    ↓
Claim Extractor
    ↓
Claim–Evidence Aligner
    ↓
Claim Risk Detector
    ├── 全部通过 ───────────────→ Final Answer
    └── 存在问题
            ↓
    Verification Question Generator
            ↓
    Targeted Evidence Retrieval
            ↓
    Claim Verification Judge
            ↓
    Localized Claim Repair
            ↓
    Final Answer + Claim Audit Report
```

### 4.1 Research Scope Builder

职责：

- 从用户问题中提取比较对象；
- 建立统一比较维度；
- 区分需要本地证据和公开证据的子问题；
- 避免一开始就生成未经验证的胜负结论。

旗舰 Demo 的建议子问题：

1. 三种方法的训练目标分别是什么？
2. 三种表征在静态语义和密集空间特征方面有什么差异？
3. 哪些方法原生建模视频时序或未来动态？
4. 论文是否提供机器人控制或规划实验？
5. 是否存在同一控制基准上的直接对照？
6. 哪些结论是论文事实，哪些只是合理推断？

### 4.2 Initial Writer

初稿必须满足：

- 使用统一维度比较三种方法；
- 重要事实附带来源；
- 明确区分事实、推断和推荐；
- 必须回答“哪个更有利于机器人控制，为什么”；
- 不因缺少直接实验而回避判断；
- 不把训练目标匹配度直接写成已证明的性能优势。

### 4.3 Claim Extractor

只提取值得验证的核心 Claim，不把所有句子都拆成 Claim。

建议每份回答限制为 6～15 条核心 Claim，并保留 Claim 在原文中的位置。

输出示例：

```json
{
  "claim_id": "C7",
  "text": "V-JEPA 2 在机器人控制上优于 DINOv3 和普通 MAE。",
  "type": "recommendation",
  "importance": 0.95,
  "risk": "high",
  "answer_span": {
    "section": "最终判断",
    "start": 1820,
    "end": 1864
  }
}
```

### 4.4 Claim–Evidence Aligner

对每条 Claim 查找直接支持、部分支持和冲突证据。

不能只检查答案中是否出现 URL 或论文名，而要判断 Evidence 是否在语义上支持对应 Claim。

输出至少包含：

- 直接证据；
- 反向证据；
- 证据来源质量；
- 是否属于同一实验设置；
- Claim 与 Evidence 的蕴含关系；
- 当前支持状态。

### 4.5 Claim Risk Detector

优先验证以下高风险 Claim：

- 最好、最强、全面优于、首次、唯一；
- 已开源、已经发布、包含完整代码；
- 明确的数字、日期和排名；
- 从表征特点直接跳到控制性能；
- 跨论文实验结果的非统一比较；
- 没有直接控制实验却声称控制效果更好；
- 来源之间出现结论冲突。

### 4.6 Targeted Evidence Retrieval

每个问题 Claim 生成一个独立、可搜索的验证问题。

例如：

```text
Claim：V-JEPA 2 在机器人控制上全面优于 DINOv3 和 MAE。

Verification Question：
是否存在 V-JEPA 2、DINOv3 和 MAE 在相同机器人控制任务、
数据规模和训练设置下的直接对照实验？
```

定向检索必须：

- 只服务于当前 Claim；
- 优先选择最权威的来源；
- 避免重复调用已经执行过的相同查询；
- 默认最多进行 1～2 次 Tool Call；
- 达到预算后停止，并保留不确定性。

### 4.7 Localized Claim Repair

修补动作分为四种：

```text
STRENGTHEN   找到充分证据，保留或增强结论
QUALIFY      加入适用条件，降低结论强度
REPLACE      用证据支持的新 Claim 替换原 Claim
REMOVE       无法支持且非必要，删除 Claim
```

修补时只允许修改：

- 问题 Claim 所在句子；
- 与该 Claim 直接相关的解释句；
- 最终结论中依赖该 Claim 的局部表述。

不得重新生成全文。未被标记的问题段落必须保持不变。

---

## 5. Claim 与 Evidence 数据结构

### 5.1 Claim

```python
class Claim(TypedDict, total=False):
    claim_id: str
    text: str
    claim_type: str
    importance: float
    risk: str
    status: str
    answer_section: str
    answer_start: int
    answer_end: int
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    verification_question: str
    repair_action: str
    repaired_text: str
```

### 5.2 Evidence

```python
class ClaimEvidence(TypedDict, total=False):
    evidence_id: str
    claim_id: str
    stance: str
    source: str
    source_type: str
    source_authority: float
    title: str
    authors: list[str]
    published_at: str
    retrieved_at: str
    page: int | str
    url: str
    quote: str
    context: str
    retrieval_score: float
    entailment_score: float
    comparable_setting: bool
    worker_id: str
```

### 5.3 Claim Audit

```python
class ClaimAudit(TypedDict, total=False):
    claim_id: str
    original_claim: str
    final_claim: str
    original_status: str
    final_status: str
    problem: str
    verification_question: str
    new_evidence_ids: list[str]
    repair_action: str
    changed: bool
```

---

## 6. Demo 中应重点展示的修补案例

### 6.1 初始过强结论

```text
V-JEPA 2 在机器人控制上优于 DINOv3 和普通 MAE。
```

### 6.2 Verifier 判断

```json
{
  "claim": "V-JEPA 2 在机器人控制上优于 DINOv3 和普通 MAE。",
  "status": "UNSUPPORTED",
  "risk": "high",
  "problem": "现有证据来自不同任务和实验设置，不能形成统一性能比较。",
  "required_evidence": "相同机器人控制基准和训练设置下的直接对照实验"
}
```

### 6.3 定向检索

```text
是否存在 V-JEPA 2、DINOv3 和普通 MAE 在相同机器人控制基准上的直接实验对比？
```

### 6.4 找不到直接证据时的局部修补

```diff
- V-JEPA 2 在机器人控制上优于 DINOv3 和普通 MAE。

+ V-JEPA 2 的时序预测目标与需要动态理解和规划的机器人任务更匹配，
+ 因而在这类任务中具有更有利的归纳偏置；但在缺少统一控制基准直接
+ 对照的情况下，不能断言它在所有机器人控制任务中都优于 DINOv3 和 MAE。
```

这段差异应成为旗舰 Demo 的视觉中心。

---

## 7. UI 展示方案

现有 Execution Trace 继续保留，但默认折叠。主要界面突出三个区域。

### 7.1 Comparison

展示三种表征在统一维度下的比较表，并允许点击单元格查看来源。

### 7.2 Claim Audit

| Claim | 类型 | 状态 | 风险 | 证据 |
|---|---|---|---|---|
| V-JEPA 2 原生建模视频动态 | Fact | Supported | 低 | 本地论文第 X 页 |
| DINOv3 提供较强密集空间特征 | Fact | Supported | 中 | 本地论文第 X 页 |
| V-JEPA 2 在所有控制任务中最好 | Recommendation | Unsupported | 高 | 无直接统一对照 |
| V-JEPA 2 更适合动态预测型控制 | Inference | Partially Supported | 中 | 多条间接证据 |

### 7.3 Repair Timeline

```text
Original Claim
    ↓
Detected Problem
    ↓
Verification Question
    ↓
New Evidence
    ↓
Repair Action
    ↓
Final Claim
```

使用 Diff 显示局部修改，并展示：

- 修改了哪些 Claim；
- 哪些段落保持不变；
- 额外执行了多少次 Tool Call；
- 验证增加了多少延迟；
- 最终结论为什么比初稿更可靠。

---

## 8. 评测方案

### 8.1 Baseline

至少比较三种模式：

1. **Writer Only**：检索后直接生成答案；
2. **Whole-Answer Verifier**：对整篇答案做 Pass/Fail 并整体重写；
3. **Claim-Level Verification + Targeted Repair**：本项目方案。

### 8.2 核心指标

| 指标 | 含义 |
|---|---|
| Unsupported Claim Rate | 最终答案中无充分证据 Claim 的比例 |
| Citation Entailment | 引用是否真正支持对应 Claim |
| Claim Coverage | 重要 Claim 中具备证据映射的比例 |
| Error Detection Recall | Verifier 发现错误 Claim 的能力 |
| Verifier Precision | 被标记 Claim 中真正存在问题的比例 |
| Repair Success Rate | 问题 Claim 被正确修补的比例 |
| Correct Claim Preservation | 修补后原本正确 Claim 的保留比例 |
| Conclusion Usefulness | 最终答案是否给出清晰、有条件的决策 |
| Extra Tool Calls | 验证阶段额外 Tool Call 数量 |
| Verification Latency | 验证和修补引入的额外延迟 |
| Token Overhead | 相对 Writer Only 增加的 Token |

### 8.3 Demo 专项标注集

围绕 V-JEPA 2、DINOv3、MAE 和机器人控制，先建立 30～50 条高质量测试问题，覆盖：

- 论文明确事实；
- 训练目标比较；
- 密集特征比较；
- 时序能力比较；
- 控制实验定位；
- 不同实验设置的错误横向比较；
- 无直接证据的“谁最好”问题；
- 最新开源、代码和权重状态；
- 故意带有错误前提的问题；
- 需要从强结论降级为条件化结论的问题。

每条样本人工标注：

- 核心 Claim；
- 支持证据；
- 反对证据；
- 可接受结论；
- 不可接受的过强表述；
- 推荐的修补动作。

### 8.4 目标结果表

最终报告应围绕以下表格展开：

| 方法 | 无依据论断率 ↓ | 修补成功率 ↑ | 正确内容保留率 ↑ | 额外 Tool Calls ↓ | 额外延迟 ↓ |
|---|---:|---:|---:|---:|---:|
| Writer Only | 待测 | — | — | 0 | 0 |
| Whole-Answer Verifier | 待测 | 待测 | 待测 | 待测 | 待测 |
| Claim-Level + Targeted Repair | 待测 | 待测 | 待测 | 待测 | 待测 |

不得预填理想化结果，所有数据由固定测试集和真实运行产生。

---

## 9. 实施计划

### 阶段一：Claim 数据层

目标：建立 Claim、ClaimEvidence、ClaimAudit 三类核心结构。

任务：

- 扩展 ResearchState；
- 实现 Claim Extractor；
- 为 Claim 保留原文位置；
- 建立 Claim–Evidence 多对多关系；
- 支持五种证据状态；
- 为现有 Evidence 生成稳定 ID。

验收标准：

- 对旗舰 Demo 初稿能稳定提取 6～15 条关键 Claim；
- 每条 Claim 能映射到零条或多条 Evidence；
- Claim Audit 可以完整序列化到运行日志。

### 阶段二：Claim 级验证

目标：从整篇答案验证升级为逐 Claim 风险判断。

任务：

- 实现 Claim 风险评分；
- 实现 Evidence 蕴含判断；
- 区分 Fact、Comparison、Inference、Recommendation；
- 识别跨论文不可比实验；
- 输出结构化验证问题。

验收标准：

- 能识别“V-JEPA 2 在所有控制任务中最好”为高风险 Claim；
- 不把“V-JEPA 2 使用视频预训练”等明确事实误判为高风险；
- Verifier 输出可以直接驱动定向检索。

### 阶段三：局部修补

目标：只修改失败 Claim 及其直接依赖内容。

任务：

- 实现稳定的答案 Span 定位；
- 实现 Strengthen、Qualify、Replace、Remove；
- 修补前后生成结构化 Diff；
- 验证未标记段落保持不变；
- 限制定向检索和修补轮次。

验收标准：

- 找不到统一对照证据时，将绝对胜负结论降级为条件化结论；
- 初稿中正确事实和引用不因修补被重写；
- 达到最大验证预算后能保留明确的不确定性说明。

### 阶段四：旗舰 Demo 与评测

目标：形成可重复演示和可信实验结果。

任务：

- 补充 MAE 原始论文及必要公开资料；
- 为三种方法建立统一检索元数据；
- 建立 30～50 条人工标注测试集；
- 实现三种 Baseline；
- 输出 Claim Audit、Repair Diff 和指标报告；
- 调整 Web UI 的视觉重点。

验收标准：

- 一条命令可完整运行旗舰 Demo；
- 三种 Baseline 使用同一数据和问题；
- 结果可复现；
- UI 能清楚展示“原 Claim—问题—补证—修补后 Claim”。

---

## 10. 明确不做的内容

为了保持项目聚焦，本阶段暂不扩展：

- 通用多 Agent 平台；
- 动态创建大量 Worker；
- 更多无关 MCP Tool；
- GitHub、数据库、文件系统等工具大全；
- 长期用户画像和跨会话记忆；
- 通用知识问答能力竞赛；
- 复杂权限和多租户系统；
- 仅为展示而增加的角色节点；
- 与 Claim 验证无关的大规模前端改版。

如果后续引入 Worker，只允许用于并行验证不同高风险 Claim，并必须通过实验说明其收益。

---

## 11. 现有模块的重新定位

| 现有模块 | 新定位 |
|---|---|
| Router | 决定是否进入深度研究，不作为创新点 |
| Planner | 建立比较维度和研究范围 |
| Researcher | 收集初始 Evidence |
| Context Manager | 控制 Claim、Evidence 和验证上下文预算 |
| MCP | 标准化外部证据来源接入 |
| Skill | 封装固定论文研究流程 |
| Writer | 生成可拆解、可引用的初稿 |
| Active Verifier | 升级为 Claim Auditor |
| Repair | 升级为 Localized Claim Repair |
| Web UI | 展示比较结果、Claim Audit 和 Repair Diff |
| Trace | 调试信息，默认折叠 |

---

## 12. 项目最终叙事

项目介绍不再以 V0、V2、V3、V4、V5 的技术演进开场，而采用以下叙事：

> 现有 Research Agent 通常能够检索资料并生成带引用的答案，但“带引用”并不意味着每个重要结论都得到证据支持。尤其在跨论文比较中，Agent 容易把不同实验条件下的结果直接比较，或者把训练目标上的合理推断写成已经被实验证明的性能优势。
>
> Evidence-First Research Agent 将答案拆解为独立 Claim，为每条 Claim 建立证据关系，主动识别无证据、弱证据和冲突证据，并只对存在问题的 Claim 进行定向检索与局部修补。系统既要形成明确判断，也要说明判断成立的条件和证据边界。

最终需要证明的核心命题是：

> 与整篇重写相比，Claim 级主动验证和局部修补能以更少的额外检索成本，降低无依据论断率，同时更好地保留原答案中的正确内容。

