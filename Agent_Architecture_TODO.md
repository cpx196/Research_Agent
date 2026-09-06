# Research Agent 后续架构优化待办

## 目标

当前 V3 已经形成基础 LangGraph Workflow：

```text
Planner
  ↓
Researcher
  ↓
Writer
  ↓
Verifier
  ↓
END / 回到 Researcher
```

这个版本作为 Baseline 保留，不删除。

后续目标不是简单增加更多角色，而是逐步验证：

```text
什么时候应该直接回答
什么时候应该使用 ReAct
什么时候需要 Planning
什么时候需要并行 Worker
什么时候需要 Verifier
什么时候应该由 Workflow 控制
什么时候应该让 LLM 自主决策
```

所有新方案必须：

```text
保留旧版本
+
新增实现
+
做对照实验
```

---

# TODO 1：加入 Router，避免所有问题都跑完整流程

## 当前问题

现在即使用户输入：

```text
你好
```

也会执行：

```text
Planner
→ Researcher
→ Writer
→ Verifier
```

这会造成不必要的：

```text
LLM Calls
Token Cost
Latency
```

## 目标

增加：

```text
Router
```

根据任务复杂度选择不同路径。

建议：

```text
                   User Query
                       ↓
                     Router
                 /      |       \
              Simple   Tool     Complex
                ↓       ↓          ↓
           Direct LLM  ReAct   Research Workflow
```

### Simple

例如：

```text
你好
Transformer 是什么
```

直接：

```text
LLM → Answer
```

### Tool

例如：

```text
3947 * 8123
查一下某个最新信息
```

进入：

```text
V0 ReAct Agent
```

### Complex

例如：

```text
比较 V-JEPA 2、DINOv3 和 JEPA-WAM
```

进入：

```text
Planner
→ Researcher
→ Writer
→ Verifier
```

## 测试

准备三类 Query：

```text
简单问答
单 Tool 任务
复杂 Research 任务
```

统计：

```text
Router Accuracy
LLM Calls
Tool Calls
Latency
Input Tokens
Task Success
```

---

# TODO 2：Planner 改成按需触发

## 当前问题

不是所有问题都需要 Planner。

例如：

```text
V-JEPA 2 发布时间是什么？
```

没必要：

```text
Planner：
步骤 1：查询发布时间
```

## 目标

Planner 只用于：

```text
需要多步骤
需要多来源
需要比较
需要多个子问题
```

的复杂任务。

简单研究任务可以：

```text
Router
↓
Researcher / ReAct
↓
Answer
```

## 测试

对同一批中等复杂度 Query：

```text
Always Planner
vs
Conditional Planner
```

比较：

```text
Task Success
LLM Calls
Latency
Token Cost
```

---

# TODO 3：Researcher 内部保留 ReAct Loop

## 目标

Researcher 不是一次函数调用后就结束。

Researcher 内部继续允许：

```text
LLM
↓
Tool Call
↓
Observation
↓
LLM
↓
继续 Tool / 返回 Evidence
```

即：

```text
外层：
Workflow

内层：
ReAct Agent
```

## 要求

继续复用：

```text
calculator
web_search
paper_search
local_search
```

不重新实现 Tool。

V0/V2 的 Tool Calling 逻辑继续保留作为 Baseline。

---

# TODO 4：加入 Orchestrator-Worker

## 当前问题

复杂任务如果只有一个 Researcher：

```text
任务 1
→ 任务 2
→ 任务 3
→ 任务 4
```

会串行执行，且一个 Researcher 需要同时维护大量信息。

## 目标

复杂任务由 Orchestrator 动态拆成多个子任务：

```text
              Orchestrator
                   ↓
       ┌───────────┼───────────┐
       ↓           ↓           ↓
    Worker 1    Worker 2    Worker 3
       ↕           ↕           ↕
     Tools       Tools       Tools
       \           |           /
        \          |          /
              Evidence
```

例如：

```text
比较 V-JEPA 2、DINOv3、JEPA-WAM
```

拆成：

```text
Worker 1：
研究 V-JEPA 2

Worker 2：
研究 DINOv3

Worker 3：
研究 JEPA-WAM
```

## 要求

Worker 可以并行执行。

Worker 内部仍然是：

```text
ReAct + Tools
```

## 测试

比较：

```text
Single Researcher
vs
Multiple Workers
```

统计：

```text
总延迟
Tool Calls
LLM Calls
Task Success
Evidence Coverage
Token Cost
```

---

# TODO 5：动态 Worker 数量

## 目标

Worker 数量不写死。

例如：

```text
简单任务
→ 1 Worker

双模型比较
→ 2 Workers

多模型综合调研
→ 3~5 Workers
```

由 Orchestrator 根据 Plan 动态创建。

## 重点观察

```text
Worker 太少
→ 信息覆盖不足

Worker 太多
→ 成本和重复搜索增加
```

需要实验寻找合理范围。

---

# TODO 6：Evidence 汇总层

## 当前问题

多个 Worker 会产生大量 Tool Results。

不能直接：

```text
所有 Tool Output
→ Writer
```

## 目标

统一形成结构化 Evidence：

```text
{
    claim
    source
    source_type
    page/url
    evidence_text
    relevance
    worker_id
}
```

然后：

```text
Workers
↓
Evidence Pool
↓
Writer
```

## 后续可接

```text
Evidence Deduplication
Evidence Ranking
Evidence Compression
```

---

# TODO 7：Verifier 改为按需触发

## 当前问题

当前所有任务都跑：

```text
Writer
↓
Verifier
```

简单任务没有必要。

## 目标

增加判断：

```text
Need Verification?
```

例如：

```text
简单问答
→ 不验证

单一明确事实
→ 可不验证

复杂比较 / 多源结论
→ 验证

证据不足 / 高风险结论
→ 必须验证
```

整体：

```text
Writer
  ↓
Need Verification?
  ├─ No → END
  └─ Yes → Verifier
```

---

# TODO 8：Verifier Fail 后动态回退

## 当前问题

Verifier Fail 后如果只是机械：

```text
Verifier
→ Researcher
```

还不够。

## 目标

Verifier 输出明确缺口：

```text
Missing Evidence:
DINOv3 robotics benchmark

Unsupported Claim:
xxx

Need:
additional local_search / web_search
```

Researcher 根据 Feedback 定向补充。

流程：

```text
Writer
↓
Verifier
↓
Fail
↓
Feedback
↓
Researcher
↓
补充 Evidence
↓
Rewrite
```

## 测试

构造：

```text
Evidence 不完整
来源冲突
关键结论缺支撑
```

检查能否正确恢复。

---

# TODO 9：限制 Verifier 循环次数

必须设置：

```text
max_verification_iterations
```

建议：

```text
2~3
```

达到上限后：

```text
输出当前最佳答案
+
标记证据不足
```

禁止无限：

```text
Research
→ Write
→ Verify
→ Research
→ ...
```

---

# TODO 10：保留 V3 四节点 Baseline

V3 固定结构必须继续存在：

```text
Planner
→ Researcher
→ Writer
→ Verifier
```

不得删除。

建议运行模式：

```bash
python main.py --mode graph_baseline
```

新方案：

```bash
python main.py --mode graph_dynamic
```

旧 ReAct：

```bash
python main.py --mode legacy
```

最终至少形成：

```text
Legacy ReAct
V3 Fixed Workflow
Dynamic Workflow
```

三个 Baseline。

---

# TODO 11：统一 Trace

动态 Agent 必须输出可读 Trace。

至少记录：

```text
Router Decision
Planner 是否触发
Plan
Worker 数量
Worker Task
Tool Call
Tool Arguments
Evidence
Writer
Verifier 是否触发
Verifier Result
Retry / Loop
Final Answer
```

示例：

```text
[Router]
Complex Research

[Planner]
3 subtasks

[Orchestrator]
Spawn 3 workers

[Worker 1]
V-JEPA 2
Tool: local_search

[Worker 2]
DINOv3
Tool: local_search

[Worker 3]
JEPA-WAM
Tool: paper_search

[Evidence Merge]
12 evidence records

[Writer]
Draft generated

[Verification Router]
Verification required

[Verifier]
Fail

Missing:
DINOv3 robotics evidence

[Worker 2]
Additional web_search

[Verifier]
Pass

[END]
```

---

# TODO 12：建立统一 Benchmark

不能只测试：

```text
能不能运行
```

必须建立测试集。

建议至少包含：

## A. 简单问题

```text
你好
什么是 Transformer
```

目的：

```text
验证 Router 是否能跳过复杂 Workflow
```

## B. 单 Tool

```text
3947 * 8123
查某篇论文
```

目的：

```text
验证 ReAct 路径
```

## C. 单主题 Research

```text
调研 V-JEPA 2 的机器人控制部分
```

## D. 多主题比较

```text
比较 V-JEPA 2 与 DINOv3
```

## E. 长 Research

```text
比较 V-JEPA 2、DINOv3、JEPA-WAM、Patch Policy，
从训练目标、视觉表征、动作建模、机器人泛化、
开源情况五个方面分析。
```

## F. Evidence 缺失

故意设计本地资料无法完全回答的问题。

## G. 冲突来源

不同来源提供不同信息。

验证 Agent 是否能识别冲突。

---

# TODO 13：统一指标

所有版本都记录：

```text
Task Success Rate
Final Answer Accuracy
Tool Selection Accuracy
Tool Calls
Repeated Tool Calls
LLM Calls
Input Tokens
Output Tokens
Total Tokens
Latency
Retrieval Calls
Evidence Count
Verifier Pass Rate
Recovery Rate
```

对于多 Worker 额外记录：

```text
Worker Count
Parallel Execution Time
Duplicate Evidence Rate
```

---

# TODO 14：主要对照实验

至少比较：

```text
A：Legacy ReAct

B：V3 Fixed Workflow
Planner
→ Researcher
→ Writer
→ Verifier

C：Dynamic Router
简单任务绕过完整流程

D：Dynamic Router
+
Conditional Planner

E：Orchestrator-Worker

F：Orchestrator-Worker
+
Conditional Verifier
```

---

# TODO 15：核心研究问题

最终项目不应该只是：

```text
“使用了 LangGraph”
```

而应该回答：

### 问题 1

```text
什么时候固定 Workflow 比 ReAct 更好？
```

### 问题 2

```text
什么时候 Planner 值得它额外增加的一次 LLM Call？
```

### 问题 3

```text
多 Worker 并行是否真的降低长 Research 延迟？
```

### 问题 4

```text
Worker 数量增加是否会导致重复搜索和 Token 浪费？
```

### 问题 5

```text
Verifier 是否提高 Task Success？
增加了多少 Cost？
```

### 问题 6

```text
什么时候应该由代码 Workflow 控制，
什么时候应该让 LLM 自主决策？
```

---

# 最终目标架构

```text
                           User
                             ↓
                           Router
                             ↓
               ┌─────────────┼─────────────┐
               ↓             ↓             ↓
          Direct LLM      ReAct       Orchestrator
                              ↕             ↓
                            Tools       Task Planning
                                            ↓
                                    Dynamic Workers
                                  ↙       ↓       ↘
                              Worker    Worker    Worker
                                ↕         ↕         ↕
                              Tools     Tools     Tools
                                  \       |       /
                                   Evidence Pool
                                        ↓
                                      Writer
                                        ↓
                                Need Verification?
                                  /            \
                                No              Yes
                                ↓                ↓
                               END           Verifier
                                               ↓
                                             Pass?
                                           /       \
                                         Yes        No
                                          ↓          ↓
                                         END     Research /
                                                 Rewrite
```

---

# 开发原则

后续所有 Agent 版本统一遵守：

```text
1. 不删除旧方案
2. 新方案使用独立模式 / 独立入口
3. 每增加一个模块，都明确它解决什么问题
4. 每增加一个模块，都设计 Baseline
5. 每增加一个模块，都记录 Token / Latency / Success
6. 不因为架构更复杂就默认它更好
7. 最终以实验结果决定模块是否保留
```

核心原则：

> **不是把 Agent 做得越来越复杂，而是通过实验确定哪些复杂度真正值得。**
