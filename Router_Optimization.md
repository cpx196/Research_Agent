# Research Agent Router 优化方案

## 1. 目标

当前 Research Agent 如果完全依赖固定规则做路由，例如：

```python
if "比较" in query:
    route = "complex"
elif "最新" in query:
    route = "web_search"
else:
    route = "simple"
```

存在明显问题：

```text
规则覆盖能力弱
自然语言表达变化大
复杂度判断不稳定
难以判断是否需要 Planning
难以判断是否需要多 Worker
难以判断是否需要 Verifier
```

因此后续 Router 不采用纯规则方案，而采用：

> **LLM Router + 少量确定性规则兜底 + LangGraph 动态执行。**

---

## 2. Router 的作用

Router 不负责真正执行任务。

Router 只负责判断：

```text
这个 Query 属于什么类型？
是否需要复杂 Workflow？
需要哪些能力？
```

建议输出：

```json
{
  "route": "direct | react | research",
  "complexity": "simple | tool | research",
  "need_planning": true,
  "need_web": true,
  "need_local_rag": false,
  "need_verification": true,
  "suggested_workers": 3,
  "confidence": 0.83
}
```

---

## 3. 整体架构

```text
                 User Query
                     ↓
              Hybrid Router
                     ↓
        ┌────────────┼────────────┐
        ↓            ↓            ↓
     Direct        ReAct       Research
                                   ↓
                              Planner?
                                   ↓
                              Workers?
                                   ↓
                                Writer
                                   ↓
                             Verifier?
```

其中：

```text
Direct = 直接回答
ReAct = LLM + Tool Calling + Observation 循环
Research = 复杂 Research Workflow
```

---

## 4. 为什么不使用纯规则 Router

纯规则示例：

```python
if "论文" in query:
    route = "research"
```

问题：

```text
“帮我看看 V-JEPA 2 的工作是怎么做的”
```

没有出现“论文”，但实际上仍然是 Research Task。

另外：

```text
“比较一下 2 和 3 哪个大”
```

虽然出现“比较”，但根本不需要 Planner。

因此：

```text
关键词 ≠ 任务语义
```

Router 需要理解：

```text
任务意图
任务复杂度
所需工具
是否需要多步推理
```

---

## 5. 混合路由策略

最终采用：

```text
确定性规则 + LLM Router
```

而不是纯规则，也不是所有请求都先走复杂大模型分类。

---

## 6. 第一层：确定性规则

只处理非常明确的情况。

### 简单寒暄

```text
你好
谢谢
你是谁
```

直接：

```text
Direct
```

### 明确数学表达式

```text
3947 * 8123
```

优先：

```text
ReAct → calculator
```

### 明确要求本地资料

例如：

```text
根据我本地论文回答
根据 papers/ 中的资料
```

必须确保：

```text
need_local_rag = true
```

### 明确要求最新信息

例如：

```text
今天
最新
现在是否开源
最近发布
```

至少标记：

```text
need_web = true
```

但最终是否走 Research 仍由 LLM 判断。

---

## 7. 第二层：LLM Router

无法由简单规则确定的 Query，交给轻量 LLM Router。

输入：

```text
User Query
+
可用工具列表
+
Router 任务定义
```

输出结构化 JSON。

建议：

```json
{
  "route": "research",
  "complexity": "research",
  "need_planning": true,
  "need_web": true,
  "need_local_rag": true,
  "need_verification": true,
  "suggested_workers": 3,
  "confidence": 0.87
}
```

---

## 8. Router 需要判断的内容

### 8.1 route

```text
direct
react
research
```

### 8.2 complexity

```text
simple
tool
research
```

### 8.3 need_planning

例如：

```text
查 V-JEPA 2 发布时间
```

通常：

```text
false
```

而：

```text
系统比较 V-JEPA 2、DINOv3、JEPA-WAM
```

通常：

```text
true
```

### 8.4 need_web

用于判断是否需要：

```text
web_search
GitHub MCP
```

### 8.5 need_local_rag

用于判断是否需要：

```text
local_search
```

### 8.6 need_verification

复杂 Research、来源冲突、长报告等：

```text
true
```

普通问答：

```text
false
```

### 8.7 suggested_workers

复杂任务可以建议：

```text
1
2
3
...
```

例如：

```text
比较 V-JEPA 2、DINOv3、JEPA-WAM
```

可以：

```text
suggested_workers = 3
```

---

## 9. 示例

输入：

```text
帮我比较 V-JEPA 2 和 DINOv3
在机器人控制上的优劣，
最好结合论文和开源情况。
```

Router：

```json
{
  "route": "research",
  "complexity": "research",
  "need_planning": true,
  "need_web": true,
  "need_local_rag": true,
  "need_verification": true,
  "suggested_workers": 3,
  "confidence": 0.83
}
```

LangGraph 根据结果执行：

```text
Router
 ↓
Planner
 ↓
Workers
 ↓
Evidence
 ↓
Writer
 ↓
Verifier
```

---

## 10. 简单任务示例

输入：

```text
你好
```

Router：

```json
{
  "route": "direct",
  "complexity": "simple",
  "need_planning": false,
  "need_web": false,
  "need_local_rag": false,
  "need_verification": false,
  "suggested_workers": 0,
  "confidence": 0.99
}
```

直接：

```text
LLM → Answer
```

禁止继续跑完整四节点流程。

---

## 11. 单 Tool 示例

输入：

```text
V-JEPA 2 官方代码现在开源了吗？
```

可能：

```json
{
  "route": "react",
  "complexity": "tool",
  "need_planning": false,
  "need_web": true,
  "need_local_rag": false,
  "need_verification": false,
  "suggested_workers": 1,
  "confidence": 0.90
}
```

执行：

```text
ReAct
 ↓
web_search / GitHub MCP
 ↓
Observation
 ↓
Answer
```

---

## 12. 复杂任务示例

输入：

```text
比较 V-JEPA 2、DINOv3 和 JEPA-WAM，
从训练目标、视觉表征、动作建模、
机器人泛化和开源情况五个方面分析。
```

可能：

```json
{
  "route": "research",
  "complexity": "research",
  "need_planning": true,
  "need_web": true,
  "need_local_rag": true,
  "need_verification": true,
  "suggested_workers": 3,
  "confidence": 0.91
}
```

执行：

```text
Planner
 ↓
Orchestrator
 ↓
Parallel Workers
 ↓
Evidence Pool
 ↓
Writer
 ↓
Verifier
```

---

## 13. Confidence

Router 可以输出：

```text
confidence
```

第一版建议只记录，不急着做复杂多级 Router。

后续可以：

```text
confidence >= 0.8 → 直接执行
0.5 <= confidence < 0.8 → 走保守路径
confidence < 0.5 → 强模型重新判断
```

---

## 14. 为什么第一版不做复杂 Confidence 策略

当前重点仍然是：

```text
先搭完整 Agent Framework
```

因此第一版：

```text
Router 输出 confidence
+
记录到 Trace
```

即可。

复杂的：

```text
双模型 Router
置信度校准
模型级联
```

留到最终优化阶段。

---

## 15. Router 使用什么模型

优先：

```text
较小 / 较快模型
```

例如：

```text
Router → 小模型
Planner / Writer → 强模型
```

原因：

Router 主要做：

```text
分类
结构化判断
```

不需要长文本生成。

如果当前只有一个 LLM API，也可以先用同一个模型配 Router Prompt。

---

## 16. Router Prompt

建议：

```text
You are a routing module for a research agent.

Classify the user query and return structured JSON.

Possible routes:
- direct
- react
- research

Determine:
- complexity
- whether planning is needed
- whether web access is needed
- whether local RAG is needed
- whether verification is needed
- suggested worker count
- confidence

Do not answer the user query.
Only output routing decisions.
```

---

## 17. LangGraph 接入

Router 作为 Graph 入口：

```text
START
 ↓
Router
```

然后通过 Conditional Edge：

```text
Router
 ↓
route?
 ├─ direct   → DirectAnswer
 ├─ react    → ReActAgent
 └─ research → Planner
```

---

## 18. Research 路径内部

```text
Planner
 ↓
Need Workers?
 ↓
Researcher / Workers
 ↓
Writer
 ↓
Need Verification?
 ├─ No → END
 └─ Yes → Verifier
```

---

## 19. ReAct 和 Router 的关系

Router 决定：

```text
是否进入 ReAct
```

ReAct 决定：

```text
进入之后下一步调用什么 Tool
```

因此：

```text
Router = 高层路径选择
ReAct = 局部动态执行
```

---

## 20. Workflow 和 Router 的关系

Router 不负责执行完整 Workflow。

Router 只负责选择走哪条 Workflow。

真正执行由 LangGraph 控制。

---

## 21. 最终控制关系

```text
规则
↓
处理确定性边界

LLM Router
↓
理解任务语义

LangGraph
↓
控制流程

ReAct
↓
执行局部 Tool Calling

Tools / MCP
↓
与外部环境交互
```

核心原则：

> **规则负责确定性边界，LLM 负责语义判断，Workflow 负责执行控制，ReAct 负责局部自主工具调用。**

---

## 22. 当前阶段测试要求

现在不做大规模 Benchmark。

只做最小 Smoke Test：

```text
1. 简单问题能正确 Direct
2. 单 Tool 问题能正确进入 ReAct
3. 复杂 Research 能正确进入 Research
4. 明确本地资料问题能标记 need_local_rag
5. 最新信息问题能标记 need_web
6. Router 输出结构化 JSON
7. Router 失败时有 fallback
```

---

## 23. Router Fallback

如果 Router 输出解析失败：

```text
Invalid JSON
Missing Field
Unknown Route
```

不要让 Agent 崩溃。

建议：

```text
Router Error
 ↓
Fallback
 ↓
ReAct
```

第一版推荐保守进入 ReAct，避免错误地直接回答复杂任务。

---

## 24. 旧方案保留

原 V3：

```text
Planner
→ Researcher
→ Writer
→ Verifier
```

必须保留。

新增：

```text
Dynamic Router Workflow
```

建议：

```bash
python main.py --mode graph_baseline
python main.py --mode graph_dynamic
```

---

## 25. 后续统一评测

最终测试阶段再统一比较：

```text
固定四节点 Workflow
vs
纯规则 Router
vs
LLM Router
vs
Hybrid Router
```

指标：

```text
Task Success
Routing Accuracy
LLM Calls
Tool Calls
Input Tokens
Latency
Token Cost
```

当前阶段暂不执行大规模测试。

---

## 26. 最终架构

```text
                         User
                           ↓
                    Deterministic Rules
                           ↓
                     LLM-based Router
                           ↓
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Direct         ReAct        Research
                            ↕             ↓
                          Tools        Planner
                                         ↓
                                     Workers
                                         ↓
                                     Evidence
                                         ↓
                                      Writer
                                         ↓
                                Need Verification?
                                  /                                            No              Yes
                                ↓                ↓
                               END           Verifier
```

---

## 27. 核心价值

Router 优化的目标不是增加一个新的“角色”，而是解决：

```text
简单任务被过度编排
复杂任务又需要更强控制
```

最终希望做到：

> **根据 Query 的真实语义和复杂度，动态选择 Direct、ReAct 或 Research Workflow，只在必要时支付 Planning、Worker、Writer、Verifier 的额外成本。**
