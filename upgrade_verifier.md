# upgrade_verifier.md

## 1. 升级目标

当前 Research Agent 的 Verifier 较简单：

```text
Writer
  ↓
Verifier
  ↓
Pass / Fail
```

这种方式主要依赖一次 LLM 判断，容易出现：

- 只做主观判断，没有真正核查证据；
- 无法明确指出具体哪条结论有问题；
- Verifier Fail 后只能让 Researcher 整体重做；
- 容易增加无效搜索和重复生成。

本次升级目标：

> 将 Verifier 从单一判断节点，升级为一个轻量 Verification Workflow。

---

## 2. 升级后的结构

```text
Writer
  ↓
Verifier Workflow
  ↓
① Error Detector
  ↓
② Verification Question Generator
  ↓
③ Verification Researcher
  ↓
④ Judge
  ↓
Pass / Fail + Feedback
```

接回主流程：

```text
Planner
  ↓
Researcher
  ↓
Writer
  ↓
Verifier Workflow
  ├─ Pass → END
  └─ Fail → Targeted Research / Rewrite
```

---

## 3. 各模块职责

### 3.1 Error Detector

输入：

```text
User Query
Final Answer
Evidence
```

输出潜在问题，例如：

```json
{
  "potential_errors": [
    {
      "claim": "V-JEPA 2 已完全开源机器人控制代码",
      "reason": "当前证据不足"
    }
  ]
}
```

重点不是直接判断答案错误，而是定位“值得进一步验证”的 Claim。

### 3.2 Verification Question Generator

把潜在错误转换成可搜索、可验证的问题。

例如：

```text
潜在问题：
V-JEPA 2 是否完全开源机器人控制代码？

↓

验证问题：
官方 GitHub 是否包含机器人控制训练与推理代码？
官方仓库是否提供对应 checkpoint？
```

### 3.3 Verification Researcher

针对验证问题调用现有工具：

```text
local_search
web_search
paper_search
GitHub MCP
```

执行方式保持轻量：

```text
Verification Question
  ↓
ReAct
  ↓
Tool
  ↓
Observation
```

默认限制较小的 tool-call budget，避免 Verifier 自己变成完整 Research Agent。

建议：

```text
max verification tool calls = 2~4
```

### 3.4 Judge

输入：

```text
Final Answer
Potential Errors
Verification Questions
Verification Evidence
```

输出：

```json
{
  "passed": false,
  "feedback": [
    {
      "claim": "...",
      "problem": "...",
      "suggested_action": "targeted_research"
    }
  ]
}
```

如果没有发现明显问题：

```json
{
  "passed": true,
  "feedback": []
}
```

---

## 4. Fail 后的处理

原方案：

```text
Verifier Fail
  ↓
整个 Researcher 重跑
```

升级后优先：

```text
Verifier Fail
  ↓
定位具体 Claim
  ↓
生成 Targeted Research Task
  ↓
只补缺失 Evidence
  ↓
Writer 修改对应内容
```

第一版不需要实现非常复杂的局部 patch。

可以先简单实现：

```text
Verifier Feedback
  ↓
Researcher 再搜索一次
  ↓
Writer 重新生成答案
```

但 Researcher 必须接收到具体 Feedback，而不是重新从零研究。

---

## 5. State 增加字段

建议增加：

```python
state = {
    ...
    "potential_errors": [],
    "verification_questions": [],
    "verification_evidence": [],
    "verifier_feedback": [],
    "verification_iteration": 0
}
```

---

## 6. 防止无限循环

增加：

```text
max_verification_iterations = 1~2
```

例如：

```text
Verifier Fail
  ↓
iteration < 2 ?
  ├─ Yes → Research Again
  └─ No  → 输出当前最佳答案
```

---

## 7. 保留旧版本

不要删除原 Verifier。

建议保留两个模式：

```text
simple_verifier
active_verifier
```

例如：

```bash
python main.py --verifier simple
python main.py --verifier active
```

便于最终做 baseline 对比。

---

# 8. 简化测试方案

当前只做 Smoke Test，不进行大规模 Benchmark。

## Test 1：明显正确答案

构造一个已有充分 Evidence 的问题。

预期：

```text
Writer
  ↓
Active Verifier
  ↓
Pass
  ↓
END
```

检查：

- Verifier 可以正常运行；
- 不触发无意义二次 Research；
- 系统不报错。

## Test 2：缺少关键证据

故意让 Writer 输出一条 Evidence 不足的 Claim。

预期：

```text
Error Detector
  ↓
找到对应 Claim
  ↓
生成 Verification Question
  ↓
调用 Tool
  ↓
Judge = Fail
```

检查：

- 是否成功识别问题；
- 是否生成具体 Feedback。

## Test 3：Verifier 触发补充研究

构造一个需要外部核查的问题。

预期：

```text
Writer V1
  ↓
Verifier Fail
  ↓
Researcher 根据 Feedback 补搜索
  ↓
Writer V2
```

检查：

- Feedback 是否传回 Researcher；
- Researcher 是否围绕缺失内容搜索；
- 最终答案是否发生合理修正。

## Test 4：循环限制

人为构造持续无法验证的问题。

检查：

```text
verification_iteration
```

达到：

```text
max_verification_iterations
```

后必须停止，不能死循环。

---

## 9. 当前阶段记录指标

只记录：

```text
Verifier Pass / Fail
Verification Tool Calls
Verification Iterations
是否触发二次 Research
最终是否成功结束
```

暂时不测试：

```text
RACE
FACT
Token Cost
Latency
大规模 Accuracy
```

这些放到最终统一 Evaluation。

---

## 10. 完成标准

```text
[ ] 原 simple_verifier 保留
[ ] active_verifier 可以独立运行
[ ] 可以识别潜在 Claim 问题
[ ] 可以生成 Verification Question
[ ] 可以调用现有 Tool 验证
[ ] 可以输出 Pass / Fail + Feedback
[ ] Fail 后可以触发一次补充 Research
[ ] 有最大 Verification Iteration 限制
[ ] 4 个 Smoke Test 基本通过
```

核心升级：

> **Verifier 从“单次 LLM Judge”升级为“发现问题 → 主动验证 → 生成反馈 → 必要时补充研究”的轻量 Verification Workflow。**
