# V4 Evaluation

## V3 vs V4 对照

| Query | System | Success | Input Tokens | Output Tokens | Total Tokens | Avg Context/Step | Max Context | LLM Calls | Tool Calls | Repeated Tool Calls | Retrieval Calls | Latency | Evidence Count | Verifier Pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 根据本地论文，V-JEPA 2-AC 是如何使用 action 的？ | V3 Full Context | True | 3948 | 198 | 4146 | 1974.0 | 3840 | 2 | 1 | 0 | 1 | 0.024 | 5 | True |
| 根据本地论文，V-JEPA 2-AC 是如何使用 action 的？ | V4 Context Manager | True | 2064 | 68 | 2132 | 1392.5 | 3615 | 2 | 1 | 0 | 1 | 0.054 | 5 | True |
| 根据本地论文，DINOv3 的表示学习特点是什么？ | V3 Full Context | True | 5541 | 177 | 5718 | 2770.5 | 5435 | 2 | 1 | 0 | 1 | 0.023 | 5 | True |
| 根据本地论文，DINOv3 的表示学习特点是什么？ | V4 Context Manager | True | 2539 | 68 | 2607 | 1763.8 | 4608 | 2 | 1 | 0 | 1 | 0.074 | 5 | True |
| 比较 V-JEPA 2 和 DINOv3 在机器人控制中的潜在优势。 | V3 Full Context | True | 11924 | 450 | 12374 | 2981.0 | 11582 | 4 | 3 | 0 | 3 | 0.061 | 15 | True |
| 比较 V-JEPA 2 和 DINOv3 在机器人控制中的潜在优势。 | V4 Context Manager | True | 10038 | 136 | 10174 | 2950.7 | 8000 | 4 | 3 | 0 | 3 | 0.312 | 15 | True |
| 比较 V-JEPA 2、DINOv3、JEPA-WAM、Patch Policy，从训练目标、视觉表征、动作建模、机器人泛化、开源情况五个方面进行系统分析。 | V3 Full Context | True | 22712 | 848 | 23560 | 3785.3 | 22074 | 6 | 5 | 0 | 5 | 0.11 | 25 | True |
| 比较 V-JEPA 2、DINOv3、JEPA-WAM、Patch Policy，从训练目标、视觉表征、动作建模、机器人泛化、开源情况五个方面进行系统分析。 | V4 Context Manager | True | 16361 | 204 | 16565 | 2981.0 | 8000 | 6 | 5 | 0 | 5 | 1.674 | 25 | True |
| 根据本地论文，V-JEPA 2-AC 使用了什么机器人训练数据？ | V3 Full Context | True | 3189 | 196 | 3385 | 1594.5 | 3079 | 2 | 1 | 0 | 1 | 0.024 | 5 | True |
| 根据本地论文，V-JEPA 2-AC 使用了什么机器人训练数据？ | V4 Context Manager | True | 1689 | 68 | 1757 | 1113.0 | 2865 | 2 | 1 | 0 | 1 | 0.048 | 5 | True |

所有 token 为估算值：英文约 4 字符/token，CJK 约 1.8 字符/token；不是 provider billing token。V3 使用原始 Graph 路径，V4 使用默认 ContextConfig。

## Context Growth

长任务上下文序列：

| Sequence | V3 input tokens | V4 final context tokens |
|---:|---:|---:|
| 1 | 127 | 60 |
| 2 | 126 | 162 |
| 3 | 126 | 2263 |
| 4 | 127 | 2756 |
| 5 | 132 | 2891 |
| 6 | 22074 | 3048 |
| 7 | - | 4668 |
| 8 | - | 8000 |

目标趋势：V3 随历史/Tool 结果增长，V4 通过 Node-specific Context、Top-K 和 Budget 控制增长。

## 消融实验

| Variant | Tokens | Selected Evidence | Source/Page Preserved | Notes |
|---|---:|---:|---:|---|
| A: V3 Full Context | 5868 | 47 | True | 所有 Evidence 直接进入 Writer context |
| B: Compression only | 3000 | 47 | True | Raw Tool Result → 规则压缩，不做 Top-K |
| C: Compression + Evidence Top-K | 2423 | 10 | True | 从 47 条 Evidence 选择 10 条 |
| D: Full V4 | 2455 | 10 | True | Compression + Top-K + Node-specific Context + Budget |

## 压缩质量与 Failure Case 检查

| Case | Check | Result |
|---|---|---:|
| Compression too aggressive | source/page retained while long body is truncated | True |
| Verifier feedback loss | researcher context retains explicit verifier feedback | True |
| Source loss | compressed local result still contains Source and Page | True |

三个检查覆盖：过度压缩时正文会被截断、Verifier Feedback 不丢失、来源与页码不丢失。真正的语义准确率仍需要人工标注或真实 LLM judge；DemoLLM 只作为确定性 smoke test。

## 当前问题

- Planner 仍使用确定性规则生成 Plan。
- Token 估算不是 tokenizer 精确计费。
- Level-2 LLM Evidence Extractor 默认关闭；开启后会增加一次额外 LLM 调用。
- 默认 Evidence Selector 使用 BGE local score + 透明 lexical overlap；未额外创建持久化 Evidence FAISS 索引。
- InMemorySaver 和本次 Context Manager 均只服务单次进程内任务，不提供跨会话 Memory。
- V4 省 token 的收益在长任务更明显，短任务可能因标题/字段包装出现轻微额外开销。
