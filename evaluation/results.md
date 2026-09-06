# Research Agent V2 实测结果

测试日期：2026-08-29。机器：macOS 26.6.2、Apple Silicon arm64，无独立显卡。

## 1. 数据集与索引

数据源是 `/Users/ocbrain/Zotero` 的 `UST` collection。由于 Zotero 正在占用主库，导入器自动使用只读备份 `zotero.sqlite.bak`；不会修改 Zotero 数据。

| 项目 | 实测值 |
| --- | --- |
| PDF Parser | PyMuPDF 1.28.2 |
| Embedding Model | `BAAI/bge-small-en-v1.5` |
| Embedding dimension | 384 |
| Device | CPU |
| Chunk size / overlap | 2000 / 200 characters |
| PDF 数量 | 10 |
| 有文本 Page 数 | 350 |
| Chunk 数量 | 832 |
| 文档 embedding 耗时 | 77.35 s（首次 CPU 建库） |
| FAISS | `IndexFlatIP`；向量归一化后等价 Cosine Similarity |
| FAISS index 大小 | 1,277,997 bytes |
| metadata 大小 | 1,480,704 bytes |
| warm Top-K retrieval 平均耗时 | 0.019 s |

10 篇 PDF 已复制到 `papers/`，名单和 Zotero provenance 在 `papers/manifest.json`。其中明确包含 4 篇 JEPA 相关论文和 DINOv3：

```text
vjepa2.pdf
vjepa2_1.pdf
jepa_images.pdf
vla_jepa.pdf
dinov3.pdf
world_action_models.pdf
fast_wam.pdf
lawam.pdf
patch_policy.pdf
causal_world_modeling.pdf
```

## 2. 五个 Retrieval Query

`Relevant Top-K` 以是否出现与问题直接相关的原文段落为准；V2 默认 BGE 是英文模型，因此含大量中文、但没有英文主题词的泛化问题会有明显语言限制。

| Query | Top-1 / 代表结果 | Correct source | Correct page area | Relevant Top-K |
| --- | --- | ---: | ---: | ---: |
| V-JEPA 2 的核心训练目标是什么？ | `vjepa2_1.pdf`, p.1；Top-5 另含 `vjepa2.pdf`, p.3 | Yes | Partial | Partial |
| V-JEPA 2-AC 是如何使用 action 的？ | `vjepa2.pdf`, p.8, score 0.7536 | Yes | Yes | Yes |
| V-JEPA 2-AC 使用了什么机器人训练数据？ | `vjepa2.pdf`, p.8, score 0.7165 | Yes | Yes | Yes |
| 比较 V-JEPA 2 和其他 JEPA-style 方法 | `vjepa2_1.pdf`, p.1, score 0.7346 | Yes | Partial | Yes |
| 根据本地资料给出论文名和页码 | 问题缺少主题约束；返回 `dinov3.pdf`, p.35 | Partial | No | No |

对 Q1、Q3 使用明确英文主题词的复核查询成功命中：

```text
V-JEPA 2 core pretraining objective → vjepa2.pdf p.3 / p.4 / p.8
V-JEPA 2-AC robot control training data Droid dataset → vjepa2.pdf p.8
```

## 3. 三条完整 Agent Trace

使用 `DemoLLM + local_search` 运行，三条均为 `steps=2、tool_calls=1、tool_errors=0`，且第二步只收到 `local_search` Observation 后生成最终回答：

1. `V-JEPA 2-AC` 的 action 使用方式：命中 `vjepa2.pdf` p.8，最终回答引用 p.4、p.8。
2. `V-JEPA 2` 的训练目标：最终回答引用 `vjepa2.pdf` p.4 的 masked-patch / EMA-target / L1 loss。
3. `V-JEPA 2-AC` 的机器人数据：命中 `vjepa2.pdf` p.8–9，最终回答给出 Droid、约 62 小时、无标注、远程操作 Franka Emika Panda 视频和末端状态信号。

完整原始控制流（包括 Arguments、RAG 状态、5 条 Observation 结果和 Final Answer）见 [traces.md](traces.md)。可重复生成：

```bash
.venv-v2/bin/python -m evaluation.run_trace
```

## 4. 测试结果

```text
Ran 13 tests in 0.005s
OK
```

覆盖：chunk overlap、页码/source metadata、FAISS 与 metadata 对齐、local_search schema、缺失向量库错误、Agent Tool Loop 以及原有 calculator/agent 行为。

## 5. 当前问题

- `BAAI/bge-small-en-v1.5` 对纯中文泛化 query 的检索能力有限；当前按任务书使用英文 BGE，带 `V-JEPA 2`、`Droid` 等英文术语的问题效果较好。若后续需要全中文 query，应换用同一模型的多语版本并重建 index。
- 当前代码优先使用新版 `import pymupdf`，并兼容旧版 `fitz` API；10 篇文本型 PDF 均可正常解析。
- `DemoLLM` 只是无 API Key 的确定性 smoke adapter；真实回答质量取决于配置的 OpenAI-compatible LLM。`local_search` 的 Observation 已包含 source/page/text，可直接交给真实 LLM。
- 首次 BGE 下载曾因 Hugging Face Xet 通道卡住，但随后缓存完成；若再次遇到该网络问题，ingest 支持 `--backend hashing` 作为不联网的同空间 fallback。
