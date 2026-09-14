---
title: vLLM 推理引擎
type: domain
tags: [vllm, llm-inference, pagedattention, kv-cache]
source:
  - "wiki: vLLM wiki | pageId 566290559 space ~zhanbin"
  - "code: vLLM 仓库 (分析基准 commit b74ac0bc6, v0.28.1rc0-40)"
  - "code: vLLM flash-attention fork | commit 506341a143fc (2026-09-04 起)"
  - "doc: tracker/vllm-tracker/vllm-推理引擎与FlashAttention-完整技术文档.md (canonical)"
date: 2026-09
---

# vLLM 推理引擎

> 领域索引。完整内容见 canonical 文档 `tracker/vllm-tracker/vllm-推理引擎与FlashAttention-完整技术文档.md`(563 行,八大主题)。
> 本页只放定位、关键事实与导航,不重复正文。

## 一句话定位

vLLM 是**高吞吐、低显存占用**的开源 LLM 推理与服务引擎(Apache 2.0,源于 UC Berkeley Sky Computing Lab,2000+ 贡献者)。

## 关键技术四块

| 技术 | 解决什么 |
|---|---|
| **PagedAttention** | 受 OS 虚拟内存启发,KV cache 分固定 block + block table 非连续分配,消除显存碎片(SOSP 2023, arXiv:2309.06180) |
| **连续批处理** | 迭代级调度,decode+prefill 混批,填满 `max_num_batched_tokens` |
| **V1 引擎架构** | 调度(CPU)与执行(GPU)解耦的多进程 + 异步调度 |
| **优化注意力内核** | FlashAttention / FlashInfer / TRTLLM-GEN / FlashMLA / Triton |

## 关键事实(快照)

- 版本:v0.28.0(2026-08-26);Stars ~90k / Forks ~21k
- **V1 架构(`vllm/v1/`)是唯一现役主线**,旧 `vllm/engine/` 仅兼容保留;网上多数资料讲的是 V0,两者调度模型本质不同
- 分析基准 commit `b74ac0bc6`;FA fork `506341a143fc`
- 主线 `vllm/v1/` 目录(canonical 文档源码一手实读自这里)

## 内容导航(canonical 八大主题)

1. 项目概述与背景 → 2. 整体架构 → 3. PagedAttention 原理 → 4. FlashAttention 技术细节 → 5. 连续批处理与调度 → 6. KV Cache 管理 → 7. 分布式推理 → 8. 性能调优 → 9. 参考链接

## 深度概念篇(concepts,2026-09 多源综合+bot 审批)

> 把 notes/ 的源码事实提炼成"打开一篇能读懂"的知识沉淀(机制原理 + 数据 + 实践)。

| 篇目 | 类型 | 一句话 |
|---|---|---|
| [PagedAttention 与分页 KV Cache](concepts/paged-attention.md) | concept | 显存管理心法:分页 + block table,利用率 33%→84%、batch 2-4×,是连续批处理/prefix caching/抢占的地基 |
| [连续批处理与统一 Token 预算](concepts/continuous-batching.md) | concept | 迭代级调度 + 统一 token 预算,decode/prefill/chunk 混批填满 max_num_batched_tokens,吞吐 2-4× |
| [V1 引擎架构](concepts/v1-architecture.md) | concept | 调度执行解耦的多进程 + 异步调度心智模型,EngineCore / Worker / GPUModelRunner 三层 |

## 相关分篇笔记(已并入本领域 `notes/` 子目录)

> 深挖 vLLM 各子域的知识笔记,已从 docs/ 移入 `obrain/llm-inference/vllm/notes/`。

- `notes/vllm-core-architecture-notes.md` 核心架构
- `notes/vllm-model-registry-notes.md` 模型注册
- `notes/vllm-quantization-gptq-awq-fp8-notes.md` 量化
- `notes/vllm-distributed-ray-notes.md` 分布式 / Ray
- `notes/vllm-sampling-penalties-notes.md` sampling / penalties
- `notes/vllm-api-server-openai-compat-notes.md` OpenAI 兼容 API server
- `notes/vllm-serving-layer-analysis.md` 服务层分析
- `notes/vllm-full-technical-report.md` 完整技术报告

## 更新与同步(由定时任务维护,本页不追)

- 更新检查:`tracker/vllm-tracker/check_updates.sh`
- Wiki 同步:`tracker/vllm-tracker/sync_wiki.py --push`(pageId 566290559,surgical 增量)
- 定时任务:cron `e0c951b60eea`
- 更新流程细节:`tracker/vllm-tracker/vllm-文档更新流程.md`、`tracker/vllm-tracker/vllm-文档更新-定时任务配置.md`
