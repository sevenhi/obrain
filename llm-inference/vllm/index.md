---
title: vLLM 推理引擎
type: domain
tags: [vllm, llm-inference, pagedattention, kv-cache]
source:
  - "code: vLLM 仓库 (分析基准 commit b74ac0bc6, v0.28.1rc0-40)"
  - "code: vLLM flash-attention fork | commit 506341a143fc (2026-09-04 起)"
date: 2026-09
---

# vLLM 推理引擎

> 领域索引。**知识本体在 `concepts/` 深度概念篇 + `notes/` 源码笔记**,内容自足,不依赖库外文件。本页放定位、关键事实与导航。

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

- 版本:v0.28.0(2026-08-26;当前已发布 v0.29.0,2026-09-09);Stars ~90k / Forks ~21k
- **V1 架构(`vllm/v1/`)是唯一现役主线**,旧 `vllm/engine/` 仅兼容保留;网上多数资料讲的是 V0,两者调度模型本质不同
- 分析基准 commit `b74ac0bc6`;FA fork `506341a143fc`
- ⚠️ **时效警示**:v0.29.0(2026-09-09)起 Model Runner V2 已成为所有模型的默认执行路径;本库架构描述基于 v0.28.1rc0-40(`b74ac0bc6`,恰在切换前),v0.29+ 上 GPUModelRunner 执行链会演进
- 主线 `vllm/v1/` 目录(本库 notes 源码一手实读自这里)

## 深度概念篇(concepts,2026-09 多源综合+bot 审批)

> 把 notes/ 的源码事实提炼成"打开一篇能读懂"的知识沉淀(机制原理 + 数据 + 实践)。

| 篇目 | 类型 | 一句话 |
|---|---|---|
| [PagedAttention 与分页 KV Cache](concepts/paged-attention.md) | concept | 显存管理心法:分页 + block table,利用率 33%→84%、batch 2-4×,是连续批处理/prefix caching/抢占的地基 |
| [连续批处理与统一 Token 预算](concepts/continuous-batching.md) | concept | 迭代级调度 + 统一 token 预算,decode/prefill/chunk 混批填满 max_num_batched_tokens,吞吐 2-4×(vs Orca,主要来自 PagedAttention 内存管理,详见正文) |
| [V1 引擎架构](concepts/v1-architecture.md) | concept | 调度执行解耦的多进程 + 异步调度心智模型,EngineCore / Worker / GPUModelRunner 三层 |
| [量化:GPTQ / AWQ / FP8](concepts/quantization-gptq-awq-fp8.md) | concept | 注册表→按层分派→kernel 选型三分派链;AWQ 大 batch 启发式、FP8 硬件/AMD 适配 |
| [分布式与 Ray 集成](concepts/distributed-ray.md) | concept | Executor 三后端选型、Ray rank 重排、通信组初始化;PP×TP 编址 |
| [模型注册 ModelRegistry](concepts/model-registry.md) | concept | 架构名→实现类懒加载映射;Transformers 兜底、后缀归一化、新增模型只改两处 |

## 相关分篇笔记(本领域 `notes/` 子目录)

> 深挖 vLLM 各子域的知识笔记,同属本领域 `notes/` 目录。

- [notes/vllm-core-architecture-notes.md](notes/vllm-core-architecture-notes.md) 核心架构
- [notes/vllm-sampling-penalties-notes.md](notes/vllm-sampling-penalties-notes.md) sampling / penalties
- [notes/vllm-api-server-openai-compat-notes.md](notes/vllm-api-server-openai-compat-notes.md) OpenAI 兼容 API server
- [notes/vllm-serving-layer-analysis.md](notes/vllm-serving-layer-analysis.md) 服务层分析
- [notes/vllm-full-technical-report.md](notes/vllm-full-technical-report.md) 完整技术报告
