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

> 领域索引。**知识本体在 `concepts/` 深度概念篇**(8 篇,由源码一手实读 + 官方文档多源综合而成;原 `notes/` 源码笔记已全部提炼入概念篇并删除),内容自足,不依赖库外文件。本页放定位、关键事实与导航。

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
- 主线 `vllm/v1/` 目录(本库概念篇的源码一手实读来自这里)
- **关键版本差异(区别于网上旧资料)**:API 层 `api_server.py` 已退化为 59 行 shim,协议端点拆独立子包;模型注册从 `Models` 枚举演进为 `registry.py` 分类字典 + `_ModelRegistry`;量化合并为 `auto_gptq.py`/`auto_awq.py` + 统一 kernel 选型;executor 迁至 `vllm/v1/executor/`;采样惩罚在 `vllm/v1/sample/ops/` + argmax 不变性分组框架(详见各概念篇)

## 深度概念篇(concepts,2026-09 多源综合 + 深度打磨)

> 源码一手实读 + 官方文档多源综合,提炼成"打开一篇能读懂"的知识沉淀(机制原理 + 数据 + 实践)。

| 篇目 | 类型 | 一句话 |
|---|---|---|
| [PagedAttention 与分页 KV Cache](concepts/paged-attention.md) | concept | 显存管理心法:分页 + block table,利用率 33%→84%、batch 2-4×,是连续批处理/prefix caching/抢占的地基 |
| [连续批处理与统一 Token 预算](concepts/continuous-batching.md) | concept | 迭代级调度 + 统一 token 预算,decode/prefill/chunk 混批填满 max_num_batched_tokens,吞吐 2-4×(vs Orca,主要来自 PagedAttention 内存管理,详见正文) |
| [V1 引擎架构](concepts/v1-architecture.md) | concept | 调度执行解耦的多进程 + 异步调度心智模型,EngineCore / Worker / GPUModelRunner 三层 |
| [量化:GPTQ / AWQ / FP8](concepts/quantization-gptq-awq-fp8.md) | concept | 注册表→按层分派→kernel 选型三分派链;AWQ 大 batch 启发式、FP8 硬件/AMD 适配 |
| [分布式与 Ray 集成](concepts/distributed-ray.md) | concept | Executor 三后端选型、Ray rank 重排、通信组初始化;PP×TP 编址;非阻塞两段式执行 |
| [模型注册 ModelRegistry](concepts/model-registry.md) | concept | 架构名→实现类懒加载映射;Transformers 兜底、后缀归一化、新增模型只改两处;注册后经 get_model_loader 注入权重 |
| [采样参数与惩罚机制](concepts/sampling-penalties.md) | concept | Sampler 管线顺序即机制;惩罚两类数学(repetition 乘性、presence/frequency 加性)、指数竞赛免同步、to_sampling_params 三级回退 |
| [API Server 与 OpenAI 兼容层](concepts/api-server-openai-compat.md) | concept | 59 行 shim + 四件套子包重构;薄路由 + app.state 注入、条件化路由注册、SO_REUSEPORT 多进程、SSE keep-alive |

> 2026-09 收尾:原 `notes/` 5 篇源码笔记(sampling-penalties / api-server / core-architecture / full-technical-report / serving-layer-analysis)已按"先迁移独有内容、后删除"提炼并入上述概念篇,`notes/` 目录随之清空;溯源以源码 commit `b74ac0bc6` 为准。

## 相关条目(跨域互链)

- **长上下文/流式 KV cache 算法侧**:PagedAttention 解决"cache 装得下就全留"的分页管理;**装不下时逐出谁**见 [`attention/algorithm/attention-sinks-streaming.md`](../attention/algorithm/attention-sinks-streaming.md)(StreamingLLM:attention sink + 滚窗,4M token 稳定生成)。两者正交,vLLM 主线未内置 sink 逐出(TensorRT-LLM/SGLang/HF 已内置)。
- **注意力内核与架构谱系**:[`attention/`](../attention/index.md)(算法/架构层 + FA 内核实现层,合并域)。
