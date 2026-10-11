---
title: vLLM 推理引擎
type: domain
tags: [vllm, llm-inference, pagedattention, kv-cache]
source:
  - "code: vLLM 仓库 (分析基准 commit b74ac0bc6, v0.28.1rc0-40)"
  - "code: vLLM flash-attention fork | commit 506341a143fc (2026-09-04 起)"
  - "web: vLLM 官方文档 | https://docs.vllm.ai/en/latest/ | 阅读日期 2026-09-27"
  - "web: vLLM Model Runner V2 Design Document | https://docs.vllm.ai/en/latest/design/model_runner_v2/ | 阅读日期 2026-09-27"
date: 2026-10
---

# vLLM 推理引擎

> 领域索引。**知识本体在 `concepts/` 深度概念篇**(9 篇,由源码一手实读 + 官方文档多源综合而成);生产部署清单在 `practices/`(4 篇,含选型对比);生态构件在 `ecosystem/`(10 篇,含 vllm-project 组织 47 仓源码实读)。原 `notes/` 源码笔记已全部提炼入概念篇并删除,内容自足,不依赖库外文件。本页放定位、关键事实与导航。

## 一句话定位

vLLM 是**高吞吐、低显存占用**的开源 LLM 推理与服务引擎(Apache 2.0,源于 UC Berkeley Sky Computing Lab,2000+ 贡献者)。

## 关键技术四块

| 技术 | 解决什么 |
|---|---|
| **PagedAttention** | 受 OS 虚拟内存启发,KV cache 分固定 block + block table 非连续分配,消除显存碎片(SOSP 2023, arXiv:2309.06180) |
| **连续批处理** | 迭代级调度,decode+prefill 混批,填满 `max_num_batched_tokens` |
| **V1 引擎架构** | 调度(CPU)与执行(GPU)解耦的多进程 + 异步调度 |
| **优化注意力内核** | FlashAttention / FlashInfer / Triton / MLA 族(FlashMLA/FlashInfer MLA/CUTLASS MLA/Triton MLA)等,可插拔 backend(见 [attention-backend-selection](practices/attention-backend-selection.md)) |

## 关键事实速记

- 当前文档快照：官方文档已进入 v0.29+ 时代；本地源码分析基线仍是 `v0.28.1rc0-40`（commit `b74ac0bc6`），不能把基线当作当前实现版本。
- **V1 架构(`vllm/v1/`)仍是本库的主线组织方式**；旧 `vllm/engine/` 仅作兼容背景，网上多数资料讲的是 V0，不能直接套用。
- 分析基准 commit `b74ac0bc6`; FA fork `506341a143fc`
- ⚠️ **Model Runner 时效警示**：官方 Model Runner V2 设计文档称其从头设计了持久 batch、async-first 执行、异步 barrier 移除和显式 CUDA Graph 管理，但同时明确写着 MRV2 尚未 feature-complete、未经过充分严格测试且仍有开放设计决策；本库原有源码分析恰在切换前，因此不能把旧 `GPUModelRunner` 细节直接套到新执行链。当前官方架构总览仍用 API Server / Engine Core / GPU Worker 解释部署进程模型，后续更新应分别核对 MRV2 设计、`vllm/v1/worker/`、API Server 与量化执行链。
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
| [投机解码](concepts/speculative-decoding.md) | concept | 草稿模型串行猜 γ 个 + 目标模型一次并行验证;期望每步 token = (1−α^(γ+1))/(1−α),EAGLE/MTP/Medusa/n-gram/DSpark 各后端支持状态 |
| [API Server 与 OpenAI 兼容层](concepts/api-server-openai-compat.md) | concept | 59 行 shim + 四件套子包重构;薄路由 + app.state 注入、条件化路由注册、SO_REUSEPORT 多进程、SSE keep-alive |

## 生产实践篇(practices)

| 篇目 | 类型 | 一句话 |
|---|---|---|
| [生产部署与观测清单](practices/production-serving.md) | checklist | 从模型/显存预算、启动参数、健康检查、并发与限流到 Prometheus 指标、日志、升级和安全边界的上线前后检查 |
| [LLM 推理引擎选型](practices/inference-engine-selection.md) | comparison | vLLM、SGLang、TensorRT-LLM、llama.cpp、LMDeploy 的定位、适用场景和统一评测方法 |
| [注意力后端选型](practices/attention-backend-selection.md) | comparison | V1 attention backend 可插拔机制(AttentionBackend + register_backend + 最小 priority 分派,无 ATTN_BACKEND 环境变量);backend 枚举无 TRT-LLM,MLA 族 FlashMLA 双平台 |
| [PD 分离部署清单](practices/pd-disaggregation.md) | checklist | Prefill/Decode 解耦:TTFT/TPOT 分离优化、KV 传输(DistServe/Splitwise/Mooncake、KVCacheConnector/NIXL/LMCache)、何时用/不用、vLLM 落地与监控坑 | ✅ 已入库(2026-10) |

> 2026-09 收尾:原 `notes/` 5 篇源码笔记(sampling-penalties / api-server / core-architecture / full-technical-report / serving-layer-analysis)已按"先迁移独有内容、后删除"提炼并入上述概念篇,`notes/` 目录随之清空;溯源以源码 commit `b74ac0bc6` 为准。

## 生态构件篇(ecosystem/,源码实读 vllm-project 组织)

> 引擎之外那一圈仓库:全模态、路由决策、集群部署、压缩工具、投机与 RL、硬件插件、评测。每篇含 mermaid 图与真实代码示例。

| 条目 | 类型 | 一句话 |
|---|---|---|
| [vllm-omni](ecosystem/vllm-omni.md) | concept | 全模态服务:text/audio/image/video/action 统一流水线,AR/DiT 统一 paged KV,全双工 session,MiniCPM-o/H3 案例 |
| [semantic-router](ecosystem/semantic-router.md) | concept | 可编程决策层:信号→布尔规则树→选模/安全/成本,Go+YAML,Decision/Vela 模型族 |
| [aibrix](ecosystem/aibrix.md) | concept | K8s 原生 GenAI 推理基础设施:gateway/controller/autoscaler/ModelAdapter,CRD 与路由算法 |
| [production-stack](ecosystem/production-stack.md) | concept | 参考部署栈:单机→集群不改应用代码,LMCache 分层缓存,Helm values 拓扑 |
| [router](ecosystem/router.md) | concept | 高性能请求路由:五种负载均衡、PD 分离、K8s 服务发现,`docs/load_balancing/` 策略速查 |
| [llm-compressor](ecosystem/llm-compressor.md) | concept | 压缩工具链:GPTQ/FP8/SmoothQuant recipe 有序流水线,compressed-tensors 格式落地 |
| [speculators](ecosystem/speculators.md) | concept | 草稿模型训练/评测/导出:EAGLE-3/MTP/DFlash/DSpark,训练→vLLM 部署全流程 |
| [vime](ecosystem/vime.md) | concept | vLLM rollout 的 RL 后训练:Megatron+vLLM 同步/异步,权重同步四路径 |
| [hardware-plugins](ecosystem/hardware-plugins.md) | concept | 七后端插件体系:platform_plugins/general_plugins entry points,Ascend/Metal/Gaudi/Neuron/TPU/XPU/OpenVINO |
| [guidellm-bench](ecosystem/guidellm-bench.md) | howto | SLO-aware 基准评测:Profile 负载模拟、TTFT/ITL/Goodput、容量规划六步 |

**生态全景**:核心引擎与内核 → 硬件后端/压缩格式/插件注入 → 路由决策 → 集群部署 → 投机与 RL → 评测。47 仓九分组地图见原 `vllm-project/index.md` 简表(下)。

| 分组 | 代表仓库 |
|---|---|
| 核心引擎与内核 | vllm, flash-attention, tml-fa4, FlashKDA, MSA, humming, vllm-nccl, dllm-plugin |
| 全模态 | vllm-omni, bart-plugin |
| 路由与决策 | router, agentic-api, semantic-router |
| 集群与部署 | production-stack, aibrix, afd-plugin, recipes, vmon, vvm |
| 压缩与格式 | compressed-tensors, llm-compressor, vllm-gguf-plugin, vllm-bnb-plugin |
| 投机与 RL | speculators, vime |
| 硬件后端 | vllm-ascend, vllm-metal, vllm-gaudi, vllm-neuron, tpu-inference, vllm-xpu-kernels, vllm-openvino |
| 评测与工具 | guidellm, perf-eval, vllm-bench, lm-evaluation-harness, perf-dashboard, vllm-dashboard, ci-infra, vllm-skills |
| 文档与 RFC | vllm-docs, rfcs, vllm-project.github.io, vllm-daily, media-kit |

**扩展位(生态待补)**:agentic-api、compressed-tensors 格式规范、afd-plugin、humming JIT、MSA/FlashKDA/tml-fa4 内核专篇、production-stack 监控清单。

## 相关条目(跨域互链)


- **长上下文/流式 KV cache 算法侧**:PagedAttention 解决"cache 装得下就全留"的分页管理;**装不下时逐出谁**见 [`attention/algorithm/attention-sinks-streaming.md`](../attention/algorithm/attention-sinks-streaming.md)(StreamingLLM:attention sink + 滚窗,4M token 稳定生成)。两者正交,vLLM 主线未内置 sink 逐出(TensorRT-LLM/HF 已内置;SGLang 主线仍支持 attention sink:sinks kwarg / learnable sink / speculative draft sink 等多个 backend)。
- **注意力内核与架构谱系**:[`attention/`](../attention/index.md)(算法/架构层 + FA 内核实现层,合并域)。

## 扩展位(2026-10 全库审查待补)

- 结构化输出/guided decoding(xgrammar/llguidance、grammar bitmask 与调度重叠)
- multi-LoRA 服务(adapter 热插拔、base 共享、LoRARequest 链路)
- ~~PD 分离与 KV transfer~~(已入库)
- RL rollout 服务(sleep/wake、async scheduling、DP 负载均衡)
- MoE EP/EPLB 服务侧(补 distributed-ray 目录空白)
- KV cache 分层/卸载(CPU/NVMe、LMCache)
- Model Runner V2 落地篇(index 警示已备)
- production-serving 补指标名与 continuous-batching 对齐

## 溯源

- 外部链接索引见 [references.md](references.md)。
