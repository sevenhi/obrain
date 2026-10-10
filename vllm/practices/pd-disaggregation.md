---
title: PD 分离：Prefill/Decode 解耦的服务架构与部署清单
type: checklist
tags: [vllm, llm-inference, serving, pd-disaggregation, kv-transfer, prefill, decode, performance]
source:
  - "paper: DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving (OSDI 2024) | arXiv:2401.09670"
  - "paper: Splitwise: Efficient Generative LLM Inference Using Phase Splitting (ISCA 2024) | arXiv:2401.13081"
  - "paper: Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving | arXiv:2407.00079"
  - "web: vLLM 官方文档 | https://docs.vllm.ai/en/latest/ | 阅读日期 2026-10"
  - "web: vLLM Production Metrics | https://docs.vllm.ai/en/latest/usage/metrics.html | 阅读日期 2026-10"
date: 2026-10
---

# PD 分离：Prefill/Decode 解耦的服务架构与部署清单

## TL;DR

PD 分离（Prefill/Decode Disaggregation）把 **compute-bound 的 prefill** 与 **memory-bound 的 decode** 拆到不同实例乃至不同设备上，是 2025–26 在线服务架构的最大主题。收益是 TTFT 与 TPOT 可独立优化、算力与带宽各自贴合负载、吞吐/延迟 SLO 同时改善；代价是 **KV cache 跨实例传输**、调度与路由复杂度、以及多一套故障面。它不是 continuous batching / chunked prefill 的替代品，而是更硬的一层分离——先弄清瓶颈在算力还是带宽，再决定要不要拆。

## 1. 为什么分离

### 1.1 两个阶段的算力画像完全不同

| 阶段 | 计算形态 | 瓶颈 | 典型指标 |
|---|---|---|---|
| **Prefill** | 大矩阵乘（prompt 整段/分块前向），算术强度高 | compute-bound（GPU FLOPS） | TTFT、prefill 吞吐 |
| **Decode** | 每步只算 1 token，权重反复读、KV 反复读 | memory-bound（显存带宽） | TPOT / ITL、output token 吞吐 |

同一实例混跑时，两类负载互相干扰：长 prompt 的 prefill 会把 batch 里的 decode 步骤拖慢（TTFT 尖刺反过来污染 TPOT），decode 的低强度步又在浪费峰值算力。指标口径上要分开看：

- **TTFT**（Time To First Token）：几乎完全由 prefill 决定；
- **TPOT / ITL**（Time Per Output Token / Inter-Token Latency）：几乎完全由 decode 决定；
- **吞吐**（request throughput / output tokens per second）：两者叠加，但混批时会被慢的一侧拖累。

### 1.2 软融合 vs 硬分离

| 方案 | 做法 | 解耦程度 | 代表 |
|---|---|---|---|
| **Chunked prefill（软融合）** | 把长 prefill 切块，和 decode 步混在同一 batch 的 token 预算里 | 弱：仍共享同一批 GPU、同一份调度 | vLLM V1 默认策略 |
| **PD 分离（硬分离）** | prefill 实例与 decode 实例物理/逻辑拆开，KV 经传输层交接 | 强：算力、带宽、调度各自独立 | DistServe、Splitwise、Mooncake |

Chunked prefill 解决的是「一步别被一个长 prompt 独占」；PD 分离解决的是「两类负载根本不是一种资源」。二者可以叠加：PD 的 prefill 实例内部仍可 chunked，decode 实例内部仍是 continuous batching。

### 1.3 理论收益从哪来

- **资源专化**：prefill 侧重算力（可选计算卡/高 FLOPS 型号），decode 侧重带宽（高 HBM 带宽、大显存）；
- **互不污染**：长 prompt 进来只打 prefill 池，decode 池的 TPOT 保持平稳；
- **独立扩缩**：prompt 长、输出短的流量加 prefill 实例；对话续写、agent 长生成加 decode 实例；
- **goodput 视角**（DistServe 的口径）：在 TTFT/TPOT 双 SLO 约束下，分离后能同时满足两者的请求比例显著更高，而不只是峰值 FLOPS 变好看。

## 2. 架构模式

### 2.1 基本拓扑（DistServe / Splitwise 思想）

```
客户端 → 路由/网关
           ├─→ Prefill 实例（compute-bound）
           │        │  KV cache 传输（RDMA / NVLink / 共享存储 / 专用互联）
           │        ▼
           └─→ Decode 实例（memory-bound）→ 流式输出 token
```

要点：

1. 一条请求先在 **prefill 实例**跑完 prompt 前向，得到全部 KV cache 与第一个 token；
2. **KV cache**（不是 logits、不是权重）经传输层送到 decode 实例；
3. decode 实例接管后续逐 token 生成，直接对客户端流式返回；
4. 路由层要保证「同一请求的 prefill 与 decode 配对」，并决定 KV 往哪送。

Splitwise 的贡献是把 phase splitting 落到共享集群（甚至同一 GPU 上按时间片切分）；DistServe 强调按 SLO 做资源放配与 goodput；Mooncake 则把 **KVCache 当一等公民**，围绕 KV 池做全局调度与分层存储（显存/CPU/SSD），把前缀复用与跨节点传输绑在一起设计。

### 2.2 KV 传输通道与组件

| 层级 | 方案 | 特点 |
|---|---|---|
| **硬件互联** | NVLink/NVSwitch（节点内）、RDMA/RoCE（跨节点）、IB GPUDirect | 延迟与带宽最好；对拓扑和驱动敏感 |
| **传输库** | NIXL（NVIDIA Inference Xfer Library）、NCCL P2P、自研 RPC | 负责张量级搬运与注册；NIXL 为 Dynamo 等框架共用 |
| **KV 池/缓存中间层** | LMCache、Mooncake KV 池、共享存储（如 3FS/SSD 后备） | 支持前缀复用、offload、跨实例共享；多一跳但解耦部署 |
| **引擎插件** | vLLM `KVCacheConnector` / `kv_transfer/`、`kv_transfer_config` | 把传输接到引擎的 KV 生命周期上，不必自改调度器 |

### 2.2.1 一条请求的 PD 路径

1. **入口路由**：网关按 prompt 长度/历史复用情况选 prefill 实例；同时向 decode 池预留位置（或先 prefill 后申请）。
2. **Prefill**：prompt 前向（可 chunked），产出 KV block 集合 + 首 token + 序列元数据。
3. **KV 交接**：按 block/slot 打包 → 走 RDMA/NVLink/共享池 → decode 端注册并写入本地 block table。
4. **Decode 接管**：在本地 continuous batching 里继续采样；客户端拿到的流从首 token 起不断。
5. **回收**：序列结束后两边释放 block；若进 KV 池，则按复用策略保留前缀块。

失败点也按这五步看：路由没预留、prefill 超时、传输慢/断、接收端缓冲满、decode 侧 slot 映射不一致——任一步出问题都要能降级回单体实例。

### 2.3 vLLM 侧落地形态（按 2026-10 文档快照）

- **多实例部署**：prefill 实例与 decode 实例是两个（或两组）`vllm serve`/engine，靠 **KV transfer 配置**（`kv_transfer_config` / KVConnector）互相交接；具体字段名与 connector 列表随版本变化，部署前用当前版本文档与 `--help` 核对。
- **代码位置**：KV 传输相关实现集中在 `kv_transfer/`（或 `vllm/distributed/kv_transfer/`，路径随版本迁移）与 `KVCacheConnector` 抽象；官方 doc 的 multi-instance / disaggregated serving 示例是最快的入口。
- **与 PagedAttention 的关系**：传输的单位通常是 **KV block / slot**（见 [PagedAttention](../concepts/paged-attention.md)），不是连续大 buffer；接收端要做 slot 映射，才能直接进本地 block table。
- **生态对照**：NVIDIA Dynamo、SGLang router、Mooncake Store 等都提供独立的 PD 路由 + KV 传输栈；跨引擎复用优先看 NIXL/共享存储方案，而不是私有格式硬编码。

> 版本提醒：库内源码实读基线是 `v0.28.1rc0-40`（commit `b74ac0bc6`），官方文档已进入 v0.29+ 时代。PD/KV transfer 是演进最快的区域，**凡涉及 connector 名、CLI 参数、目录路径的表述均属「按 2026-10 文档快照」，上线前必须以当前版本为准**。

## 3. 何时用 / 何时不用

### 3.1 决策表

| 流量/部署特征 | 建议 | 理由 |
|---|---|---|
| 长 prompt、输出中短、并发中等（文档问答、RAG、agent 入口） | **优先上 PD 分离** | prefill 重、decode 轻；分离后 TTFT 与 TPOT 同时改善，DistServe/Splitwise 场景吻合 |
| 短 prompt、高并发、输出短（分类、改写、轻聊天） | **暂不必要** | prefill 占比低，KV 传输与调度开销可能吃掉收益；先把 continuous batching 调好 |
| 小模型（≤7B）/ 单卡就能吃满流量 | **通常不值得** | 传输与运维成本超过收益；chunked prefill + 分页已够用 |
| 超长上下文、多轮共享前缀 | **值得，且配 KV 池** | 前缀复用 + 跨实例共享收益大；Mooncake/LMCache 类方案匹配 |
| 输出远长于输入（长文续写、代码补全长尾） | **看 TPOT 是否已 SLO 不达标** | 解码侧先考虑带宽与投机解码，再考虑 PD |
| 已有大规模多实例/异构 GPU | **值得** | 异构专化（高算力卡跑 prefill、高带宽卡跑 decode）是分离的直接红利 |
| 团队无法承担多一套监控/故障面 | **先软后硬** | 从 chunked prefill 与请求分类做起，再上 PD |

### 3.2 与 continuous batching / chunked prefill 的关系

- **不是互相替代**：continuous batching 是「迭代级把 GPU 填满」；chunked prefill 是「长 prompt 别独占一步」；PD 分离是「两类负载不在同一个池子里跑」。三者正交，可叠加。
- **调参顺序**：先确认调度与 token 预算是否健康（见 [连续批处理](../concepts/continuous-batching.md)），再看 TTFT/TPOT 是否被长 prompt 互相污染，最后才上 PD。
- **反向指标**：若 KV cache 使用率与抢占才是瓶颈（`vllm:num_preemptions_total` 高），先修显存与分页（见 [PagedAttention](../concepts/paged-attention.md)、[生产清单](production-serving.md) §2），PD 不解决「装不下」。

### 3.2.1 与相邻优化的关系（避免叠错 buff）

| 技术 | 和 PD 分离的关系 | 备注 |
|---|---|---|
| **投机解码 / MTP** | 正交，落在 decode 池内 | 草稿验证提高 decode 算术强度，不改变 prefill/decode 资源画像 |
| **Prefix caching / KV 复用** | 强相关 | 分离后更需要跨实例共享前缀，否则每次 prefill 都重算 system prompt |
| **KV 量化** | 正交且降低传输量 | 传输的是 KV，量化后带宽压力同步下降；但两端格式必须一致 |
| **并行（TP/PP/EP）** | 可组合但更复杂 | prefill 与 decode 可各自选不同的 TP/EP 度；传输要跨过并行切片边界 |
| **多 LoRA / 多模型** | 路由变难 | 配对键从「序列」变成「序列 + adapter/模型」，容易配错 |

### 3.3 成本模型（定性）

| 成本项 | PD 分离后 |
|---|---|
| GPU 算力 | 更贴负载，可异构；总卡数未必减少 |
| 显存/带宽 | decode 池要预留接收 KV 的缓冲与本地 block 池 |
| 网络 | 多出 **KV 传输带宽与尾延迟**，跨节点时不可忽视 |
| 运维 | 两套实例、两套指标、一套配对路由，故障面翻倍 |
| 软件 | connector/版本/序列化格式锁定，升级要回归传输链路 |

## 4. 部署 checklist

### 4.1 版本与能力确认

- [ ] 确认 vLLM 版本对 **KVConnector / 多实例 PD** 的支持范围（0.28+ 持续演进，**按 2026-10 文档快照**：字段与 connector 名以当前 release 为准）。
- [ ] 阅读当前版本的 disaggregated / multi-instance serving 官方示例，不要照搬博客里的旧 CLI。
- [ ] 明确 connector 类型：进程内 NCCL P2P、NIXL、共享存储、LMCache 等，带宽与延迟差一个数量级。
- [ ] 锁定 prefill 侧与 decode 侧的 **block size、dtype、模型 revision** 必须一致，否则 KV 无法互认。

### 4.2 拓扑与缓冲

- [ ] prefill ↔ decode 走 **最短互联**：同节点 NVLink 优先，跨节点用 RDMA 并验证实际带宽。
- [ ] 为 KV 传输预留 **接收端缓冲**与 GPU 显存水位（缓冲不足会反压 prefill，把 TTFT 拖回来）。
- [ ] 路由策略：长 prompt 路由到 prefill 池；配对失败或超时要有降级（回退单体实例）路径。
- [ ] 两侧 `gpu_memory_utilization`、`max_model_len`、`max_num_batched_tokens` 分别按自身负载调，不要抄同一份配置。
- [ ] 若用共享 KV 池/LMCache/Mooncake Store，单独监控命中率、池内层（HBM/CPU/SSD）与淘汰。

### 4.3 监控指标（分离后必须两边分开看）

| 侧 | 优先指标 |
|---|---|
| Prefill 池 | TTFT p50/p95/p99、prefill 吞吐、排队时间、chunk 步数、计算利用率 |
| Decode 池 | TPOT/ITL p50/p95/p99、output tokens/s、batch 填充、KV cache 使用率 |
| 传输层 | KV 传输时延、带宽、失败/重试、序列化开销、接收缓冲水位 |
| 业务 | goodput（双 SLO 同时达标比例）、端到端延迟、错误率、降级次数 |

- [ ] 传输时延占比要有仪表：**若 KV 传输 > TTFT 的可感知预算，分离得不偿失**。
- [ ] 把「配对成功率」和「路由降级率」纳入告警，而不是只看 HTTP 200。
- [ ] 指标命名随版本变化，用当前 metrics 文档核对，不要永久沿用旧 dashboard 字段（同 [生产清单](production-serving.md) §5）。

### 4.4 常见坑

1. **只拆不传**：没做传输带宽规划，跨节点走 TCP/普通 socket，PD 变成「慢一倍的单体」。
2. **缓冲反压**：decode 端接收缓冲小，prefill 完成后阻塞，TTFT 回退到混批水平。
3. **口径混看**：把 prefill 实例的 GPU 利用率和 decode 实例的利用率平均，掩盖一侧闲置。
4. **版本漂移**：connector 或 KV 布局随 vLLM 升级变化，灰度时必须双跑传输链路 smoke test。
5. **把 PD 当扩容药**：KV 装不下、抢占暴涨时，PD 无能为力，先修分页与显存预算。
6. **忽略前缀复用**：长共享 system prompt 场景不做 prefix/KV 池共享，PD 实例之间反复算同一段前缀。
7. **单体回退缺失**：传输层故障直接全局失败，应能一键切回非 PD 路径。
8. **首 token 与流式语义断裂**：prefill 产出的首 token 与 decode 流衔接不当，客户端看到「卡顿后再连发」，TTFT 体感被传输延迟二次污染。
9. **序列化/CPU 开销被忽略**：KV 在主机侧打包、压缩或校验过重时，PCIe/CPU 成为新瓶颈——传输链路要用 GPU-direct 或批量注册，避免每请求小包往返。

### 4.5 推荐落地顺序

1. **基线**：单体 + continuous batching + chunked prefill，测出 TTFT/TPOT/吞吐三条曲线（按真实 prompt 长度分布）。
2. **诊断**：长 prompt 是否在污染 TPOT？prefill 是否长期占满算力而 decode 饿？若是，进入下一步。
3. **仿真**：先同节点双实例 + 最快互联，把 KV 传输时延打到日志里，验证「分离收益 > 传输成本」。
4. **扩到跨节点**：换 RDMA/NIXL，重测 goodput；不达标就停在同节点或回退。
5. **加 KV 池/前缀共享**：多租户、共享 system prompt 时再上 LMCache/Mooncake 类组件。
6. **固化**：把双侧指标、降级开关、版本兼容矩阵写进部署仓库，灰度发布时双跑传输 smoke test。

> 口袋规则：**先证明「混批在互相污染」，再证明「传输不贵」，最后才谈异构专化**。顺序颠倒就会买一整套网络与运维复杂度，却换不回 TTFT。

## 溯源

- **DistServe**：《DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving》，OSDI 2024，arXiv:2401.09670——PD 分离 + goodput/SLO 口径与资源放配。
- **Splitwise**：《Splitwise: Efficient Generative LLM Inference Using Phase Splitting》，ISCA 2024，arXiv:2401.13081——phase splitting（含共享集群/混部）动机与实测。
- **Mooncake**：《Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving》，arXiv:2407.00079——KVCache 为中心的分离架构、KV 池与全局调度。
- **vLLM 官方文档**（阅读日期 2026-10）：Architecture Overview、multi-instance / disaggregated serving 相关页面、Production Metrics（https://docs.vllm.ai/en/latest/）。PD/KV transfer 字段与目录以当前版本文档为准。
- **库内源码基线**：`v0.28.1rc0-40`（commit `b74ac0bc6`）——`kv_transfer/` / `KVCacheConnector` 的一手细节需在该版本或更新 main 上复查，**不把基线当作当前实现**。
- **相关方向（非 PD 主线）**：Nexus（Intra-GPU Prefill/Decode 分离，arXiv:2507.06608）；SLOWeave（Deadline-Aware Adaptive Prefill Chunking，arXiv:2609.07883）——见 [连续批处理](../concepts/continuous-batching.md) 溯源。

## 相关条目(库内互链)

- **[连续批处理与统一 Token 预算](../concepts/continuous-batching.md)**：软融合路径（chunked prefill + 混批）。PD 分离是其「更硬的下一刀」，两者正交可叠加；调参顺序见本篇 §3.2。
- **[V1 引擎架构](../concepts/v1-architecture.md)**：调度/执行解耦的进程模型。PD 多实例是在其之上再拆一层「prefill 进程组 / decode 进程组」，API Server 与 Engine Core 的角色边界仍适用。
- **[PagedAttention 与分页 KV Cache](../concepts/paged-attention.md)**：KV 以 block 为单位分配与传输；PD 的 KV 交接语义建立在 block table / slot mapping 之上。
- **[生产部署与观测清单](production-serving.md)**：显存预算、指标分位、升级回滚的通用清单；PD 场景在其上叠加「传输层指标 + 配对路由」（见本篇 §4.3）。
- **[`deepseek-infra/`](../../deepseek-infra/index.md) 推理系统缺口**：该域扩展位已点名 **inference-system-v3-r1**（PD 分离、跨节点 EP 批扩展、73.7k/14.8k 与 545% 口径）**尚未入库**。工业级 PD + 大规模 EP 的一手口径要等该篇补全，本篇只覆盖 vLLM 生态与公开论文层，**不要用本文数字去填 DeepSeek 官方口径**。
