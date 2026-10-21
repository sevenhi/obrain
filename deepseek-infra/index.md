---
title: deepseek-infra — DeepSeek 开源算力基础设施栈(昇腾 + NVIDIA 双平台 + 并行/存储)
type: domain
tags: [deepseek, ascend, nvidia, cuda, tilelang, deepgemm, deepep, deepjit, tilekernels, flashmla, deepselect, dualpipe, eplb, lplb, esft, engram, 3fs, smallpond, deepinfer, serving, npu, huawei, infra]
date: 2026-10
---

# deepseek-infra — DeepSeek 开源算力基础设施栈(昇腾 + NVIDIA 双平台 + 并行/存储)

> 本页是 deepseek-infra 领域的**导航起始页**。往下走:定位与关键事实 → 条目索引 → 更新维护。
> **真正的可读正文在 `concepts/` 子目录**,每篇一个专题、可独立打开读懂。本页只做导航,不重复正文。

---

## 一句话定位

**deepseek-infra** 记录 DeepSeek 开源的**算力基础设施栈**(2025-02 开源周放训练/推理/存储三件套;2026-09-30 补昇腾适配;稀疏 MQA 打分内核 NVIDIA 侧 09-30(v2.8.1)引入、10-14 v2.8.2 对齐修复,Ascend 2026/10 加入;mHC 流形超连接 2026-10 单列一篇,含 DeepGEMM `mega_mhc` 融合内核与 10-20 的 CUDA graph 屏障修复;**2026-10-20 DeepInfer 开源,把前面的算子/通信/负载均衡装进一个生产推理引擎**)。全景是「算子怎么写(TileLang)→ 矩阵怎么算(DeepGEMM)→ 专家怎么通信(DeepEP)→ 训练怎么并行(DualPipe/EPLB/LPLB)→ 残差流怎么连(mHC)→ 微调怎么省(ESFT)→ 稀疏记忆怎么查(Engram)→ 数据怎么存/算(3FS/smallpond)→ **推理怎么落到生产(DeepInfer)**」这条从语言、算子、通信、并行到存储、服务的全链路,**NVIDIA 与昇腾双平台**跑出近硬件峰值。

它定位在**算子实现层 / 基础设施层**,与库内 model-architecture(架构)、attention(算法)、vllm(引擎)、cutlass(NVIDIA 造内核方法论)四个领域正交互补:那些域讲「模型长什么样、注意力为什么省显存、引擎怎么调度、NVIDIA 怎么造内核」,本域讲「**这些算子/通信/并行/存储怎么实现、怎么适配、性能到哪**」。

**本域两条主线**:
1. **昇腾适配线**(2026-09-30):NVIDIA 原版 → 昇腾版的「同 API 不同后端」移植(TileLang/DeepGEMM/DeepEP/TileKernels/FlashMLA/DeepSelect),见「昇腾适配」条目组;
2. **NVIDIA/并行/工具线**(该批入库):DeepGEMM/DeepEP 的 NVIDIA 原版对照、DeepJIT、并行负载均衡(DualPipe/EPLB/LPLB)、微调(ESFT)、稀疏记忆(Engram)、存储与数据(3FS/smallpond)、开源总索引(open-infra-index)。

---

## 关键事实速记(快照,2026-10)

- **开源时间**:2026-09-30,DeepSeek 开源昇腾算力基础设施(训练+推理),华为团队支持;目标平台为昇腾 950 系列 NPU 及 **128 卡超节点 UBL128**(全互联,**待核/据昇腾官方资料**)。
- **六件套覆盖链路**:语言层 → 矩阵库 → 通信库 → 注意力算子。
- **三条 P0 事实(本域写作最易踩,务必按此记)**:
  1. **TileLang 是 tile-ai 的第三方 DSL,不是 DeepSeek 的仓库**;昇腾支持分三层:主仓 tile-ai/tilelang 现支持 Ascend 950 native backend(2026-09-30 起)、独立仓 tile-ai/tilelang-ascend(A2/A3 的 ascendc/pto/npuir)、tilelang-mlir-ascend(MLIR 路线)。
  2. **FlashMLA 昇腾版 2026-09-30 已合入 main(commit 2e5429f)**;该版是 breaking change:移除 Hopper 支持、移除 V3/V3.2/V4.0 旧模型、改 FP8/FP4 KV cache 格式,只服务 V4.1,旧模型切旧 commit `ba89a346`——以上均已由 README 明写,无需待核。
  3. **TileKernels 昇腾支持从 PyPI v2.0.0 起**,main 现为 v2.1.0(HEAD c1cf914a,含 Ascend 950 + CANN 9.2.0 双平台)。
- **术语(本域核定拼写;标「待核」者为一手库未附链接)**:fractal(分形**布局**)、MAD(matrix multiply-add 原语)、UBMEM、URMA、HCCL(集合通信)、HCOMM(控制面 API,**待核/据昇腾官方资料**)、bisheng(毕昇编译器)、DeepJIT(运行时 JIT)、torch_npu、**CANN 9.2.0**(DeepGEMM README 原文 "9.20" 即此版本,不写 9.20)、**950PR ≠ 950DT**(同一 950 Die 两款型:PR=Prefill&Recommendation、DT=Decode&Training,**待核/据昇腾官方资料**)。
- **性能数字单源标注**:本域各算子性能数字(DeepGEMM 99.8%/856/862/850 TFLOPS、DeepEP 90–95% 带宽、FlashMLA 410/360、TileLang FA/mHC 比值)均出自各仓 README/deep-dive 一手源,暂无独立第三方或硬件实测复核,引用时保留原始口径、待硬件实测复核。
- **mHC 三句话**:① 收益来自**残差流 n 路化 + 流间混合**(n=4),稳定性来自把混合矩阵**投影到双随机流形**(Sinkhorn,t_max=20)——双随机保证谱范数 ≤1 且对乘法封闭,复合增益 HC ~3000 → mHC ~1.6;② 代价是访存 ~10× 与 PP 通信 ~n×,靠 kernel 融合 + 按 `L_r*≈√(nL/(n+2))` 分块重算 + DualPipe 重叠压到**净增 6.7% 训练时间**;③ NVIDIA 侧落地是 DeepGEMM `mega_mhc`(SM100 单 kernel,6 WG,split-K),昇腾侧是依赖 tilelang 的 HC Prenorm GEMM。详见 [concepts/mhc-manifold-hyper-connections.md](concepts/mhc-manifold-hyper-connections.md)。

### 六件套 ↔ 上游映射表(已核实)

| 昇腾新增/适配 | 上游/对应 NVIDIA 版 | 性质 | 真实 repo |
|---|---|---|---|
| TileLang(昇腾后端) | tile-ai/tilelang(TileLang 原生) | 新增昇腾后端 | tile-ai/tilelang + tile-ai/tilelang-ascend |
| DeepGEMM-Ascend | DeepGEMM | 新建独立仓,API 完全兼容 | deepseek-ai/DeepGEMM-Ascend |
| DeepEP-Ascend | DeepEP | 新建独立仓,API 完全兼容 | deepseek-ai/DeepEP-Ascend |
| TileKernels | TileKernels 原生 | 新增昇腾后端(v2.0.0) | deepseek-ai/TileKernels |
| FlashMLA | FlashMLA 原生 | 昇腾版 2026-09-30 已合入 main(commit 2e5429f) | deepseek-ai/FlashMLA |
| DeepSelect | DeepSelect 原生 | 新增昇腾后端 | deepseek-ai/DeepSelect |

### 技术栈分层(文字图)

```
┌─────────────────────────────────────────────────────────┐
│  语言层    TileLang(tile-ai/tilelang)+ tilelang-ascend  │  算子「怎么写」
├─────────────────────────────────────────────────────────┤
│  计算库    DeepGEMM-Ascend(GEMM)  TileKernels(算子合集) │  矩阵「怎么算」
├─────────────────────────────────────────────────────────┤
│  通信库    DeepEP-Ascend(MoE EP all-to-all)             │  专家「怎么通信」
├─────────────────────────────────────────────────────────┤
│  算子      FlashMLA(dense+sparse 注意力) DeepSelect(TopK)│  注意力「怎么算」
└─────────────────────────────────────────────────────────┘
                ↓ 跑在:昇腾 950 系列 NPU + CANN 9.2.0 + 128 卡超节点 UBL128
```

**DSA 链路(注意力三件套的关系)**:DSA(DeepSeek Sparse Attention,机制)→ 先由 **DeepSelect**(TopK)选出每个 token 该 attend 哪些 KV 的 indices → 再由 **FlashMLA sparse** 按 indices 算稀疏注意力。DeepSelect 出索引,FlashMLA 消费索引,FlashMLA = dense + sparse 两套内核库,是 DSA 的**实现**,不是 DSA 本身。

---

## 条目索引

> 每个专题一篇,按「先语言基础 → 跨库专题 → 计算/通信库 → 算子库 → 注意力算子」排。

| 专题 | 文件 | 一句话 | 状态 |
|---|---|---|---|
| TileLang 与昇腾后端 | [concepts/tilelang.md](concepts/tilelang.md) | tile 级 DSL(TVM 之上),昇腾支持分三层(主仓 Ascend 950 native / tilelang-ascend A2/A3 / tilelang-mlir-ascend);三层编程接口 | ✅ 已入库 |
| Ascend 适配原理 | [concepts/ascend-port-principles.md](concepts/ascend-port-principles.md) | NVIDIA→昇腾关键抽象对照表(CUDA→Ascend C、TMA→稀疏加载/协程、NCCL→HCCL/HCOMM、fractal 布局等)+ 移植难点 | ✅ 已入库 |
| DeepGEMM-Ascend | [concepts/deepgemm-ascend.md](concepts/deepgemm-ascend.md) | 昇腾 GEMM 库,MAD 原语轻量抽象,藏 fractal/对齐/地址计算;稀疏加载 + 协程流水线;scaling factor 格式与 NVIDIA 不同;2026/10 新增稀疏 MQA logits | ✅ 已入库 |
| DeepEP-Ascend | [concepts/deepep-ascend.md](concepts/deepep-ascend.md) | MoE 专家并行通信库,EP all-to-all(dispatch/combine),EP≤32 dispatch 达 90–95% 物理带宽 | ✅ 已入库 |
| TileKernels | [concepts/tilekernels.md](concepts/tilekernels.md) | TileLang 之上的 LLM 算子库(MoE routing/量化/Engram/mHC),v2.0.0 起支持昇腾(已合入 main),v2.1.0(2026-10)量化/Engram bug 修复,同一 Python API 跑 NVIDIA+华为 | ✅ 已入库 |
| DeepSelect | [concepts/deepselect.md](concepts/deepselect.md) | DSA/Sampler 的定制 TopK 算子,2–20× vs torch.topk(口径:有效内存带宽);昇腾仅 bfloat16 | ✅ 已入库 |
| FlashMLA | [concepts/flashmla.md](concepts/flashmla.md) | MLA 注意力内核库(dense+sparse),昇腾版 2026-09-30 已合入 main(commit 2e5429f,breaking change 三条已由 README 明写);410 TFlops 双义(平台要标清);2026-10-15 ABI 稳定化(构建层) | ✅ 已入库 |

### NVIDIA 版对照 + 并行/工具/存储(NVIDIA 主线)

| 专题 | 文件 | 一句话 | 状态 |
|---|---|---|---|
| DeepGEMM NVIDIA vs Ascend | [concepts/deepgemm-nvidia-vs-ascend.md](concepts/deepgemm-nvidia-vs-ascend.md) | NVIDIA 原版:轻量 tensor core 内核库(FP8/FP4/BF16、Mega MoE、MQA),H800 1550 TFLOPS;与昇腾版同 API 不同后端(WGMMA/TMA vs MAD/协程);稀疏 MQA logits:NVIDIA 09-30(v2.8.1)引入、10-14 v2.8.2 对齐修复,Ascend 2026/10 新增;mHC 融合内核 10-20 修 CUDA graph 屏障 bug(#478) | ✅ 已入库 |
| **DeepInfer** | [concepts/deepinfer.md](concepts/deepinfer.md) | DeepSeek 生产推理框架首次开源(2026-10-20):服务 V4.1-Flash,EPD 分离 + Wide-EP(EP64–128)、零开销调度 ZOS(post-process 推迟一步 + GPU 侧 seqlens 修正)、DSpark 变长投机验证、YoCo + SWA bounded replay、三级 KV(3FS 分页/host SWA/GPU state)、原生 EPLB;**KV 存储客户端与跨 DP 负载均衡器未开源**;SM100 + Ascend 950 双平台。2026-10-21 增量:§15 补 vendored pin 对照(DeepGEMM 落后 #478、FlashMLA 落后 #230)+ 构建链细节 | ✅ 已入库(2026-10-21 更新) |
| mHC Manifold Hyper-Connections | [concepts/mhc-manifold-hyper-connections.md](concepts/mhc-manifold-hyper-connections.md) | 残差流 n 路扩展 + Birkhoff 双随机约束(Sinkhorn 20 轮)恢复恒等映射:HC 复合增益 ~3000→mHC ~1.6,n=4 仅增 6.7% 训练时间;DeepGEMM `mega_mhc` 单 kernel(6 WG 静态分工/split-K/跨 CTA Mix+Norm 屏障),2026-10-20 #478 用 epoch 取代 `%gridid` 修 CUDA graph 重放旧 partial bug | ✅ 已入库(2026-10-20) |
| DeepEP NVIDIA vs Ascend | [concepts/deepep-nvidia-vs-ascend.md](concepts/deepep-nvidia-vs-ascend.md) | NVIDIA 原版:EP 通信库,V2 迁 NCCL Gin 后端、EPBuffer 统一接口、SM 占用 24→4–6(旧 V2 公告口径);与昇腾版同 API 不同后端;2026-10-21 `f6aa38fe` 起 `third-party/deep_jit` submodule 走 HTTPS(PR #788),同日 PR #781 删 lint 工具链 | ✅ 已入库(2026-10-21 更新) |
| DeepJIT | [concepts/deepjit.md](concepts/deepjit.md) | xPU kernel 运行时 JIT(CUDA + 昇腾),header-only;缓存键穷尽 source/include/编译版本;支持跨节点共享缓存 | ✅ 已入库 |
| Engram | [concepts/engram.md](concepts/engram.md) | 条件记忆:O(1) N-gram 查找,与 MoE 互补的稀疏轴;确定性寻址可 offload 到 host memory;27B 全面领先 iso-param MoE | ✅ 已入库 |
| DualPipe | [concepts/dualpipe.md](concepts/dualpipe.md) | 双向流水线并行,V3/R1 训练;气泡减半(参数 2× 换吞吐),DualPipeV 设备数减半 | ✅ 已入库 |
| 训练并行编排 V3 | [concepts/training-parallelism-v3.md](concepts/training-parallelism-v3.md) | 671B MoE 训练总账:PP16×EP64×ZeRO-1(不用 TP)+ DualPipe/DeepEP 重叠 + 显存预算;与 Megatron 3D 并行对照 | ✅ 已入库(2026-10) |
| deepseek-recipe | [concepts/deepseek-recipe.md](concepts/deepseek-recipe.md) | V4/V4.1 多格式 API 与 prompt 编解码库(Rust+Python bindings):Messages/Chat Completions/Responses↔Conversation↔token IDs;v0.2.0 新增 ModelVersion/结构化输出/logprobs | ✅ 已入库(2026-10,v0.2.0) |
| LPLB + EPLB | [concepts/lplb-epb-load-balancing.md](concepts/lplb-epb-load-balancing.md) | 专家并行负载均衡:EPLB 静态冗余专家复制(已部署)、LPLB 动态线性规划逐 batch 分流(IPM+cuSolverDx) | ✅ 已入库 |
| ESFT | [concepts/esft.md](concepts/esft.md) | 专家特化微调:打分→选任务相关 expert→只训相关 expert + adapter(EMNLP 2024) | ✅ 已入库 |
| open-infra-index | [concepts/open-infra-index.md](concepts/open-infra-index.md) | DeepSeek 开源基础设施总索引:开源周一天一 repo + 论文 + V3/R1 线上数据(73.7k/14.8k tok/s) | ✅ 已入库 |
| 3FS | [concepts/3fs.md](concepts/3fs.md) | Fire-Flyer 分布式文件系统:disaggregated + CRAQ 强一致 + FoundationDB;6.6 TiB/s 聚合读、GraySort 3.66 TiB/min | ✅ 已入库 |
| smallpond | [concepts/smallpond.md](concepts/smallpond.md) | DuckDB + 3FS 轻量数据处理框架,无长驻服务,PB 级;GraySort 与 3FS 同口径 | ✅ 已入库 |

### 实践清单(practices)

| 专题 | 文件 | 一句话 | 状态 |
|---|---|---|---|
| Ascend 部署检查清单 | [practices/ascend-deployment-checklist.md](practices/ascend-deployment-checklist.md) | 环境两线分开(950: CANN 9.2.0+PyTorch 2.13.0;A2/A3: CANN 8.3.RC1+);算子跑通顺序 TileLang→DeepGEMM→DeepEP→FlashMLA→DeepSelect + 各算子实操坑 | ✅ 已入库 |

---

## 相关条目(跨域互链)

- **[`cutlass/`](../cutlass/index.md)**:NVIDIA 侧「怎么造 GEMM 内核」的方法论(WGMMA/TMA/CuTe)。本域是同一问题在昇腾侧的对照:昇腾的 MAD/fractal/协程流水线 vs NVIDIA 的 WGMMA/TMA,见 [ascend-port-principles.md](concepts/ascend-port-principles.md) 与 [hopper-wgmma.md](../cutlass/concepts/hopper-wgmma.md)。
- **[`model-architecture/`](../model-architecture/index.md)**:MLA 低秩压缩机制([attention-heads-mha-mqa-gqa.md](../model-architecture/concepts/attention-heads-mha-mqa-gqa.md))、MoE 架构([open-model-landscape.md](../model-architecture/concepts/open-model-landscape.md))是本域 FlashMLA/DeepEP 的「架构层需求」来源。
- **[`attention/`](../attention/index.md)**:FlashAttention 内核机制([flash-attention-implementation.md](../attention/implementation/flash-attention-implementation.md))是 FlashMLA 稀疏内核(在线 softmax)的方法论前身。
- **[`vllm/`](../vllm/index.md)**:这些算子最终集成进推理引擎兑现吞吐;FP8 量化见 [quantization-gptq-awq-fp8.md](../vllm/concepts/quantization-gptq-awq-fp8.md),采样见 [sampling-penalties.md](../vllm/concepts/sampling-penalties.md)。

---

## 扩展位(2026-10 全库审查待补)

- ~~training-parallelism-v3~~(已入库)
- fp8-training-stack(细粒度 scaling、UE8M0、训练侧 DeepGEMM)
- inference-system-v3-r1(PD 分离、跨节点 EP 批扩展、73.7k/14.8k 与 545% 口径)
- profiling-overlap(profile-data 仓库、nsight/bench_msprof 方法论)
- fire-flyer-hardware(SC24 集群/网络/成本协同设计)
- 3fs-internals(USRBIO/CRAQ/FoundationDB/EC)
- deepep 低延迟解码路径、DreamCraft3D++ 增量、OCR 数据合成管线

## 溯源

- 外部链接索引见 [references.md](references.md)。
- 昇腾六件套仓库 + NVIDIA 原版对照仓库 + 并行/微调/存储各仓库 + 昇腾官方文档 + 各 deep-dive 技术报告的完整链接清单见 [references.md](references.md)。
