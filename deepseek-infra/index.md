---
title: deepseek-infra — DeepSeek 开源算力基础设施栈(昇腾/双平台)
type: domain
tags: [deepseek, ascend, tilelang, deepgemm, deepep, tilekernels, flashmla, deepselect, npu, huawei, infra]
source:
  - "code: deepseek-ai/DeepGEMM-Ascend | https://github.com/deepseek-ai/DeepGEMM-Ascend"
  - "code: deepseek-ai/DeepEP-Ascend | https://github.com/deepseek-ai/DeepEP-Ascend"
  - "code: deepseek-ai/TileKernels | https://github.com/deepseek-ai/TileKernels"
  - "code: deepseek-ai/FlashMLA | https://github.com/deepseek-ai/FlashMLA"
  - "code: deepseek-ai/DeepSelect | https://github.com/deepseek-ai/DeepSelect"
  - "code: tile-ai/tilelang + tile-ai/tilelang-ascend | https://github.com/tile-ai/tilelang"
date: 2026-09
---

# deepseek-infra — DeepSeek 开源算力基础设施栈(昇腾/双平台)

> 本页是 deepseek-infra 领域的**导航起始页**。往下走:定位与关键事实 → 条目索引 → 更新维护。
> **真正的可读正文在 `concepts/` 子目录**,每篇一个专题、可独立打开读懂。本页只做导航,不重复正文。

---

## 一句话定位

**deepseek-infra** 记录 DeepSeek 于 2026-09-30 开源的**算力基础设施栈的华为昇腾(NPU)适配与双平台实现**——把「算子怎么写(TileLang)→ 矩阵怎么算(DeepGEMM-Ascend / TileKernels)→ 专家怎么通信(DeepEP-Ascend)→ 注意力怎么算(FlashMLA / DeepSelect)」这条从语言到算子的全链路,在昇腾 950 超节点上跑出接近硬件峰值的效果。它定位在**算子实现层 / 基础设施层**,与库内 model-architecture(架构)、attention(算法)、vllm(引擎)、cutlass(NVIDIA 造内核方法论)四个领域正交互补:那些域讲「模型长什么样、注意力为什么省显存、引擎怎么调度、NVIDIA 怎么造内核」,本域讲「**这些算子搬到昇腾上怎么写、怎么适配、性能到哪**」。

---

## 关键事实速记(快照,2026-09)

- **开源时间**:2026-09-30,DeepSeek 开源昇腾推理基础设施,华为团队支持;目标平台为昇腾 950 系列 NPU 及 **128 卡超节点 UBL128**(全互联)。
- **六件套覆盖链路**:语言层 → 矩阵库 → 通信库 → 注意力算子。
- **三条 P0 事实(本域写作最易踩,务必按此记)**:
  1. **TileLang 是 tile-ai 的第三方 DSL,不是 DeepSeek 的仓库**;昇腾后端拆在独立仓 `tile-ai/tilelang-ascend`(双路线:Ascend C & PTO / AscendNPU IR)。
  2. **FlashMLA 昇腾版在独立分支 `20260930-ascend-open-source` 且是 breaking change**:移除 Hopper 支持、移除 V3/V3.2/V4.0 旧模型、改 FP8/FP4 KV cache 格式,只服务 V4.1;旧模型切旧 commit `ba89a346`。
  3. **TileKernels 昇腾支持从 PyPI v2.0.0 起**,但 main README 滞后(仍只写 NVIDIA SM90/SM100)。
- **术语(本域核定拼写)**:fractal(分形**布局**)、MAD(matrix multiply-add 原语)、UBMEM、URMA、HCCL(集合通信)、HCOMM(控制面 API)、bisheng(毕昇编译器)、DeepJIT(运行时 JIT)、torch_npu、**CANN 9.2.0**(DeepGEMM README 原文 "9.20" 即此版本,不写 9.20)、**950PR ≠ 950DT**(同一 950 Die 两款型:PR=Prefill&Recommendation、DT=Decode&Training)。

### 六件套 ↔ 上游映射表(已核实)

| 昇腾新增/适配 | 上游/对应 NVIDIA 版 | 性质 | 真实 repo |
|---|---|---|---|
| TileLang(昇腾后端) | tile-ai/tilelang(TileLang 原生) | 新增昇腾后端 | tile-ai/tilelang + tile-ai/tilelang-ascend |
| DeepGEMM-Ascend | DeepGEMM | 新建独立仓,API 完全兼容 | deepseek-ai/DeepGEMM-Ascend |
| DeepEP-Ascend | DeepEP | 新建独立仓,API 完全兼容 | deepseek-ai/DeepEP-Ascend |
| TileKernels | TileKernels 原生 | 新增昇腾后端(v2.0.0) | deepseek-ai/TileKernels |
| FlashMLA | FlashMLA 原生 | 昇腾版在独立分支 | deepseek-ai/FlashMLA |
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
| TileLang 与昇腾后端 | [concepts/tilelang.md](concepts/tilelang.md) | tile 级 DSL(TVM 之上),昇腾后端独立仓双路线(Ascend C & PTO / AscendNPU IR),三层编程接口;基准 GEMM 0.98× 手写 AscendC | ✅ 已入库 |
| Ascend 适配原理 | [concepts/ascend-port-principles.md](concepts/ascend-port-principles.md) | NVIDIA→昇腾关键抽象对照表(CUDA→Ascend C、TMA→稀疏加载/协程、NCCL→HCCL/HCOMM、fractal 布局等)+ 移植难点 | ✅ 已入库 |
| DeepGEMM-Ascend | [concepts/deepgemm-ascend.md](concepts/deepgemm-ascend.md) | 昇腾 GEMM 库,MAD 原语轻量抽象,藏 fractal/对齐/地址计算;稀疏加载 + 协程流水线;scaling factor 格式与 NVIDIA 不同 | ✅ 已入库 |
| DeepEP-Ascend | [concepts/deepep-ascend.md](concepts/deepep-ascend.md) | MoE 专家并行通信库,EP all-to-all(dispatch/combine),EP≤32 dispatch 达 90–95% 物理带宽 | ✅ 已入库 |
| TileKernels | [concepts/tilekernels.md](concepts/tilekernels.md) | TileLang 之上的 LLM 算子库(MoE routing/量化/Engram/mHC),v2.0.0 起支持昇腾,同一 Python API 跑 NVIDIA+华为 | ✅ 已入库 |
| DeepSelect | [concepts/deepselect.md](concepts/deepselect.md) | DSA/Sampler 的定制 TopK 算子,2–20× vs torch.topk(口径:有效内存带宽);昇腾仅 bfloat16 | ✅ 已入库 |
| FlashMLA | [concepts/flashmla.md](concepts/flashmla.md) | MLA 注意力内核库(dense+sparse),昇腾版在独立分支且 breaking change;410 TFlops 双义(平台要标清) | ✅ 已入库 |

---

## 相关条目(跨域互链)

- **[`cutlass/`](../cutlass/index.md)**:NVIDIA 侧「怎么造 GEMM 内核」的方法论(WGMMA/TMA/CuTe)。本域是同一问题在昇腾侧的对照:昇腾的 MAD/fractal/协程流水线 vs NVIDIA 的 WGMMA/TMA,见 [ascend-port-principles.md](concepts/ascend-port-principles.md) 与 [hopper-wgmma.md](../cutlass/concepts/hopper-wgmma.md)。
- **[`model-architecture/`](../model-architecture/index.md)**:MLA 低秩压缩机制([attention-heads-mha-mqa-gqa.md](../model-architecture/concepts/attention-heads-mha-mqa-gqa.md))、MoE 架构([open-model-landscape.md](../model-architecture/concepts/open-model-landscape.md))是本域 FlashMLA/DeepEP 的「架构层需求」来源。
- **[`attention/`](../attention/index.md)**:FlashAttention 内核机制([flash-attention-implementation.md](../attention/implementation/flash-attention-implementation.md))是 FlashMLA 稀疏内核(在线 softmax)的方法论前身。
- **[`vllm/`](../vllm/index.md)**:这些算子最终集成进推理引擎兑现吞吐;FP8 量化见 [quantization-gptq-awq-fp8.md](../vllm/concepts/quantization-gptq-awq-fp8.md),采样见 [sampling-penalties.md](../vllm/concepts/sampling-penalties.md)。

---

## 溯源

- 外部链接索引见 [references.md](references.md)。
- 六件套仓库 + 昇腾官方文档 + 各 deep-dive 技术报告的完整链接清单见 [references.md](references.md)。
