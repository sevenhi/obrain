---
title: DeepEP-Ascend — MoE 专家并行通信库
type: concept
tags: [deepep, ascend, moe, expert-parallel, all-to-all, dispatch, combine, hccl, hcomm, urma, ubmem, npu]
source:
  - "code: deepseek-ai/DeepEP-Ascend | https://github.com/deepseek-ai/DeepEP-Ascend ↑primary"
  - "code: deepseek-ai/DeepEP(原版,API 对齐基准)| https://github.com/deepseek-ai/DeepEP"
  - "code: deepseek-ai/DeepJIT(运行时 JIT)| https://github.com/deepseek-ai/DeepJIT"
date: 2026-09
---

# DeepEP-Ascend — MoE 专家并行通信库

## TL;DR

**DeepEP-Ascend** 是面向华为昇腾 NPU 的高性能训练/推理**通信库**,提供 MoE 的专家并行(EP)**all-to-all** 操作(dispatch/combine),支持 **FP8 dispatch** 和**延迟 epilogue**;此外还提供 PP、CP/DP 的 Bucket 集合通信,以及 Engram 远端内存访问(开发中)。**公开 buffer API 与 NVIDIA 版 DeepEP 对齐**。

底层:**Ascend C 内核** + **HCCL/HCOMM** + **UBMEM** + **URMA**,通过 **DeepJIT** 运行时编译。

**关键数据(带完整口径)**:Ascend 950DT + CANN 9.2.0 + 手工配置的 PoC HDK,netlayer 1(超节点外部 Clos 网络)。16384 tok/rank、hidden 7168、top-6 routing / 256 experts、32 AI cores / 64 AIVs、expert alignment 128、无 bias;dispatch FP8(row-major scales)、combine BF16。EP≤32 时 dispatch 达物理载荷带宽 **90–95%**。

---

## 1. 机制:MoE 专家并行的 all-to-all 通信

MoE 模型里,每个 token 由 router 决定进哪几个 expert。专家并行(EP)下 expert 分布在不同 rank 上,于是每步都要做两类 all-to-all:

- **dispatch**:把 token 从「其所在 rank」发给「负责其 expert 的 rank」;
- **combine**:expert 算完后,把结果从「expert rank」发回「token 原 rank」。

DeepEP-Ascend 就是这两类通信的高性能实现,支持:
- **FP8 dispatch**(发前把 token 量化成 FP8,省带宽)+ **延迟 epilogue**(epilogue 计算延后,与通信重叠);
- **PP/CP/DP Bucket 集合通信**(流水线并行、上下文并行、数据并行的分桶集合通信);
- **Engram 远端内存访问**(开发中)。

### 底层技术栈

| 组件 | 作用 |
|---|---|
| **Ascend C 内核** | 通信内核实现语言 |
| **HCCL** | 集合通信库(对标 NCCL) |
| **HCOMM** | 950 控制面通信 API(替代旧 HCCP RA) |
| **UBMEM** | 片上统一缓冲内存 |
| **URMA** | Unified Remote Memory Access,950 上 AIV 直驱 UB 远程内存 |
| **DeepJIT** | 运行时 JIT 编译内核(独立仓 deepseek-ai/DeepJIT) |

---

## 2. 性能数据(带完整环境口径,与 README 完全一致)

**环境**:Ascend **950DT** + CANN **9.2.0** + 手工配置的 **PoC HDK**,netlayer 1(超节点外部 Clos 网络);16384 tok/rank、hidden 7168、top-6 routing / 256 experts、32 AI cores / 64 AIVs、expert alignment 128、无 bias;dispatch FP8(row-major scales)、combine BF16;每 rank 10 warmups + 50 samples(cache flushing)。**带宽范围覆盖所有 rank,计时含 issue+drain、不含 final epilogue**。

| EP 规模 | Dispatch (GB/s) | Combine (GB/s) |
|---|---|---|
| EP8 | 373–375 | 345–347 |
| EP16 | 348–352 | 338–341 |
| EP32 | 335–340 | 320–324 |
| EP64 | 323–327 | 294–298 |
| EP128 | 313–320 | 272–278 |

**结论**:EP≤32 时 dispatch 达物理载荷带宽 **90–95%**;更大 EP 与 combine 仍在优化——combine 有本地 reduction 开销 + 与 URMA 的 HBM 竞争,华为计划固件改进以降低竞争。

---

## 3. 实践要点 / 坑

- **环境要求**:昇腾 950(A5)带 UBMEM 连通 + UBC_CTP/URMA 通道、参与 rank 数为偶数;CANN(Ascend C、bisheng 编译器、HCCL/HCOMM 头文件)、Python 3.10+、匹配的 PyTorch/`torch_npu`、C++20 编译器。验证栈:Ascend 950DT、CANN 9.2.0、Python 3.12、PyTorch 2.13.0+cpu、torch_npu 2.13.0rc1。
- **⚠ PoC HDK vs 商用 HDK(P1 坑)**:README 性能用的是**手工配置的 PoC HDK,未公开分发**;**Atlas 850E 商用 HDK(约 2026-10 中旬)将含完整带宽配置**——PoC 与商用性能有差异,生产使用前建议关注官方后续更新。
- **术语雷区**:
  - **HCCL ≠ HCOMM**:HCCL=集合通信库,HCOMM=950 控制面 API;
  - **UBMEM/URMA ≠ UDMA**:URMA=远程内存访问语义,UDMA=SHMEM 消费 URMA 的实现;
  - **EP 带宽数字别漏口径**:PoC HDK、netlayer 1、dispatch 不含 final epilogue。

---

## 溯源

- **DeepEP-Ascend README**(主源:机制、性能表+环境口径、PoC/商用 HDK、HCCL/HCOMM/UBMEM/URMA/DeepJIT):https://github.com/deepseek-ai/DeepEP-Ascend
- **DeepEP 原版**(buffer API 对齐基准):https://github.com/deepseek-ai/DeepEP
- **DeepJIT**(运行时 JIT):https://github.com/deepseek-ai/DeepJIT
- **库内交叉**:[../../model-architecture/concepts/open-model-landscape.md](../../model-architecture/concepts/open-model-landscape.md)(MoE 架构背景)、[../../vllm/concepts/distributed-ray.md](../../vllm/concepts/distributed-ray.md)(并行通信)。
