---
title: Ascend 适配原理 — 从 NVIDIA 到昇腾的关键抽象对照
type: concept
tags: [ascend, port, cann, bisheng, fractal, mad, hccl, hcomm, urma, ubmem, deepjit, npu, porting]
source:
  - "web: 昇腾官方文档站 | https://www.hiascend.com/ ↑primary"
  - "code: tile-ai/tilelang-ascend(Programming Guide,术语/对照)| https://github.com/tile-ai/tilelang-ascend"
  - "code: deepseek-ai/DeepGEMM-Ascend(README,MAD/fractal/scaling factor)| https://github.com/deepseek-ai/DeepGEMM-Ascend"
  - "code: deepseek-ai/DeepEP-Ascend(README,HCCL/HCOMM/UBMEM/URMA/DeepJIT)| https://github.com/deepseek-ai/DeepEP-Ascend"
date: 2026-09
---

# Ascend 适配原理 — 从 NVIDIA 到昇腾的关键抽象对照

## TL;DR

昇腾(NPU)与 NVIDIA GPU 是两套不同的硬件/软件栈,把 NVIDIA kernel 搬到昇腾不是「换几个宏」,而是要**换一整套底层抽象**。本文给一张**对照表**,把每条 NVIDIA 概念对应到昇腾侧(拼写经一手源核实;标注「待核」者为一手库未附链接、据昇腾官方资料的术语):

| NVIDIA | 昇腾 |
|---|---|
| CUDA | **Ascend C**(编程语言)/ **TileLang**(tilelang-ascend 后端) |
| TMA / async copies | **稀疏数据加载 / 协程(coroutine)流水线** |
| cuBLAS | **MAD(matrix multiply-add)矩阵乘加原语** |
| NCCL | **HCCL**(集合通信)/ **HCOMM**(950 控制面 API,**待核/据昇腾官方资料**) |
| shared memory tile | **fractal 分形布局** |
| NVSHMEM | **URMA**(Unified Remote Memory Access,950 上 AIV 直驱 UB 远程内存)/ **UBMEM** |
| JIT(PTX/NVRTC) | **DeepJIT**(运行时 JIT) |

移植的真正难点不在「换名字」,而在:分形布局/对齐/地址计算被库层隐藏(DeepGEMM-Ascend/TileLang 做了这层抽象)、scaling factor 格式差异(UE8M0 每对打包 int16)、EP 带宽受 HDK 固件限制(PoC vs 商用)。

---

## 1. 昇腾软硬件背景(先认硬背景)

- **硬件代际**:昇腾 950 系列 NPU(华为 A5/Atlas 代际),同一颗 950 Die 出两款型(**待核/据昇腾官方资料**):
  - **950PR**(Prefill & Recommendation,性价比取向);
  - **950DT**(Decode & Training,高访存带宽取向)。
  - ⚠ **950PR ≠ 950DT**,两款型定位不同,别混用。
- **超节点**:128 卡超节点 **UBL128**(全互联,**待核/据昇腾官方资料**),本域六件套的目标部署形态。
- **软件栈**:CANN 9.2.0(统一版本;**DeepGEMM-Ascend README 原文写 "9.20" 即此版本,写法不规范**)、Ascend C 编程语言、bisheng(毕昇编译器)、torch_npu(PyTorch 昇腾适配)。

---

## 2. 对照表逐条展开(拼写,部分术语待核)

| NVIDIA | 昇腾 | 说明 |
|---|---|---|
| **CUDA** | **Ascend C** / **TileLang**(tilelang-ascend) | Ascend C 是手写内核语言;TileLang 是 tile 级 DSL,昇腾后端独立仓 tilelang-ascend |
| **TMA / async copies** | **稀疏数据加载 / 协程流水线** | DeepGEMM-Ascend 用这两项代替原版 Hopper 的 TMA/persistent 机制 |
| **cuBLAS** | **MAD**(matrix multiply-add)原语 | 昇腾的矩阵乘累加硬件原语;DeepGEMM-Ascend 在 MAD 上做轻量抽象 |
| **NCCL** | **HCCL** + **HCOMM** | HCCL=集合通信库(对标 NCCL);HCOMM=950 控制面通信 API(替代旧 HCCP RA,**待核/据昇腾官方资料**),**两者不是同义词** |
| **shared memory tile** | **fractal 分形布局** | fractal 是**数据布局**(分形排列),不是算子;DeepGEMM/TileLang 把它藏起来 |
| **NVSHMEM** | **URMA** / **UBMEM** | URMA=Unified Remote Memory Access(950 上 AIV 直驱 UB 远程内存);UBMEM=片上统一缓冲内存;URMA≠UDMA(UDMA 是消费 URMA 的 SHMEM 实现,**待核/据昇腾官方资料**) |
| **JIT(PTX/NVRTC)** | **DeepJIT** | 运行时 JIT,独立仓 deepseek-ai/DeepJIT,DeepEP-Ascend 用它运行时编译 |

---

## 3. 移植难点(库层已隐藏的部分)

1. **分形布局 / 对齐约束 / 地址计算**:昇腾的矩阵数据按 fractal 分形布局存放,对齐与地址计算繁琐——DeepGEMM-Ascend(在 MAD 上)、TileLang(在 tile 级)分别把这块藏起来,让 kernel 保持简洁。
2. **scaling factor 格式差异**:昇腾上每对 UE8M0 scaling factor 沿 K 维打包进一个 `int16`,并按 MN-major 存储(为硬件效率),与 NVIDIA 的格式不同。
3. **EP 带宽受 HDK 固件限制**:DeepEP-Ascend 实测用的是手工配置的 PoC HDK;商用 Atlas 850E HDK(约 2026-10 中旬)才会带完整带宽配置——PoC 与商用性能有差异。

---

## 4. 关键数据(带口径,汇总自各篇)

| 指标 | 数值 | 口径 |
|---|---|---|
| GEMM 利用率 | **99.8%** | BF16×BF16,M=4096/N=7168/K=16384(DeepGEMM-Ascend README Performance 表,950DT + CANN 9.20 + bench_msprof cold L2) |
| EP dispatch 带宽 | **90–95%** 物理载荷带宽 | EP≤32 时(950DT + CANN 9.2.0 + PoC HDK) |
| FlashMLA 昇腾 prefill | **95%** 硬件峰值(410 TFlops) | 昇腾 950,sparse prefill |
| FlashMLA 昇腾 decode | **83%** 硬件峰值(360 TFlops) | 昇腾 950,sparse decoding |

---

## 5. 实践要点 / 坑

- **跑昇腾组件需 950 + 独占 HDK**:本域六件套均面向昇腾 950 系列;DeepEP-Ascend 还需 UBMEM 连通 + UBC_CTP/URMA 通道、偶数 rank。
- **128 卡超节点 UBL128** 是 DeepSeek 昇腾方案的部署形态。
- **术语雷区速记**:
  - **950PR ≠ 950DT**(PR=Prefill&Recommendation、DT=Decode&Training,同一 950 Die 两款型);
  - **CANN 9.20 与 9.2.0 是同一版本**,统一写 CANN 9.2.0;
  - **fractal 是布局不是算子**;
  - **HCOMM 不是 HCCL 的别名**(HCCL=集合通信,HCOMM=控制面,**待核/据昇腾官方资料**);
  - **URMA ≠ UDMA**(URMA=远程内存访问语义,UDMA=SHMEM 消费 URMA 的实现,**待核/据昇腾官方资料**)。

---

## 溯源

- **昇腾官方文档**:https://www.hiascend.com/ (Ascend C、CANN、bisheng、torch_npu 权威口径)
- **tilelang-ascend Programming Guide**:术语与三层编程接口、CANN 环境 —— https://github.com/tile-ai/tilelang-ascend/blob/main/docs/TileLang-Ascend%20Programming%20Guide.md
- **DeepGEMM-Ascend README**:MAD/fractal/稀疏加载/协程/scaling factor —— https://github.com/deepseek-ai/DeepGEMM-Ascend
- **DeepEP-Ascend README**:HCCL/HCOMM/UBMEM/URMA/DeepJIT、950DT+CANN 9.2.0+PoC HDK 口径 —— https://github.com/deepseek-ai/DeepEP-Ascend
- **库内交叉**:[../../cutlass/concepts/hopper-wgmma.md](../../cutlass/concepts/hopper-wgmma.md)(TMA/WGMMA 对照)、[../../cutlass/concepts/cute-layout-algebra.md](../../cutlass/concepts/cute-layout-algebra.md)(布局代数对照)。
