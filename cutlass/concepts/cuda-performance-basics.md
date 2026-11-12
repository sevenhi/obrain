---
title: CUDA GPU 性能基础：从带宽、占用率到 Roofline
type: concept
tags: [cuda, gpu, performance, memory-bandwidth, occupancy, roofline, profiler]
source:
  - "web: CUDA 12.6.3 Best Practices Guide（shared memory broadcast） | https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks | reviewed 2026-11-04"
  - "web: NVIDIA CUDA C++ Programming Guide | https://docs.nvidia.com/cuda/cuda-c-programming-guide/ | 阅读日期 2026-09-27"
  - "web: NVIDIA CUDA Best Practices Guide | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ | 阅读日期 2026-09-27"
  - "web: NVIDIA Nsight Compute Documentation | https://docs.nvidia.com/nsight-compute/ | 阅读日期 2026-09-27"
  - "web: CUTLASS 官方文档 | https://docs.nvidia.com/cutlass/latest/ | 阅读日期 2026-09-27"
date: 2026-09
reviewed: 2026-11-04
---

# CUDA GPU 性能基础：从带宽、占用率到 Roofline

> **TL;DR**：GPU kernel 慢，先判断是计算受限、显存带宽受限、片上存储受限，还是同步/启动/负载不均衡。FLOPs 越高不等于越快；真正要看数据移动、并发是否足够、Tensor Core 是否被喂饱，以及 kernel 是否把多个算子融合成更少的 HBM 往返。

## 1. 四层存储账单

| 层级 | 特征 | 常见用途 | 主要风险 |
|---|---|---|---|
| Registers | 每线程私有、最快、容量小 | 累加器、临时值 | 寄存器过多降低 occupancy，甚至 spill 到 local memory |
| Shared Memory / L1 | CTA 内共享、低延迟、容量有限 | tile、数据复用、协作加载 | bank conflict、同步、tile 放不下 |
| L2 | 跨 SM 缓存、容量中等 | 重用、读写合并后的缓存 | 命中率和访问模式影响大 |
| HBM/Global Memory | 容量大、带宽高但延迟高 | 权重、激活、KV cache | 非合并访问、重复读写、带宽饱和 |

性能优化的基本方向是：让数据尽量在更近的层级复用，减少 HBM 访问；但不能无限增加 shared memory 和 registers，否则会减少同时驻留的 CTA。

## 2. Arithmetic Intensity 与 Roofline

算术强度定义为：

```text
arithmetic intensity = FLOPs / bytes moved
```

Roofline 用两个上限判断 kernel：

```text
可达到性能 ≤ min(峰值计算吞吐, 内存带宽 × 算术强度)
```

- 低算术强度：通常是 memory-bound，优化重点是合并访问、复用、融合和减少读写。
- 高算术强度但达不到计算峰值：检查 Tensor Core 使用、指令形状、数据类型、流水线和 occupancy。
- 位于拐点附近：改变 tile、融合策略和 batch 可能改变受限类型。

Roofline 是定位工具，不是性能证明；实际还会受到指令发射、同步、L2、分支和负载不均衡影响。

跨域比较复杂度时固定三个口径：**训练/预填充的全序列总工作量、解码每步工作量、状态/额外空间**。例如固定头维 d 的 full attention，长度 n 的训练注意力工作量为 O(n²d)，单步 decode 读取 KV 为 O(nd)，KV 状态为 O(nd)；这不包含投影/FFN 等其它成本。FlashAttention 减少中间矩阵物化与访存，并不把 dense attention 的算术量改为线性。显存账还应区分输入/输出、持久状态与临时 workspace；测量结果须附 shape、dtype、软硬件版本和计时口径。

## 3. Occupancy 不是越高越好

Occupancy 通常表示一个 SM 上活跃 warp 数相对硬件最大 warp 数的比例。它受以下资源共同限制：

- 每个线程的 registers
- 每个 CTA 的 shared memory
- 每个 CTA 的线程数
- SM 可容纳的线程、warp、CTA 数

高 occupancy 有助于隐藏内存延迟，但并不保证 kernel 更快：

- 计算密集型 GEMM 可能只需足够的 warp 来填满 Tensor Core pipeline。
- 提高 occupancy 可能迫使 tile 变小，降低数据复用。
- 过多寄存器导致 spill 时，理论 occupancy 的提升可能换来更慢的 local memory 流量。

因此应把 occupancy 与 achieved occupancy、SM throughput、memory throughput、stall reason 一起看。

## 4. 合并访问与数据布局

一个 warp 访问连续且对齐的地址时，global memory 更容易合并为少量 transaction。常见问题：

- stride 太大或线程按列访问 row-major 矩阵
- AoS 布局导致同一字段不连续，改为 SoA 后更易合并
- 未对齐地址或不规则 gather
- padding 与 tile 形状不匹配
- shared memory 中不同线程访问同一 bank 的不同地址，可能产生 bank conflict；同一 warp 读取同一 32-bit 地址可广播，不应误判为冲突。具体还要看访问宽度、指令与硬件代际。

CUTLASS 的 Layout、Copy Atom、TiledCopy 和 `partition` 抽象，本质上是在编译期描述线程、数据和硬件 tile 的对应关系；参见 [CuTe Layout 代数](cute-layout-algebra.md)。

## 5. GEMM 与 attention 的不同瓶颈

### GEMM

- 重点是 Tensor Core 指令利用率、M/N/K tile、shared-memory pipeline、epilogue 融合。
- 大矩阵通常有足够工作量；小 batch、瘦矩阵和尾块会产生浪费。
- split-K、Grouped GEMM 和 Stream-K 是处理并行度与负载不均衡的不同路线。

### Attention

- QKᵀ、softmax、PV 如果把中间矩阵写回 HBM，会有大量不必要流量。
- FlashAttention 通过 tiling、online softmax 和融合减少 HBM 往返，数学结果仍是精确 attention。
- 长上下文下，KV cache 的布局、分页和量化会直接改变带宽与容量账单。

参见 [分层 GEMM tiling](gemm-tiling.md)、[FlashAttention 实现](../../attention/implementation/flash-attention-implementation.md) 和 [KV Cache 量化](../../quantization/concepts/kv-cache-quantization.md)。

## 6. 推荐 profiling 顺序

1. 先测端到端指标：延迟、吞吐、显存峰值和 kernel 时间占比。
2. 用 Nsight Systems 看 CPU/GPU 时间线、kernel 间隙、同步和数据拷贝。
3. 用 Nsight Compute 选关键 kernel，查看 achieved FLOPS、DRAM throughput、L2 命中、occupancy、寄存器、shared memory 和 stall reason。
4. 判断瓶颈类别后一次只改一个变量：tile、layout、stage、融合、数据类型或 launch 配置。
5. 用正确性测试和多个输入形状回归，避免只优化单一 benchmark。

## 7. 典型症状到假设

| 症状 | 优先检查 |
|---|---|
| GPU 利用率低但 kernel 很多 | kernel launch、CPU 调度、同步、碎片化小算子 |
| DRAM throughput 接近峰值 | 访存合并、数据复用、融合、数据类型和布局 |
| Tensor Core 利用率低 | dtype、矩阵形状、对齐、MMA tile、kernel dispatch |
| registers 很高、occupancy 低 | 累加器和 tile 规模、unroll、spill |
| 单个 kernel 很快但端到端不快 | kernel 间读写、launch gap、框架转换和同步 |
| 大矩阵快、小矩阵慢 | 并行度、尾块、Grouped/Stream-K、批处理方式 |

## 相关条目

- [CUTLASS 分层 GEMM](gemm-tiling.md)
- [CuTe Layout 代数](cute-layout-algebra.md)
- [CUTLASS 性能调优](../frameworks/cutlass-profiling-tuning.md)
- [Hopper WGMMA](hopper-wgmma.md)
- [FlashAttention 实现](../../attention/implementation/flash-attention-implementation.md)

## 溯源

- 主源:NVIDIA CUDA C++ Programming Guide(https://docs.nvidia.com/cuda/cuda-c-programming-guide/)与 CUDA Best Practices Guide(https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)。
- 交叉源:Nsight Compute 文档(https://docs.nvidia.com/nsight-compute/,profiler 口径)、CUTLASS 官方文档(https://docs.nvidia.com/cutlass/latest/)。
