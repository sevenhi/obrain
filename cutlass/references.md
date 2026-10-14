---
title: cutlass 外部链接索引
type: reference
tags: [cutlass, cute, gemm, references]
date: 2026-09
---

# References — cutlass

> 只放链接索引,不复制全文(库规约)。聚合自各篇 frontmatter 的 source;条目级溯源见各篇文末『溯源』。

## 官方文档

- CUTLASS 官方文档站 — https://docs.nvidia.com/cutlass/latest/
- CUTLASS Python / CuTe DSL 概览 — https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html
- Quickstart — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html
- Efficient GEMM in CUDA — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
- CuTe Layouts — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html
- CuTe Layout Algebra — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html
- CuTe MMA instructions — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html
- Grouped Kernel Schedulers — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html
- Blackwell persistent clusters / Cluster Launch Control (CLC) — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html
- Profiler — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html
- Custom epilogue fusions for GEMMs — https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/001_gemm_with_fused_epilogue.html
- PTX ISA(tcgen05.mma 章节)— https://docs.nvidia.com/cuda/parallel-thread-execution/

## 论文 / 技术报告

- Stream-K: Work-centric Parallel Decomposition — arXiv:2301.03598
- A Generalized Micro-kernel Abstraction for GPU Linear Algebra (BLIS Retreat 2023) — https://www.cs.utexas.edu/~flame/BLISRetreat2023/slides/Thakkar_BLISRetreat2023.pdf
- Understanding GEMM Performance and Energy on NVIDIA Ada Lovelace — arXiv:2411.16954

## NVIDIA / Colfax 博客

- CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions — https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/
- CUTLASS: Principled Abstractions for Handling Multidimensional Data — https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels/
- Improving GEMM Kernel Auto-Tuning Efficiency (CUTLASS 4.2) — https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/
- Colfax — Fast Matrix-Multiplication with WGMMA on Hopper — https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
- Colfax — Design of a GEMM Kernel (Pipelining) — https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/

## 代码与仓库

- CUTLASS GitHub — https://github.com/NVIDIA/cutlass
- examples/tutorial/wgmma_sm90.cu、examples/49_hopper_gemm_with_collective_builder、examples/71_blackwell_gemm_with_collective_builder、examples 12/35/37/45(epilogue 融合示例)
- sm90_tile_scheduler_stream_k.hpp / sm100_tile_scheduler_stream_k.hpp
- include/cute/arch/ tcgen05/UMMA/TMEM 头文件:mma_sm100_umma.hpp、mma_sm100_desc.hpp、tmem_capacity_sm100.hpp、tmem_allocator_sm100.hpp、mma_sm107_umma.hpp、mma_sm120.hpp、mma_sm120_sparse.hpp(详见 blackwell-tcgen05.md 溯源)

## 社区教程

- Lei Mao — CuTe Thread-Value Layout — https://leimao.github.io/blog/CuTe-Thread-Value-Layout/
- Lei Mao — CuTe Tiled MMA — https://leimao.github.io/blog/CuTe-Tiled-MMA/
- 杨远航《CUTLASS 笔记》系列 — https://www.zhihu.com/column/c_1938664963049763058
- PyTorch Blog — TorchInductor's CuteDSL backend — https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/

## 硬件规格

- H100 Datasheet — https://resources.nvidia.com/en-us-data-center-overview/nvidia-tensor-core-gpu-datasheet
- A100 Datasheet — https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
