---
title: FlashAttention
type: domain
tags: [flashattention, attention, gpu, cuda]
source:
  - "code: github.com/Dao-AILab/flash-attention | Star 24.7k"
  - "pdf: FlashAttention FA1 | arXiv:2205.14135 NeurIPS 2022"
  - "pdf: FlashAttention-2 FA2 | arXiv:2307.08691 ICLR 2024"
  - "pdf: FlashAttention-3 | arXiv:2407.08608 NeurIPS 2024"
date: 2026-09
---

# FlashAttention

> 领域索引(导航起始页)。**领域知识本体在 `concepts/` 子目录,每篇一个专题、可独立打开读懂**;本页放定位、关键事实与导航,不重复正文。各篇溯源见文末,内容自足、不依赖库外文件。


## 一句话定位

FlashAttention 是 Tri Dao 主导的开源高效 Attention 实现,核心思想是 **IO-aware tiling**:把 attention 分块(tile)放进 SRAM,用 online softmax 让 softmax 可增量合并,从而**不在 HBM 实例化完整的 N×N attention 矩阵**——不损失精度(精确)地**加速 + 省显存**。从 2022 起已四代演进。

**一句话收益**:标准 attention 的 HBM 访存 Θ(Nd + N²)(元素访问) → FlashAttention Θ(N²d²/M)(M=SRAM),显存 O(N²)→O(N)。**瓶颈是内存搬运,不是计算**。

## 四代演进

| 代次 | 时间 | 论文 | 目标 GPU | 核心实现技术 | MFU(前向) |
|---|---|---|---|---|---|
| FA1 | 2022.05 | NeurIPS 2022 | A100 | IO-aware tiling + online softmax | 25-40% |
| FA2 | 2023.07 | ICLR 2024 | A100+H100 | 更好 work partitioning, 减 non-matmul FLOPs | 50-73% |
| FA3 | 2024.07 | NeurIPS 2024 | H100 | WGMMA async, TMA, FP8 | ~75% FP16 / ~61-66% FP8 |
| FA4 | 2026.03 | arXiv:2603.05451 (MLSys 2026) | B200 | CuTe DSL, 异步 UMMA/tcgen05, 软 exp+软 softmax rescale, TMEM+2-CTA MMA | ~71% (BF16, up to 1613 TFLOPS) |

## 本库沉淀条目(obrain 内可读)

> 领域 index.md 只放导航;**正文在 `concepts/` 下**,随浏览随时打开。

| 条目 | 类型 | 一句话 | 状态 |
|---|---|---|---|
| [IO-aware tiling 与 Online Softmax](concepts/io-aware-tiling.md) | concept | 为什么标准 attention 慢(内存 N²)、softmax 不结合→online softmax、完整算法、IO 复杂度、FA2+ 演进 | ✅ 已入库 |
| [并行与 Work Partitioning(FA2)](concepts/work-partitioning.md) | concept | FA2 为什么快 2×:减 non-matmul FLOPs、warp 按 Q 行切分免通信、序列维 split-K | ✅ 已入库 |
| [Hopper 异步内核(FA3)](concepts/fa3-hopper-async.md) | concept | FA3:异步 WGMMA、TMA、softmax/matmul/load 重叠 pipeline、FP8 | ✅ 已入库 |

## 为什么快(标准 Attention 的问题)

标准 attention 中 `S = Q@Kᵀ·scale` 和 `P = softmax(S)` 都是 N×N 矩阵,需写入 HBM 再读回:

- N=8192, d=128 时,`S` 占 `8192²×2B ≈ 128MB`,远超 GPU SRAM(~192KB)
- IO 开销:标准 attention 需 **Θ(Nd + N²)** 次 HBM 读写(元素访问,完整 N×N 矩阵实例化两次)
- FA 通过 tiling + online softmax,把 HBM 访问降到 **Θ(N²d²/M)**(M=SRAM 容量;对比标准 Θ(Nd+N²),大 N 时数量级下降)

## 相关论文(公开可溯源)

- FA1: Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022, arXiv:2205.14135
- FA2: Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, ICLR 2024, arXiv:2307.08691
- FA3: Dao et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, NeurIPS 2024, arXiv:2407.08608
