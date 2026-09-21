---
title: GPU 随机数生成
type: domain
tags: [gpu, rng, prng, philox, mtgp, curand]
source:
  - "pdf: Parallel Random Numbers: As Easy as 1, 2, 3 (SC11 Best Paper) | DOI 10.1145/2063384.2063405 | ~/zhanbin/library/papers/Parallel Random Numbers: As Easy as 1, 2, 3.pdf"
  - "pdf: Variants of Mersenne Twister Suitable for Graphic Processors | arXiv:1005.4973 | ~/zhanbin/library/papers/Variants of Mersenne Twister Suitable for Graphic Processors.pdf"
date: 2026-09
---

# GPU 随机数生成 (gpu-rng)

> 领域索引。主题:GPU 上高质量伪随机数怎么生成——counter-based(无状态双射)与 F2-linear 大块并行(MTGP)两条路线,及 cuRAND 工程落地。
> 域定位刻意收窄为 RNG(原计划 gpu-numerics,单个 concept 撑不起"数值算法"域,审批修正为 gpu-rng)。后续若新增 GPU 数值算法条目(如采样/噪声/量化随机化),可再评估扩域。

## 一句话核心

> 传统 PRNG 是串行状态递推,GPU 并行化的答案是:**counter-based**(第 n 个数 = 双射函数(计数器,key),零状态天然并行,Philox 最快)或 **MTGP**(一个 block 多线程协作推进一个大状态,超长周期)。cuRAND 默认 XORWOW 快但过不了 BigCrush,严肃场景选 `PHILOX4_32_10` 或 `MTGP32`。

## 内容索引(1 篇)

| 专题 | 文件 | 定位 | 状态 |
|---|---|---|---|
| GPU RNG 两条路线 | [concepts/gpu-random-number-generation.md](concepts/gpu-random-number-generation.md) | 懂:counter-based/Philox 机制 + MTGP 机制 + cuRAND API 用法 + SC11/MTGP 性能与 BigCrush 数据表 + 选型实践 | ✅ |
| 参考资料索引 | [references.md](references.md) | 溯源:官方库/文档链接 | ✅ |

## 选型速记

1. **极致吞吐 + 可复现** → Philox(counter-based);用应用变量拼 counter/key 得机器无关流
2. **超长周期 + 高维等分布** → MTGP(一 block 一 state,≤256 线程,参数集离线预生成)
3. **XORWOW 只配当 baseline**:BigCrush 系统性 reject,勿用于严肃蒙特卡洛
4. F2-linear(MT/MTGP/WELL)在 LinearComp/MatrixRank 测试上失败是共性,常规实数模拟一般无碍;密码学或线性结构敏感场景必须 counter-based

## 相关条目(跨域)

- [gpu-memory-lifecycle](../gpu-memory-lifecycle/index.md):cuRAND state 存 global memory 跨 kernel 复用时受 host-vs-device 生命周期约束
- [cutlass](../cutlass/index.md):同属 GPU kernel 工程域——Philox 融合进自定义 kernel 时的性能考量同"计算 vs 搬运"权衡
