---
title: gpu-rng 参考资料索引
type: reference
tags: [gpu, rng, prng, curand, references]
date: 2026-09
---

# gpu-rng 参考资料

> 链接索引,不复制全文(URL 均于 2026-09 实测存活)。库内原始 PDF 见 `~/zhanbin/library/papers/`(只读存档,引用不复制)。

## 论文(库内 PDF 的权威在线版)

- Parallel Random Numbers: As Easy as 1, 2, 3 (SC11 Best Paper) — https://www.thesalmons.org/john/random123/papers/random123sc11.pdf (ACM DL 版 DOI 10.1145/2063384.2063405,需订阅)
- Variants of Mersenne Twister Suitable for Graphic Processors — https://arxiv.org/abs/1005.4973

## 官方实现/参数集

- Random123(ARS/Threefry/Philox,C/C++/CUDA/OpenCL)— https://www.thesalmons.org/john/random123/ ;GitHub: https://github.com/deshawresearch/random123
- MTGP / MTGPDC 源码与参数集(広島大)— https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/MTGP/
- Mersenne Twister 主页(MT19937/SFMT 背景)— https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/emt.html

## 工程文档(cuRAND)

- cuRAND Host API Overview(生成器类型: XORWOW/MRG32K3A/MTGP32/PHILOX4_32_10/MT19937)— https://docs.nvidia.com/cuda/curand/host-api-overview.html
- cuRAND Device API Overview(kernel 内 curand_init/curand 用法、Performance Notes)— https://docs.nvidia.com/cuda/curand/device-api-overview.html

## 测试工具

- TestU01(SmallCrush/Crush/BigCrush)— 论文口径;"Crush-resistant" = 全过零 Failed
