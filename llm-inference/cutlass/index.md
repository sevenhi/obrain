---
title: CUTLASS
type: domain
tags: [cutlass, gpu, gemm, cute, cuda]
source:
  - "wiki: CUTLASS wiki | pageId 566290114 space ~zhanbin"
  - "code: github.com/NVIDIA/cutlass | 版本 4.8.0 (2026-08, 支持至 Rubin SM107)"
  - "web: NVIDIA CUTLASS 官方文档站 | https://docs.nvidia.com/cutlass/latest/"
  - "doc: archive/CUTLASS-完整技术文档.md (canonical, 库内二次整理的完整文档, 上游为 NVIDIA CUTLASS 官方站)"
date: 2026-09
---

# CUTLASS

> 本页是 CUTLASS 领域的**导航起始页**(类比网站首页)。往下走:定位与关键事实 → 分领域内容索引 → 更新维护。
> **真正的可读正文在 `concepts/` / `frameworks/` 子目录**,每篇一个专题、可独立打开读懂。本页只做导航,不重复正文。

---

## 一句话定位

**CUTLASS**(CUDA Templates for Linear Algebra Subroutines)是 NVIDIA 开源的 **header-only CUDA C++ 模板库 + Python DSL(CuTe DSL)**,既是 cuBLAS/cuDNN 等闭源库的**算法蓝本**,也是开发者构建自定义高性能 kernel 的**脚手架**。已被 PyTorch / TensorFlow / TensorRT-LLM / FlashAttention / FlashInfer 广泛集成。

**核心思想一句话**:把「一次矩阵乘」沿硬件并发层级**层层分块(tiling)**,每层做成可独立定制的模板组件,靠 shared memory + registers 两层局部性喂饱 Tensor Core。

---

## 关键事实(快照,2026-09)

| 项 | 值 |
|---|---|
| 版本 | 4.8.0(2026-08,支持至 Rubin SM107)|
| Stars / Forks / License | ~10.3k / ~2k / BSD-3-Clause(2017-11-30 创建)|
| 定位对比 | **WMMA**(最底层硬件接口)/ **cuBLAS**(黑盒成品)/ **CUTLASS**(开源模板+可定制)|
| 性能 | 与 cuBLAS 同量级,部分超越;Stream-K 峰值最高 **14×**(vs CUTLASS)/ **6.7×**(vs cuBLAS)|
| 架构 | Volta → Turing → Ampere → Ada → Hopper → Blackwell |

**四代演进**:1.x(2018,基本 GEMM 分解)→ 2.x(2019-2023,TF32/稀疏/Grouped/Hopper)→ 3.x(2023,CuTe 核心库架构级重构)→ 4.x(2026,CuTe DSL Python 接口,Blackwell 与手写 C++ 差距 <2%)。

---

## 分领域内容索引

> 每个专题一篇,按「先原理、后工具」排。带 🔒 的是规划中未写,带 ✅ 是已入库。

### 核心概念(懂)
| 专题 | 文件 | 一句话 | 状态 |
|---|---|---|---|
| 分层 GEMM:tiling 分解 | [concepts/gemm-tiling.md](concepts/gemm-tiling.md) | 硬件三层并发 → 三层 tiling → mainloop/epilogue → pipelining → Stream-K → CuTe | ✅ 已审批入库 |
| Epilogue 收尾融合 | [concepts/epilogue-fusion.md](concepts/epilogue-fusion.md) | 为什么 epilogue 是性能关键、融合模式(bias/act/scale/转置)、与主循环分界 | ✅ 已写 |
| TileScheduler 与负载均衡 | [concepts/tile-scheduler.md](concepts/tile-scheduler.md) | tile-based / Stream-K / Grouped 三种调度,SM 占用率,tail effect,fixup | ✅ 已写 |
| CuTe Layout 代数 | [concepts/cute-layout-algebra.md](concepts/cute-layout-algebra.md) | Layout 抽象、代数运算(divide/product/coalesce)、Atom/TiledMMA、TV layout | ✅ 已写 |

### 工具与上手(用)
| 专题 | 文件 | 一句话 | 状态 |
|---|---|---|---|
| 构建与初上手 | [frameworks/cutlass-build-and-usage.md](frameworks/cutlass-build-and-usage.md) | 构建/编译优化、CollectiveBuilder 范式、CuTe DSL Python、examples 导航 | ✅ 已写 |
| 性能调优 | [frameworks/cutlass-profiling-tuning.md](frameworks/cutlass-profiling-tuning.md) | Profiler、roofline 判受限、关键调优维度、Hopper/Blackwell 专属 | ✅ 已写 |

---

## 内容雷区(读/写 CUTLASS 时的坑速记)

- **wgmma 是 Hopper 的计算指令**,不是搬运;producer 搬数据用 TMA `cp.async.bulk.tensor`
- **`cp.async` 自 Ampere 引入**,Ada 属 Ampere 阵营;无异步指令的一代是 Volta/Turing
- **CuTe 运算名**:是 `logical_divide`/`zipped_divide`/`tiled_divide`/`coalesce`/`composition`/`complement`,不是 `left_padding`/`partitioned_into_TV`
- **Stream-K 有 fixup 代价**:跨 CTA 归约 + workspace;默认 `ReductionMode::Deterministic` 已是确定性归约(非"天然不可复现")

> 以上为三 bot 审批(2026-09)纠错沉淀,详见各篇正文。

---

## 相关资源

- canonical 完整文档:`archive/CUTLASS-完整技术文档.md`(400 行,二次整理)
- `archive/CUTLASS-技术调研笔记.md` / `CUTLASS-资源清单.md`(已归档)
- 官方文档站:https://docs.nvidia.com/cutlass/latest/
- GitHub:https://github.com/NVIDIA/cutlass

## 更新与同步

- 更新检查:`tracker/cutlass-tracker/check_updates.sh`
- Wiki 同步:`tracker/cutlass-tracker/sync_wiki.py`(pageId 566290114)
- 内容源:`tracker/cutlass-tracker/wiki_v9_content.md` / `wiki_v9_clean.md`
