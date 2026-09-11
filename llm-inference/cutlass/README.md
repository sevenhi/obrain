---
title: CUTLASS
type: domain
tags: [cutlass, gpu, gemm, cute, cuda]
source:
  - "wiki: CUTLASS wiki | pageId 566290114 space ~zhanbin"
  - "code: github.com/NVIDIA/cutlass | 版本 4.8.0 (2026-08, 支持至 Rubin SM107)"
  - "web: NVIDIA CUTLASS 官方文档站 | https://docs.nvidia.com/cutlass/latest/"
  - "doc: docs/CUTLASS-完整技术文档.md (canonical)"
date: 2026-09
---

# CUTLASS

> 领域索引。完整内容见 canonical 文档 `docs/CUTLASS-完整技术文档.md`(376 行)。旧调研草稿已归档到 `archive/`。
> 本页只放定位、关键事实与导航,不重复正文。

## 一句话定位

CUTLASS(CUDA Templates for Linear Algebra Subroutines)是 NVIDIA 开源的 **header-only CUDA C++ 模板库 + Python DSL(CuTe DSL)**,既是 cuBLAS/cuDNN 等闭源库的**算法蓝本**,也是开发者构建自定义高性能 kernel 的**脚手架**。已被 PyTorch / TensorFlow / TensorRT-LLM / FlashAttention / FlashInfer 广泛集成。

## 关键事实(快照)

- 版本 4.8.0(2026-08);Stars ~10.3k / Forks ~2k;License BSD-3-Clause;2017-11-30 创建
- 定位对比:WMMA(最底层硬件接口)/ cuBLAS(黑盒成品)/ CUTLASS(开源模板 + 可定制)
- 与 cuBLAS 同量级性能,部分场景超越;Stream-K 在 32,824 种 GEMM 几何上峰值提升最高 14x(对比 CUTLASS)/ 6.7x(对比 cuBLAS)
- 架构支持:Volta → Turing → Ampere → Ada → Hopper → Blackwell

## 核心思想:分层分解 + 模块化组合

把「一次 GEMM」沿硬件并发层级**层层分块(tiling)**,每层对应一种并发粒度,各层「活动部件」做成可独立定制的 C++ 模板组件:

- **并发三层**:threadblocks / warps / CUDA·Tensor Cores
- **局部性两层**:shared memory / registers
- 每层 tile 尺寸、数据类型、算法策略均是模板参数,可独立调优

**四代演进**:1.x(2018,基本 GEMM 分解)→ 2.x(2019-2023,TF32/稀疏/Grouped/Hopper)→ 3.x(2023,CuTe 核心库架构级重构)→ 4.x(2026,CuTe DSL Python 接口,Blackwell 性能与手写 C++ 差距 <2%,public beta)

## 内容导航(canonical 章节)

1. 项目简介与定位 → 2. 整体架构与核心概念 → 3. CuTe 详解 → 4. GEMM 算法与策略 → 5. 卷积与扩展 → 6. 参考资源

## 相关资源

- `archive/CUTLASS-技术调研笔记.md` — 调研笔记(已归档)
- `archive/CUTLASS-资源清单.md` — 官方/论文/社区博客资源索引(2026-08-28,已归档)
- CUTLASS 官方文档站:https://docs.nvidia.com/cutlass/latest/
- GitHub:https://github.com/NVIDIA/cutlass

## 更新与同步

- 更新检查:`cutlass-tracker/check_updates.sh`
- Wiki 同步:`cutlass-tracker/sync_wiki.py`(pageId 566290114)
- 内容源:`cutlass-tracker/wiki_v9_content.md` / `wiki_v9_clean.md`(可读版,含 floatpilot 宏)
- 周报:每日产出在 `cutlass-tracker/daily_reports/`
