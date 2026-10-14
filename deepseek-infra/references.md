---
title: deepseek-infra 外部链接索引
type: reference
tags: [deepseek, ascend, nvidia, cuda, tilelang, deepgemm, deepep, deepjit, tilekernels, flashmla, deepselect, dualpipe, eplb, lplb, esft, engram, 3fs, smallpond, npu, references]
date: 2026-10
---

# References — deepseek-infra

> 只放链接索引,不复制全文(库规约)。聚合自各篇 frontmatter 的 source;条目级溯源见各篇文末『溯源』。
> 本域事实断言(性能数字、支持矩阵、术语拼写)均核对一手源(各 GitHub README/raw、deep-dive 文档、PyPI、arXiv 论文页)。

## 昇腾六件套仓库

- **TileLang(上游 DSL,多后端:CUDA/ROCm/Metal)** — https://github.com/tile-ai/tilelang ;官方站 https://tilelang.com/
- **TileLang-Ascend(昇腾专用后端仓,双路线 Ascend C & PTO / AscendNPU IR)** — https://github.com/tile-ai/tilelang-ascend
- **DeepGEMM-Ascend(昇腾 GEMM 库)** — https://github.com/deepseek-ai/DeepGEMM-Ascend
- **DeepEP-Ascend(昇腾 MoE 通信库)** — https://github.com/deepseek-ai/DeepEP-Ascend
- **TileKernels(TileLang 算子库)** — https://github.com/deepseek-ai/TileKernels ;PyPI v2.0.0(昇腾版) — https://pypi.org/project/tile-kernels/2.0.0/
- **FlashMLA(注意力内核库,昇腾版 2026-09-30 已合入 main,commit 2e5429f)** — https://github.com/deepseek-ai/FlashMLA
- **DeepSelect(DSA/Sampler 的 TopK 算子)** — https://github.com/deepseek-ai/DeepSelect

## NVIDIA 原版对照 + 并行/工具/存储仓库(该批入库)

- **DeepGEMM(NVIDIA 原版)** — https://github.com/deepseek-ai/DeepGEMM
- **DeepEP(NVIDIA 原版)** — https://github.com/deepseek-ai/DeepEP
- **DeepJIT(运行时 JIT,CUDA + 昇腾双后端)** — https://github.com/deepseek-ai/DeepJIT
- **Engram(条件记忆/稀疏查找)** — https://github.com/deepseek-ai/Engram ;论文 arXiv:2601.07372 — https://arxiv.org/abs/2601.07372
- **DualPipe(双向流水线并行)** — https://github.com/deepseek-ai/DualPipe
- **LPLB(线性规划专家并行负载均衡)** — https://github.com/deepseek-ai/LPLB
- **EPLB(专家并行负载均衡)** — https://github.com/deepseek-ai/EPLB
- **ESFT(专家特化微调)** — https://github.com/deepseek-ai/ESFT ;论文 arXiv:2407.01906 — https://arxiv.org/abs/2407.01906
- **open-infra-index(开源基础设施总索引)** — https://github.com/deepseek-ai/open-infra-index
- **3FS(Fire-Flyer 分布式文件系统)** — https://github.com/deepseek-ai/3FS
- **smallpond(DuckDB + 3FS 数据处理)** — https://github.com/deepseek-ai/smallpond ;官方文档 https://deepseek-ai.github.io/smallpond/

## 论文(arXiv,curl 实测有效)

- DeepSeek-V3 Technical Report(DualPipe 出处)— https://arxiv.org/abs/2412.19437
- Engram《Conditional Memory via Scalable Lookup》— https://arxiv.org/abs/2601.07372 (ACL 2026 long)
- ESFT《Let the Expert Stick to His Last》— https://arxiv.org/abs/2407.01906 (EMNLP 2024)
- 《Insights into DeepSeek-V3》(ISCA25 Industry Track)— https://arxiv.org/abs/2505.09343
- 《Fire-Flyer AI-HPC》(SC24)— https://arxiv.org/abs/2408.14158

> ⚠ DeepEP **无独立 arXiv 论文**,公开引用是 GitHub 仓库(bibtex `deepep2025`);勘误说明。

## 技术 deep-dive 文档(一手,机制细节来源)

- FlashMLA 昇腾稀疏 prefill deep-dive — https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20260930-ascend-prefill-deep-dive.md (中文同目录 `-zh.md`)
- FlashMLA FP8 sparse decoding deep-dive(2025-09-29)—— https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md
- FlashMLA dense decoding deep-dive(2025-04-22)—— https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md
- DeepSelect 算法 deep-dive — https://github.com/deepseek-ai/DeepSelect/blob/main/docs/DeepSelect-deep-dive.md (中文 `DeepSelect-deep-dive.zh.md`)
- TileLang-Ascend Programming Guide — https://github.com/tile-ai/tilelang-ascend/blob/main/docs/TileLang-Ascend%20Programming%20Guide.md

## 官方/媒体报道(开源背景出处)

- 昇腾官方文档站 — https://www.hiascend.com/

## PyPI 版本页

- tile-kernels(含 v2.0.0 Ascend 支持说明)—— https://pypi.org/project/tile-kernels/
- tilelang(上游 DSL)—— https://pypi.org/project/tilelang/
- smallpond — https://pypi.org/project/smallpond/

### 训练并行/API 编解码(2026-10 补)

- DeepSeek-V3 仓库 — https://github.com/deepseek-ai/DeepSeek-V3
- deepseek-recipe — https://github.com/deepseek-ai/deepseek-recipe

(Technical Report 与 DualPipe 链接已在上方「论文」「NVIDIA 原版对照」两节,不重复列。)
