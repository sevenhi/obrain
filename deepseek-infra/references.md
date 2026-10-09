---
title: deepseek-infra 外部链接索引
type: reference
tags: [deepseek, ascend, tilelang, deepgemm, deepep, tilekernels, flashmla, deepselect, npu, references]
date: 2026-09
---

# References — deepseek-infra

> 只放链接索引,不复制全文(库规约)。聚合自各篇 frontmatter 的 source;条目级溯源见各篇文末『溯源』。
> 本域事实断言(性能数字、支持矩阵、术语拼写)均于 2026-09-30 开源当周核对一手源(各 GitHub README/raw、deep-dive 文档、PyPI)。

## 六件套仓库

- **TileLang(上游 DSL,多后端:CUDA/ROCm/Metal)** — https://github.com/tile-ai/tilelang ;官方站 https://tilelang.com/
- **TileLang-Ascend(昇腾专用后端仓,双路线 Ascend C & PTO / AscendNPU IR)** — https://github.com/tile-ai/tilelang-ascend
- **DeepGEMM-Ascend(昇腾 GEMM 库)** — https://github.com/deepseek-ai/DeepGEMM-Ascend
- **DeepEP-Ascend(昇腾 MoE 通信库)** — https://github.com/deepseek-ai/DeepEP-Ascend
- **TileKernels(TileLang 算子库)** — https://github.com/deepseek-ai/TileKernels ;PyPI v2.0.0(昇腾版) — https://pypi.org/project/tile-kernels/2.0.0/
- **FlashMLA(注意力内核库,昇腾版已合入 main,原开发分支 20260930-ascend-open-source)** — https://github.com/deepseek-ai/FlashMLA
- **DeepSelect(DSA/Sampler 的 TopK 算子)** — https://github.com/deepseek-ai/DeepSelect
- **DeepJIT(运行时 JIT,DeepEP-Ascend 依赖)** — https://github.com/deepseek-ai/DeepJIT

## 上游 NVIDIA 版(API 对齐基准)

- DeepGEMM(原版)—— https://github.com/deepseek-ai/DeepGEMM
- DeepEP(原版)—— https://github.com/deepseek-ai/DeepEP

## 技术 deep-dive 文档(一手,机制细节来源)

- FlashMLA 昇腾稀疏 prefill deep-dive — https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20260930-ascend-prefill-deep-dive.md (中文同目录 `-zh.md`)
- FlashMLA FP8 sparse decoding deep-dive(2025-09-29)—— https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md
- FlashMLA dense decoding deep-dive(2025-04-22)—— https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md
- DeepSelect 算法 deep-dive — https://github.com/deepseek-ai/DeepSelect/blob/main/docs/DeepSelect-deep-dive.md (中文 `DeepSelect-deep-dive.zh.md`)
- TileLang-Ascend Programming Guide — https://github.com/tile-ai/tilelang-ascend/blob/main/docs/TileLang-Ascend%20Programming%20Guide.md

## 官方/媒体报道(开源背景出处)

- 昇腾官方文档站 — https://www.hiascend.com/
- 华为 CANN 社区(昇腾开源成果:SuperPoD Flex / UBL128 超节点)联合公告
- DeepSeek 官方开源公告(凤凰网 / 网易 / 七牛云等转载报道)

## PyPI 版本页

- tile-kernels(含 v2.0.0 Ascend 支持说明)—— https://pypi.org/project/tile-kernels/
- tilelang(上游 DSL)—— https://pypi.org/project/tilelang/
