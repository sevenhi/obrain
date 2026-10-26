---
title: attention 外部链接索引
type: reference
tags: [transformer, attention, references]
date: 2026-09
---

# References — attention

> 只放链接索引,不复制全文(库规约)。全部 URL 于 2026-09-21 curl 实测存活(标注除外)。原始论文链接见本页 references，不依赖本地文件。

## 原论文(arXiv / 会议)

- Attention Is All You Need — https://arxiv.org/abs/1706.03762 (NIPS 2017)
- Efficient Streaming Language Models with Attention Sinks — https://arxiv.org/abs/2309.17453 (ICLR 2024)
- LASER: Attention with Exponential Transformation — https://arxiv.org/abs/2411.03493 (ICML 2025, PMLR 267:14902-14917)
- SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning — https://arxiv.org/abs/2012.09852 (HPCA 2021;IEEE Xplore doc 9407256 对 curl 返回 202 反爬,以 arXiv 为准)
- MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers — https://arxiv.org/abs/2305.07185 (NeurIPS 2023)
- FlashAttention FA1 — https://arxiv.org/abs/2205.14135;FA2 — https://arxiv.org/abs/2307.08691;FA3 — https://arxiv.org/abs/2407.08608;FA4 (MLSys 2026) — arXiv:2603.05451(四代全部链接/博客口径见 [implementation/flash-attention-implementation.md](implementation/flash-attention-implementation.md) §7 溯源)
- Self-Attention does not Need O(n²) Memory (Rabe & Staats, FA1 前驱) — https://arxiv.org/abs/2112.05682
- Online normalizer calculation for softmax (Milakov & Gimelshein, online softmax 最早出处) — https://arxiv.org/abs/1805.02867
- LLaMA/RoPE/RMSNorm/SwiGLU 等模型架构组件论文已移至 [model-architecture/references.md](../model-architecture/references.md)

## 官方页 / 会议版

- LASER PMLR 官方页 — https://proceedings.mlr.press/v267/duvvuri25a.html
- MEGABYTE NeurIPS 2023 官方 PDF — https://proceedings.neurips.cc/paper_files/paper/2023/file/f8f78f8043f35890181a824e53a57134-Paper-Conference.pdf
- MEGABYTE OpenReview — https://openreview.net/forum?id=JTmO2V9Xpz
- StreamingLLM ICLR 2024 OpenReview — https://openreview.net/forum?id=NG7sS51zVF
- StreamingLLM ICLR proceedings — https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html

## 项目页 / 代码

- StreamingLLM 项目页(MIT Han Lab)— https://hanlab.mit.edu/projects/streamingllm
- StreamingLLM 代码 — https://github.com/mit-han-lab/streaming-llm
- SpAtten 代码(SpinalHDL RTL + PyTorch 算法)— https://github.com/mit-han-lab/spatten
- Rabe & Staats 官方 JAX 实现(FA1 前驱,memory_efficient_attention 目录,含 colab 复现)— https://github.com/google-research/google-research/tree/master/memory_efficient_attention
- Rabe & Staats PyTorch 复现(lucidrains)— https://github.com/lucidrains/memory-efficient-attention-pytorch
- FlashAttention 官方实现仓库(含 FA1/FA2 kernel 与 `benchmarks/`)— https://github.com/Dao-AILab/flash-attention
- MEGABYTE 官方仓库已失效(github.com/facebookresearch/megabyte → 404,2026-09-21 实测);第三方参考实现 — https://github.com/lucidrains/MEGABYTE-pytorch
- Transformer 官方原始代码(tensor2tensor,已归档,后继 Trax)— https://github.com/tensorflow/tensor2tensor

## 教程 / 解读

- The Illustrated Transformer(Jay Alammar,图解)— https://jalammar.github.io/illustrated-transformer/
- The Annotated Transformer(Harvard NLP,PyTorch 逐段复现)— https://nlp.seas.harvard.edu/annotated-transformer/

## FlexAttention(框架层,2026-10 收录,URL 于 2026-10-14 实测存活)

- FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention(官方博客,2024-08-07 发布 / 2025-05-30 更新)— https://pytorch.org/blog/flexattention/
- FlexAttention + FlashAttention-4: Fast and Flexible(官方博客,2026-03,FLASH 后端)— https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
- Flex Attention: A Programming Model for Generating Optimized Attention Kernels(论文)— https://arxiv.org/abs/2412.05496
- Colfax Research — A User's Guide to FlexAttention in FlashAttention CuTe DSL(FA4 CuTeDSL 侧一手指南)— https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/
- PyTorch 官方文档 torch.nn.attention.flex_attention(API 签名)— https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html
- PyTorch 源码:torch/nn/attention/flex_attention.py — https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py;Inductor Flex 内核目录(flex_attention.py / flex_flash_attention.py / flex_decoding.py)— https://github.com/pytorch/pytorch/tree/main/torch/_inductor/kernel/flex
- attention-gym(官方示例库:FLASH vs TRITON 对比、paged/ring attention、mask/mod 库)— https://github.com/meta-pytorch/attention-gym
- FA4 CuTeDSL 实现(flash_attn/cute/,含 FlexAttention 扩展点)— https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute

## FA CuTe 后端的 ragged/paged 加载与 TMA descriptor 治理(2026-10)

- 用 per-CTA GMEM descriptor 约束 ragged K/V TMA 加载(替代 wraparound descriptor,内含预取足迹与 128 KB 越界说明)— https://github.com/Dao-AILab/flash-attention/pull/3071
- SM100 反向按 seqlen_k 约束 K/V 加载,避免未用行 NaN 污染 dQ — https://github.com/Dao-AILab/flash-attention/pull/3012
- SM90 前向/反向按 seqlen_k 约束 TMA 加载(对齐 SM100)— https://github.com/Dao-AILab/flash-attention/pull/3013
- SM100 前向:未用 V cache 行的 NaN 污染防治 + 抽出 reusable ragged TMA helper — https://github.com/Dao-AILab/flash-attention/pull/2983
- typed forward config(`FwdConfig`/`FwdHeuristicInputs`,host 侧纯计算的选择器契约)— https://github.com/Dao-AILab/flash-attention/pull/2732
- hd256 反向统一到通用 SM100 内核(opt-in 灰度)— https://github.com/Dao-AILab/flash-attention/pull/2966
- CUTLASS v4.8.1 release notes(`tensormap_replace` 缩小 descriptor 的越界修复 + `keep_oob_fetch_window` 退回开关;CUDA 已知问题页链接)— https://github.com/NVIDIA/cutlass/releases/tag/v4.8.1
- CUTLASS CuTeDSL `TensorMapManager` 源码(GMEM/SMEM 两种 update 模式与 fence 语义)— https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/tensor_utils/tensormap.py
- 反向 pack_gqa:把 GQA 的 q-head 折进 Q tile(免 fp32 dK/dV 归约与 postprocess;含 `_bwd_pack_gqa_heuristic` 两道闸)— https://github.com/Dao-AILab/flash-attention/pull/2967

## attention sink 后续研究(延伸,均 curl 200)

- When Attention Sink Emerges in Language Models(ICLR 2025)— https://openreview.net/forum?id=78Nn4QJTEN
- What are you sinking? A geometric approach on attention sink — https://openreview.net/forum?id=OiC78C68sJ
