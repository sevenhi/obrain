---
title: attention-and-transformers 外部链接索引
type: reference
tags: [transformer, attention, references]
date: 2026-09
---

# References — attention-and-transformers

> 只放链接索引,不复制全文(库规约)。全部 URL 于 2026-09-21 curl 实测存活(标注除外)。本地 PDF 原件在 `~/zhanbin/library/papers/`(只读)。

## 原论文(arXiv / 会议)

- Attention Is All You Need — https://arxiv.org/abs/1706.03762 (NIPS 2017)
- Efficient Streaming Language Models with Attention Sinks — https://arxiv.org/abs/2309.17453 (ICLR 2024)
- LASER: Attention with Exponential Transformation — https://arxiv.org/abs/2411.03493 (ICML 2025, PMLR 267:14902-14917)
- SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning — https://arxiv.org/abs/2012.09852 (HPCA 2021;IEEE Xplore doc 9407256 对 curl 返回 202 反爬,以 arXiv 为准)
- MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers — https://arxiv.org/abs/2305.07185 (NeurIPS 2023)
- LLaMA: Open and Efficient Foundation Language Models — https://arxiv.org/abs/2302.13971
- 组件原始论文:RoPE/RoFormer — https://arxiv.org/abs/2104.09864;RMSNorm — https://arxiv.org/abs/1910.07467;SwiGLU(GLU Variants)— https://arxiv.org/abs/2002.05202;FlashAttention — https://arxiv.org/abs/2205.14135

## 官方页 / 会议版

- LASER PMLR 官方页 — https://proceedings.mlr.press/v267/duvvuri25a.html
- MEGABYTE NeurIPS 2023 官方 PDF — https://proceedings.neurips.cc/paper_files/paper/2023/file/f8f78f8043f35890181a824e53a57134-Paper-Conference.pdf
- MEGABYTE OpenReview — https://openreview.net/forum?id=JTmO2V9Xpz
- StreamingLLM ICLR 2024 OpenReview — https://openreview.net/forum?id=NG7sS51zVF
- StreamingLLM ICLR proceedings — https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html
- Meta AI 博客:Introducing LLaMA(2023-02-24)— https://ai.meta.com/blog/large-language-model-Llama-meta-ai/ (注意路径大小写敏感,`Llama` 大写;小写路径 curl 失败)

## 项目页 / 代码

- StreamingLLM 项目页(MIT Han Lab)— https://hanlab.mit.edu/projects/streamingllm
- StreamingLLM 代码 — https://github.com/mit-han-lab/streaming-llm
- SpAtten 代码(SpinalHDL RTL + PyTorch 算法)— https://github.com/mit-han-lab/spatten
- LLaMA 官方推理参考实现 — https://github.com/facebookresearch/llama
- MEGABYTE 官方仓库已失效(github.com/facebookresearch/megabyte → 404,2026-09-21 实测);第三方参考实现 — https://github.com/lucidrains/MEGABYTE-pytorch
- Transformer 官方原始代码(tensor2tensor,已归档,后继 Trax)— https://github.com/tensorflow/tensor2tensor

## 教程 / 解读

- The Illustrated Transformer(Jay Alammar,图解)— https://jalammar.github.io/illustrated-transformer/
- The Annotated Transformer(Harvard NLP,PyTorch 逐段复现)— https://nlp.seas.harvard.edu/annotated-transformer/

## attention sink 后续研究(延伸,均 curl 200)

- When Attention Sink Emerges in Language Models(ICLR 2025)— https://openreview.net/forum?id=78Nn4QJTEN
- What are you sinking? A geometric approach on attention sink — https://openreview.net/forum?id=OiC78C68sJ
