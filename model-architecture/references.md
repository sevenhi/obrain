---
title: model-architecture 外部链接索引
type: reference
tags: [llm-architecture, llama, references]
date: 2026-09
---

# References — model-architecture

> 只放链接索引,不复制全文(库规约)。全部 URL 于 2026-09-21 验证存活(标注除外)。本地 PDF 原件在 `~/zhanbin/library/papers/`(只读)。

## 原论文(arXiv / 会议)

- LLaMA: Open and Efficient Foundation Language Models — https://arxiv.org/abs/2302.13971
- Llama 2: Open Foundation and Fine-Tuned Chat Models — https://arxiv.org/abs/2307.09288
- The Llama 3 Herd of Models — https://arxiv.org/abs/2407.21783 (HTML 全文: https://arxiv.org/html/2407.21783v3 )
- Language Models are Few-Shot Learners (GPT-3) — https://arxiv.org/abs/2005.14165
- Training Compute-Optimal Large Language Models (Chinchilla) — https://arxiv.org/abs/2203.15556
- Scaling Laws for Neural Language Models (Kaplan) — https://arxiv.org/abs/2001.08361

## 组件原始论文

- 归一化:On Layer Normalization in the Transformer Architecture (pre-LN 理论, Xiong et al., ICML 2020) — https://arxiv.org/abs/2002.04745 ;Root Mean Square Layer Normalization (Zhang & Sennrich, NeurIPS 2019) — https://arxiv.org/abs/1910.07467
- 激活:GLU Variants Improve Transformer (SwiGLU, Shazeer 2020) — https://arxiv.org/abs/2002.05202
- 位置编码:RoFormer (RoPE, Su et al. 2021) — https://arxiv.org/abs/2104.09864 ;Train Short, Test Long (ALiBi, Press et al., ICLR 2022) — https://arxiv.org/abs/2108.12409
- GQA: Grouped-Query Attention (Ainslie et al. 2023) — https://arxiv.org/abs/2305.13245

## 官方页 / 代码

- Meta AI 博客:Introducing LLaMA(2023-02-24)— https://ai.meta.com/blog/large-language-model-Llama-meta-ai/ (注意路径大小写敏感,`Llama` 大写;小写路径 curl 失败)
- Meta AI 博客:Introducing Meta Llama 3(2024-04)— https://ai.meta.com/blog/meta-llama-3/
- LLaMA 官方推理参考实现 — https://github.com/facebookresearch/llama
- Llama 3 MODEL_CARD — https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md

## attention 机制基础(跨域)

见 [attention/references.md](../attention/references.md)(AIAYN/StreamingLLM/FlashAttention 等)。
