---
title: model-architecture 外部链接索引
type: reference
tags: [llm-architecture, llama, gqa, mqa, mla, context-extension, yarn, longrope, moe, mixtral, deepseek, qwen, rwkv, references]
date: 2026-09
---

# References — model-architecture

> 只放链接索引,不复制全文(库规约)。全部 URL 于 2026-09-21 curl 实测存活(标注除外)。本地 PDF 原件在 `~/zhanbin/library/papers/`(只读)。mistral.ai 官方博客在本网络环境连接超时(非 404),故 Mistral 系以 arXiv 论文为准。

## 原论文(arXiv / 会议)

- LLaMA: Open and Efficient Foundation Language Models — https://arxiv.org/abs/2302.13971
- Llama 2: Open Foundation and Fine-Tuned Chat Models — https://arxiv.org/abs/2307.09288
- The Llama 3 Herd of Models — https://arxiv.org/abs/2407.21783 (HTML 全文: https://arxiv.org/html/2407.21783v3 )
- Language Models are Few-Shot Learners (GPT-3) — https://arxiv.org/abs/2005.14165
- Attention Is All You Need (原始 Transformer) — https://arxiv.org/abs/1706.03762
- Training Compute-Optimal Large Language Models (Chinchilla) — https://arxiv.org/abs/2203.15556
- Scaling Laws for Neural Language Models (Kaplan) — https://arxiv.org/abs/2001.08361

## 注意力头结构(MHA/MQA/GQA/MLA)

- Fast Transformer Decoding: One Write-Head is All You Need (MQA, Shazeer 2019) — https://arxiv.org/abs/1911.02150
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al. 2023) — https://arxiv.org/abs/2305.13245
- DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model (MLA 首篇) — https://arxiv.org/abs/2405.04434
- DeepSeek-V3 Technical Report (MLA + MoE 当前口径) — https://arxiv.org/abs/2412.19437

## 上下文扩展(RoPE 外推)

- RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE 原始) — https://arxiv.org/abs/2104.09864
- Extending Context Window of LLMs via Position Interpolation (PI, Chen et al., Meta) — https://arxiv.org/abs/2306.15595
- YaRN: Efficient Context Window Extension of Large Language Models (Peng et al., Nous Research) — https://arxiv.org/abs/2309.00071
- LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens (Ding et al., MSR) — https://arxiv.org/abs/2402.13753
- Train Short, Test Long: Attention with Linear Biases (ALiBi, Press et al., ICLR 2022) — https://arxiv.org/abs/2108.12409

## 开源模型谱系与 MoE

- Mistral 7B — https://arxiv.org/abs/2310.06825
- Mixtral of Experts — https://arxiv.org/abs/2401.04088
- DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models — https://arxiv.org/abs/2401.06066
- Qwen Technical Report (初代) — https://arxiv.org/abs/2309.16609
- Qwen2 Technical Report — https://arxiv.org/abs/2407.10671
- Qwen2.5 Technical Report — https://arxiv.org/abs/2412.15115
- RWKV: Reinventing RNNs for the Transformer Era — https://arxiv.org/abs/2305.13048

## 组件原始论文

- 归一化:On Layer Normalization in the Transformer Architecture (pre-LN 理论, Xiong et al., ICML 2020) — https://arxiv.org/abs/2002.04745 ;Root Mean Square Layer Normalization (Zhang & Sennrich, NeurIPS 2019) — https://arxiv.org/abs/1910.07467
- 激活:GLU Variants Improve Transformer (SwiGLU, Shazeer 2020) — https://arxiv.org/abs/2002.05202

## 官方页 / 代码 / 模型卡

- Meta AI 博客:Introducing LLaMA(2023-02-24)— https://ai.meta.com/blog/large-language-model-Llama-meta-ai/ (注意路径大小写敏感,`Llama` 大写;小写路径 curl 失败)
- Meta AI 博客:Introducing Meta Llama 3(2024-04)— https://ai.meta.com/blog/meta-llama-3/ (2026-09-21 本机 curl 超时,未复核;引用时以 arXiv:2407.21783 为准)
- LLaMA 官方推理参考实现 — https://github.com/facebookresearch/llama
- Llama 模型仓库(llama-models,含 MODEL_CARD/许可) — https://github.com/meta-llama/llama-models
- Qwen 官方博客:Qwen2.5 发布 — https://qwenlm.github.io/blog/qwen2.5/
- Mistral 官方参考实现 — https://github.com/mistralai/mistral-src
- YaRN 官方实现与 checkpoint — https://github.com/jquesnelle/yarn
- LongRoPE 官方实现 — https://github.com/microsoft/LongRoPE
- DeepSeek-V3 官方仓库(MIT LICENSE) — https://github.com/deepseek-ai/DeepSeek-V3
- DeepSeek-V2 官方仓库 — https://github.com/deepseek-ai/DeepSeek-V2
- RWKV 官方仓库 — https://github.com/BlinkDL/RWKV-LM
- HF 模型页(config.json 口径核对用):
  - Mistral-7B-v0.1 — https://huggingface.co/mistralai/Mistral-7B-v0.1
  - Mixtral-8x7B-v0.1 — https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
  - Meta-Llama-3.1-8B — https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
  - Qwen2.5-72B-Instruct — https://huggingface.co/Qwen/Qwen2.5-72B-Instruct
  - DeepSeek-V3 — https://huggingface.co/deepseek-ai/DeepSeek-V3
  - Yarn-Llama-2-7b-128k(Nous,社区 YaRN 扩展例) — https://huggingface.co/NousResearch/Yarn-Llama-2-7b-128k
  - LLaMA-2-7B-32K(Together,社区 PI 扩展例,YaRN 论文 Table 1 对照模型) — https://huggingface.co/togethercomputer/LLaMA-2-7B-32K

## attention 机制基础(跨域)

见 [attention/references.md](../attention/references.md)(AIAYN/StreamingLLM/FlashAttention 等)。

## 推理引擎(跨域)

见 vllm 域:[paged-attention](../vllm/concepts/paged-attention.md)(KV cache 分页)、[continuous-batching](../vllm/concepts/continuous-batching.md)、[quantization-gptq-awq-fp8](../vllm/concepts/quantization-gptq-awq-fp8.md)(KV/权重量化)。
