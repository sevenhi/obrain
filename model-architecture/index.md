---
title: Model Architecture(通用 LLM 架构)
type: domain
tags: [llm-architecture, llama, rmsnorm, swiglu, rope, alibi, gqa, chinchilla, scaling-laws, gpt]
source:
  - "pdf: LLaMA | arXiv:2302.13971 | ~/zhanbin/library/papers/LLaMA: Open and Efficient Foundation Language Models.pdf"
  - "pdf: Llama 2 | arXiv:2307.09288"
  - "pdf: The Llama 3 Herd of Models | arXiv:2407.21783"
  - "pdf: Chinchilla (Training Compute-Optimal LLMs) | arXiv:2203.15556"
date: 2026-09
---

# Model Architecture(通用 LLM 架构)

> 领域索引(导航起始页)。领域知识本体在 `concepts/` 子目录,每篇一个专题、可独立打开读懂;本页放定位与导航。外部链接索引见 [references.md](references.md)。
> 本域 2026-09 新建:原 attention 域的 `llama-architecture.md` 经 git mv 移入并扩展为**通用 LLM 架构篇**(保留 LLaMA 全部内容,补齐组件演进谱系与 Llama 2/3)。

## 一句话定位

回答"一个现代 decoder-only LLM 长什么样、为什么长这样":以 LLaMA 三件套(pre-RMSNorm + SwiGLU + RoPE)为定型锚点,覆盖归一化/激活/位置编码三条组件演进线、Kaplan→Chinchilla→LLaMA 训练哲学(scale 什么、喂多少数据),以及 GPT→LLaMA→Llama 2/3 的演进对照(GQA、128K vocab、15T tokens、128K context)。

## 条目

| 条目 | 类型 | 一句话 | 状态 |
|---|---|---|---|
| [llm-architecture](concepts/llm-architecture.md) | concept | 通用 LLM 架构:post/pre-norm→RMSNorm、ReLU→GeLU→SwiGLU((2/3)4d)、正弦→learned→RoPE→ALiBi 三条演进线 + Chinchilla/推理预算训练哲学 + GPT→LLaMA→Llama 2/3 演进表(GQA/θ=500k/128K vocab/15T tokens) | ✅ 已入库 |

## 关键事实速记

- **定型配方**:pre-RMSNorm + SwiGLU + RoPE 是 2023 年后开源 LLM 事实标准(LLaMA/Qwen/Mistral 全沿用);
- **RMSNorm** = LayerNorm 去掉减均值,只按均方根缩放——少一次统计量、更快,质量无损;pre-norm 让残差主干恒等直通、深层稳训(可不用 warmup);
- **SwiGLU** 三个权重矩阵 → 内维取 (2/3)·4d 对齐参数量;更优是经验性的(T5 规模系统评测),无干净理论;
- **RoPE**:旋转矩阵把绝对位置编进 Q/K,内积只依赖相对距离;改 base θ 即可扩上下文(Llama 3: 10k→500k);ALiBi 是另一条零参数外推路线(BLOOM/MPT);
- **训练哲学**:Chinchilla 算力最优 ~20 tokens/参数;LLaMA 把推理预算写进目标函数 → 7B×1T(143 tokens/参数);13B 多数基准超 GPT-3(175B);
- **Llama 2→3**:2T→15T tokens、4k→128K context、70B GQA→全系 8 KV 头、32k SentencePiece→128K tiktoken vocab(压缩率 3.17→3.94)——架构不动,收益来自数据。

## 相关条目(跨域互链)

- **[`attention/`](../attention/index.md)**:本域架构篇的 attention 机制基础(softmax 数学/O(n²) 账单/多头等)在 attention 域 algorithm 层;高效内核(FA 四代)在 [attention/implementation/flash-attention-implementation.md](../attention/implementation/flash-attention-implementation.md)——LLaMA 训练用的 xformers attention = Rabe & Staats 前向 + FA1 反向;GQA 缩小的 KV cache 由 [`vllm/`](../vllm/index.md) PagedAttention 管理。
- **[`quantization/`](../quantization/index.md)**:LLaMA 权重是 GPTQ/AWQ/SpQR 等量化的最常用测试床。

## 相关论文(公开可溯源)

链接索引见 [references.md](references.md)。
