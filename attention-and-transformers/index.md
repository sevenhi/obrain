---
title: Attention 与 Transformer 架构
type: domain
tags: [transformer, attention, self-attention, streamingllm, megabyte, llama, laser, spatten]
source:
  - "pdf: Attention Is All You Need | arXiv:1706.03762 NIPS 2017 | ~/zhanbin/library/papers/Attention is All You Need.pdf"
  - "pdf: Efficient Streaming Language Models with Attention Sinks | arXiv:2309.17453 ICLR 2024 | ~/zhanbin/library/papers/Efficient Streaming Language Models With Attention Sinks.pdf"
  - "pdf: LASER: Attention with Exponential Transformation | arXiv:2411.03493 ICML 2025 | ~/zhanbin/library/papers/LASER: Attention with Exponential Transformation.pdf"
  - "pdf: SpAtten | arXiv:2012.09852 HPCA 2021 | ~/zhanbin/library/papers/SpAtten-- Efficient Sparse Attention Architecture with Cascade Token and Head Pruning.pdf"
  - "pdf: MEGABYTE | arXiv:2305.07185 NeurIPS 2023 | ~/zhanbin/library/papers/MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.pdf"
  - "pdf: LLaMA | arXiv:2302.13971 | ~/zhanbin/library/papers/LLaMA: Open and Efficient Foundation Language Models.pdf"
date: 2026-09
---

# Attention 与 Transformer 架构

> 领域索引(导航起始页)。**领域知识本体在 `concepts/` 子目录,每篇一个专题、可独立打开读懂**;本页放定位、关键事实与导航,不重复正文。外部链接索引见 [references.md](references.md)。

## 一句话定位

以 2017 年 Transformer 原论文为根,收录**注意力机制本体与架构级演化**:原始设计(QKV/多头/位置编码)、softmax 和为 1 约束的三个推论(sink→StreamingLLM、梯度饱和→LASER、长尾稀疏→SpAtten)、长序列的架构分解(MEGABYTE)、以及把三件套改进定型为开源标准的 LLaMA。

## 本库沉淀条目(concepts,5 篇)

| 条目 | 类型 | 一句话 | 状态 |
|---|---|---|---|
| [attention-is-all-you-need](concepts/attention-is-all-you-need.md) | concept | Transformer 原始设计:√d_k 缩放的推导、多头为何免费、O(1) 路径 vs O(n²) 账单、base/big 训练配方与 WMT14 数据 | ✅ 已入库 |
| [attention-sinks-streaming](concepts/attention-sinks-streaming.md) | concept | attention sink 因果链(softmax 和为 1 → 垃圾注意力倒给开头 token)+ StreamingLLM sink+滚窗:4M token 稳定 PPL、22.2× 加速、learnable sink token | ✅ 已入库 |
| [attention-variants-laser-spatten](concepts/attention-variants-laser-spatten.md) | concept | 双线索:LASER 对 exp(V) 做 attention 修梯度饱和(训练侧,−9.4% 时间/−15.65% 参数)× SpAtten 级联 token/head 剪枝 + progressive quant(硬件侧,10× DRAM/162× GPU 加速) | ✅ 已入库 |
| [megabyte-multiscale](concepts/megabyte-multiscale.md) | concept | 多尺度分解:Global(patch 间)+ Local(patch 内字节),attention O(T^4/3)、FFN per-patch → 同算力 P 倍大模型;tokenization-free 追平 subword | ✅ 已入库 |
| [llama-architecture](concepts/llama-architecture.md) | concept | RMSNorm/SwiGLU((2/3)4d)/RoPE 三件套 + 推理预算导向的"小模型 1T+ tokens"哲学;13B>GPT-3、65B≈Chinchilla-70B | ✅ 已入库 |

## 领域地图(一条主线读)

```
Attention Is All You Need (2017)  ← 根:QKV/多头/√d_k/位置编码/O(n²) 账单
        │
        ├── softmax 和为 1 的三个推论
        │     ├─ KV cache 侧: attention sink → StreamingLLM (ICLR 2024)
        │     ├─ 训练侧: 概率长尾 → 梯度饱和 → LASER (ICML 2025)
        │     └─ 推理硬件侧: 概率尖/稀疏 → SpAtten (HPCA 2021)
        │
        ├── 长序列架构分解: MEGABYTE (NeurIPS 2023)  ← FFN 才是大头
        │
        └── 架构改进定型: LLaMA (2023)  ← RMSNorm/SwiGLU/RoPE + 多数据小模型
```

## 相关条目(跨域互链)

- **[`vllm/`](../vllm/index.md)**:StreamingLLM(逐出谁)与 PagedAttention/KV cache 分页(装得下就全留,见 [`vllm/concepts/paged-attention.md`](../vllm/concepts/paged-attention.md))正交互补——长上下文/常驻流式服务的两个半边;vLLM 未内置 sink 逐出,TensorRT-LLM/SGLang/HF 已内置。
- **[`flash-attention/`](../flash-attention/index.md)**:O(n²) 账单的**内核路线**精确解(IO-aware tiling,不改数学);与本域 MEGABYTE(架构分解)、SpAtten(专用硬件稀疏)构成同一 profiling 结论("attention 是访存受限")的三条路线。LLaMA 训练用的 xformers 高效 attention = Rabe & Staats 前向 + FA1 反向。
- **[`quantization/`](../quantization/index.md)**:SpAtten progressive quantization(attention 输入 MSB/LSB)与 LLM 权重量化(SpQR/GPTQ/AWQ)是量化谱系的两侧;attention sink 与量化 outlier 同源(softmax 和为 1 制造极端 token)。
- **[`cutlass/`](../cutlass/index.md)**:FA 内核的底座;本域 SpAtten 代表"绕开通用 GEMM 库造专用 attention 硬件"的对照路线。

## 关键事实速记

- 原论文 Table 1:self-attention 每层 O(n²·d)、串行 O(1)、路径 O(1);n<d 时反而比 RNN 便宜——quadratic 当年是卖点;
- StreamingLLM:4 个 sink token 饱和(1~2 不够),PPL 5158→5.40,4M tokens 稳定;
- LASER:80% 注意力概率 <10⁻³、20% <10⁻⁷(Jacobian ∝ a 的直接后果);Log-Weighted-Sum-Exp trick 后 FlashAttention 零修改可用;
- SpAtten:attention 占 GPT-2 延迟 >50% / FLOPs ~10%,73% 时间在搬数据;仅 5.9% 样本需要 LSB;
- MEGABYTE:GPT-3 尺寸 attention 只占 1.4% FLOPs;P=T^{1/3} → O(T^{4/3});
- LLaMA:7B×1T tokens(143 tokens/参数,约 7× Chinchilla 配比)仍单调变好;65B = 2048×A100 × 21 天。
