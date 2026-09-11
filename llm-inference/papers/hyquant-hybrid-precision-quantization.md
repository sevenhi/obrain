---
title: HyQuant — 面向 LLM Attention 的混合精度量化
type: concept
tags: [attention, quantization, kv-cache, llm-inference, long-context, hyquant]
source:
  - "pdf: HyQuant 论文全文 | arXiv:2608.27875v1 | library/2608.27875v1.pdf"
  - "web: HyQuant 代码仓库 | https://github.com/jerrysfls/HyQuant"
date: 2026-09
---

# HyQuant:混合精度量化 LLM Attention

> 论文阅读笔记(arXiv 2608.27875v1,2026-08-28;SJTU / 西安交大 / 腾讯 PPL 等)。

## TL;DR

LLM 长上下文推理时,attention 高度不平衡——**极少量 key 位置(<5%)构成持久的"竖线"(vertical-line),承载绝大多数 attention mass**。HyQuant 的思路:只把这批竖线位置 + 一个局部滑动窗口**保留全精度**,其余绝大部分 KV 压到低比特(如 K4V4),Prefill 做混合精度算子、Decode 做混合精度 KV cache + 融合反量化。用极小全精度预算,换来接近全精度的准确率 + 明显的 decode 加速。

一句话:**均匀低比特量化对"高 attention 分数位置"的误差被反复放大;纠正办法不是把全部位宽调高,而是针对这些关键位置保留精度**。

## 动机(为什么)

1. **attention 高度非均匀**:横跨 Qwen3-8B / Gemma4-31B / Qwen3.5-4B / Llama3-8B,attention heatmap 都有持久竖线(类似 MInference 的 vertical-line、StreamingLLM 的 attention sink)。这些位置被大量 query token 反复关注。
   - 数据(Qwen3-8B / Llama-3.1-8B):全局 top-1% key 位置盖住 **47~59%** attention mass;top-5% 盖住 **54~64%**;top-5% + 局部窗口(W=128)盖住 **82~86%**。
2. **量化误差在"高分数位置"被放大**:K/V 扰动经大量 query 反复访问而放大。均匀 4-bit 误差显著大于 8-bit;但**保留 top-1%/top-5% 高分数位置为全精度后,其余 4-bit 的误差一路逼近 8-bit 水平**(1K~32K 序列都成立)。这直接催生混合精度设计。
3. 现有方法(KV eviction 如 H2O/SnapKV/PyramidKV,或 KV cache 量化如 KIVI/KVTuner)大多假设 token 同质,对"知识型长 CoT"里 token 重要性非平稳的情况(被 evict 的后来可能变关键)不够稳。竖线是 query 分布级现象(大量 query 都关注某 key),比单 query 关键更持久。

## 方法(怎么做)

把每个 layer/head 的 key 分成互斥三段:`K = K_VL ∪ K_Win ∪ K_Q`:
- **K_VL 竖线位置**:从"非窗口前缀"里选 top-ρ(ρ≈5%,竖线占比)。竖线分数 = 累计列方向 attention mass `S(k)=Σ_t a_t,k`,用轻量 reduction 增量维护,**几乎零预处理开销**(3%~5% 总耗时)。
- **K_Win 局部窗口**:最近 W 个 key(固定窗口大小,W 为预设常数)。
- **K_Q 其余多数**:低比特量化。

**Prefill(计算密集)**:混合精度量化 attention 算子 —— 其余上下文(K_Q)走 INT8/FP8 或 INT4/FP4,竖线+窗口(K_FP)走 FP16/BF16,两条路径**融合进单个 FlashAttention-style 算子**(online softmax + 分块扫描),结构不变、额外开销极小。

**Decode(带宽/显存密集)**:KV cache 混合精度存储 —— K_VL∪K_Win 全精度,其余 K4V4 低比特;**将反量化融进 attention kernel**(扫描量化块时即时 DeQuant,不物化全精度中间态),减少内存搬运。对比 KIVI/KVTuner 的"先 dequantize 再 attend"策略,HyQuant 是"dequantize-in-attention"单遍 online-softmax。

竖线识别额外开销:3%~5% 运行时;aux 内存最多 64×H_Q×d×2 字节(H_Q = num Q heads,FP16 缓冲 64 个 query 向量);5% 竖线全精度使非窗口 KV 增大约 15%。

## 结果

- 模型:Qwen3-8B / Llama-3.1-8B-Instruct / GLM-4-9B-0414 / Qwen3-32B;测试:LongBench v1 + GSM8K + MATH500(NVIDIA H100)。
- 对比 KIVI / KVTuner / SageAttention / FA2(全精度基准):在多数场景保持接近 FA2 精度,且优于严格低比特基线;部分数据集略超 FA2(视为方差)。
- **Decode kernel 加速**:1.32×~3.58×(随前缀增长而增大,32K 前缀达 3.58×);**端到端 decode 加速 1.04×~1.17×**。
- **Prefill**:与 SageAttention 速度相当,但层间 MSE 明显更低(保留竖线+窗口对抑制低比特误差有效)。
- **高并行**:batch=32 时 HyQuant 是唯一不 OOM 的方法(KV 压缩 + 融合内核降带宽和峰值显存)。

## 与既有技术的关系

| 相关工作 | 区别 |
|---|---|
| MInference | 用竖线当**稀疏 mask**(跳过块);HyQuant 用它分**精度**,低信息 token 仍保留在低比特而不是丢弃 |
| KVTuner | sensitivity-aware 混合精度但按 **layer** 分配;HyQuant 按 **token** 分配 |
| KIVI / KVQuant | 纯 KV cache 量化(降误差),没显式利用持久竖线做 token 级全精度保留 |
| 前作(如 H2O) | 只做了 Decode 端;HyQuant 是 **Prefill+Decode 两端一体化** |

## 局限

- 长上下文场景收益最明显;短上下文可能提升有限(仍持平全精度)。
- 只在 NVIDIA H100 实验;短上下文时 decode 未完全带宽受限,端到端加速不明显。
- 未测 >Qwen3-32B 大模型;agent / coding 场景未知。

---
*代码:https://github.com/jerrysfls/HyQuant*
