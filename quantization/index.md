---
title: 量化算法
type: domain
tags: [quantization, llm-inference, kv-cache]
source:
  - "pdf: HyQuant 论文全文 | arXiv:2608.27875v1"
date: 2026-09
---

# 量化算法

> 领域索引。定位:**LLM 量化算法**(权重/KV cache 量化的算法侧:混合精度、低比特、误差分析方法),2026-09 由原 `llm-inference` 域下的论文笔记目录(papers/)升级而来。

## 条目索引

| 篇目 | 类型 | 一句话 |
|---|---|---|
| [HyQuant — LLM Attention 混合精度量化](concepts/hyquant-hybrid-precision-quantization.md) | concept | 竖线+局部窗口留全精度,其余 KV 压低比特(arXiv 2608.27875) |

## 相关条目(跨域互链)

- [vLLM 量化:GPTQ / AWQ / FP8](../vllm/concepts/quantization-gptq-awq-fp8.md)——**engine 视角**:量化的注册/分派/内核集成如何在推理引擎里落地;本领域是**算法视角**(怎么量化、误差怎么控制),两者互补。

## 扩展位(待补充候选)

- 待补充:KIVI / KVTuner / AWQ / GPTQ / KVQuant 等量化算法(尚未入库,先登记不造假索引行)
