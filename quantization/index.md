---
title: 量化与压缩
type: domain
tags: [quantization, compression, llm-inference, kv-cache, low-rank, outlier]
source:
  - "pdf: HyQuant 论文全文 | arXiv:2608.27875v1"
  - "pdf: SpQR 论文全文 | arXiv:2306.03078 (ICLR 2024) | ~/zhanbin/library/papers/SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression.pdf"
  - "pdf: On the Effectiveness of Low-Rank Matrix Factorization for LSTM Model Compression | arXiv:1908.09982 (PACLIC 2019) | ~/zhanbin/library/papers/On the Effectiveness of Low-Rank Matrix Factorization for LSTM Model Compression.pdf"
date: 2026-09
---

# 量化与压缩

> 领域索引。定位:**LLM/NN 的量化与压缩**——权重/KV cache 量化的算法侧(混合精度、低比特、误差分析方法)+ 权重压缩的结构路线(稀疏+量化混合表示、低秩分解/剪枝)。2026-09 由原 `llm-inference` 域下的论文笔记目录(papers/)升级而来;同月批C扩充后定位从「量化算法」扩为「量化与压缩」(SpQR=量化+稀疏混合路线,LSTM 低秩=分解/剪枝路线)。

## 条目索引

| 篇目 | 类型 | 一句话 |
|---|---|---|
| [HyQuant — LLM Attention 混合精度量化](concepts/hyquant-hybrid-precision-quantization.md) | concept | 竖线+局部窗口留全精度,其余 KV 压低比特(arXiv 2608.27875) |
| [SpQR — 稀疏+量化混合表示](concepts/spqr-sparse-quantized.md) | concept | ~1% 高敏感 outlier 存 16-bit(CSR)+ 其余 3-4bit + bilevel 量化统计量,首个近无损(<1% PPL)3-4bit 权重压缩(arXiv 2306.03078, ICLR 2024) |
| [LSTM 低秩压缩 — 分解 vs 剪枝](concepts/lstm-lowrank-compression.md) | concept | SVD/Semi-NMF vs 剪枝:压 W_h(天生低秩)远比压 W_i 安全,核范数是可压缩性先验(arXiv 1908.09982, PACLIC 2019) |

**共同心法(三篇一条线)**:压缩误差不是均匀分布的——HyQuant 按 attention mass 分精度、SpQR 按敏感度分位宽、LSTM 篇按谱结构(核范数)分压缩方式;**都是"把预算花在杠杆点上",而非全局均匀砍**。

## 相关条目(跨域互链)

- [vLLM 量化:GPTQ / AWQ / FP8](../vllm/concepts/quantization-gptq-awq-fp8.md)——**engine 视角**:量化的注册/分派/内核集成如何在推理引擎里落地;本领域是**算法视角**(怎么量化、误差怎么控制),两者互补。
- [flash-attention/io-aware-tiling](../flash-attention/concepts/io-aware-tiling.md)——FA3 的 FP8 block-quantize(Q/K/V 分块量化 + incoherent processing)是"量化进 attention kernel"的实例;HyQuant 的 dequantize-in-attention 同属此路线。
- SpQR 的推理 kernel(稀疏 CSR matmul + dense 3/4-bit matmul)是 CUTLASS 域 GEMM 定制的应用场景(参见 [cutlass/](../cutlass/index.md))。

## 扩展位(待补充候选)

- 待补充:KIVI / KVTuner / AWQ / GPTQ(算法侧专篇,现散见于 SpQR 篇与 vllm 域)/ KVQuant 等量化算法(尚未入库,先登记不造假索引行)
- 待补充:低秩路线的 Transformer 时代后继——LoRA / QLoRA(NF4+双重量化,与 SpQR 同作者脉络)、AQLM / QuIP#(2-bit 以下)
