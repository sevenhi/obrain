---
title: 量化与压缩
type: domain
tags: [quantization, compression, llm-inference, kv-cache, low-rank, outlier]
source:
  - "pdf: HyQuant 论文全文 | arXiv:2608.27875v1"
  - "pdf: SpQR 论文全文 | arXiv:2306.03078 (ICLR 2024)"
  - "pdf: On the Effectiveness of Low-Rank Matrix Factorization for LSTM Model Compression | arXiv:1908.09982 (PACLIC 2019)"
  - "pdf: GPTQ | arXiv:2210.17323"
  - "pdf: AWQ | arXiv:2306.00978"
  - "pdf: SmoothQuant | arXiv:2211.10438"
  - "pdf: KIVI | arXiv:2402.02750"
  - "web: vLLM Quantization | https://docs.vllm.ai/en/latest/features/quantization/ | 阅读日期 2026-09-27"
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
| [PTQ: GPTQ / AWQ / SmoothQuant](concepts/ptq-gptq-awq-smoothquant.md) | concept | 权重/激活后训练量化的三条路线:二阶误差补偿、激活感知权重保护、平滑离群值后做 W8A8;按显存、硬件和质量目标选型 |
| [KV Cache 量化](concepts/kv-cache-quantization.md) | concept | 把运行时 KV cache 从权重压缩问题单独拆出:按 token/channel 分组、对称/非对称量化、混合精度与长上下文权衡 |

**共同心法(三篇一条线)**:压缩误差不是均匀分布的——HyQuant 按 attention mass 分精度、SpQR 按敏感度分位宽、LSTM 篇按谱结构(核范数)分压缩方式;**都是"把预算花在杠杆点上",而非全局均匀砍**。

## 相关条目(跨域互链)

- [vLLM 量化:GPTQ / AWQ / FP8](../vllm/concepts/quantization-gptq-awq-fp8.md)——**engine 视角**:量化的注册/分派/内核集成如何在推理引擎里落地;本领域是**算法视角**(怎么量化、误差怎么控制),两者互补。
- [attention/flash-attention-implementation](../attention/implementation/flash-attention-implementation.md)——FA3 的 FP8 block-quantize(Q/K/V 分块量化 + incoherent processing)是"量化进 attention kernel"的实例;HyQuant 的 dequantize-in-attention 同属此路线。
- SpQR 的推理 kernel(稀疏 CSR matmul + dense 3/4-bit matmul)是 CUTLASS 域 GEMM 定制的应用场景(参见 [cutlass/](../cutlass/index.md))。

## 扩展位(待补充候选)

- 已补充:GPTQ / AWQ / SmoothQuant(算法侧统一篇)与 KV cache 量化(KIVI + vLLM 工程视角)。
- 待补充:KVQuant / KVTuner / AQLM / QuIP#(2-bit 以下),以及不同 GPU 架构上的实测基准。
