---
title: KV Cache 量化
type: concept
tags: [quantization, kv-cache, llm-inference, long-context, memory]
source:
  - "pdf: KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache | arXiv:2402.02750"
  - "web: vLLM Quantization | https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/ | 阅读日期 2026-09-27"
  - "web: vLLM 官方文档 | https://docs.vllm.ai/en/latest/ | 阅读日期 2026-09-27"
date: 2026-09
---

# KV Cache 量化

> **TL;DR**：KV cache 是每个请求运行时增长的状态，不应和静态权重量化混为一谈。长上下文、高并发和 beam/prefix caching 场景中，KV cache 可能比权重更快成为显存瓶颈。量化 KV cache 能提高可容纳 token 数，但会引入注意力分数误差；必须同时确定量化粒度、校准/scale、K/V 是否同精度，以及 attention kernel 是否真正支持该格式。

## 1. 显存账单

对 decoder-only Transformer，单个 token 的 KV cache 近似占用：

```text
bytes/token ≈ 2(K,V) × layers × kv_heads × head_dim × bytes_per_element
```

批量服务的总占用还要乘以请求 token 数。GQA/MQA 通过减少 `kv_heads` 已经降低了账单；KV 量化是在此基础上进一步降低每个元素的字节数。

因此，下面三件事是不同层面的优化：

- GQA/MQA：改变模型结构中的 K/V 头数量
- PagedAttention：改变 KV cache 的分配和复用方式
- KV quantization：改变 cache 元素的存储精度

参见 [GQA/MQA/KV cache](../../model-architecture/concepts/attention-heads-mha-mqa-gqa.md) 与 [PagedAttention](../../vllm/concepts/paged-attention.md)。

## 2. 为什么 K/V 不能简单地均匀量化

K 和 V 的误差影响路径不同：K 参与注意力 logits，误差会改变 softmax 权重；V 直接参与加权求和，误差更像输出值扰动。不同 token、channel、head 和层的敏感度也不均匀。

常见设计维度：

- **粒度**：per-tensor、per-token、per-channel、per-head 或分组
- **映射**：对称量化或非对称量化
- **位宽**：FP8/INT8 较稳妥，INT4/INT2 压缩更激进
- **布局**：scale/zero-point 如何存储，是否能和 attention kernel 融合
- **混合精度**：保留敏感层、近期 token 或部分 channel 的高精度

KIVI 的代表性结论是 K 和 V 的最佳量化粒度并不相同，非对称低比特 KV cache 需要针对统计分布设计，而不是直接套用权重量化流程。

**KIVI 核心机制（2-bit 免调参的非对称 KV 量化）**：KIVI 的关键是把 K 和 V 的统计分布差异显式建模，用两套不同粒度：

- **K 用 per-channel 量化、V 用 per-token 量化**（为何不同）：
  - **K 的离群值出现在固定的少数 channel（特征维）**：某个 key 维度上某些 channel 持续出现大值（key 的“channel-wise outlier”），因此按 channel 分组量化、给每个 channel 一个 scale，能把这些离群 channel 单独用更大 scale 保护，避免离群值拖垮整块量化范围。
  - **V 的离群值出现在少数 token（序列维）**：value 的幅度是“token-wise”尖峰（少数 token 的 value 幅度极大），因此按 token 分组量化、per-token scale，才能精确框住这些尖峰 token。
  - 换句话说：**K 的离群沿 channel 方向，V 的离群沿 token 方向**——这是 K/V 各自参与的计算（K 进 logits、V 进加权和）与其数据布局共同决定的分组选择。
- **非对称量化（有 zero-point）**：K/V 分布不是以 0 为中心对称，用非对称量化（scale + zero-point）比对称量化（仅 scale）更贴分布，2-bit 下误差更小。
- **recent-token 全精度窗口(two-part cache)**:最近的 token 保持 FP16 不量化,更早的 token 压到 2-bit——用极小的高精度窗口保护近期 KV,其余按分组低比特存储。(残差式离群保留——per-vector 保留离群分量——属 KVQuant 的 Dense-and-Sparse,见 kvquant 篇;KIVI 无此机制。)
- **scale 布局与 group 说明**：per-channel 的 K 每个 channel 一个 (scale, zero-point)，per-token 的 V 每个 token 一个；group size 决定 scale 的存储密度（group 越小越精细、scale 本身存储开销越大），部署时要与 attention kernel 的 layout 约定对齐（scale 布局是 KIVI 类方法最容易踩的坑）。

## 3. 与权重量化的关系

| 项目 | 权重量化 | KV Cache 量化 |
|---|---|---|
| 生命周期 | 模型加载前/加载时 | 请求运行时持续增长 |
| 主要收益 | 减少模型权重显存和带宽 | 增加上下文长度/并发容量 |
| 校准对象 | 权重和激活样本 | 运行时 K/V 分布 |
| 风险 | 层输出误差、模型质量下降 | 注意力分布和生成稳定性下降 |
| 依赖 | 权重格式与 GEMM/MoE kernel | attention kernel、cache layout、scale 管理 |

实际服务通常可以组合：W4A16/W8A8 压缩权重，再用 FP8/INT8 KV cache 控制长上下文显存。

## 4. 部署检查清单

- 明确当前瓶颈是权重、KV cache、激活还是 workspace。
- 记录 GPU 型号、计算能力、CUDA/PyTorch/vLLM 版本。
- 确认 vLLM 版本支持目标 KV cache dtype 和硬件。
- 比较相同模型、相同 prompt、相同输出长度下的质量和延迟。
- 至少测量 TTFT、ITL、吞吐、显存峰值、可容纳 token 数和 OOM 边界。
- 覆盖短上下文、长上下文、并发请求和 prefix caching 场景。
- 对聊天、代码、长文档和结构化输出分别做回归样例。

## 5. 常见误区

- KV cache 量化不会减少模型权重显存。
- 把 `torch_dtype` 改成低精度不等于启用了 KV cache 量化。
- 只看单请求显存，无法预测多并发服务的收益。
- 低比特 KV cache 需要 kernel 配合；仅改变存储格式可能增加反量化开销。
- 长上下文下的质量退化可能在短样本上看不出来，必须使用真实长度分布。

## 相关条目

- [PTQ：GPTQ / AWQ / SmoothQuant](ptq-gptq-awq-smoothquant.md)
- [HyQuant：Attention 混合精度量化](hyquant-hybrid-precision-quantization.md)
- [GQA / MQA / MLA](../../model-architecture/concepts/attention-heads-mha-mqa-gqa.md)
- [PagedAttention](../../vllm/concepts/paged-attention.md)
- [vLLM GPTQ / AWQ / FP8](../../vllm/concepts/quantization-gptq-awq-fp8.md)

## 溯源

- 主源:KIVI 论文(arXiv:2402.02750)。
- 交叉源:vLLM 文档 KV cache 量化页(https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)。
