---
title: PTQ：GPTQ / AWQ / SmoothQuant
type: concept
tags: [quantization, llm-inference, gptq, awq, smoothquant, weight-only, w8a8]
source:
  - "pdf: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | arXiv:2210.17323"
  - "pdf: AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | arXiv:2306.00978"
  - "pdf: SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | arXiv:2211.10438"
  - "web: vLLM Quantization | https://docs.vllm.ai/en/latest/features/quantization/ | 阅读日期 2026-09-27"
date: 2026-09
---

# PTQ：GPTQ / AWQ / SmoothQuant

> **TL;DR**：三者都属于后训练量化(PTQ)，但优化对象不同：GPTQ 用近似二阶误差补偿做权重逐层量化；AWQ 用激活统计识别少量重要权重并保护它们；SmoothQuant 把激活离群值通过等价缩放迁移到权重侧，适合 W8A8 的权重+激活量化。选型不能只看 bit 数，还要同时看硬件 kernel、batch、上下文长度和质量目标。

## 1. 先区分量化对象

| 路线 | 主要对象 | 常见格式 | 典型目标 |
|---|---|---|---|
| Weight-only | 权重，激活保持 FP16/BF16 | W4A16、W3A16 | 最大限度压缩显存，适合中低 batch |
| Weight + activation | 权重和激活 | W8A8、W4A8 | 让 Tensor Core/专用 INT8/FP8 kernel 提高吞吐 |
| KV cache quantization | 运行时 K/V cache | KV8、KV4、混合精度 | 长上下文和高并发下扩大可容纳 token 数 |

PTQ 不等于“把 checkpoint 除以一个常数”。量化误差会经过残差、注意力和 MLP 层累积；校准数据、离群值处理和实际 kernel 共同决定最终效果。

## 2. GPTQ：二阶误差补偿的 weight-only 路线

GPTQ 把一层权重视为要逐步量化的矩阵，并用 Hessian 的近似信息估计“量化某个权重会给层输出带来多大损失”。每次量化后，误差会补偿到还未量化的权重上，从而避免独立舍入带来的累积误差。

### 适合场景

- 目标是 W4A16/W3A16 等 weight-only checkpoint
- 主要瓶颈是模型权重显存，而不是激活或 KV cache
- 使用已有 GPTQ/Marlin 等受支持 kernel
- 能接受离线校准和模型转换时间

### 需要记录的实验变量

- group size、是否对称量化、zero point 设置
- 校准样本数量、序列长度和领域分布
- 每层是否跳过 embedding、lm_head 或敏感层
- 实际 kernel，而不是只看 checkpoint 的文件大小

## 3. AWQ：保护激活敏感的少量权重

AWQ 的核心观察是：少量通道的权重对激活幅度特别敏感，直接均匀量化会造成较大误差。它用校准激活统计寻找重要通道，通过等价缩放保护这些权重，再对整体权重做低比特量化。它不需要把全部权重都当作同等重要，也不依赖反向传播更新模型。

### 适合场景

- 需要 W4A16 且希望使用高效 GPU kernel
- 关注推理速度和部署便利，而不仅是压缩率
- 模型有明显激活离群通道，希望按激活敏感度分配误差预算

AWQ 的最终收益取决于对应硬件是否有高效实现；没有匹配 kernel 时，低比特 checkpoint 可能只减少存储，却不一定减少端到端延迟。

## 4. SmoothQuant：把激活离群值迁移到权重侧

SmoothQuant 利用线性层的等价变换，在输入通道上使用缩放因子：把难以量化的激活离群值“平滑”到权重中，使激活和权重都更容易做 INT8 量化。它的目标是 W8A8，而不是单纯的 weight-only 压缩。

### 适合场景

- 硬件对 INT8 W8A8 有成熟 Tensor Core 或专用 kernel
- 高 batch / 高吞吐服务比极限显存压缩更重要
- 希望权重和激活都落在整数或低精度计算路径

需要注意：缩放会改变权重分布，也可能放大某些权重通道；部署时必须让校准得到的 scale、权重格式和 kernel 实现保持一致。

## 5. 选型决策

1. **只想让模型装进显存**：先比较 GPTQ/AWQ 的 W4A16，并验证真实 kernel。
2. **高吞吐、硬件 INT8 友好**：评估 SmoothQuant W8A8。
3. **真正的长上下文瓶颈是 KV cache**：不要只量化权重，转向 [KV Cache 量化](kv-cache-quantization.md)。
4. **使用 vLLM**：以官方支持矩阵和实际 GPU 架构为准，不能仅凭论文名称判断可用性；参见 [vLLM 量化实现](../../vllm/concepts/quantization-gptq-awq-fp8.md)。
5. **需要可复现结论**：至少同时记录质量指标、显存峰值、TTFT、ITL、吞吐、batch、上下文长度和 kernel/backend。

## 6. 常见误区

- 低 bit 不必然更快：解量化、非融合 kernel 或小 batch 可能抵消收益。
- 权重文件变小不等于服务显存按同样比例下降：激活、KV cache、workspace 和 CUDA graph 仍占显存。
- 校准集不能随便用短英文样本代替真实业务分布。
- 不同 backend 对 group size、zero point、scale layout 的约定不同，转换成功不代表数值和性能正确。

## 相关条目

- [KV Cache 量化](kv-cache-quantization.md)
- [SpQR：稀疏+量化混合表示](spqr-sparse-quantized.md)
- [vLLM GPTQ / AWQ / FP8](../../vllm/concepts/quantization-gptq-awq-fp8.md)
- [FlashAttention 实现](../../attention/implementation/flash-attention-implementation.md)

## 溯源

- 主源:GPTQ(arXiv:2210.17323)、AWQ(arXiv:2306.00978)、SmoothQuant(arXiv:2211.10438)三篇原始论文。
- 交叉源:vLLM Quantization 文档(https://docs.vllm.ai/en/latest/features/quantization/,引擎侧注册分派口径)。
