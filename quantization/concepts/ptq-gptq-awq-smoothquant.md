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

**核心机制（transformer 论文 Eq.2/3 的闭式解）**：对一层权重 W，设量化前后的层输出误差目标为 ‖W·X − Ŵ·X‖²。用 Hessian H = 2XXᵀ 度量误差，把问题写成逐列（按权重列 w_q）贪心量化 + 补偿的闭式迭代：

- **误差补偿式**：量化第 q 列后，其余未量化列按 `δ_q = (w_q − ŵ_q) / [H⁻¹]_qq · (H⁻¹)_:,q` 补偿——即把该列的舍入误差“分摊”到还未处理的列上，权重其逆 Hessian（H⁻¹ 的第 q 列）。
- **列补偿的直觉**：直接逐元素 round 会让每列的误差独立累积；GPTQ 利用 H⁻¹ 的相关结构，把当前列的量化误差沿“对后续输出影响最大”的方向提前抵消，等效于在贪心量化里做了一次二阶校正（近似于逐步最优的贪心，而非全局解）。
- 这也是为何 GPTQ 需要一次前向跑校准数据得到 H（或 Cholesky 分解的 H⁻¹）再离线量化——H 来自校准集，量化顺序逐列推进。

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

**优化目标（为什么用缩放保护）**：AWQ 不直接量化 W，而是给每列引入一个 per-channel scale s，量化 `Q(W·s)`、前向时再除以 s，使缩放前后输出等价：`Q(W·s) · (X/s) ≈ W·X`。目标就是**最小化缩放-量化-还原后的输出误差**：

```
min_s ‖ Q(W·s) · (X/s) − W·X ‖²
```

关键直觉：对激活幅度大的通道（outlier channel），量化误差相对被放大；通过给这些通道一个更大的 scale，把“误差预算”从激活敏感通道挪到不敏感通道。s 直接用校准激活的幅度统计（如各通道激活的均值/最大值）启发式确定，无需梯度或反向传播——这也是它比 GPTQ 更快、更省校准的原因。缩放只改 scale 不改数学结果（scale 可吸收进前后层的归一化或权重里），因此是“等价变换”而非近似。

### 适合场景

- 需要 W4A16 且希望使用高效 GPU kernel
- 关注推理速度和部署便利，而不仅是压缩率
- 模型有明显激活离群通道，希望按激活敏感度分配误差预算

AWQ 的最终收益取决于对应硬件是否有高效实现；没有匹配 kernel 时，低比特 checkpoint 可能只减少存储，却不一定减少端到端延迟。

## 4. SmoothQuant：把激活离群值迁移到权重侧

SmoothQuant 利用线性层的等价变换，在输入通道上使用缩放因子：把难以量化的激活离群值“平滑”到权重中，使激活和权重都更容易做 INT8 量化。它的目标是 W8A8，而不是单纯的 weight-only 压缩。

**迁移因子（离群值怎么挪）**：SmoothQuant 观察激活 X 有离群值（幅度远大于同通道其他值），直接 INT8 量化会因 range 过大而精度崩。它引入 per-channel 迁移因子 s，把 `Y = W·X` 等价改写为 `Y = (W·diag(s)) · (diag(s)⁻¹·X)`——把 X 的幅度“搬到” W 上。s 的闭式取法：

```
s_j = max(|X_j|) / max(|W_j|)      # 第 j 通道:激活最大幅度 / 权重最大幅度
```

- **直觉**：离群通道的 X 幅度大 → s_j 大 → 该通道激活被除以 s_j 压平，同时权重乘以 s_j 吸收幅度 → 二者量化范围都落在 INT8 可表示区间。
- **幂次缩放**：`diag(s)⁻¹` 可实现为整数右移/幂次缩放（2 的幂），避免引入额外浮点乘法。
- **代价**：缩放改变权重分布、可能放大某些权重通道；部署时 scale 必须与权重格式、kernel 实现保持一致（这也是本域反复强调的“scale 约定”问题）。

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
