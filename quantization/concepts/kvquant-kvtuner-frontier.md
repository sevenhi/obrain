---
title: KV cache 极低比特前沿:KVQuant / KVTuner
type: concept
tags: [quantization, kv-cache, llm-inference, long-context, kvquant, kvtuner, mixed-precision, pre-rope, non-uniform]
source:
  - "pdf: KVQuant 论文全文 | arXiv:2401.18079 (UC Berkeley/SqueezeAILab, Hooper/Kim/Gholami et al., NeurIPS 2024, ↑primary)"
  - "pdf: KVTuner 论文全文 | arXiv:2502.04420 (华为 Noah's Ark + CUHK, ICML 2025)"
  - "pdf: KIVI(交叉核对,同期并行工作) | arXiv:2402.02750"
  - "web: KVQuant 官方代码 | https://github.com/SqueezeAILab/KVQuant (2026-10 实测 200)"
  - "web: KVTuner 官方代码 | https://github.com/cmd2001/KVTuner (2026-10 实测 200)"
date: 2026-10
---

# KV cache 极低比特前沿:KVQuant / KVTuner

> **TL;DR**:KVQuant(UC Berkeley,2401.18079)把 KV cache 压到 **3-bit 近无损(<0.1 PPL 退化)、2-bit 可接受**,靠四招:Per-Channel Key、**Pre-RoPE Key(RoPE 之前量化 K,关键创新)**、Non-Uniform per-layer datatype、Per-Vector Dense-and-Sparse;标题卖点是**单卡 A100 跑 1M、8 卡跑 10M context**。KVTuner(华为/CUHK,2502.04420)是**独立**的后续工作(baseline 用 KIVI,非 KVQuant),把「每层 KV 精度对」的粗粒度混合精度搜出来:**Llama-3.1-8B ≈3.25-bit、Qwen2.5-7B ≈4.0-bit 近无损**,吞吐比 KIVI-KV8 最高 +21.25%。两者一条线:**KV cache 的省显存杠杆在长上下文,2-3 bit 就能把 cache 显存砍掉 4-8 倍**。

---

## 0. 与已有篇的分工(避免复述)

本库 [kv-cache-quantization.md](kv-cache-quantization.md) 已讲清 K/V 的通用设计维度(per-channel K / per-token V / 非对称 / 混合精度),以及 KIVI 的完整机制。本篇**只聚焦 KVQuant / KVTuner 相对 KIVI 新增的东西**,并明确:KIVI(2402.02750)与 KVQuant(2401.18079)是**同期并行**工作,不是继承关系。

## 1. KVQuant:四招把 KV 压到 3-bit 近无损

### 1.1 动机:为什么 sub-4-bit KV 之前没人做得动

KV cache 激活值有**结构化 outlier**,且与权重的 outlier 不同(见 §1.4):Key 在 RoPE 前有「固定 channel 上的持久 outlier」,RoPE 之后这种一致性被打散;Value 的 outlier 无固定模式(channel 和 token 上都有)。之前的 int4/nf4 方案在 4-bit 尚可,但 int3/nf3 直接崩(LLaMA-7B int3 PPL 从 5.68 崩到 10.87),int2 更是 11779。KVQuant 的目标就是把 3-bit 乃至 2-bit 做可用。

### 1.2 四大组件(机制原理)

**(i) Per-Channel Key Quantization。** Key 有「outlier channel」——少数 channel 的平均幅度显著大于其他 channel。按 channel 分组量化(每 channel 一个 scale/zero-point),把「相似幅度的值」天然聚到一组,outlier channel 不再拖垮整块量化范围。Value 的 outlier 则 channel/token 都有、更弱,故 Value 仍按 token 量化(与 KIVI「K per-channel、V per-token」的结论一致)。

**(ii) Pre-RoPE Key Quantization(关键创新)。** RoPE 是「位置相关的成对旋转」:它对 K 的每对 channel 做旋转变换,导致 **RoPE 后原本清晰的 outlier channel 幅度一致性被打散**(图 2 实证),量化难度陡增。KVQuant 的做法:**在 RoPE 之前就量化 K,推理时先反量化再套 RoPE**。为此写了一个 fused kernel,在 dequantize 之后 on-the-fly 应用 RoPE。收益实打实:3-bit LLaMA-7B 上 Pre-RoPE 比 Post-RoPE **PPL 低 0.82**(6.23 vs 7.05)。

```
K_n = W_k · x_n                      # 原始 key 向量(有清晰 outlier channel)
  │  ① 量化:K̃_n = Q(K_n)   (per-channel,在 RoPE 之前)
  ▼
K̃_n (低比特存储)
  │  ② 推理:fused kernel dequant → 再套 RoPE:R_θ,n · dequant(K̃_n)
  ▼
K̃_n(旋转后,进 attention)
```

**(iii) Non-Uniform per-layer datatype(NUQ)。** 不用均匀量化、也不复用固定的 nf4 datatype,而是**每层**用「sensitivity-weighted」的非均匀 datatype:在 calibration 集上,用**对角 Fisher 信息矩阵**对量化误差加权(而非只看幅度),配合 k-means 求解每层 K/V 各自的量化 signpost(台阶位置)。这是把「哪个值域误差要紧」显式建模——与权重侧 SpQR/QuIP# 的 Hessian/Fisher 加权同一哲学,但作用于 KV 激活。校准成本很低(Fisher 计算 2.8 分钟,每层 k-means 4.5/2.7/1.9 分钟,可并行)。

**(iv) Per-Vector Dense-and-Sparse Quantization。** 借鉴权重侧 dense-and-sparse 思路,把数值 outlier 单独挑出存高精度(稀疏矩阵),其余 dense 低比特。**关键改进:outlier 阈值 per-vector**(per-channel 量化时每 channel 一个阈值,per-token 时每 token 一个),而非全层一个全局阈值——因为「某 channel 的普通值可能是另一 channel 的 outlier」。只挑 1% outlier 就能把 3-bit PPL 再降 0.19(5.94→5.75,距 FP16 仅 0.07)。

### 1.3 数据(LLaMA 系 PPL,WikiText2,论文 Table 1)

| 位宽(方法) | LLaMA-7B | LLaMA-13B | LLaMA-30B | LLaMA-65B |
|---|---|---|---|---|
| FP16 | 5.68 | 5.09 | 4.10 | 3.53 |
| KVQuant-4bit(-1% out) | 5.69 | 5.10 | 4.11 | 3.54 |
| KVQuant-3bit(-1% out) | **5.75** | **5.14** | **4.15** | **3.57** |
| KVQuant-2bit(-1% out) | 6.01 | 5.36 | 4.78(≈) | 4.03(≈) |
| (对照) int3 / int2 | 10.87 / 11779 | 8.69 / 69965 | — | — |

- **位宽-精度-显存的量化关系**:4-bit <0.02、3-bit <0.1、2-bit <0.5 PPL 退化(WikiText2,跨模型),对应显存省 **3.7× / 4.8× / 6.9×**。
- **长上下文能力**(标题卖点):LLaMA-7B 单卡 A100-80GB 跑 **1M context**,8-GPU 跑 **10M context**(表 7:7B 10M context 的 KV cache 从 4883GB 降到 610GB)。
- **吞吐**:custom CUDA kernel 相对 fp16 matvec 最高 ~1.7× 加速。

### 1.4 与 KIVI 的关系(写明并行)

KIVI(2402.02750)与 KVQuant(2401.18079)几乎同期提交,是**独立并行**工作:两者都发现「K 用 per-channel、V 用 per-token」以及「非对称量化」对 KV 有效;KVQuant 独有的是 **Pre-RoPE 量化 + NUQ 非均匀 datatype + per-vector sparse**,KIVI 独有的是免调参的运行时分组与 recent-token 全精度窗口(two-part cache)。不要把「Pre-RoPE」误记到 KIVI 头上。

## 2. KVTuner:layer-wise 混合精度搜索(独立工作)

### 2.1 三个痛点与思路

KVQuant/KIVI 解决「怎么把 KV 压到低比特」,但 KVTuner 指出三个未解问题:① 各层对 KV 量化的敏感度差异被忽视(统一 KV8 或统一 KV4 都在浪费预算);② 在线细粒度决策开销高;③ 对不同 LLM/约束的灵活性差。KVTuner 的方案:**离线搜索每层的「K/V 精度对」(coarse-grained),在线直接套用**。

### 2.2 机制:粗粒度混合精度 + 两阶段剪枝 + MOO

**粒度**:不逐 token/逐列,而是每层一整个 KV cache 用一个精度对,如 `K8V4`、`K4V2`(硬件友好,无在线开销)。搜索空间是 {2,4,8}×{2,4,8} 的每层选择,即 S^L(L 层)。

**两个理论观察(为什么 K 比 V 更重要)**:低比特 key 量化导致的准确率下降主要来自**误差跨层累积**和 **attention 分布偏移**;量化误差大的 head 往往是非稀疏、非集中的 attention 模式(稀疏的 streaming head 对量化更鲁棒,retrieval head 更敏感)。因此「key cache 通常比 value cache 对量化误差更敏感」,精度预算应优先给 K。实证:K8V2(5-bit)精度 ≥ K4V8(6-bit)且省 12.5% 显存。

**两阶段搜索空间剪枝**:
1. **intra-layer pruning**:每层内,把不在「等效位宽 vs attention 输出误差」Pareto 前沿上的精度对剪掉(如保留 {KV8, K8V4, KV4, K4V2, KV2},砍掉 K2V8 这类高误差低收益项)。
2. **inter-layer clustering**:按「裁剪后的候选精度对集合」对层分组,再按量化敏感度(relative attention output error)聚成 G 个类,同类的层共享精度对 → 搜索空间从 5^32≈2.3×10²² 缩到 5^6=15625。

**MOO 搜索**:目标 `min (f_m 显存, f_a 精度损失) s.t. f_m≤M, f_a≤ΔA`,离线用多目标优化求出 Pareto 前沿,得到不同约束(C3.25 / C4.0 / C5.44 等)下的配置,在线直接查表。校准时用「反量化 KV cache 参与 prefill 自注意力」的技巧放大误差累积,以区分候选精度对。

### 2.3 数据

**近无损点**:Llama-3.1-8B-Instruct **≈3.25-bit**(KIVI 基线,GSM8K 数学推理几乎无损,表 8:BF16 0.7635 vs KVTuner-C3.25 0.7483~0.7604);敏感模型 Qwen2.5-7B-Instruct **≈4.0-bit**(K4V8 对已崩到 0.0076,KVTuner-C4.0 恢复到 0.3315~0.3333 的 GPQA 近无损区间)。

**吞吐**(表 8,Llama-3.1-8B-Instruct,相对 KIVI-KV8):KVTuner-C3.25 在 batch=64/len=128 下 **+21.25%**,batch=16/len=512 +17.55%,batch=8/len=1024 +16.79%——省显存换来更大 batch/更长上下文。

## 3. 对长上下文的意义(为什么这条线重要)

KV 量化省的是 **cache 显存**,与权重量化省的是两笔账:长上下文下 KV cache 反超权重成为显存第一大户(1M context 的 LLaMA-7B cache 达 512GB)。把 KV 压到 2-3 bit(省 4-8×),直接换来「更长上下文 / 更大 batch」,而不是更快。配套的显存账单与 PagedAttention 的关系见 [kv-cache-quantization.md](kv-cache-quantization.md) §1;dequantize-in-attention 的带宽账见 [HyQuant](hyquant-hybrid-precision-quantization.md)。

## 4. 实践要点

1. **与权重量化叠加的误差耦合**:W4A16 + KV3 两层误差不是简单相加——KV 量化误差经 attention 分布偏移放大,权重误差经逐层累积,叠加后可能超过单独任一者。上线前做**联合**校准与回归,别分开验证。
2. **引擎侧现实(与 2-bit 权重同理)**:vLLM 对 3-bit KV cache 的支持有限——`Fp8KVCacheMethod` 是 FP8 路径(见 [vLLM 量化篇](../../vllm/concepts/quantization-gptq-awq-fp8.md) §3),KVQuant 的 NUQ/KVTuner 的 layer-wise 精度对**不能靠一个 `--kv-cache-dtype` 参数直接开出来**,需自备 kernel/集成。
3. **校准集验证**:KVQuant 的 NUQ 用对角 Fisher 加权、KVTuner 离线搜索都用 calibration 集;校准分布若与线上不匹配,精度对选择会失效。用真实业务长度分布(含长 CoT)校准。
4. **选型决策**:纯省显存且要稳 → KVQuant(3-bit 近无损,2-bit 可接受);模型对 KV 量化极敏感(如 Qwen2.5-7B)→ KVTuner 的 layer-wise 精度对(把预算给敏感层,近无损下压到 4-bit);只需温和压缩 → 8-bit/4-bit 均匀量化即可,别过度工程。
5. **注意「近无损」的度量对象**:KVQuant 的 <0.1 PPL 是 perplexity 口径;KVTuner 的近无损是数学推理/GPQA 口径。二者不可直接横比,落到自己的评测集上复测。

## 相关条目

- [KV Cache 量化(KIVI)](kv-cache-quantization.md)——K/V 通用机制与 KIVI,本篇的前置
- [HyQuant:Attention 混合精度量化](hyquant-hybrid-precision-quantization.md)——token 级精度分配 + dequantize-in-attention,与 KVTuner 的 layer 级互补
- [2-bit 权重量化前沿:QuIP# / AQLM](two-bit-weight-frontier.md)——另一条「极低比特」战线(权重)
- [vLLM 量化实现](../../vllm/concepts/quantization-gptq-awq-fp8.md)——引擎侧 KV cache 量化支持口径
- [PagedAttention](../../vllm/concepts/paged-attention.md)、[GQA/MQA](../../model-architecture/concepts/attention-heads-mha-mqa-gqa.md)

## 溯源

- 主源 ↑primary:KVQuant(arXiv:2401.18079,UC Berkeley/SqueezeAILab,NeurIPS 2024)。
- 第二主源:KVTuner(arXiv:2502.04420,华为 Noah's Ark + CUHK,ICML 2025)。
- 交叉核对:KIVI(arXiv:2402.02750)——与 KVQuant 同期并行,非继承。
- 代码:github.com/SqueezeAILab/KVQuant、github.com/cmd2001/KVTuner(2026-10 均实测 200)。
