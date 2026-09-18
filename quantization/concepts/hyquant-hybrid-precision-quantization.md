---
title: HyQuant — 面向 LLM Attention 的混合精度量化
type: concept
tags: [attention, quantization, kv-cache, llm-inference, long-context, hyquant]
source:
  - "pdf: HyQuant 论文全文 | arXiv:2608.27875v1"
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

### 机制推导:为什么"竖线保留精度"能兜住均匀低比特的误差

这是 HyQuant 成立的关键因果链,拆成三步:

**① 竖线位置是 attention 输出误差的"杠杆点"。** 输出 `O_i = Σ_j a_ij V_j`,其中 `a_ij = softmax(S_ij)`。若某个 key 位置 `j*` 被大量 query 给予很高的 `a_ij`,那么 `V_j*` 的量化噪声 `ΔV_j*` 会以**加权和**的形式进入多个输出:`ΔO_i ≈ a_ij* · ΔV_j*`。对竖线位置 `a_ij*` 大 ⇒ 同样的 `ΔV` 被放大成更大的输出误差。**均匀量化把所有位置一视同仁,恰恰在这个杠杆点上放了最大风险**。

**② 保留竖线 = 在杠杆点上去掉噪声源。** 对竖线位置保留全精度(`ΔV=0`),即使其余 `K_Q` 位被压到 4-bit,噪声只从低注意力权重的 token 进来,对输出的贡献被 `a_ij` 天然加权稀释。

**③ 为什么"低比特误差逼近 8-bit"是必然。** 竖线(top-5%)+ 窗口盖住 82~86% 的 attention mass,意味着输出几乎全部来自"已保留精度"的部分;其余 14~18% 的低注意力部分即使 4-bit,其误差经小权重进入输出,贡献与"全 8-bit"相当。**所以不是"2-bit 免费的午餐",而是"用极小的全精度预算(只护住高杠杆点),把有效量化噪声压到 8-bit 水平"**。

### 机制推导:为什么 decode 的"反量化融合进 attention"省带宽

Decode 是带宽受限(显存/带宽密集),瓶颈是**从 HBM 把 KV cache 搬进 SRAM** 的字节数,不是 FLOPs。

- **KIVI/KVTuner 的做法**(dequantize-then-attend):先从 HBM 读低比特 KV → 反量化成 FP16 中间张量(写回或占用更多寄存器/SRAM) → 再 attend。多一趟反量化中间态的搬运。
- **HyQuant 的做法**(dequantize-in-attention):扫描被量化的 KV 块时,**在它即将作为 matmul 操作数进 Tensor Core 前那一刻才反量化**,不物化整份全精度中间态。

省带宽的关键:**低比特 KV 在 HBM 里占的字节少(4-bit vs 16-bit = 4× 压缩)**,反量化发生在片上(读出来即用),不把扩大的中间态来回搬。所以 decode 加速 `1.32×~3.58×` 正是"少搬了低比特那 4× 的字节 + 融合避免中间态往返"的量化体现。

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

## 可复现细节(示意)

**KV 三段划分**(每个 layer/head 的 key 序列,互斥三段):

```
K 序列:   [0 ............... 非窗口前缀 ..............) [最近 W 个 key]
            ├─ K_VL:竖线 top-ρ≈5%(分散挑出)→ 全精度 FP16
            └─ K_Q:其余 ~95% → 低比特 K4V4
               (prefill 算子内为 INT8/FP8 或 INT4/FP4)   K_Win → 全精度 FP16

K = K_VL ∪ K_Win ∪ K_Q(互斥三段)
竖线分数 S(k) = Σ_t a_t,k(列向累计 attention mass,轻量 reduction 增量维护,开销 3%~5%)
```

**Decode 混合精度 kernel(dequantize-in-attention,示意伪代码)**:

```python
# (示意)按论文描述语义概括,非论文原码:单遍 online-softmax,
# 量化块在"即将进 Tensor Core 前"才反量化,不物化全精度中间态。
def hyquant_decode_step(q, kv_blocks):
    m, l, acc = -inf, 0, zeros_like(q)          # online softmax 状态
    for blk in kv_blocks:                       # 扫描 KV cache 块
        k_raw, v_raw = load_hbm(blk)            # 低比特块为 4-bit 打包存储
        if blk.tag in ("VL", "Win"):            # K_VL 竖线 ∪ K_Win 窗口
            k, v = k_raw, v_raw                 #   全精度,直通
        else:                                   # K_Q 低比特块
            k, v = dequant(k_raw), dequant(v_raw)   # 片上即时反量化(dequantize-in-attention)
        s = q @ k.T                             # 分块打分(Tensor Core)
        m_new = max(m, rowmax(s)); p = exp(s - m_new)
        r = exp(m - m_new)                      # online softmax rescale
        acc = acc * r + p @ v
        l = l * r + rowsum(p); m = m_new
    return acc / l
```

> 与 KIVI/KVTuner 的对比:后者先反量化出全精度中间张量再 attend(dequantize-then-attend),多一趟搬运;HyQuant 只在算子内即时反量化,decode kernel 加速 1.32×~3.58×(随前缀增长而增大)。

## 选型要点(何时用)

- **用 HyQuant 类方案**:长上下文(数 K~32K+ 前缀)decode 带宽受限场景;attention 有明显竖线/sink 的 decoder LM(主流模型均有);要压 KV cache 显存又怕均匀 4-bit 精度崩。
- **不必用**:短上下文(decode 未到带宽瓶颈,端到端仅 1.04×~1.17×);需要 bitwise 严格复现的离线评测;非 H100 类硬件(论文未验证)。
- **同族对比**:只要省显存 → KIVI/KVQuant(纯量化,更简单);可接受丢 token 换稀疏加速 → MInference/H2O 系;layer 级混合精度够用 → KVTuner。HyQuant 差异化 = **token 级精度分配 + Prefill/Decode 一体化**。

## 局限

- 长上下文场景收益最明显;短上下文可能提升有限(仍持平全精度)。
- 只在 NVIDIA H100 实验;短上下文时 decode 未完全带宽受限,端到端加速不明显。
- 未测 >Qwen3-32B 大模型;agent / coding 场景未知。

---
*代码:https://github.com/jerrysfls/HyQuant*
