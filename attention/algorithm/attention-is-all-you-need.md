---
title: Attention Is All You Need — Transformer 架构原始设计
type: concept
tags: [transformer, self-attention, qkv, multi-head, positional-encoding, scaled-dot-product, nmt]
source:
  - "pdf: Attention Is All You Need (primary) | arXiv:1706.03762 NIPS 2017"
  - "web: The Illustrated Transformer — Jay Alammar | https://jalammar.github.io/illustrated-transformer/"
  - "web: The Annotated Transformer — Harvard NLP | https://nlp.seas.harvard.edu/annotated-transformer/"
date: 2026-09
---

# Transformer:纯注意力序列转导架构(2017)

## TL;DR

Transformer(原论文《Attention Is All You Need》,Vaswani et al., NIPS 2017)是**第一个完全抛弃 recurrence 和 convolution、仅靠 self-attention 构建的序列转导模型**。核心组件三件套:**缩放点积注意力**(softmax(QKᵀ/√d)V)+ **多头机制**(h 份低维投影并行,弥补单头平均化的分辨率损失)+ **位置编码**(正弦函数注入顺序信息)。收益:任意两位置的路径长度 O(1)(RNN 是 O(n))、每层可完全并行(sequential operations O(1))。WMT14 EN-DE 28.4 BLEU / EN-FR 41.8 BLEU,当时 SOTA,且训练成本只有先前最佳模型的零头(big 模型 2.3×10¹⁹ FLOPs,8×P100 训 3.5 天)。后续 BERT/GPT/LLaMA/vision 全部是它的直系后代。

---

## 1. 机制原理:为什么"只要注意力"就够

### 1.1 被解决的两个根本问题

原论文用三个指标比较层类型(Table 1),这是理解 Transformer 动机的钥匙:

| 层类型 | 每层复杂度 | 最少串行操作 | 最长路径 |
|---|---|---|---|
| Self-Attention | O(n²·d) | **O(1)** | **O(1)** |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k(n)) |
| Restricted Self-Attention(邻域 r) | O(r·n·d) | O(1) | O(n/r) |

- **RNN 的病灶是串行**:h_t 依赖 h_{t-1},训练样本内无法并行,长序列下 batch 又受显存限制 → 训练慢、长程依赖难学(梯度要沿 O(n) 路径传播)。
- **CNN 的病灶是路径长度**:kernel 宽 k<n 时,连接任意两位置需要堆 O(n/k) 层(膨胀卷积 O(log_k(n)) 层),远距离依赖的学习信号同样被路径拉长。
- **Self-attention 一次全连接**:任意两位置一层内直接交互,路径 O(1);代价是 O(n²·d) 复杂度——当 n < d(机器翻译的句长 vs 512 维,通常成立)时反而比 RNN 的 O(n·d²) 便宜。**这就是"quadratic attention 当年不是缺点而是卖点"的历史语境**;n≫d 后 O(n²) 才成为主要矛盾,催生整个高效注意力研究线(见本篇 §5)。

### 1.2 缩放点积注意力:为什么要除 √d_k

Attention(Q,K,V) = softmax(QKᵀ/√d_k)V。Q 是"我要找什么",K 是"我是什么",V 是"我携带的信息";点积算 query 与每个 key 的兼容度,softmax 归一成权重,对 V 加权求和。

**缩放因子 √d_k 的推导**(原论文脚注 4):设 q、k 分量独立、均值 0 方差 1,则 q·k = Σqᵢkᵢ 的方差是 d_k。d_k 大时点积量级 ~√d_k,把 softmax 推进饱和区(梯度极小)。**除以 √d_k 把方差拉回 1**,保住梯度。对比 additive attention(前馈网络算兼容度):理论复杂度相近,但点积可以直接用高度优化的 matmul 库实现,**实践中快得多、省内存得多**——这是工程选择压倒理论优雅的经典案例。d_k 小时两者性能相当,d_k 大时不缩放的点积反而输给 additive(饱和),缩放是补丁。

### 1.3 多头:为什么要 h 份 64 维而不是 1 份 512 维

单头 attention 的输出是 value 的**加权平均**——平均会抹掉信息,模型只能表达"一种"关注模式。多头把 Q/K/V 用 h 组不同的学习投影各自降到 d_k=d_v=d_model/h 维,并行做 h 次 attention,拼接后再过 Wᴼ 投影:

MultiHead(Q,K,V) = Concat(head₁,…,head_h)Wᴼ,headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)

- **动机**:让模型"同时关注来自不同表示子空间、不同位置的信息"(原论文 §3.2.2)。可视化(原论文附录 Figure 3-5)证实:不同头确实学会不同任务——有的头做长距离短语绑定("making…more difficult"),有的做指代消解("its"  sharp 地指向先行词),有的头行为对应句法结构。
- **成本不变**:h=8、d_k=d_v=64 时总计算量 ≈ 单个全维度头,多头几乎免费。
- **消融数据**(Table 3 行 A,newstest2013 dev BLEU):h=1 → 24.9(比最优低 0.9);h=8 → 25.8(最优);h=32(d_k=16)→ 25.4。**头太少表达不够,太多则每头维度过低、兼容度函数算不准**——中间有最优区间。行 B 进一步证实:压缩 d_k 伤质量("判断兼容不容易,可能需要比点积更精细的兼容函数")。

### 1.4 位置编码:没有 recurrence/convolution 后顺序从哪来

self-attention 本身是置换等变的,必须显式注入位置。原论文用固定正弦编码:

PE(pos,2i) = sin(pos/10000^(2i/d_model)),PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

波长构成 2π → 10000·2π 的几何级数。**选择理由**:对任意固定偏移 k,PE_{pos+k} 可表示为 PE_pos 的线性函数 → 模型容易学会"按相对位置 attend"。消融(Table 3 行 E):learned positional embedding 与正弦几乎同分(dev BLEU 25.7 vs 25.8),选正弦是因为**理论上可外推到比训练更长的序列**(后续 RoPE/ALiBi 的相对位置编码路线取代了它,见 [llm-architecture](../../model-architecture/concepts/llm-architecture.md) 篇)。

### 1.5 整体结构:encoder-decoder + 三种 attention 用法

- **Encoder**:N=6 层,每层两个子层(多头 self-attention → position-wise FFN),每个子层套残差 + LayerNorm:`LayerNorm(x + Sublayer(x))`(post-norm;LLaMA 时代改为 pre-norm + RMSNorm)。全模型子层与 embedding 统一输出 d_model=512,残差才能直通。
- **Decoder**:N=6 层,比 encoder 多第三个子层——对 encoder 输出做 cross-attention(Q 来自 decoder 前一层,K/V 来自 encoder)。decoder self-attention 加**因果掩码**(非法连接置 −∞ 再 softmax),配合输出 embedding 右移一位,保证位置 i 的预测只依赖 <i 的已知输出(autoregressive 性质)。
- **三种 attention 用法**(§3.2.3):encoder self-attention(全双向)、decoder masked self-attention(因果)、encoder-decoder cross-attention(解码端查源端全序列)。
- **FFN**:每位置独立同参的两层全连接 + ReLU:`FFN(x)=max(0,xW₁+b₁)W₂+b₂`,内层 d_ff=2048(4×d_model)。等价于两个 kernel=1 的卷积。参数量上 FFN 远大于 attention,这个比例问题后来被 MEGABYTE 抓住做文章(见 megabyte-multiscale 篇)。
- **权重共享**:输入/输出 embedding 与 pre-softmax 线性层共用一个矩阵;embedding 层乘 √d_model。

## 2. 可复现细节

### 2.1 base/big 配置(原论文 Table 3)

| | N | d_model | d_ff | h | d_k=d_v | P_drop | ϵ_ls | steps | params |
|---|---|---|---|---|---|---|---|---|---|
| base | 6 | 512 | 2048 | 8 | 64 | 0.1 | 0.1 | 100K | 65M |
| big | 6 | 1024 | 4096 | 16 | 64 | 0.3 | 0.1 | 300K | 213M |

### 2.2 训练配方(可照抄的关键超参)

- **Adam** β₁=0.9, **β₂=0.98**(比常用 0.999 小,配合 warmup 稳住早期二阶矩), ϵ=10⁻⁹;
- **Noam 学习率**:lrate = d_model^(−0.5)·min(step^(−0.5), step·warmup^(−1.5)),warmup_steps=4000——先线性升后按 1/√step 降;
- **batch 按 token 数定**(近似等长句分组):每 batch ~25000 source + 25000 target tokens,不是按句子数;
- **正则三件套**:residual dropout 0.1(EN-FR big 用 0.1,EN-DE big 用 0.3)+ embedding/PE 之和上 dropout + label smoothing 0.1(牺牲 PPL 换 BLEU/准确率);
- **推理**:checkpoint 平均(base 取最后 5 个、间隔 10 分钟;big 取最后 20 个),beam=4,长度惩罚 α=0.6;
- **硬件/时长**:8×P100;base 每步 0.4s × 100K 步 ≈ 12 小时;big 每步 1.0s × 300K 步 ≈ 3.5 天。

### 2.3 最小 PyTorch 核心(单头缩放点积 + 因果掩码,~10 行)

```python
import torch, math

def attention(q, k, v, mask=None):        # q,k,v: (B, S, D)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)   # (B,S,S)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    return torch.softmax(scores, -1) @ v

causal = torch.tril(torch.ones(S, S))     # 因果掩码: 下三角为 1
# 多头 = reshape (B,S,h,d_k) 后在 h 维并行调用,输出 concat 过 W_O
```

完整可运行参考实现:Harvard NLP 的 [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)(PyTorch 逐段对应原论文,已 curl 验证存活);直观图解见 Jay Alammar 的 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)。原论文官方代码在 tensor2tensor(github.com/tensorflow/tensor2tensor,已归档,后继为 Trax)。

## 3. 数据对比

### 3.1 WMT14 翻译(原论文 Table 2)

| 模型 | EN-DE BLEU | EN-FR BLEU | 训练 FLOPs (EN-DE) |
|---|---|---|---|
| GNMT+RL | 24.6 | 39.92 | 2.3×10¹⁹ |
| ConvS2S | 25.16 | 40.46 | 9.6×10¹⁸ |
| MoE | 26.03 | 40.56 | 2.0×10¹⁹ |
| GNMT+RL Ensemble | 26.30 | 41.16 | 1.8×10²⁰ |
| ConvS2S Ensemble | 26.36 | 41.29 | 7.7×10¹⁹ |
| **Transformer (base)** | **27.3** | 38.1 | **3.3×10¹⁸** |
| **Transformer (big)** | **28.4** | **41.8** | 2.3×10¹⁹ |

要点:**base 模型(3.3×10¹⁸ FLOPs)就超过了所有先前单模型和集成模型**,成本是 GNMT+RL ensemble 的 ~1/50;big 再 +1.1 BLEU。注意 v5 版正文 §6.1 写 EN-FR big 为 41.0、Table 2 写 41.8——两版本数字在 arXiv 修订间有出入,引用时以 Table 2 的 41.8(摘要同款)为准并标注版本。

### 3.2 泛化性:成分句法分析(Table 4)

4 层、d_model=1024 的 Transformer 几乎不调参,在 WSJ 上 WSJ-only 判别式 91.3 F1、半监督 92.7 F1,超过除 RNNG(93.3)外的全部先前模型——证明架构不是翻译特化。

### 3.3 消融小结(Table 3,dev BLEU)

- 头数:h=1→24.9,h=8→25.8,h=32→25.4(见 §1.3);
- d_k 压缩:64→16 掉 0.4 BLEU(兼容度函数需要维度);
- 模型规模:d_model 1024/d_ff 4096 → 26.0(更大更好);
- dropout:0 → 24.6,**0.1 → 25.8**(dropout 对防过拟合非常关键);
- label smoothing:去掉(0.0)→ 25.3,加 0.1 → 25.8(PPL 变差但 BLEU 变好)。

## 4. 实践要点

- **√d_k 缩放不可省**:大 d_k 无缩放会把 softmax 推进饱和区,梯度消失、训练发散。现代实现(QK-Norm、LASER 的 per-dim temperature)都是在同一问题上继续打补丁(见 attention-variants-laser-spatten 篇)。
- **d_ff/d_model ≈ 4× 的配比**是原论文定下的惯例(512/2048、1024/4096),后续 GPT/LLaMA 沿用(LLaMA 因 SwiGLU 三矩阵改为 (2/3)·4d)。
- **post-norm vs pre-norm**:原论文是 `LayerNorm(x+Sublayer(x))`(post-norm),深层训练不稳;GPT-3 起主流改 pre-norm,LLaMA 再换 RMSNorm——读老代码注意两种残差结构不可混。
- **beam search 超参对翻译质量影响显著**:beam=4 + α=0.6 + checkpoint 平均,三件套是当年复现 SOTA 的必要条件,只训不平均会掉 ~0.5 BLEU 量级。
- **batch 按 token 预算而非样本数**:25000+25000 tokens/batch 的思路延续到今天 LLM 训练(LLaMA 按 4M tokens/batch)。
- **O(n²) 是留给后人的账单**:原论文已预告 restricted self-attention(邻域 r)作为长序列出路。今天的高效注意力谱系——FlashAttention(IO-aware 精确算法,见 [flash-attention-implementation](../implementation/flash-attention-implementation.md))、StreamingLLM(sink+滚窗,见 [attention-sinks-streaming](attention-sinks-streaming.md))、稀疏剪枝硬件(SpAtten,见 [attention-variants-laser-spatten](attention-variants-laser-spatten.md))、多尺度分解(MEGABYTE,见 [megabyte-multiscale](megabyte-multiscale.md))——全部是在还这笔账。

## 溯源

- 主源:论文原文(arXiv v5,2017-12-06),正文/表格数字均直接核对原文。
- 交叉源:Illustrated Transformer(图解心智模型)、Annotated Transformer(PyTorch 复现参考),URL 已 curl 验证存活(2026-09-21)。
- 关联条目:[attention-sinks-streaming](attention-sinks-streaming.md)(softmax 分母视角的延伸)、[llm-architecture](../../model-architecture/concepts/llm-architecture.md)(三项架构改进)、[flash-attention-implementation](../implementation/flash-attention-implementation.md)(O(n²) 内存问题的精确解)。
