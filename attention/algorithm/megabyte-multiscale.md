---
title: MEGABYTE — 多尺度 decoder 与百万字节级 tokenization-free 序列建模
type: concept
tags: [megabyte, multiscale, byte-level, long-sequence, patch, global-local, sub-quadratic-attention]
source:
  - "pdf: MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers (primary) | arXiv:2305.07185 NeurIPS 2023"
  - "web: NeurIPS 2023 官方 PDF | https://proceedings.neurips.cc/paper_files/paper/2023/file/f8f78f8043f35890181a824e53a57134-Paper-Conference.pdf"
  - "web: OpenReview (NeurIPS 2023) | https://openreview.net/forum?id=JTmO2V9Xpz"
date: 2026-09
---

# MEGABYTE:大 Global 模型跨 patch + 小 Local 模型 patch 内逐字节

## TL;DR

MEGABYTE(Meta AI,NeurIPS 2023)把长字节序列切成固定大小的 **patch**,用**两层嵌套的 decoder-only Transformer** 建模:大 **Global 模型**在 patch 序列上做 self-attention(每 patch 一次,维度 P·D_G),小 **Local 模型**在 patch 内逐字节自回归预测(以 Global 输出为条件)。三个收益:① self-attention 从 O(T²) 降到 O(T²/P² + TP),取 P=T^{1/3} 得 **O(T^{4/3})**;② FFN 从 per-token 变 per-patch——**GPT-3 尺寸下 attention 只占 1.4% FLOPs、FFN 占绝对大头**,同样的算力可以给 Global 模型配 P 倍大的 FFN;③ 解码可 P 路并行,1.5B 的 MEGABYTE 比 350M 的标准 Transformer **生成快 40% 且 PPL 更低**。结果:**tokenization-free** 的字节级模型首次在长上下文语言建模上追平 subword 模型(PG-19 400B 字节规模 test PPL 36.4,vs BlockRecurrent subword 26.5——仍有差距但字节模型此前是 69+),ImageNet 64×64 密度估计以一半算力追平 Perceiver AR SOTA(3.40 bpb),2TB 原始音频直接建模(3.477 bpb)。定位:**架构分解路线**(与 FlashAttention 的内核路线、StreamingLLM 的 cache 路线正交)——不改 attention 数学,改"attention 应用在什么粒度的序列上"。

---

## 1. 机制原理

### 1.1 为什么直接建模字节不行、MEGABYTE 怎么绕开

字节序列百万级长度,两个致命成本:
- self-attention O(T²);
- **更要命的是 per-position FFN**:GPT-3 架构下二次 attention 只占 1.4% FLOPs(Kaplan 近似:前向 ≈ 2mT FLOPs,m 为非嵌入参数)——长序列的大头不是 attention,是每个位置都要过一遍的大 FFN。

MEGABYTE 的分解:
1. **Patch Embedder**:每字节查嵌入表(D_G 维)+ 位置嵌入,然后 reshape 成长 K=T/P、维 P·D_G 的 patch 序列(**无损拼接**,不学 patch 词表);
2. **Global 模型**:标准 decoder-only Transformer(因果掩码),吃 patch 序列——attention 与 FFN 都只跑 T/P 步;
3. **Local 模型**:小 Transformer(维 D_L),patch 内逐字节自回归;每个位置输入 = Global 输出的对应切片过线性投影 w_GL + **前一字节的 local 嵌入**(偏移一位,p=0 用可训练 padding 嵌入)。K 个 patch 的 Local 训练时完全并行。

**为什么不能只用 Global**(像 decoder 版 ViT):patch 的联合分布输出空间是 256^P,P 稍大就不可解;若分解成条件独立 p(x_{t+1}|x_{0..t})···p(x_{t+P}|x_{0..t}),表达力严重受限——**"50% cat / 50% dog"的分布会被迫给 cag、dot 这类杂交串分概率**。Local 模型在 patch 内条件化前字节,只给合法串分概率。**为什么不能只用 Local**:长程信息全靠 Global 提供(变体 §2.3 的 cross-patch attention 允许 Local 看上一 patch 尾部 r 个元素,缓解边界)。

### 1.2 效率推导(可复算)

**Attention**:Global O((T/P)²·…) = O(T²/P²),Local 是 T/P 个长 P 序列 = O(T·P²/P)=O(P·T),合计 **O(T²/P² + T·P)**;取 P=T^{1/3} → **O(T^{4/3})**(比 Sparse/Routing Transformer 的 O(T^{3/2}) 更低;P=T^{1/5} 则 O(T^{8/5}))。任意 1<P<T 都比标准 Transformer 便宜。

**FFN/参数效率**:Global m_g 参数跑 T/P 个位置 + Local m_l 跑 T 个位置 ≈ 2T(m_g/P + m_l) FLOPS;m_g≫m_l 时 ≈ 2T·m_g/P → **同算力可养 P 倍大的模型**。这是"98% FLOPs 在 FFN"观察的直接变现。

**解码并行**:生成一个 patch,MEGABYTE 串行深度 O(L_global + P·L_local),标准 Transformer O(P·L_global + P·L_local);L_global≫L_local 时**推理串行深度降 ~P 倍**。

### 1.3 与"高效 attention"谱系的关系

线性 attention(Performer/Linear Transformer)改 softmax 的核近似,但大规模语言建模上无竞争性结果;Perceiver AR 用 cross-attention 潜变量压缩上下文,是同代最强基线(MEGABYTE 复现其 ImageNet64 3.53 bpb vs 原文 3.54 以校准实现)。MEGABYTE 不改 attention 数学、只做**序列分解**,因此完全复用标准 Transformer 组件(论文明说这是设计目标:继承 Transformer 的 scaling 性质),与 FlashAttention 类内核优化可叠加。

## 2. 可复现细节

### 2.1 前向公式(论文 Figure 2 全套,照此可实现)

```
h_embed_t   = E_global-embed[x_t] + E_pos_t                      # t∈[0,T)
h_global-in_k = E_global-pad (k=0) | h_embed[(k-1)P : kP]        # 右移一 patch 防泄漏
h_global-out  = Transformer_global(h_global-in)                   # 因果掩码
h_local-in_{k,p} = w_GL · h_global-out[k, pD_G:(p+1)D_G]
                 + (E_local-pad (p=0) | E_local-embed[x_{kP+p-1}])
h_local-out  = Transformer_local(h_local-in)                      # K 份并行
p(x_t | x_{0:t}) = softmax(E_local-embedᵀ h_local-out_{k,p}),  t=kP+p
```

要点:Global 输入左端 pad 一个可训练 patch 嵌入、右端去掉最后一个 patch;Local 输入偏移一个字节;**输出层与 local 嵌入共享权重**(softmax(E_local-embed·h))。训练用 Metaseq/PyTorch + fairscale FSDP + 混合精度。

### 2.2 论文实验配置(直接可抄)

- **文本控制实验**(80B 字节、同算力):Transformer 320M/ctx 1024;Perceiver AR 248M/ctx 8192(latent 1024);MEGABYTE **Global 758M / Local 262M,ctx 8192,P=8**;
- **大规模 FLOPs 对比**(Fig.3):P=8,三档 452M/151M、5.8B/604M、170B/3.2B,对 660M~173B 模型与至 1M 上下文全面低于 Transformer/Linear Transformer;
- **PG-19 400B 字节**:同配置放大,Global/Local 输出转 word-level PPL 与 subword 模型对比;
- **图像**:ImageNet 64×64 大模型 Global 2.7B / Local 350M,1.4T tokens,P=12;ImageNet256 P=192(640×640 时序列 1.2M tokens,单次前向);
- **音频**:16kHz/16-bit 原始字节(每秒 32KB),2TB(≈18000 小时),seq len 524288,P=32,batch 32;
- **变体**:因果卷积 patch 编码器(filter 3/5/7 堆叠,缓解 patch 相位不平移不变)、cross-patch attention(Local 拼接上一 patch 尾部 r 个 K/V,RoPE 相对位置,全可微——与 Transformer-XL 的区别)、strided inference(两次前向错开 P/2,各取 patch 前半,推理成本×2 换 PPL)。

## 3. 数据对比

### 3.1 文本(控制算力,bits-per-byte,越低越好;Table 2/7)

| 数据集 | Transformer | Perceiver AR | **MEGABYTE** |
|---|---|---|---|
| PG-19 | 1.057 | 1.104 | **1.000** |
| Stories | 1.064 | 1.070 | **0.978** |
| Books | 1.097 | 1.104 | **1.007** |
| arXiv | 0.816 | 0.791 | **0.678** |
| Code | 0.575 | 0.546 | **0.411** |

### 3.2 PG-19 放大到 400B 字节(word-level PPL,Table 3)

| | 分词 | test PPL |
|---|---|---|
| Transformer XL | subword 32k | 36.3 |
| Compressive Transformer | subword | 33.6 |
| Perceiver AR | subword | 28.9 |
| BlockRecurrent | subword | **26.5** |
| byte Transformer(本文) | 字节 | 69.4 |
| byte Perceiver AR(本文) | 字节 | 88.8 |
| **MEGABYTE** | 字节 | **36.4** |

字节模型此前(小尺度 Block-Recurrent 46.5 / Memorizing 49.5)与 subword 差一大截;MEGABYTE 36.4 已"competitive with subword SOTA"(论文措辞;严格看仍不及 BlockRecurrent 26.5,但对方用 subword+recurrence)。

### 3.3 图像/音频

- **ImageNet 64×64**:MEGABYTE 3.40 bpb = Perceiver AR SOTA,**训练 GPU 时数不到其一半**;同算力多分辨率对比(Table 5):Image64 3.52 / Image256 3.158 / Image640 2.282,全面优于 Transformer(3.62/3.801/2.847)与 Perceiver AR(3.55/3.373/2.345),且 640×640(1.2M tokens)只有 MEGABYTE 能单次前向;
- **音频(2TB 原始字节)**:3.477 bpb vs Perceiver AR 3.543、Transformer 3.567;
- **生成速度**(PG-19 同算力,Global 1.3B + Local 218M vs 350M Transformer):MEGABYTE PPL 更低、生成快 ~40%(intro 数字为 1.5B vs 350M)。

## 4. 实践要点

- **P 的选择**:文本 P=8、高分辨率图像 P=192、音频 P=32——P 是"Local 能扛住的 patch 内建模难度"与"Global 序列缩短倍数"的权衡;复杂度最优 P=T^{1/3},但工程上按模态局部可预测性定;
- **patch 相位问题真实存在**:同一段字节在 patch 内偏移不同会得到不同表示(模型要重学偏移后的"词"),卷积 patch 编码器是论文给的缓解;做字节级模型时先检查这一点;
- **strided inference 是免费的 PPL 补丁**:patch 尾部逐字节损失升高(越靠后越依赖弱 Local),错开 P/2 两次前向取前半,2× 推理成本换稳定质量——评估 bpb 时注意是否用了它;
- **适用判断**:MEGABYTE 赢在**长序列 + 高局部冗余**(字节文本/像素/音频);短序列 subword 场景没有优势。它证明的是"tokenization-free 可行",不是"tokenization 该死"——工业界 LLM 主线仍走 BPE,字节路线后续由 H-Net 等接力;
- **与 KV cache 路线的分工**:MEGABYTE 的 Global cache 只按 patch 存(T/P 条),天然省 KV 显存;与 StreamingLLM(sink+滚窗)不冲突,一个减少 cache 条目数、一个决定逐出策略。

## 溯源

- 主源:论文原文(arXiv v1,2023-05-12),公式/复杂度推导/Table 2/3/5/音频节数字直接核对原文。
- 交叉源:NeurIPS 2023 proceedings 官方 PDF(确认会议版)、OpenReview forum。官方 facebookresearch/megabyte 仓库已不存在(curl 404,2026-09-21 实测),不列入;第三方参考实现 lucidrains/MEGABYTE-pytorch 存活可查。URL 均已 curl 验证(2026-09-21)。
- 关联条目:[attention-is-all-you-need](attention-is-all-you-need.md)(per-position FFN 成本的结构来源)、[flash-attention-implementation](../implementation/flash-attention-implementation.md)(内核路线,与本篇架构路线正交可叠加)、[attention-sinks-streaming](attention-sinks-streaming.md)(cache 路线)。
