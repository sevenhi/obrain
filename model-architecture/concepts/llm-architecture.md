---
title: 通用 LLM 架构:归一化/激活/位置编码的演进与 LLaMA→Llama 2/3 定型线
type: concept
tags: [llama, llm-architecture, rmsnorm, layernorm, swiglu, gelu, rope, alibi, pre-normalization, positional-encoding, scaling-laws, chinchilla, gqa, gpt, open-llm, meta-ai]
source:
  - "pdf: LLaMA: Open and Efficient Foundation Language Models (primary) | arXiv:2302.13971 (2023-02)"
  - "pdf: Llama 2: Open Foundation and Fine-Tuned Chat Models | arXiv:2307.09288 (2023-07) | https://arxiv.org/abs/2307.09288"
  - "pdf: The Llama 3 Herd of Models | arXiv:2407.21783 (2024-07) | https://arxiv.org/html/2407.21783v3 (2026-09-21 实测可达)"
  - "pdf: Training Compute-Optimal Large Language Models (Chinchilla/Hoffmann) | arXiv:2203.15556 (2022-03) | https://arxiv.org/abs/2203.15556"
  - "pdf: On Layer Normalization in the Transformer Architecture (pre-LN, Xiong et al. ICML 2020) | arXiv:2002.04745 | https://arxiv.org/abs/2002.04745"
  - "pdf: Root Mean Square Layer Normalization (Zhang & Sennrich, NeurIPS 2019) | arXiv:1910.07467 | https://arxiv.org/abs/1910.07467"
  - "pdf: GLU Variants Improve Transformer (SwiGLU, Shazeer 2020) | arXiv:2002.05202 | https://arxiv.org/abs/2002.05202"
  - "pdf: RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE, Su et al. 2021) | arXiv:2104.09864 | https://arxiv.org/abs/2104.09864"
  - "pdf: Train Short, Test Long: Attention with Linear Biases (ALiBi, Press et al. ICLR 2022) | arXiv:2108.12409 | https://arxiv.org/abs/2108.12409"
  - "pdf: Language Models are Few-Shot Learners (GPT-3, Brown et al. NeurIPS 2020) | arXiv:2005.14165 | https://arxiv.org/abs/2005.14165"
  - "web: Meta AI 官方博客 Introducing LLaMA | https://ai.meta.com/blog/large-language-model-Llama-meta-ai/"
  - "code: facebookresearch/llama (官方推理参考实现) | https://github.com/facebookresearch/llama"
date: 2026-09
---

# 通用 LLM 架构:组件演进(归一化/激活/位置编码)+ 训练哲学 + GPT→LLaMA→Llama 2/3 定型线

> 本篇原为《LLaMA 架构》,2026-09 移入 model-architecture 域并扩展为**通用 LLM 架构篇**:以 LLaMA 三件套(pre-RMSNorm + SwiGLU + RoPE)为定型锚点,向前补齐每个组件的演进谱系(GPT/GPT-2/GPT-3 基线)、向后延伸到 Llama 2(GQA/4k)与 Llama 3(128K vocab/15T tokens/RoPE θ=500k)。**原 LLaMA 篇全部内容保留**(§3 LLaMA 主体、Table/数据对比)。

## 0. TL;DR(全篇)

今天新起一个 decoder-only LLM,默认配方已被 LLaMA 定型:**pre-RMSNorm 归一化 + SwiGLU FFN((2/3)·4d 内维)+ RoPE 位置编码 + 因果 attention + "小模型多数据"训练哲学**。这套配方不是凭空来的,而是三条组件演进线在 2023 年的汇合:

- **归一化**:post-norm(原 Transformer)→ pre-norm(GPT-2 起,深层稳训)→ RMSNorm(LLaMA,去掉减均值,更快质量无损);
- **激活**:ReLU(原 Transformer FFN)→ GeLU(GPT/BERT)→ SwiGLU(PaLM/LLaMA,门控乘性,经验性更优);
- **位置编码**:正弦绝对(原 Transformer)→ learned 绝对(GPT-2/3)→ RoPE 相对旋转(LLaMA/GPT-NeoX,可外推)→ ALiBi 线性偏置(另一条外推路线,零参数);
- **训练哲学**:Kaplan 堆参数 → Chinchilla 算力最优(~20 tokens/参数)→ LLaMA 推理预算最优(7B×1T tokens,~143 tokens/参数);
- **演进线**:GPT-3(175B,learned PE,pre-LN(GPT-2 引入))→ LLaMA(7-65B,三件套,公开数据)→ Llama 2(2T tokens,4k context,70B 上 GQA)→ Llama 3(15T tokens,128k context,8 KV 头 GQA,tiktoken 128K vocab,RoPE θ=500k)——架构本身几乎不变,**收益主要来自数据规模与质量**(Llama 3 论文自述)。

---

## 1. 归一化演进:post-norm → pre-norm → RMSNorm

下图把三次挪动串成一条链:归一化从残差相加之后移到子层输入,再砍掉减均值。每次改动都只为一个具体问题买单——读任何新架构,先看这一处放在哪。

<div class="archify-figure">
<iframe src="../../../diagrams/ma-llm-norm-lineage.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="归一化三次挪动:post-norm → pre-norm → RMSNorm"></iframe>
</div>


### 1.1 post-norm vs pre-norm

原 Transformer(Vaswani 2017)是 **post-norm**:`x = LayerNorm(x + Sublayer(x))`——归一化在残差相加**之后**。问题:深层网络里梯度必须反复穿过每层的 LayerNorm,残差主干不再"恒等直通",**深层训练不稳,通常需要 learning-rate warmup 才能收敛**(Xiong et al., ICML 2020, arXiv:2002.04745 给出理论分析:pre-LN 的梯度量级与层数无关,post-LN 的梯度在最底层附近被层数放大)。

**pre-norm**(GPT-2 起成为主流):`x = x + Sublayer(LayerNorm(x))`——归一化挪到子层输入,残差主干是干净的恒等通路,梯度可以无衰减地从顶层直达底层。**代价**:最后输出前需要补一个 final norm;理论上 pre-norm 的层间"有效深度"略低(各层贡献更均匀),但工程上稳定压倒一切——GPT-2/3、LLaMA、几乎所有现代 LLM 都是 pre-norm。

> **残差这条线 2025-26 的进展**:pre-norm 之所以稳,是因为残差主干满足**恒等映射**;后续 Hyper-Connections(HC)把残差流扩成 n 条并允许流间混合拿到性能收益,却丢掉了恒等性(深层复合增益失控),mHC 用「双随机矩阵投影(Sinkhorn)」把它约束回 Birkhoff 多面体恢复恒等性。机制与工程见 [`deepseek-infra/concepts/mhc-manifold-hyper-connections.md`](../../deepseek-infra/concepts/mhc-manifold-hyper-connections.md)。

### 1.2 RMSNorm:去掉减均值

LayerNorm 做两件事:减均值(re-centering)+ 除标准差(re-scaling)。Zhang & Sennrich(NeurIPS 2019)发现 **re-centering 不是关键,re-scaling 才是**,于是 RMSNorm 只保留缩放:

```
LayerNorm(x) = (x − μ) / sqrt(σ² + ε) · g + b        # μ=mean(x), σ²=var(x)
RMSNorm(x)   = x / RMS(x) · g,   RMS(x) = sqrt(mean(x²) + ε)   # 不减均值、无 bias b
```

- **为何省计算**:每层少一次均值统计与减法(激活的每元素运算减少);参数量少一个 bias 向量;
- **质量无损**:Zhang & Sennrich 在翻译/语言建模任务上与 LayerNorm 持平或更好;PaLM、LLaMA 起被大规模验证,成为开源 LLM 事实标准;
- LLaMA 的用法:`x + Attention(RMSNorm(x))`、`x + FFN(RMSNorm(x))`(pre-RMSNorm),输出前 final RMSNorm。

---

## 2. 激活函数演进:ReLU → GeLU → SwiGLU

下图对比标准 FFN 与 SwiGLU 的数据路径:标准 FFN 是「升维→激活→压回」;SwiGLU 是两条并行线性投影,一条经过 Swish 当门、一条原样当内容,逐元素相乘后压回。注意内维为什么必须压到 (2/3)·4d。

<div class="archify-figure">
<iframe src="../../../diagrams/ma-llm-swiglu-ffn.workflow.html"
        style="width:100%;height:760px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="SwiGLU 与标准 FFN 的结构差异"></iframe>
</div>


FFN(逐位置前馈层)是 Transformer 里参数最多的组件(`4d×d×2` 每层),激活函数选择直接影响容量利用:

| 激活 | 公式 | 用于 |
|---|---|---|
| ReLU | `max(0, x)` | 原 Transformer FFN |
| GeLU | `x · Φ(x)`(Φ 为标准正态 CDF,常用 tanh 近似) | GPT-2/3、BERT、Llama 2 论文对比基线 |
| Swish/SiLU | `x · sigmoid(βx)` | PaLM 的 SwiGLU 内件 |
| **SwiGLU** | `(Swish_β(xW₁) ⊙ xV) W₂`(⊙ 为逐元素乘) | PaLM、LLaMA、Qwen、Mistral |

**GLU(Gated Linear Unit)族的核心思想**:FFN 不再是"一个线性层 + 激活",而是**两条线性投影,一条做门控**——`GLU(x,W,V) = (xW ⊙ xV)`,信息流经乘法门控,表达力更强。SwiGLU = Swish 激活 + GLU 门控。

- **参数对齐**:SwiGLU 有**三个**权重矩阵(W₁、V、W₂),标准 FFN 只有两个;为保持总参数量可比,**内维取 (2/3)·4d 而非 4d**(LLaMA 做法;实践常再圆整到硬件友好的倍数,如 LLaMA-7B 的 11008 ≈ (2/3)·4·4096 圆整到 256 的倍数);
- **为何更好**:Shazeer(2020, arXiv:2002.05202)在 T5 规模上系统评测 GLU 变体,SwiGLU 一致更优——**收益是经验性的,没有干净理论**(论文自己说 "no pretense of being exhaustive… they work")。乘性门控提供了 ReLU 类"硬截断"没有的平滑选择性;
- LLaMA 原话路线:照抄 PaLM 配方,把原 FFN `max(0,xW₁+b₁)W₂` 换成 `FFN_SwiGLU(x,W₁,V,W₂) = (Swish_β(xW₁) ⊙ xV)W₂`。

---

## 3. LLaMA 主体:三处改进 + 推理预算导向的训练哲学

(以下保留原《llama-architecture.md》全部内容,2026-09 由 attention 域移入。)

### 3.0 LLaMA TL;DR

LLaMA(Meta AI,2023-02)证明**只用公开数据**就能训出第一梯队基础模型:7B~65B 四档,1.0~1.4T tokens。**架构上对原始 Transformer 只做三处修改**——① pre-normalization + **RMSNorm**(GPT-3 路线,稳训练);② ReLU FFN → **SwiGLU**(PaLM 路线,内维取 (2/3)·4d 保持参数量对齐);③ 绝对正弦位置编码 → **RoPE** 每层旋转(GPT-NeoX 路线)。这三件套自此成为开源 LLM 事实标准(Llama 2/3、Qwen、Mistral 等全部沿用)。**训练哲学**是核心贡献:Hoffmann(Chinchilla)scaling law 只优化训练算力,LLaMA 把**推理预算**纳入——给定目标性能,推理最快的模型胜出,所以"小模型 + 远超 Chinchilla 建议量的数据"(7B 训 1T tokens,Chinchilla 建议 7B 只配 ~140B);结果 7B 在 1T tokens 后 loss 仍在降,**LLaMA-13B 多数基准超 GPT-3(175B,10× 大),LLaMA-65B 与 Chinchilla-70B/PaLM-540B 同档**。工程侧:65B 在 2048×A100-80G 上 380 tokens/s/GPU,1.4T tokens ≈ 21 天;attention 用 xformers 的高效因果实现(前向思想来自 Rabe & Staats 的 O(1) 注意力内存,反向来自 FlashAttention),加手工反向的选择性激活重计算 + 模型/序列并行 + 通信重叠。

### 3.1 为什么"小模型多数据":把推理预算写进 scaling law

下图展示 scaling law 目标函数的三次改写:Kaplan 只看参数,Chinchilla 发现参数与数据要同速翻倍,LLaMA 再把「每个 token 的推理成本」塞进目标函数——于是 7B 可以训 1T tokens。

<div class="archify-figure">
<iframe src="../../../diagrams/ma-llm-token-budget.workflow.html"
        style="width:100%;height:730px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="从 Kaplan 到 LLaMA:推理预算如何进入 scaling law"></iframe>
</div>


- Kaplan(2020):性能主要随参数量涨 → 大家拼命堆大模型;
- Hoffmann/Chinchilla(2022, arXiv:2203.15556):固定**训练**算力下,最优是参数量与数据量同速增长(~20 tokens/参数;论文原话 "for every doubling of model size the number of training tokens should also be doubled",训练 400+ 个 70M~16B 模型、5~500B tokens 拟合得出)→ 10B 配 200B tokens;Chinchilla-70B(4× Gopher 数据)在 MMLU 上 67.5%,超 GPT-3/Gopher-280B/MT-NLG-530B,且推理便宜得多;
- **LLaMA 的观察**:Chinchilla 目标函数里没有推理成本。服务规模下,给定目标性能,**首选模型不是训练最快的,而是推理最快的**;大模型训练便宜但每个 token 的推理永远贵,小模型多训数据一次性成本换永久推理便宜。实证支撑:7B 模型训到 1T tokens(143 tokens/参数,约 7× Chinchilla 配比)loss 仍单调下降(Figure 1),且 13B/33B/65B(1.4T)曲线亦未见饱和(Llama 2 论文同样报告 2T tokens 后无饱和迹象)。

### 3.2 三处架构修改(逐条给"为什么")

**① Pre-normalization + RMSNorm**(来源 GPT-3 / Zhang & Sennrich 2019):
- post-norm(原 Transformer `LayerNorm(x+Sublayer(x))`)在深层训练不稳,梯度需在残差主干上反复归一;pre-norm 把归一化挪到子层输入 `x + Sublayer(Norm(x))`,主干恒等直通,深层训练稳定(数学与谱系详见本篇 §1);
- RMSNorm 进一步去掉 LayerNorm 的减均值步骤,只按均方根缩放:`RMSNorm(x) = x / sqrt(mean(x²)+ε) · g`——少一次统计量、更快,质量无损(Zhang & Sennrich 结论,PaLM 起被大规模验证)。

**② SwiGLU FFN**(来源 PaLM / Shazeer 2020):
- 原 FFN `max(0,xW₁+b₁)W₂`;SwiGLU 把 ReLU 换成门控:`FFN_SwiGLU(x,W₁,V,W₂) = (Swish_β(xW₁) ⊙ xV)W₂`,三个权重矩阵(激活族谱系详见本篇 §2);
- 为与标准 FFN 参数量可比,**内维取 (2/3)·4d 而非 4d**(两个 d×d_ff 矩阵变三个);
- 收益是经验性的(GLU 变体一族在 T5 规模上系统性更优),无干净理论——照抄 PaLM 配方。

**③ RoPE 旋转位置嵌入**(来源 GPT-NeoX / Su et al. 2021):
- 删除绝对位置编码;每层对 Q、K 施加与位置相关的旋转:q_m → R_m q_m,k_n → R_n k_n,使 q_m·k_n 只依赖相对距离 m−n(旋转矩阵内积性质);
- 与正弦编码的动机一脉相承(相对位置线性可学),但 RoPE 作用在每层 QK 上而非只加在输入嵌入,且不占参数(完整数学见本篇 §4.3);
- 对长上下文的后续意义:RoPE 的相对性使 NTK/位置插值/YaRN 等上下文扩展成为可能(Llama 2/3 长上下文都靠改 RoPE base);也是 StreamingLLM 能按 cache 内位置重编号的前提(见 [attention-sinks-streaming](../../attention/algorithm/attention-sinks-streaming.md))。

### 3.3 数据:只用公开数据(与 GPT-3/PaLM/Chinchilla 的关键区别)

1.4T tokens 混合(采样比例):English CommonCrawl 67%(CCNet 管线 + Wikipedia 引用页分类器过滤)、C4 15%、Github 4.5%(Apache/BSD/MIT 许可)、Wikipedia 4.5%(20 种拉丁/西里尔语言)、Gutenberg+Books3 4.5%(书级去重 >90% 重叠)、ArXiv 2.5%(去头尾/展开宏)、StackExchange 2%(按分排序)。多数 token 只用一遍(Wikipedia/Books 约 2 epochs)。**BPE 分词用 SentencePiece,数字逐位拆分,未知 UTF-8 回退到字节**。可复现性(公开数据)本身就是贡献——这是它能点燃开源生态的原因(博客视角:democratize access,13B 单 V100 可推理)。

### 3.4 可复现细节

**模型配置(LLaMA Table 2)**

| params | dimension | n heads | n layers | LR | batch | tokens |
|---|---|---|---|---|---|---|
| 6.7B | 4096 | 32 | 32 | 3.0e−4 | 4M | 1.0T |
| 13.0B | 5120 | 40 | 40 | 3.0e−4 | 4M | 1.0T |
| 32.5B | 6656 | 52 | 60 | 1.5e−4 | 4M | 1.4T |
| 65.2B | 8192 | 64 | 80 | 1.5e−4 | 4M | 1.4T |

**优化器配方(可照抄)**:AdamW β₁=0.9, **β₂=0.95**;cosine schedule 终值 = 10% max LR;weight decay 0.1;grad clip 1.0;warmup 2000 步。

**高效实现(LLaMA §2.4,与 FA 域直接衔接)**:
- **因果 attention 用 xformers 实现**:不存注意力权重、不算被因果掩码遮住的 QK 分数;论文明说前向"inspired by **Rabe and Staats (2021)**"(O(1) 内存注意力),反向用 **Dao et al. (2022) = FlashAttention** 的 recomputation——即 FA1 的内存分析前驱(详见 [exact-attention-memory-lineage](../../attention/algorithm/exact-attention-memory-lineage.md));FA 内核全解见 [flash-attention-implementation](../../attention/implementation/flash-attention-implementation.md);
- **选择性激活重计算**:不用 PyTorch autograd,手写 transformer 层反向,只重算便宜激活、保存贵的(线性层输出)——比"全保存"省显存、比"全重算"省时间;
- **模型并行 + 序列并行**(Korthikanti et al. 2022)+ all_reduce 通信与激活计算重叠;
- **实测吞吐**:65B,2048×A100-80G,**380 tokens/s/GPU** → 1.4T tokens ≈ **21 天**。

### 3.5 LLaMA 数据对比

**常识推理 zero-shot(Table 3,准确率 %)**

| | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA |
|---|---|---|---|---|---|---|---|---|
| GPT-3 175B | 60.5 | 81.0 | – | 78.9 | 70.2 | 68.8 | 51.4 | 57.6 |
| Chinchilla 70B | 83.7 | 81.8 | 51.3 | 80.8 | 74.9 | – | – | – |
| PaLM 540B | 88.0 | 82.3 | – | 83.4 | 81.1 | 76.6 | 53.0 | 53.4 |
| **LLaMA-13B** | 78.1 | 80.1 | 50.4 | 79.2 | 73.0 | 74.8 | 52.7 | 56.4 |
| **LLaMA-65B** | 85.3 | 82.8 | 52.3 | 84.2 | 77.0 | 78.9 | 56.0 | 60.2 |

**LLaMA-13B 在 8 项中多数超 GPT-3(10× 参数)**;LLaMA-65B 除 BoolQ 外全面超 Chinchilla-70B,除 BoolQ/WinoGrande 外超 PaLM-540B。

**闭卷 QA(Table 4/5,exact match)**:
- NaturalQuestions 64-shot:LLaMA-65B **39.9** vs PaLM-540B 39.6、Chinchilla-70B 35.5;
- TriviaQA zero-shot:LLaMA-65B **68.2** vs Chinchilla-70B 55.4(Gopher-280B 43.5);
- 13B 即与 GPT-3/Chinchilla 同档(TriviaQA 0-shot 56.6 vs Chinchilla 55.4),单 V100 可推理。

**MMLU(Table 6 语境)**:5-shot MMLU LLaMA-65B 63.4——低于 PaLM-540B(69.3)/Chinchilla(67.5),论文归因训练数据不含足够学术文本(ArXiv 只占 2.5%);这是"纯公开数据"路线当时的短板,Llama 2/3 靠扩数据补上。

---

## 4. 位置编码演进:正弦 → learned → RoPE → ALiBi

下图并列四条路线,判断只看两个轴:超出训练长度还能不能用、要不要占参数。RoPE 与 ALiBi 走的是「改造 QK 内积」这条路,所以能外推;两种绝对编码不能。

<div class="archify-figure">
<iframe src="../../../diagrams/ma-llm-positional-encoding.workflow.html"
        style="width:100%;height:820px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="位置编码四条路线对比"></iframe>
</div>


### 4.1 正弦绝对编码(原 Transformer, 2017)

`PE(pos,2i)=sin(pos/10000^{2i/d})`、`PE(pos,2i+1)=cos(...)`,波长 2π → 10000·2π 几何级数。选择理由:对任意固定偏移 k,PE_{pos+k} 可表示为 PE_pos 的线性函数 → 容易学"按相对位置 attend";理论上可外推到比训练更长的序列(AIAYN Table 3 行 E:learned 与正弦几乎同分,dev BLEU 25.7 vs 25.8)。

### 4.2 learned 绝对编码(GPT-2/3)

直接学一张 `[max_len, d]` 的位置嵌入表,加在 token 嵌入上(GPT-3:learned positional embeddings,max context 2048)。**简单、训练内表现好,但硬上限**——超出 max_len 没有嵌入可查,**无法外推**;且只在输入层注入一次,深层位置信息衰减。这是 GPT-3 时代 context 卡在 2k 的架构原因之一。

### 4.3 RoPE 旋转位置嵌入(Su et al. 2021, arXiv:2104.09864)

**核心思想**:不加位置向量,而是**用旋转矩阵把绝对位置编码进 Q/K,使内积只依赖相对距离**。把 d 维向量按相邻两维配对成 d/2 个二维平面,位置 m 的 q 在第 j 个平面旋转角度 mθ_j:

```
θ_j = base^(−2j/d)      # base 常取 10000,即 RoPE θ
R_m = block_diag( Rot(mθ_1), Rot(mθ_2), … )    # Rot(φ) = [[cos φ, −sin φ],[sin φ, cos φ]]
q_m → R_m q_m ,   k_n → R_n k_n
⟹ (R_m q)·(R_n k) = q·R_{n−m} k        # 旋转矩阵正交:R_mᵀ R_n = R_{n−m}
```

- **性质**(RoFormer 论文):用绝对位置的旋转实现**显式相对位置依赖**;不占参数;随相对距离增大内积有**远程衰减**趋势;每层 QK 都注入位置(不只在输入层);序列长度灵活;
- **长上下文扩展的抓手**:改 base(θ)即可拉伸波长——Llama 3 把 θ 从 10000 提到 **500,000** 以支持 32k+ 上下文(Xiong et al. 2023);NTK-aware scaling/位置插值(PI)/YaRN 都是在 RoPE 频率上动手术;
- RoPE 的相对性也是 StreamingLLM 按 cache 内位置重编号的前提(见 [attention-sinks-streaming](../../attention/algorithm/attention-sinks-streaming.md))。

### 4.4 ALiBi 线性偏置(Press et al. ICLR 2022, arXiv:2108.12409)

**另一条外推路线:完全不加位置嵌入**,直接在 attention 分数上减去与距离成正比的惩罚:

```
softmax( q·kᵀ − m·|i−j| )      # m 为每个 head 预设的固定斜率(几何级数,不学习)
```

- 1.3B 模型在 1024 长度训练,可直接外推到 2048,PPL 与"在 2048 上训练的正弦模型"持平,且**训练快 11%、省 11% 内存**;归纳偏置偏向近因(recency);
- 与 RoPE 的取舍:ALiBi 零参数、外推强,但把"距离衰减"硬编码进分数(远距离信息被系统性压制);RoPE 更灵活(可学任意相对模式)、生态更大(LLaMA/Qwen/Mistral 全用它)。**LLaMA 系选 RoPE,ALiBi 见于 BLOOM/MPT 等**——两条相对位置路线并存。

---

## 5. 演进线:GPT → LLaMA → Llama 2/3

下图把四代模型放在同一条时间轴上,按「架构组件」和「数据与上下文」两条泳道对照——能直接看出三件套自 LLaMA 起没有再动,所有增长都在 tokens 量、context 长度和 tokenizer 词表上。

<div class="archify-figure">
<iframe src="../../../diagrams/ma-llm-evolution.timeline.html"
        style="width:100%;height:760px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="GPT-3 → LLaMA → Llama 2/3 演进时间轴"></iframe>
</div>


| | GPT-3 (2020) | LLaMA 1 (2023-02) | Llama 2 (2023-07) | Llama 3 (2024-07) |
|---|---|---|---|---|
| 参数档 | 175B(单档) | 7/13/33/65B | 7/13/70B(34B 未发) | 8/70/405B |
| 归一化 | pre-LN(LayerNorm) | pre-RMSNorm | pre-RMSNorm | pre-RMSNorm |
| 激活 | GeLU | SwiGLU | SwiGLU | SwiGLU |
| 位置编码 | learned 绝对 | RoPE (θ=10k) | RoPE | RoPE **θ=500,000** |
| attention | MHA(交替 dense/局部带状) | MHA | MHA;**70B 用 GQA** | **GQA,8 个 KV 头(全系)** |
| context | 2048 | 2048 | **4096(2×)** | **预训练 8192,长上下文档 128K** |
| tokenizer | BPE 50k | SentencePiece BPE 32k,数字逐位拆 | 同 LLaMA 1 | **tiktoken 基 128K vocab**(100K tiktoken + 28K 非英语),压缩率 3.17→3.94 字符/token |
| 训练 tokens | ~300B | 1.0-1.4T | **2.0T(+40%)** | **~15T(Llama 2 的 7.5×)** |
| 训练哲学 | Kaplan 大参数 | 推理预算小模型多数据 | 数据质量+RLHF 对话 | 数据规模/质量+多语+代码,405B 用 3.8×10²⁵ FLOPs(Llama 2 最大版的 ~50×) |
| 许可 | 闭源 API | 研究许可(非 OSI) | **可商用**(月活 <7 亿) | Llama 3 Community License |

关键差异的"为什么":

- **GQA(Grouped-Query Attention, Ainslie et al. 2023)**:多个 query 头共享一组 K/V 头(MHA 与 MQA 的折中)。**动机是推理侧 KV cache**:70B 上 GQA 使 KV cache 缩小(8 组 vs 64 头),解码时内存与带宽大省,质量接近 MHA;Llama 3 全系用 8 个 KV 头(论文自述:"improve inference speed and reduce the size of key-value caches during decoding")——直接对接推理引擎的 KV cache 管理(见 [`vllm/`](../../vllm/index.md));
- **Llama 3 架构几乎不动**:论文原话 "we did not make any major changes ... in terms of model architecture; our performance gains are primarily driven by improvements in data quality and diversity as well as by increased training scale"——组件定型后,**scaling 的主战场从架构转到数据**;
- **Llama 3 训练三阶段**:初始预训练(8k context)→ 长上下文预训练(~800B tokens,context 六阶段从 8K 渐进拉到 128K)→ 退火(最后 40M tokens LR 线性到 0 + 上采样高质量数据);
- **文档内 attention mask**:Llama 3 新增"同序列内跨文档不互相 attend"的 mask——标准预训练影响有限,但**超长序列继续预训练时重要**(拼接文档互相污染会被长上下文放大);
- **数字处理**:LLaMA 1/2 的"数字逐位拆分"BPE 影响算术(逐位可学但长数字变长),Llama 3 换 tiktoken 时改了这一点;
- **许可演进**:LLaMA 1 研究许可 → Llama 2 可商用 → Llama 3 Community License,"开源"用词要注意(LLaMA 1 非 OSI 开源,官方博客自己也写 publicly releasing)。

---

## 6. 实践要点(通用架构 checklist)

- **三件套是默认起点**:今天新起一个 decoder-only LLM,pre-RMSNorm + SwiGLU((2/3)4d)+ RoPE 就是 LLaMA 定型的答案;读任何"XX 架构"先对照这三处差异(外加:GQA 与否、context 长度、tokenizer);
- **β₂=0.95 不是笔误**:大模型训练常用 0.95(而非 Adam 默认 0.999)以缩短二阶矩记忆、稳大步长训练,GPT-3/LLaMA 同款;
- **配预算先问推理**:Chinchilla 最优 ≠ 部署最优;若模型要服务海量请求,倾向更小模型 + 更多 tokens(LLaMA 的核心教训,后来 7B/8B 档开源繁荣的底层逻辑);推理侧 KV cache 大时选 GQA 模型;
- **长上下文=改 RoPE base + 渐进拉长**:θ 10k→500k(Llama 3)、分阶段扩 context(8K→128K,800B tokens)——不要指望 learned PE 外推;
- **tokenizer 是被低估的架构组件**:vocab 32k→128k、压缩率 3.17→3.94 字符/token,等于"同样算力多读 25% 文本";数字拆分策略直接影响算术能力;
- **数字逐位拆分**的 BPE 细节影响算术能力(逐位可学但长数字变长),Llama 3 换 tiktoken 时改了这一点;
- **许可注意**:LLaMA 1 是"研究许可发布"(非 OSI 开源),Llama 2 起才放宽商用——引用"开源"时用词要准;
- **与量化/推理域的衔接**:LLaMA 权重是 GPTQ/AWQ 等量化的最常用测试床(见 [`quantization/`](../../quantization/index.md) 与 [`vllm/concepts/quantization-gptq-awq-fp8.md`](../../vllm/concepts/quantization-gptq-awq-fp8.md));其 KV cache 行为(sink、RoPE 重编号)见 [attention-sinks-streaming](../../attention/algorithm/attention-sinks-streaming.md);attention 内核实现见 [flash-attention-implementation](../../attention/implementation/flash-attention-implementation.md)。

## 溯源

- **LLaMA 主源**:arXiv:2302.13971 v1(2023-02-27) https://arxiv.org/abs/2302.13971,Table 2-6/§2 配方数字直接核对原文;§3.5 MMLU 63.4 为 Table 6 值(论文 v1)。交叉源:Meta AI 官方博客(发布口径/democratize 叙事,2023-02-24,URL 大小写敏感——小写 llama 路径 curl 000 失败,官方 `Llama` 大写路径 200)、facebookresearch/llama 官方仓库、RoPE/RMSNorm/SwiGLU 三篇原始论文(组件出处核对)。全部 URL curl 验证存活(2026-09-21)。
- **Llama 2**(arXiv:2307.09288,2026-09-21 web 核对):2T tokens(+40%)、context 2k→4k、70B GQA、pre-RMSNorm+SwiGLU+RoPE 沿用、"2T tokens 后训练 loss 无饱和"、SentencePiece 同 LLaMA 1。
- **Llama 3**(arXiv:2407.21783v3 HTML 版,2026-09-21 web 核对):~15T tokens、GQA 8 KV 头、128K vocab(tiktoken 100K+28K,压缩率 3.17→3.94)、RoPE θ=500,000、Table 3 超参(8B/70B/405B:层数 32/80/126、维度 4096/8192/16384、FFN 14336/28672/53248、头 32/64/128、KV 头均 8)、三阶段训练(8k→128K 六阶段渐进,长上下文档 ~800B tokens;退火 40M tokens)、文档间 attention mask、405B 训练算力 3.8×10²⁵ FLOPs(~50× Llama 2)。
- **Chinchilla**(arXiv:2203.15556):400+ 模型(70M~16B)× 5~500B tokens;算力最优=参数与数据同倍率翻倍(~20 tokens/参数);Chinchilla-70B 4× Gopher 数据,MMLU 67.5%。
- **组件论文**:pre-LN 理论(Xiong et al., arXiv:2002.04745, ICML 2020)、RMSNorm(arXiv:1910.07467, NeurIPS 2019)、SwiGLU(Shazeer, arXiv:2002.05202)、RoPE(Su et al., arXiv:2104.09864)、ALiBi(Press et al., arXiv:2108.12409, ICLR 2022;1024→2048 外推、快 11%、省 11% 内存)、GPT-3(Brown et al., arXiv:2005.14165)。
- 关联条目:[attention-is-all-you-need](../../attention/algorithm/attention-is-all-you-need.md)(被改的原始基线:正弦 PE/post-norm/ReLU FFN 的出处)、[exact-attention-memory-lineage](../../attention/algorithm/exact-attention-memory-lineage.md)(xformers 高效 attention 的算法来源:Rabe & Staats 前向 + FA1 反向)、[megabyte-multiscale](../../attention/algorithm/megabyte-multiscale.md)(同期 Meta 的另一条 scaling 探索)。
