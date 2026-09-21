---
title: LLaMA 架构 — RMSNorm/SwiGLU/RoPE 与"小模型多数据"训练哲学
type: concept
tags: [llama, rmsnorm, swiglu, rope, pre-normalization, scaling-laws, open-llm, meta-ai]
source:
  - "pdf: LLaMA: Open and Efficient Foundation Language Models (primary) | arXiv:2302.13971 (2023-02) | ~/zhanbin/library/papers/LLaMA: Open and Efficient Foundation Language Models.pdf"
  - "web: Meta AI 官方博客 Introducing LLaMA | https://ai.meta.com/blog/large-language-model-Llama-meta-ai/"
  - "code: facebookresearch/llama (官方推理参考实现) | https://github.com/facebookresearch/llama"
  - "pdf: RoPE 原论文 | arXiv:2104.09864 (RoFormer)"
  - "pdf: RMSNorm 原论文 | arXiv:1910.07467"
  - "pdf: SwiGLU (GLU Variants Improve Transformer) | arXiv:2002.05202"
date: 2026-09
---

# LLaMA:Transformer 三处改进 + 推理预算导向的训练规模选择

## TL;DR

LLaMA(Meta AI,2023-02)证明**只用公开数据**就能训出第一梯队基础模型:7B~65B 四档,1.0~1.4T tokens。**架构上对原始 Transformer 只做三处修改**——① pre-normalization + **RMSNorm**(GPT-3 路线,稳训练);② ReLU FFN → **SwiGLU**(PaLM 路线,内维取 (2/3)·4d 保持参数量对齐);③ 绝对正弦位置编码 → **RoPE** 每层旋转(GPT-NeoX 路线)。这三件套自此成为开源 LLM 事实标准(Llama 2/3、Qwen、Mistral 等全部沿用)。**训练哲学**是核心贡献:Hoffmann(Chinchilla)scaling law 只优化训练算力,LLaMA 把**推理预算**纳入——给定目标性能,推理最快的模型胜出,所以"小模型 + 远超 Chinchilla 建议量的数据"(7B 训 1T tokens,Chinchilla 建议 7B 只配 ~140B);结果 7B 在 1T tokens 后 loss 仍在降,**LLaMA-13B 多数基准超 GPT-3(175B,10× 大),LLaMA-65B 与 Chinchilla-70B/PaLM-540B 同档**。工程侧:65B 在 2048×A100-80G 上 380 tokens/s/GPU,1.4T tokens ≈ 21 天;attention 用 xformers 的高效因果实现(前向思想来自 Rabe & Staats 的 O(1) 注意力内存,反向来自 FlashAttention),加手工反向的选择性激活重计算 + 模型/序列并行 + 通信重叠。

---

## 1. 机制原理

### 1.1 为什么"小模型多数据":把推理预算写进 scaling law

- Kaplan(2020):性能主要随参数量涨 → 大家拼命堆大模型;
- Hoffmann/Chinchilla(2022):固定**训练**算力下,最优是参数量与数据量同速增长(~20 tokens/参数)→ 10B 配 200B tokens;
- **LLaMA 的观察**:Chinchilla 目标函数里没有推理成本。服务规模下,给定目标性能,**首选模型不是训练最快的,而是推理最快的**;大模型训练便宜但每个 token 的推理永远贵,小模型多训数据一次性成本换永久推理便宜。实证支撑:7B 模型训到 1T tokens(143 tokens/参数,约 7× Chinchilla 配比)loss 仍单调下降(Figure 1),且 13B/33B/65B(1.4T)曲线亦未见饱和。

### 1.2 三处架构修改(逐条给"为什么")

**① Pre-normalization + RMSNorm**(来源 GPT-3 / Zhang & Sennrich 2019):
- post-norm(原 Transformer `LayerNorm(x+Sublayer(x))`)在深层训练不稳,梯度需在残差主干上反复归一;pre-norm 把归一化挪到子层输入 `x + Sublayer(Norm(x))`,主干恒等直通,深层训练稳定;
- RMSNorm 进一步去掉 LayerNorm 的减均值步骤,只按均方根缩放:`RMSNorm(x) = x / sqrt(mean(x²)+ε) · g`——少一次统计量、更快,质量无损(Zhang & Sennrich 结论,PaLM 起被大规模验证)。

**② SwiGLU FFN**(来源 PaLM / Shazeer 2020):
- 原 FFN `max(0,xW₁+b₁)W₂`;SwiGLU 把 ReLU 换成门控:`FFN_SwiGLU(x,W₁,V,W₂) = (Swish_β(xW₁) ⊙ xV)W₂`,三个权重矩阵;
- 为与标准 FFN 参数量可比,**内维取 (2/3)·4d 而非 4d**(两个 d×d_ff 矩阵变三个);
- 收益是经验性的(GLU 变体一族在 T5 规模上系统性更优),无干净理论——照抄 PaLM 配方。

**③ RoPE 旋转位置嵌入**(来源 GPT-NeoX / Su et al. 2021):
- 删除绝对位置编码;每层对 Q、K 施加与位置相关的旋转:q_m → R_m q_m,k_n → R_n k_n,使 q_m·k_n 只依赖相对距离 m−n(旋转矩阵内积性质);
- 与正弦编码的动机一脉相承(相对位置线性可学),但 RoPE 作用在每层 QK 上而非只加在输入嵌入,且不占参数;
- 对长上下文的后续意义:RoPE 的相对性使 NTK/位置插值/YaRN 等上下文扩展成为可能(Llama 2/3 长上下文都靠改 RoPE base);也是 StreamingLLM 能按 cache 内位置重编号的前提(见 [attention-sinks-streaming](attention-sinks-streaming.md))。

### 1.3 数据:只用公开数据(与 GPT-3/PaLM/Chinchilla 的关键区别)

1.4T tokens 混合(采样比例):English CommonCrawl 67%(CCNet 管线 + Wikipedia 引用页分类器过滤)、C4 15%、Github 4.5%(Apache/BSD/MIT 许可)、Wikipedia 4.5%(20 种拉丁/西里尔语言)、Gutenberg+Books3 4.5%(书级去重 >90% 重叠)、ArXiv 2.5%(去头尾/展开宏)、StackExchange 2%(按分排序)。多数 token 只用一遍(Wikipedia/Books 约 2 epochs)。**BPE 分词用 SentencePiece,数字逐位拆分,未知 UTF-8 回退到字节**。可复现性(公开数据)本身就是贡献——这是它能点燃开源生态的原因(博客视角:democratize access,13B 单 V100 可推理)。

## 2. 可复现细节

### 2.1 模型配置(Table 2)

| params | dimension | n heads | n layers | LR | batch | tokens |
|---|---|---|---|---|---|---|
| 6.7B | 4096 | 32 | 32 | 3.0e−4 | 4M | 1.0T |
| 13.0B | 5120 | 40 | 40 | 3.0e−4 | 4M | 1.0T |
| 32.5B | 6656 | 52 | 60 | 1.5e−4 | 4M | 1.4T |
| 65.2B | 8192 | 64 | 80 | 1.5e−4 | 4M | 1.4T |

### 2.2 优化器配方(可照抄)

AdamW β₁=0.9, **β₂=0.95**;cosine schedule 终值 = 10% max LR;weight decay 0.1;grad clip 1.0;warmup 2000 步。

### 2.3 高效实现(§2.4,与 FA 域直接衔接)

- **因果 attention 用 xformers 实现**:不存注意力权重、不算被因果掩码遮住的 QK 分数;论文明说前向"inspired by **Rabe and Staats (2021)**"(O(1) 内存注意力),反向用 **Dao et al. (2022) = FlashAttention** 的 recomputation——即 FA1 的内存分析前驱(详见 [`flash-attention/`](../../flash-attention/index.md) 域,批 C 将补 Rabe & Staats 专篇);
- **选择性激活重计算**:不用 PyTorch autograd,手写 transformer 层反向,只重算便宜激活、保存贵的(线性层输出)——比"全保存"省显存、比"全重算"省时间;
- **模型并行 + 序列并行**(Korthikanti et al. 2022)+ all_reduce 通信与激活计算重叠;
- **实测吞吐**:65B,2048×A100-80G,**380 tokens/s/GPU** → 1.4T tokens ≈ **21 天**。

## 3. 数据对比

### 3.1 常识推理 zero-shot(Table 3,准确率 %)

| | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA |
|---|---|---|---|---|---|---|---|---|
| GPT-3 175B | 60.5 | 81.0 | – | 78.9 | 70.2 | 68.8 | 51.4 | 57.6 |
| Chinchilla 70B | 83.7 | 81.8 | 51.3 | 80.8 | 74.9 | – | – | – |
| PaLM 540B | 88.0 | 82.3 | – | 83.4 | 81.1 | 76.6 | 53.0 | 53.4 |
| **LLaMA-13B** | 78.1 | 80.1 | 50.4 | 79.2 | 73.0 | 74.8 | 52.7 | 56.4 |
| **LLaMA-65B** | 85.3 | 82.8 | 52.3 | 84.2 | 77.0 | 78.9 | 56.0 | 60.2 |

**LLaMA-13B 在 8 项中多数超 GPT-3(10× 参数)**;LLaMA-65B 除 BoolQ 外全面超 Chinchilla-70B,除 BoolQ/WinoGrande 外超 PaLM-540B。

### 3.2 闭卷 QA(Table 4/5,exact match)

- NaturalQuestions 64-shot:LLaMA-65B **39.9** vs PaLM-540B 39.6、Chinchilla-70B 35.5;
- TriviaQA zero-shot:LLaMA-65B **68.2** vs Chinchilla-70B 55.4(Gopher-280B 43.5);
- 13B 即与 GPT-3/Chinchilla 同档(TriviaQA 0-shot 56.6 vs Chinchilla 55.4),单 V100 可推理。

### 3.3 MMLU(Table 6 语境):5-shot MMLU LLaMA-65B 63.4——低于 PaLM-540B(69.3)/Chinchilla(67.5),论文归因训练数据不含足够学术文本(ArXiv 只占 2.5%);这是"纯公开数据"路线当时的短板,Llama 2/3 靠扩数据补上。

## 4. 实践要点

- **三件套是默认起点**:今天新起一个 decoder-only LLM,pre-RMSNorm + SwiGLU((2/3)4d)+ RoPE 就是 LLaMA 定型的答案;读任何"XX 架构"先对照这三处差异;
- **β₂=0.95 不是笔误**:大模型训练常用 0.95(而非 Adam 默认 0.999)以缩短二阶矩记忆、稳大步长训练,GPT-3/LLaMA 同款;
- **配预算先问推理**:Chinchilla 最优 ≠ 部署最优;若模型要服务海量请求,倾向更小模型 + 更多 tokens(LLaMA 的核心教训,后来 7B/8B 档开源繁荣的底层逻辑);
- **数字逐位拆分**的 BPE 细节影响算术能力(逐位可学但长数字变长),Llama 3 换 tiktoken 时改了这一点;
- **许可注意**:LLaMA 1 是"研究许可发布"(非 OSI 开源),Llama 2 起才放宽商用——引用"开源"时用词要准(官方博客自己也写 publicly releasing);
- **与量化/推理域的衔接**:LLaMA 权重是 GPTQ/AWQ 等量化的最常用测试床(见 [`quantization/`](../../quantization/index.md) 与 [`vllm/concepts/quantization-gptq-awq-fp8.md`](../../vllm/concepts/quantization-gptq-awq-fp8.md));其 KV cache 行为(sink、RoPE 重编号)见 [attention-sinks-streaming](attention-sinks-streaming.md)。

## 溯源

- 主源:本地 PDF(arXiv v1,2023-02-27),Table 2-6/§2 配方数字直接核对原文;§3.3 MMLU 63.4 为 Table 6 值(论文 v1)。
- 交叉源:Meta AI 官方博客(发布口径/democratize 叙事,2023-02-24,URL 大小写敏感——小写 llama 路径 curl 000 失败,官方 `Llama` 大写路径 200)、facebookresearch/llama 官方仓库、RoPE/RMSNorm/SwiGLU 三篇原始论文(组件出处核对)。全部 URL curl 验证存活(2026-09-21)。
- 关联条目:[attention-is-all-you-need](attention-is-all-you-need.md)(被改的原始基线)、[`flash-attention/`](../../flash-attention/index.md)(xformers 高效 attention 的算法来源)、[megabyte-multiscale](megabyte-multiscale.md)(同期 Meta 的另一条 scaling 探索)。
