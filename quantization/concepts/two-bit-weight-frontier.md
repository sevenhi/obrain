---
title: 2-bit 权重量化前沿:QuIP# / AQLM(及 QuIP 前作、BitNet 划界)
type: concept
tags: [quantization, llm-compression, 2-bit, quip, aqlm, weight-only, ptq, incoherence, lattice-codebook, additive-quantization]
source:
  - "pdf: QuIP# 论文全文 | arXiv:2402.04396 (ICML 2024, Tseng/Chee/Sun/Kuleshov/De Sa, Cornell, ↑primary)"
  - "pdf: QuIP 前作论文 | arXiv:2307.13304 (Chee/Cai/Kuleshov/De Sa, Cornell;'#'由来,引入 incoherence processing)"
  - "pdf: AQLM 论文全文 | arXiv:2401.06118 (Egiazarian/Panferov/Kuznedelev/Frantar/Babenko/Alistarh;GPTQ 作者 Frantar 参与)"
  - "web: QuIP# 官方代码 | https://github.com/Cornell-RelaxML/quip-sharp (2026-10 实测 200)"
  - "web: AQLM 官方代码 | https://github.com/Vahe1994/AQLM (2026-10 实测 200)"
date: 2026-10
---

# 2-bit 权重量化前沿:QuIP# / AQLM

> **TL;DR**:2-bit 权重量化是 PTQ 的「极限压缩」前沿,两大 SOTA 分别是 **QuIP#**(随机 Hadamard 变换打散 outlier + E8 格码本 + fine-tuning)和 **AQLM**(多低秩码本加和,GPTQ 作者参与)。二者在 2-bit 下常并列、AQLM 略胜精度,但 QuIP# 码本小能进 L1 cache、推理更快。关键认知:**3-bit 近无损、2-bit 仍有可感知 PPL 退化**——「2-bit 近无损」是假命题。2-bit 的价值在把超大模型塞进显存受限卡(2-bit vs 4-bit 权重存储省 ~50%),但引擎侧(vLLM)对 2-bit 权重支持有限,不是 `--dtype` 能直接点出来的。

---

## 1. 为什么 2-bit 以下这么难(机制原理)

4-bit 对 GPTQ/AWQ/SpQR 已经是成熟区,但继续下探到 2-bit 会撞上**质变**而非量变。两条因果链:

**① 单一尺度的表示容量坍缩。** 1-bit 权重的取值只有 {±1} × scale,等价于把一整行/整块权重**近似成一条直线**;2-bit 也只有 4 个取值。LLM 权重里「大 outlier + 大量小普通值」共存:一个 scale 框住 outlier,普通值就全被压到量化台阶的缝隙里;反之 outlier 被截断。于是误差由少数 outlier 决定,不再是均匀舍入误差。

**② Hessian 加权误差在 <3-bit 时传播失控。** GPTQ 系方法的核心是「逐列贪心量化 + 误差补偿到未量化列」,依赖逆 Hessian 对角 `[H⁻¹]_qq` 度量各权重的敏感度并据此分摊误差。当量化网格只剩 4 个点时,单列的舍入误差已经大到一个 scale 补不回来,补偿量会「超调」到相邻列,贪心局部最优与全局最优的差距被拉开。QuIP# 论文的因果表述:误差目标 `tr((Ŵ−W)H(Ŵ−W)ᵀ)` 里 H 是激活二阶矩,权重越不「对齐坐标轴」、方向越分散,贪心舍入越准——这正是 QuIP 系列从理论入手的原因。

**结论**:2-bit 下,「选量化网格」和「消除 outlier 结构」这两件事必须一起解决,靠更细的 group 或更聪明的舍入都不够。

## 2. QuIP#:旋转消灭 outlier + 格码本 + fine-tuning

### 2.1 前作 QuIP(2307.13304):incoherence processing

QuIP 的核心洞察:**量化受益于「不相干」(incoherent)的权重与 Hessian**——即权重大小均匀、且「哪些方向舍入误差要紧」不对齐坐标轴。直觉上,incoherence 是「有原则地做 outlier 缩减」:乘以随机正交阵 U/V,把 `W → W̃ = U W V` 变换得每个元素都接近平均幅度,再量化 W̃,前向时 `W x = Uᵀ quant(W̃)(V x)` 抵消。它给出了 LLM 级量化的首个理论保证,并用自适应舍入(LDLQ,极小化 Hessian 加权二次代理)得到**第一个「2-bit 可用」的 LLM 量化方法**。「#」即源于对前作的三处升级。

### 2.2 QuIP# 三招(2402.04396,ICML 2024)

1. **随机 Hadamard 变换(RHT)替换普通正交阵**。Hadamard 阵可用快速 Walsh-Hadamard(FWHT,O(n log n),无显式矩阵乘法)实现,快且理论上 μ 界更好;对角加随机 ±1 缩放(即 `H·diag(s)`)是「随机化」来源,避免固定 Hadamard 与某类数据的偶然对齐。文中对比:纯 FFT 稍差但仍可用(Table 1:Hadamard vs Fourier,2-7B 8.22 vs 8.30)。
2. **E8 格码本(向量量化)**。incoherent 处理后的权重呈「球状 sub-Gaussian」分布,天然适合**格码**。E8 是 8 维最优单位球填充格:
   ```
   E8 = (Z⁸ ∪ (Z⁸ + ½·1)) ∩ { x | 1ᵀx 为偶数 }
   ```
   在此基础上作者设计了 **E8P 码本**:2^16 个条目,却只需查一个 2^8 条目的表——剩余 8 bit 存符号与 shift。这是硬件效率的关键(见 §4)。
3. **fine-tuning(量化时微调)**。前面两招是纯 PTQ;再加一层轻量 fine-tuning(量化过程中更新权重以拟合原模型,`quantize_finetune_llama.py`)捕捉跨层交互,把 2-bit/3-bit 的尾部误差进一步压下去(消融见 Table 10:no-FT 2-bit Llama-2-70B Wiki2 4.16,加 FT 到 3.91 附近;量化过程即 fine-tuning,非单独蒸馏步骤)。

### 2.3 定位

**≤4-bit 的 SOTA;3-bit 近无损(可超越理论无损的 4-bit);2-bit 有可感知退化。** QuIP# 是第一个「3-bit 模型比理论上无损的 4-bit 模型 scale 得更好」的 PTQ 方法(反驳了 Dettmers & Zettlemoyer「4-bit 最优」的旧论断)。

## 3. AQLM:多码本加和(Additive Quantization)

### 3.1 机制

AQLM 把经典信息检索里的 Multi-Codebook Quantization(MCQ)泛化到 LLM 权重。核心思想:**每个权重不再映射到单一量化点,而是写成 M 个码本条目之和**:

```
Ŵ_i = Σ_{m=1..M} C_m · b_{i,m}     (b 是独热选择的码本索引)
```

M 个码本各含 2^B 个向量,一个权重 = M 个向量的逐元素和 → 表示能力随 M 指数增长,用「加法」把 2-bit 的 4 个点扩成 2^(M·B) 种组合。两个创新:

1. **输入自适应(input-adaptive)的码本学习**:不是对权重做通用聚类,而是把每层输出误差 `min ||WX − ŴX||²` 作为目标、用校准激活 X 的统计(XXᵀ)加权去学码本与分配——码本「知道」哪些方向的误差要紧(与 GPTQ 用 Hessian 的哲学同源,作者正是 GPTQ 的 Frantar)。
2. **跨层联合优化**:整块 transformer 层(而非逐层独立)联合优化码本参数,捕捉层间误差耦合。

### 3.2 定位与结论

**AQLM 是第一个在 <3-bit 区间「精度 vs 模型体积」Pareto 最优的方案**,2-bit 下通常 ≥ QuIP#(两篇论文各自在自家设置下略胜,见 §5 交叉核对)。同时提供 GPU/CPU 的快速 kernel,2×8-bit 码本在 GPU 上相对 FP16 加速 1.2~3×,CPU 上 2.7~4×。

## 4. 两条相反路线:QuIP# vs SpQR(一句话锚点)

> **SpQR 保留 outlier 到 16-bit,QuIP# 旋转消灭 outlier**——这是 2-bit 以下处理 outlier 的两条对跖路线。SpQR 认为 outlier 有结构(行/列/head/RoPE 条纹),值得单独高精度存(CSR);QuIP# 认为 outlier 是「权重没对齐」的症状,用旋转把它抹平到 uniform 分布。前者是「隔离并利用」,后者是「预处理消解」。(SpQR 详见 [spqr-sparse-quantized.md](spqr-sparse-quantized.md))

## 5. 数据对比

**Wikitext2 / C4 perplexity(ctx 4096,QuIP# 论文 Table 4/11 口径;FP16 基线):**

| 模型 | FP16 W2/C4 | QuIP# 2-bit | AQLM 2.0x | QuIP# 3-bit | AQLM 3.0x |
|---|---|---|---|---|---|
| Llama-2 7B | 5.12 / 6.63 | 6.19 / 8.16 | 6.93 / 8.84 | 5.41 / 7.04 | 5.46 / 7.10 |
| Llama-2 13B | 4.57 / 6.05 | 5.35 / 7.20 | 5.70 / 7.59 | 4.78 / 6.35 | 4.83 / 6.37 |
| Llama-2 70B | 3.12 / 4.97 | 3.91 / 5.71 | 3.94 / 5.72 | 3.35 / 5.15 | 3.36 / 5.17 |

- **2-bit 的退化是可感知的**:70B 从 3.12 → 3.91(+25%),7B 从 5.12 → 6.19(+21%)。**3-bit 才谈得上近无损**(70B 3.35,离 FP16 仅 7%)。这正是「2-bit 近无损」是伪命题的实证。
- **交叉核对的分歧**:AQLM 论文自家 Table 1 报 AQLM-2bit 7B W2=6.59、QuIP#=8.22(倒过来);QuIP# 论文报 QuIP# 6.19、AQLM 6.93。差异来自 context length / 校准设置不同(QuIP# 用 ctx 4096,AQLM 早期表用 ctx 2048+不同校准)。**公平口径:二者 2-bit 常并列,AQLM 略胜;不必纠结单表胜负**。
- **零样本**(QuIP# Table 3,ArcC/ArcE/PiQA/Wino,2-70B):FP16 51.1/77.7/81.1/77.0 → QuIP# 2-bit 48.7/77.3/80.3/75.9、AQLM 2-bit 47.9/77.7/80.4/75.9——退化集中在 ArcC(-2.4)。

**推理吞吐(RTX 6000 Ada,QuIP# README / 论文 Table 6):**

| 方法 | 2-7B | 2-70B |
|---|---|---|
| FP16 | 57.7 tok/s | OOM |
| AQLM 2-bit | 81.1 | 8.72 |
| QuIP# 2-bit | **176** | **21.9** |

**QuIP# 快于 AQLM 的根因**:AQLM 的码本(≈1 MiB)塞不进任何现代 GPU 的 L1 cache(H100 仅 256 KB),反量化时 cache miss 高,甚至慢于 FP16;QuIP# 的 E8P 码本刻意做成 2^8 查表 + 符号/shift 编码,**小到能常驻 L1**,推理时反量化零 cache 抖动。这是「码本设计要迁就缓存层级」的教科书案例。

## 6. 可复现细节

### 6.1 QuIP# 的量化/推理流程(结构图)

```
原权重 W (含 outlier,非均匀)
   │  ① RHT 两两非相干处理:H_L · S_L · W · S_Rᵀ · H_Rᵀ → W̃(均匀化)
   ▼
W̃ (球状 sub-Gaussian)
   │  ② 向量量化:E8P 格码本(2^16 条目 → 2^8 查表 + 8bit 符号/shift)
   ▼
Q(W̃) 2-bit 存储
   │  ③ fine-tuning 阶段:量化过程中微调 Q(W̃),拟合原模型输出
   ▼
推理前向:W x = Uᵀ · dequant(Q(W̃)) · (V x),U/V = H·diag(s)
```

关键实现点:Hessian 用 `fp64` 累加(`hessian_offline_llama.py`,慢 fp64 的卡要改 fp32);FT 复现早期结果传 `--ft_epochs 0`;RHT 依赖 fast-hadamard-transform 包。

### 6.2 AQLM 的核心目标(伪代码骨架)

```python
# 每层权重 W 分解为 M 个码本之和,码本 C 与分配 b 联合学习:
#   Ŵ_i = Σ_m C_m · b_{i,m}     (b_{i,m} 是 one-hot,选中码本 m 的第 k 个条目)
# 目标:min_{C,b} || W X - (Concat Σ_m C_m b_m) X ||²
#   展开后 = ||WX||² - 2Σ_m <W, C_m b_m>_XXᵀ + Σ_ij <C_i b_i, C_j b_j>_XXᵀ
#   交叉项 <·,·>_XXᵀ 用校准激活的 XXᵀ 加权 → 输入自适应
# 优化:坐标下降交替更新 b(码本分配)与 C(码本向量),跨层联合。
```

## 7. BitNet b1.58 划界(一句话)

BitNet b1.58 是**训练时三元权重(training-aware ternary,{-1,0,+1}×scale)**,靠从零训练而非事后量化;与本文 2-bit PTQ(QuIP#/AQLM)是**不同家族**——PTQ 作用于已训练权重,BitNet 改变训练本身。二者不在同一条比较基准上,不展开。

## 8. 实践要点

1. **何时 2-bit 值得**:超大模型(70B+)塞显存受限卡;自回归 decode 是 memory-bound,压缩即加速。2-bit vs 4-bit 权重存储省 **~50%**(bits 减半)。小模型(7B 级)2-bit 退化明显(>20% PPL),3-bit 才是性价比甜点。
2. **引擎侧现实(别被论文误导)**:vLLM 的 GPTQ/AWQ 分派链只支持 **4-bit/8-bit 对称量化**(`TYPE_MAP (num_bits,is_sym) → uint4b8/uint8b128`,见 [vLLM 量化篇](../../vllm/concepts/quantization-gptq-awq-fp8.md)),**QuIP#/AQLM 的 2-bit 权重不能靠 `--quantization` 直接点出**;AQLM 有独立 serving 支持(仓库含 vLLM 集成示例),QuIP# 靠自带 CUDA kernel。落地前先确认引擎/backend 是否真正支持该 2-bit 格式。
3. **硬件/kernel 成熟度**:QuIP# 码本小、推理快但生态窄;AQLM 有 GPU+CPU kernel 与 Rust/WASM 端侧 demo,生态更全。选型要同时看精度与「谁的 kernel 能上你的卡」。
4. **校准集敏感**:2-bit 对校准数据分布敏感(AQLM 论文 Table 17:校准样本 128→4096,PPL 6.99→6.27 仍在收敛);务必用真实业务分布校准。
5. **联合部署**:2-bit 权重省的是权重显存,长上下文场景的 KV cache/激活另算——参见 [KV cache 量化](kv-cache-quantization.md)与 [KVQuant/KVTuner 篇](kvquant-kvtuner-frontier.md)。

## 相关条目

- [PTQ:GPTQ / AWQ / SmoothQuant](ptq-gptq-awq-smoothquant.md)——4-bit 成熟区三条路线,2-bit 由 QuIP#/AQLM 接棒
- [SpQR:稀疏+量化混合表示](spqr-sparse-quantized.md)——与 QuIP# 相反的「保留 outlier」路线
- [vLLM 量化实现](../../vllm/concepts/quantization-gptq-awq-fp8.md)——引擎侧对 2-bit 权重支持有限的口径来源
- [KVQuant / KVTuner:KV cache 极低比特](kvquant-kvtuner-frontier.md)——另一条「极低比特」战线(KV 而非权重)

## 溯源

- 主源 ↑primary:QuIP#(arXiv:2402.04396,ICML 2024)与 AQLM(arXiv:2401.06118,ICML 2024)论文全文(正文 + 附表)。
- 前作:QuIP(arXiv:2307.13304,incoherence processing 起源)。
- 代码:github.com/Cornell-RelaxML/quip-sharp(README 注明已让位于后继 QTIP)、github.com/Vahe1994/AQLM(2026-10 均实测 200)。
- 发表状态:QuIP# ICML 2024;AQLM ICML 2024(后继 PV-Tuning NeurIPS 2024 oral)。
