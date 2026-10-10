---
title: 混合线性注意力/SSM 架构:Mamba、RWKV、Gated DeltaNet 与 Qwen3-Next
type: concept
tags: [linear-attention, ssm, mamba, rwkv, gated-deltanet, griffin, jamba, qwen3-next, hybrid-architecture, state-space, recurrent, hybrid-linear-attention]
source:
  - "pdf: Mamba: Linear-Time Sequence Modeling with Selective State Spaces | arXiv:2312.00752 (Gu & Dao, 2023-12)"
  - "pdf: Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2) | arXiv:2405.21060 (Dao & Gu, 2024-05)"
  - "pdf: RWKV: Reinventing RNNs for the Transformer Era | arXiv:2305.13048 (Peng et al., 2023-05)"
  - "pdf: Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models | arXiv:2402.19427 (De et al., Google, 2024-02)"
  - "pdf: Jamba: A Hybrid Transformer-Mamba Language Model | arXiv:2403.19887 (AI21 Labs, 2024-03)"
  - "pdf: Gated Delta Networks: Improving Mamba2 with Delta Rule | arXiv:2412.06464 (Yang et al., 2024-12)"
  - "pdf: Linear Transformers Are Secretly Fast Weight Memory Systems (Linear Transformer 谱系) | arXiv:2102.11174 (Schlag et al., 2021)"
date: 2026-10
---

# 混合线性注意力/SSM 架构:Mamba、RWKV、Gated DeltaNet 与 Qwen3-Next

## TL;DR

softmax attention 之外的**第三条架构主轴**(前两条:MHA→GQA/MLA 的头共享、dense→MoE 的专家化)。核心动机是一笔两难账:**训练要能并行扫描**(GPU 吃满)、**推理要 O(1) 状态**(不随序列涨显存/延迟)——RNN 后者强前者弱,softmax attention 反之。线性注意力/SSM 家族用「核化或状态递推」把序列建模压成**线性复杂度 + 固定大小状态**,代价是**精确检索/召回能力弱**(attention 的 softmax 归一化天然做「尖锐聚焦」)。2025-26 的工程答案不是全线性,而是 **hybrid:少数 full attention 层 + 多数线性/SSM 层**——Qwen3-Next(Gated DeltaNet + full attention 混合 + 超稀疏 MoE)、Jamba/Zamba、Griffin 都是这个形态。**识别要点**:config 里出现 `mamba`、`n_layers_full_attention`、`layer_types` 混排,或 RWKV 的 `time_mix`/`time_decay`,即离开纯 softmax attention 谱系。

---

## 1. 问题:训练并行 vs 推理常数状态的两难

| 路线 | 训练 | 推理(自回归) | 精确召回 |
|---|---|---|---|
| softmax attention | O(n²) 可并行(全序列一次算) | 每步读全部 KV,O(n) 状态 | 强(直接内积定位) |
| RNN/LSTM | 顺序递推,难并行 | **O(1) 状态**,每步常数 | 弱(信息挤进定长状态) |
| 线性/SSM/hybrid | 多数可并行(线性复杂度) | O(1) 或 O(k) 状态 | 弱~中(靠 hybrid 补) |

softmax attention 的 O(n²) 在长上下文预训练是算力账单、在解码是**带宽账单**(KV cache 逐 token 线性涨,见库内 [`attention-heads-mha-mqa-gqa`](attention-heads-mha-mqa-gqa.md) 的带宽分析)。线性家族想把两条账单都砍掉。

---

## 2. 机制谱系

### 2.1 线性注意力(核化)

把 `softmax(QKᵀ)V` 换成 `φ(Q)φ(K)ᵀ V`,利用结合律改写为 `φ(Q)(φ(K)ᵀV)`——先算「状态」`S=φ(K)ᵀV` (d×d),再 `φ(Q)S`,复杂度 O(n·d²) 与序列长度无关。Performer 用随机特征近似 softmax 核;Linear Transformer(Schlag 2021)把它解释为**快速权重记忆系统**(fast weights)。问题:没有 softmax 归一化,长程检索/精确匹配明显变差。

### 2.2 RWKV:时间混合的线性递推

RWKV-4 把 attention 的 QKV 换成**时间衰减加权的线性插值**(time mixing),FFN 用 channel mixing;递推形式 `wkv_t = (w·wkv_{t-1} + k_t v_t) / (w·a_{t-1} + k_t)`(带归一)。训练用并行扫描(chunked),推理用递推 O(1)。RWKV-5/6 引入多头状态与数据依赖衰减(lar);RWKV-7 走向 delta rule(数据依赖的写入/遗忘),与 Gated DeltaNet 同一趋势。**强项**:纯 RNN 推理、Apache 生态;**弱项**:精确检索仍靠提示工程/混合层。

### 2.3 Mamba/S6 与 Mamba-2/SSD

Mamba(S6)的突破是**选择性(输入依赖)的离散化步长 Δ、投影 B/C、衰减 A**——让「记住什么、忘掉什么」依赖当前 token,解决固定 LTI SSM(如 S4)不能做内容感知检索的问题;硬件上用**kernel 融合的并行扫描**(A 依赖输入后不能预计算卷积)。Mamba-2 揭示 **Structured State Space Duality (SSD)**:选择性 SSM 与一种(带掩码的)attention 形式对偶,可用矩阵乘分块实现,效率再上一台阶。**强项**:语言/音频/DNA 序列,长序列效率;**弱项**:需要精确 in-context 检索的任务,纯 Mamba 弱于 Transformer——这是 hybrid 的直接动因。

### 2.4 Griffin / Jamba / Zamba:混合层比例的实验

- **Griffin**(Google):RG-LRU 递归层与**局部 attention** 交错(如比例 1:1 或 2:1),6144 窗口局部 attention 补「近期精确聚焦」,递归管长程。
- **Jamba**(AI21):Transformer + Mamba 交错 + MoE,「1:7」一类层配比在长上下文吞吐与质量间取舍;上下文可到 256K。
- **Zamba** 系列:共享 attention 层 + Mamba 块,进一步压参数共享。
实验结论趋同:**每隔若干线性/SSM 层插一层 full 或 local attention**,是质量-效率甜点。

### 2.5 Gated DeltaNet 与 Qwen3-Next

Gated DeltaNet 在 Mamba-2 的 delta rule 形式上加**门控遗忘**,用可并行的 chunkwise 训练 + 线性推理;在检索类任务上优于纯 SSM。**Qwen3-Next** 把 Gated DeltaNet 与 full attention 混合(如大部分层 GDN、少数层 full attention),再配**超稀疏 MoE**(激活参数极低),代表 2025-26「hybrid 第三轴」进入旗舰开源模型的现实形态。库内 [`open-model-landscape-2025-2026`](open-model-landscape-2025-2026.md) 的谱系可对照。

---

## 3. 为什么是 hybrid,而不是全线性

| 任务特征 | 全线性/SSM | hybrid(少量 full attn) | 纯 softmax |
|---|---|---|---|
| 长文档摘要/流式 | 优(状态恒定) | 优 | 差(O(n) 状态) |
| 精确检索(needle) | 差 | 中~良 | 优 |
| 训练吞吐(长序列) | 优 | 良 | 差 |
| 解码带宽(长上下文) | 优 | 良 | 差 |

**工程含义**:长上下文、高吞吐、成本敏感(尤其 decode)的场景收益最大;强 RAG/精确 recall 的场景必须保留一定比例 full attention,或外挂检索。

---

## 4. 与库内条目的关系

- [`megabyte-multiscale`](../../attention/algorithm/megabyte-multiscale.md) 是**多尺度/块分解**另一条省算力路线(与线性递推正交,可叠加)。
- [`attention-heads-mha-mqa-gqa`](attention-heads-mha-mqa-gqa.md) / MLA 解决 **KV 带宽与显存**;线性架构直接消灭 KV cache 的序列维度——两者是不同杠杆。
- [`llm-architecture`](llm-architecture.md) 的定型三件套(RMSNorm/SwiGLU/RoPE)在 hybrid 模型中仍常见,变的是序列混合层。
- [`open-model-landscape`](open-model-landscape.md) 中 RWKV 旁支、[`open-model-landscape-2025-2026`](open-model-landscape-2025-2026.md) 中 Qwen3-Next 应用本篇术语。

**config 识别速记**:
- Mamba 系: `mamba`/`ssm` 块、`d_state`、`d_conv`、`time_step_rank`
- 混合: `layer_types: ["linear_attention","full_attention",...]` 或 `n_layers_full_attention`
- RWKV: `time_mix_*`、`time_decay`、`time_first`
- Gated DeltaNet: `gated_delta_net`、`conv_kernel_size` 等

---

## 5. 实践要点(checklist)

1. **先量检索需求**:若产品有精确 needle/QA,不要上全线性;hybrid 中 full attention 比例先保 1/8~1/4。
2. **推理后端**:线性层 O(1) 状态是服务优势,但**主流推理引擎对 Mamba/GDN 的 kernel 支持远不如 attention 成熟**(vLLM 等对 Mamba 有实验支持,混合模型的调度/量化更复杂)——选型前查该模型的 runtime 支持矩阵。
3. **长上下文外推**:线性递推天然无位置相位崩溃(RoPE 外推问题),但训练长度之外的**分布**仍可能漂移;不要把「状态 O(1)」等同于「任意长度零退化」。
4. **量化**:对 SSM 的 A/Δ/门控做低比特要小心时间常数精度;与 [`fp8-fp4-formats`](../../quantization/concepts/fp8-fp4-formats.md) 的硬件原生路径未必直接覆盖递推核。
5. **与 KV cache 管理的关系**:hybrid 中 full attention 层仍有 KV cache(可走 PagedAttention),线性层无——显存预算按「最坏层」算。

---

## 溯源

- 外部链接索引见 [`../../model-architecture/references.md`](../references.md);本篇主要来源见 frontmatter。
- Mamba-2 SSD 与 attention 对偶的「矩阵分块」实现思路,与库内 CUTLASS tiling 思想同构,可对照 [`../../cutlass/concepts/gemm-tiling.md`](../../cutlass/concepts/gemm-tiling.md)。

## 相关条目(库内互链)

- [`attention/algorithm/megabyte-multiscale.md`](../../attention/algorithm/megabyte-multiscale.md) — 序列建模的多尺度路线
- [`attention-heads-mha-mqa-gqa.md`](attention-heads-mha-mqa-gqa.md) — KV cache 杠杆(与消灭 KV 的对比)
- [`llm-architecture.md`](llm-architecture.md) — 定型三件套与演进线
- [`open-model-landscape-2025-2026.md`](open-model-landscape-2025-2026.md) — Qwen3-Next 等采用现状
- [`../../vllm/concepts/paged-attention.md`](../../vllm/concepts/paged-attention.md) — hybrid 中 full attention 层的 KV 管理
