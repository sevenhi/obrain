---
title: 混合线性注意力/SSM 架构:Mamba、RWKV、Gated DeltaNet 与 Qwen3-Next
type: concept
tags: [linear-attention, ssm, mamba, rwkv, gated-deltanet, griffin, jamba, qwen3-next, hybrid-architecture, state-space, recurrent, hybrid-linear-attention]
source:
  - "code: Mamba v2.2.2（A_log 参数与输入依赖的离散化 dA） | https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/mamba_ssm/modules/mamba_simple.py | reviewed 2026-11-04"
  - "pdf: Mamba: Linear-Time Sequence Modeling with Selective State Spaces | arXiv:2312.00752 (Gu & Dao, 2023-12)"
  - "pdf: Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2) | arXiv:2405.21060 (Dao & Gu, 2024-05)"
  - "pdf: RWKV: Reinventing RNNs for the Transformer Era | arXiv:2305.13048 (Peng et al., 2023-05)"
  - "pdf: Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models | arXiv:2402.19427 (De et al., Google, 2024-02)"
  - "pdf: Jamba: A Hybrid Transformer-Mamba Language Model | arXiv:2403.19887 (AI21 Labs, 2024-03)"
  - "pdf: Gated Delta Networks: Improving Mamba2 with Delta Rule | arXiv:2412.06464 (Yang et al., 2024-12)"
  - "pdf: Linear Transformers Are Secretly Fast Weight Memory Systems (Linear Transformer 谱系) | arXiv:2102.11174 (Schlag et al., 2021)"
  - "code: fla-org/flash-linear-attention(FLA,kernel 库;GDN 的 `chunk_gated_delta_rule` packed 输入 bug 在 0.5.2 修复)| https://github.com/fla-org/flash-linear-attention"
  - "code: NVIDIA/Model-Optimizer PR #2912(2026-10-23 合并:FLA 版本 × packed GDN 精度/可复现性实测表,practice 要点 0 的数据出处)| https://github.com/NVIDIA/Model-Optimizer/pull/2912"
date: 2026-10
reviewed: 2026-11-04
---

# 混合线性注意力/SSM 架构:Mamba、RWKV、Gated DeltaNet 与 Qwen3-Next

## 速览

softmax attention 之外的**序列混合路线**:线性注意力/选择性 SSM 用核化或递归状态,在固定模型宽度下获得线性全序列工作量与常数大小递归状态,并研究如何高效并行训练。hybrid 将递归层与 full 或 local attention 结合,用检索能力与状态成本作权衡;Qwen3-Next/Jamba 的 full-attention 层仍保存 O(n) KV,Griffin 的局部 attention 则按窗口预算。**常数状态是递归层性质,不是所有混合模型的性质**。config 中的 `layer_types`、`mamba`、`time_mix` 等只是识别入口,实际层型与状态还需对照 forward。

---

## 1. 问题:训练并行 vs 推理常数状态的两难

| 路线 | 训练 | 推理(自回归) | 精确召回 |
|---|---|---|---|
| softmax attention | O(n²) 可并行(全序列一次算) | 每步读全部 KV,O(n) 状态 | 强(直接内积定位) |
| RNN/LSTM | 顺序递推,难并行 | **O(1) 状态**,每步常数 | 弱(信息挤进定长状态) |
| 纯线性/SSM | 多数可并行(固定宽度下线性工作量) | O(1) 序列状态 | 依模型与任务而定 |
| 局部窗口 hybrid | 递归层 + 窗口 attention | 固定窗口 k 时 O(k) KV + 固定递归状态 | 补近期检索 |
| full-attention hybrid | 递归层 + full attention | full-attention 层仍有 O(n) KV | 用 KV 成本换检索能力 |

**复杂度三口径**：固定宽度与层数,全序列训练/预填充、单步 decode、持久状态要分开。full attention 的三者分别随 n 为 O(n²)、O(n)、O(n);纯递归线性层为 O(n)、O(1)、O(1)。含固定数量 full-attention 层的 hybrid 仍有对应的二次总工作量与线性 KV,只降低系数,不会因多数层线性就变成全模型常数状态。训练激活、反传保存与临时 workspace 另计。

softmax attention 的 O(n²) 在长上下文预训练是算力账单、在解码是**带宽账单**(KV cache 逐 token 线性涨,见库内 [`attention-heads-mha-mqa-gqa`](attention-heads-mha-mqa-gqa.md) 的带宽分析)。线性家族想把两条账单都砍掉。

---

## 2. 机制谱系

### 2.1 线性注意力(核化)

核化注意力以 `φ(Q)φ(K)ᵀ` 替代显式 softmax 权重,利用结合律先累计 `S=Σ φ(k)vᵀ`、归一化统计量 `z=Σ φ(k)`,再输出 `φ(q)ᵀS / (φ(q)ᵀz)`。因果版本仅累计当前及历史位置;这里假设特征维与 value 维均为 d。**全序列工作量 O(n·d²) 随 n 线性增长**,单步更新 O(d²)、状态 O(d²) 才与历史长度无关。Performer 用随机特征近似 softmax 核;Schlag 2021 从 fast-weight memory 解释线性 Transformer。不同核/更新规则的检索表现需具体评测,不能将其统称为“没有归一化”。

### 2.2 RWKV:时间混合的线性递推

RWKV-4 把 attention 的 QKV 换成**时间衰减加权的线性插值**(time mixing),FFN 用 channel mixing;递推形式 `wkv_t = (w·wkv_{t-1} + k_t v_t) / (w·a_{t-1} + k_t)`(带归一)。训练用并行扫描(chunked),推理用递推 O(1)。RWKV-5/6 引入多头状态与数据依赖衰减(lar);RWKV-7 走向 delta rule(数据依赖的写入/遗忘),与 Gated DeltaNet 同一趋势。**强项**:纯 RNN 推理、Apache 生态;**弱项**:精确检索仍靠提示工程/混合层。

### 2.3 Mamba/S6 与 Mamba-2/SSD

Mamba(S6)让**Δ、B、C 依赖输入**,但连续状态参数 A 本身是可训练、非逐 token 生成的参数。官方实现 `A=-exp(A_log)`;离散化后 `A_bar=exp(ΔA)` 才随输入变化。这使系统不再是固定 LTI 卷积,实现采用融合扫描。Mamba-2 的 **Structured State Space Duality (SSD)** 将选择性 SSM 与特定掩码 attention 联系起来,用矩阵乘分块提高效率。固定状态具有长序列效率优势,但精确检索等任务仍需对照测试;这是探索 hybrid 的动机之一,不是所有任务的性能定理。

### 2.4 Griffin / Jamba / Zamba:混合层比例的实验

- **Griffin**(Google):RG-LRU 递归层与**局部 attention** 交错(如比例 1:1 或 2:1),6144 窗口局部 attention 补「近期精确聚焦」,递归管长程。
- **Jamba**(AI21):Transformer + Mamba 交错 + MoE,「1:7」一类层配比在长上下文吞吐与质量间取舍;上下文可到 256K。
- **Zamba** 系列:共享 attention 层 + Mamba 块,进一步压参数共享。
实验结论趋同:**每隔若干线性/SSM 层插一层 full 或 local attention**,是质量-效率甜点。

### 2.5 Gated DeltaNet 与 Qwen3-Next

Gated DeltaNet 将 **DeltaNet 的 delta update 与门控遗忘结合**,并设计可并行的 chunkwise 训练算法;不应称为“Mamba-2 的 delta rule 形式”。原论文把 Mamba2 和 DeltaNet 作为不同对照,并探索与 Mamba2/滑窗 attention 的混合。**Qwen3-Next** 将 Gated DeltaNet 与 full attention 混合并配 MoE;full-attention 层仍保留随序列增长的 KV。具体层比例与服务支持应绑定 checkpoint/config,谱系见 [`open-model-landscape-2025-2026`](open-model-landscape-2025-2026.md)。

---

## 3. 为什么是 hybrid,而不是全线性

| 任务特征 | 全线性/SSM | hybrid(少量 full attn) | 纯 softmax |
|---|---|---|---|
| 长文档摘要/流式 | 优(状态恒定) | 优 | 差(O(n) 状态) |
| 精确检索(needle) | 差 | 中~良 | 优 |
| 训练吞吐(长序列) | 优 | 良 | 差 |
| 解码带宽(长上下文) | 优 | 良 | 差 |

**工程含义**:长上下文 decode 是值得测试的场景;对精确 recall 要求高时,应比较 full/local hybrid 与外部检索的质量和成本,不能预设固定层比例是所有任务的必要条件。上表为选型假设,不是跨模型 benchmark 认证。

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

0. **⚠️ 用 `flash-linear-attention` 跑 packed(`cu_seqlens`)训练前先查版本**:FLA 的 Gated DeltaNet `chunk_gated_delta_rule` 在 **0.5.2 之前对 packed 输入是错的,而且不可复现**——同一份输入两次跑结果不同,teacher/student 都受影响。NVIDIA ModelOpt 的实测(随机输入,packed vs 逐序列;**数字照抄其 PR,未独立复现**):

   | FLA 版本 | 最大误差(占输出 max) | 偏差 >1% 的 token 占比 | 可复现 |
   |---|---|---|---|
   | 0.4.2(nemo:26.08 内置) | 33% | 8.4% | 否 |
   | 0.5.0 | 63% | 10.9% | 否 |
   | 0.5.1 | 59% | 14.2% | 否 |
   | **0.5.2** | **0** | **0** | **是** |

   对照结论仅限该 PR 报告的版本与测试配置:其中 KDA(`chunk_kda`)及非 packed 对照在报告容差内通过,不等于“所有版本/输入都正确”。若 GDN varlen 训练出现异常,先固定 FLA/kernel/模型版本并比较 packed 与逐序列结果;升级后仍须回归。以上 PR/依赖 pin 为 2026-10 的来源快照,本次未重新复现其数值表或确认最新依赖。
1. **先量检索需求**:若产品有精确 needle/QA,不要上全线性;hybrid 中 full attention 比例先保 1/8~1/4。
2. **推理后端**:按目标模型与引擎版本核对 Mamba/GDN、状态缓存、投机与量化支持;不要把旧版“实验支持”标签当当前所有实现的状态。
3. **长上下文外推**:线性递推天然无位置相位崩溃(RoPE 外推问题),但训练长度之外的**分布**仍可能漂移;不要把「状态 O(1)」等同于「任意长度零退化」。
4. **量化**:对 SSM 的 A/Δ/门控做低比特要小心时间常数精度;与 [`fp8-fp4-formats`](../../quantization/concepts/fp8-fp4-formats.md) 的硬件原生路径未必直接覆盖递推核。
5. **与 KV cache 管理的关系**:预算应累加所有 full-attention 层 KV、窗口层 KV 与递归层状态,再加 batch/beam、副本、临时缓冲及分配器开销;不能只按一个“最坏层”估总显存。递归层没有随历史增长的 KV,但仍有状态,抢占/回滚时也需管理。

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
