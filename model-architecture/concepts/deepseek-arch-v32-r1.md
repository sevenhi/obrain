---
title: DeepSeek-V3.2(DSA) + R1 网络结构剖析
type: concept
tags: [deepseek, v3.2, dsa, sparse-attention, lightning-indexer, mla, moe, r1, rl, architecture]
series: deepseek-model-evolution
source:
  - "code: inference/model.py(v32_model.py)| https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference"
  - "paper: DeepSeek-V3.2-Exp(DeepSeek_V3_2.pdf)| https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf"
  - "paper: DeepSeek-V3.2(arXiv:2512.02556)| https://arxiv.org/abs/2512.02556"
  - "paper: DeepSeek-V3(arXiv:2412.19437)| https://arxiv.org/abs/2412.19437"
  - "paper: DeepSeek-R1(arXiv:2501.12948)| https://arxiv.org/abs/2501.12948"
  - "config: config.json| https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/resolve/main/config.json"
date: 2026-10-09
---

# DeepSeek-V3.2(DSA) + R1 网络结构剖析

> **主线定位**:本代是 DeepSeek 谱系的**破 O(n²)** 段——**唯一结构增量是 DSA 稀疏注意力**,以及正交支线 R1(纯 RL 推理时扩展,不动结构)。完整演进总览见主篇 [deepseek-model-evolution](deepseek-model-evolution.md);奠基段 V1/DeepSeekMoE/V2 见 [deepseek-arch-v1-moe-v2](deepseek-arch-v1-moe-v2.md);V3(规模 + MTP + aux-loss-free)见 [deepseek-arch-v3](deepseek-arch-v3.md)。

## TL;DR

**DeepSeek-V3.2-Exp 相对 V3.1-Terminus(即 V3 谱系)的唯一结构增量是 DSA(DeepSeek Sparse Attention,深度稀疏注意力)**,由**闪电索引器(Lightning Indexer)+ 细粒度 top-k token 选择**构成:索引器用一组**极轻量、FP8 可实现**的小头(`index_n_heads=64`、`index_head_dim=128`)计算索引分数 `I_{t,s} = Σ_j w^I_{t,j}·ReLU(q^I_{t,j}·k^I_s)`,选出每个 query 要 attend 的 **top-k=2048** 个 KV 条目,主注意力只对稀疏选中的 KV 做 MQA,把核心注意力复杂度从 **O(L²) 降到 O(L·k)**。**R1 没有新网络结构**——它是 V3-Base 上的纯 RL 推理时扩展(冷启动 + GRPO + 蒸馏)。

一句话结论(V3.2 DSA 结构):**在 MLA 的 MQA 模式上并联插入一个"闪电索引器",用哈达玛旋转(rotate_activation)消激活离群点、FP8 量化做索引打分、top-2048 稀疏选择 KV,把长上下文注意力从 O(L²) 破到 O(L·2048),主干其余结构(MLA/MoE/MTP)一字未动。**

---

## 1. 代际定位:V3.2 在 DeepSeek 谱系中的位置

| 代 | arXiv | 核心结构增量 | 相对上一代改了什么 |
|---|---|---|---|
| V2 | 2405.04434 | MLA(低秩 KV 压缩 + 解耦 RoPE)+ 细粒度稀疏 MoE | dense → MLA + MoE |
| V3 | 2412.19437 | MLA + 256 专家 MoE(top-8)+ MTP(多 token 预测)+ aux-loss-free 负载均衡 | V2 基础上加 MTP、无 aux-loss |
| V3.1-Terminus | — | V3 + 长上下文(128K)+ RL/agent 后训练 | 结构不变,训到 128K |
| **V3.2-Exp** | DeepSeek_V3_2.pdf | **DSA(稀疏注意力)** | **唯一增量 = DSA** |
| V3.2(正式版) | 2512.02556 | DSA + 可扩展 RL 框架 + Agentic 合成数据 | Exp 基础上放大后训练 |
| R1 | 2501.12948 | **无新结构** | V3-Base 上纯 RL(推理时扩展) |

**关键事实(论文原文)**:*"Compared with DeepSeek-V3.1-Terminus … the only architectural modification of DeepSeek-V3.2-Exp is the introduction of DeepSeek Sparse Attention (DSA) through continued training."*

即:**V3.2 的 MLA/MoE/MTP 全继承 V3,V3.2 结构 diff 就一件事——DSA。**

---

## 2. V3.2 相对 V3 的结构 diff(代码级逐项对照)

对照 `/home/cambricon/zhanbin/obrain/.sources/deepseek/` 下 `v3_model.py`(808 行)与 `v32_model.py`(922 行),`v32_model.py` 新增/改动的全部条目:

### 2.1 新增类/函数(全为 DSA 服务)

| 新增项 | 作用 | 对应 DSA 环节 |
|---|---|---|
| `class Indexer(nn.Module)` | 闪电索引器:算索引分数、取 top-k | DSA 核心 |
| `def rotate_activation(x)` | 哈达玛旋转(hadamard_transform,scale=hidden^-0.5) | 消激活离群点 |
| `class LayerNorm(nn.Module)` | V3 只有 RMSNorm;新增 LN 用于索引器 `k_norm` | 索引器 K 归一化 |
| `def weight_dequant(weight, scale)` | V3 从 kernel 导入;V3.2 内联到 model.py(签名改为直接传 scale) | 解码期 wkv_b 反量化 |

### 2.2 ModelArgs 新增字段(仅 DSA 引入的算子维度)

```python
# v32_model.py 新增(相对 v3_model.py):
    # index (DSA)
    index_n_heads: int = 64      # 索引器头数 H_I
    index_head_dim: int = 128    # 索引器头维 d_I
    index_topk: int = 2048       # 每个 query 选出的 KV 数 k
```

### 2.3 MLA 的变化

```python
# v3_model.py —— MLA 没有 indexer
# v32_model.py —— MLA.__init__ 末尾新增:
        self.indexer = Indexer(args)          # 每个注意力层内嵌一个索引器
        self.dequant_wkv_b = None             # 懒反量化缓存

# MLA.forward: 算完 scores 之后,两条路径(MHA prefill / MQA decode)都插入了同一段:
        # indexer (DSA)
        topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
        index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"),
                                device=x.device).scatter_(-1, topk_indices, 0)
        index_mask += mask
        scores += index_mask.unsqueeze(2)     # 非 top-k 位置 score 置 -inf → 稀疏 softmax
```

### 2.4 apply_rotary_emb 的变化

```python
# v3:   def apply_rotary_emb(x, freqs_cis)            # 仅 interleaved
# v32:  def apply_rotary_emb(x, freqs_cis, interleaved: bool = True)
#       索引器里用 interleaved=False(非交织 RoPE),主注意力仍 interleaved=True
```

### 2.5 Block 残差流的变化

```python
# v3_model.py(经典 post-norm 单流):
    x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
    x = x + self.ffn(self.ffn_norm(x))

# v32_model.py(双残差流 / residual-carrying pre-norm):
    x, residual = self.attn_norm(x, residual)   # RMSNorm 支持带 residual 的融合
    x = self.attn(x, start_pos, freqs_cis, mask)
    x, residual = self.ffn_norm(x, residual)
    x = self.ffn(x)
```

> 说明:双残差流是官方 inference 参考实现的行文风格(与注意力/FFN 的 fp8/流水细节耦合),**不是 DSA 本身引入的数学变化**;真正改变网络表达能力的是 Indexer 的插入。

### 2.6 kernel 依赖的变化

```python
# v3:  from kernel import act_quant, weight_dequant, fp8_gemm
# v32: from kernel import act_quant, fp8_gemm, fp8_index   # 新增 fp8_index;weight_dequant 内联
```

`fp8_index` 是 DSA 的专用核:在 FP8 域内算 `Σ_j w_j·ReLU(q_j·k)`(即论文公式 (1) 的索引分数)。

---

## 3. DSA 机制详解

### 3.1 两个组件

DSA = **闪电索引器(Lightning Indexer)** + **细粒度 token 选择(fine-grained token selection)**。

**① 闪电索引器** —— 论文公式 (1),计算 query `h_t` 与历史 token `h_s` 之间的索引分数:

```
I_{t,s} = Σ_{j=1}^{H_I}  w^I_{t,j} · ReLU( q^I_{t,j} · k^I_s )
```

- `H_I` = 索引器头数(**64**)
- `q^I_{t,j} ∈ R^{d_I}`(d_I = **128**)由 query token 派生(每头一个)
- `w^I_{t,j} ∈ R` 由 query token 派生的**每头标量权重**
- `k^I_s ∈ R^{d_I}` 由历史 token 派生
- 用 **ReLU** 激活(为了吞吐;FP8 下实现高效)
- 头数少 + 可 FP8 实现 → 索引器计算开销远小于主 MLA

**② 细粒度 token 选择** —— 论文公式 (2):

```
u_t = Attn( h_t, { c_s : I_{t,s} ∈ Top-k( I_{t,:} ) } )
```

只取 top-k(**2048**)个索引分数对应的 KV 条目,主注意力只在这 k 个条目上做。

### 3.2 代码里 DSA 的数据流(Indexer.forward,逐行对照)

```python
def forward(self, x, qr, start_pos, freqs_cis, mask):
    # x:   输入 hidden (bsz, seqlen, dim=7168)
    # qr:  主 MLA 的 query latent = q_norm(wq_a(x))   ← 复用主注意力的低秩 query 投影!
    q = self.wq_b(qr)                       # (bsz, seqlen, H_I*d_I)  H_I=64, d_I=128
    q = q.view(bsz, seqlen, H_I, d_I)
    q_pe, q_nope = split(q, [rope_dim=64, d_I-rope_dim=64])   # 部分 RoPE,与 MLA 同款
    q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=False)
    q = cat([q_pe, q_nope])

    k = self.wk(x)                          # 独立 key 投影: dim → d_I(=128)
    k = self.k_norm(k)                      # LayerNorm(128)  ← 新增 LN
    k_pe, k_nope = split(k, [64, 64]); k_pe = RoPE(k_pe)
    k = cat([k_pe, k_nope])

    q = rotate_activation(q)                # 哈达玛旋转,消 FP8 量化离群点
    k = rotate_activation(k)

    q_fp8, q_scale = act_quant(q)           # 激活量化为 FP8
    k_fp8, k_scale = act_quant(k)
    self.k_cache[...] = k_fp8               # 索引器 K 单独一份 FP8 cache

    weights = self.weights_proj(x.float()) * H_I ** -0.5   # 每头标量权重 w^I_{t,j}
    weights = weights.unsqueeze(-1) * q_scale * softmax_scale  # FP8 缩放折入
    index_score = fp8_index(q_fp8, weights, k_cache, k_scale_cache)  # 核内算 Σ w·ReLU(q·k)

    if mask is not None: index_score += mask
    topk_indices = index_score.topk(min(index_topk, end_pos), dim=-1)[1]  # 取 top-2048
    return topk_indices
```

**三个关键设计细节(源码直接证据)**:

1. **索引器 query 复用主 MLA 的 `qr` latent**:`self.indexer(x, qr, ...)` 里的 `qr = q_norm(wq_a(x))` 与主注意力完全同一个变量。索引器只新增自己的 `wq_b`(把 1536 维 latent 投到 64×128),**key 侧则是独立的 `wk`**(dim→128)。即索引器与主注意力**共享低秩 query 投影 `wq_a`**,只有 key 投影与打分独立——参数增量极小。

2. **哈达玛旋转消离群点**:`rotate_activation` = `hadamard_transform(x, scale=hidden^-0.5)`(128 维是 2 的幂,满足 Hadamard 维度要求)。在 RoPE **之后、FP8 量化之前**对 q_I、k_I 做正交旋转,把激活里的大离群值(softmax/attention 常见)摊平到所有维度,使 FP8 量化误差可控。这是 DSA 能把索引器塞进 FP8 的前提。

3. **索引器 K 独立 FP8 cache**:`k_cache`/`k_scale_cache` 是索引器专属的一份 fp8 cache,与主 MLA 的 `kv_cache`(latent)分开——因为索引器 K 维度(128)≠ 主 MLA latent 维度(512)。

### 3.3 主注意力的两条路径(稀疏化如何落地)

```python
# v32_model.py MLA.forward 的两种模式:
if mask is not None:        # MHA prefill(短序列,论文:masked MHA 模拟 DSA)
    scores = einsum("bshd,bthd->bsht", q, k) * scale     # 全注意力先算
    topk_indices = self.indexer(x, qr, ...)              # 再叠加 top-k mask
    index_mask = full((bsz,seqlen,seqlen), -inf).scatter_(-1, topk_indices, 0)
    scores += index_mask.unsqueeze(2)                    # 非 top-k → -inf
    scores = softmax(scores); x = einsum(scores, v)
else:                       # MQA decode(长序列,真稀疏路径)
    # 吸收 wkv_b 的 MLA 解码:latent 直接内积 q_nope·kv_cache + q_pe·pe_cache
    topk_indices = self.indexer(x, qr, ...)
    index_mask = full((bsz,1,end_pos), -inf).scatter_(-1, topk_indices, 0)
    scores += index_mask.unsqueeze(2)
    scores = softmax(scores)                              # 只对 2048 个位置非零
    x = einsum(scores, kv_cache[:bsz,:end_pos])           # 稀疏加权(KV latent 全头共享)
    x = einsum(x, wkv_b[:, -v_head_dim:])                 # 吸收 v 投影
```

- **MQA 模式**是 DSA 的"真身":latent vector(KV 条目)跨 query 的**所有注意力头共享**(对应论文 Figure 1 的 MQA 版 MLA),每个 query 只 attend 它 top-k 选中的 2048 个 latent。
- **masked MHA 模式**用于短序列 prefill,是"模拟 DSA"的高效实现(全注意力 + index mask 屏蔽),等价结果但短序列下更省。

### 3.4 复杂度:O(L²) → O(L·k)

论文原文:*"DSA reduces the core attention complexity of the main model from O(L²) to O(L·k), where k(≪ L) is the number of selected tokens."*

- 主注意力(核心):O(L·k),k=2048
- 索引器:仍是 O(L²),但头数少(64)、头维小(128)、FP8 实现,计算量远小于主 MLA → 总端到端长上下文显著加速
- H800 实测:长序列 prefill/decode 每百万 token 成本显著下降(Figure 3);API 价格下调 50%+

---

## 4. DSA 在 DecoderBlock 里的插入结构图

### 4.1 模型总览(结构不变,只 MLA 内嵌 Indexer)

```
DeepSeek-V3.2-Exp 全模型
====================================================================
  tokens → ParallelEmbedding(vocab=129280, dim=7168)
         → ×61 个 Block
         → RMSNorm
         → lm_head(dim→vocab) → logits
====================================================================

  Block(layer_id) —— 双残差 pre-norm
  ┌─────────────────────────────────────────────────────────────┐
  │  h ──attn_norm(RMSNorm)──► MLA(内含 DSA/Indexer)──ffn_norm──► FFN │
  │                                    │                          │
  │  FFN = MLP(第一层 dense) 或 MoE(256 routed + 1 shared, top-8) │
  └─────────────────────────────────────────────────────────────┘
```

### 4.2 MLA 内部:DSA 如何并联插入

```
MLA.forward 内部(V3.2,含 DSA)
════════════════════════════════════════════════════════════════
  x ── wq_a ── q_norm ──► qr (1536)   ◄── 复用同一 latent
       │
       ├──(主注意力) wq_b ──► q_A: 128 头 × (128 nope + 64 rope)
       │
       └──(索引器)  Indexer.wq_b ──► q_I: 64 头 × 128
                   (与主注意力共享 wq_a)

  x ── wkv_a ──► { kv(512 latent) , k_pe(64) }  ──► kv_cache(fp8) + pe_cache
  x ── Indexer.wk ── LayerNorm ──► k_I(128) ──► Indexer.k_cache(fp8)

  ┌──────────────────────────────────────────────────────────────┐
  │  Lightning Indexer                                           │
  │   q_I ──RoPE(非交织)──► rotate_activation ──► act_quant(fp8)  │
  │   k_I ──RoPE(非交织)──► rotate_activation ──► act_quant(fp8)  │
  │   w_I = weights_proj(x)  (每头标量, 64 个)                    │
  │   I_{t,s} = fp8_index:  Σ_j w_I_j · ReLU(q_I_j · k_I_s)      │
  │   topk_indices = Top-k(I, k=2048)                             │
  └──────────────────────────────────────────────────────────────┘
                           │
                           ▼
  scores(q_A·k_A) + index_mask(topk)  →  softmax  →  稀疏加权 Σ p·c_s
  (非 top-k 位置 score = -inf → 只 attend 2048 个 KV latent)
```

### 4.3 DSA 数据流(PyTorch 伪代码)

```python
class DeepSeekV32Block:
    def forward(self, h):
        # 1) 主 query latent(与索引器共享)
        qr = q_norm(wq_a(h))                        # (b, s, 1536)
        # 2) 主注意力 query
        q = wq_b(qr).view(b, s, 128, 192)           # 128 头 × (128+64)
        q_nope, q_pe = q[..., :128], q[..., 128:]   # 解耦 RoPE
        # 3) KV latent(MLA 低秩压缩)
        kv, k_pe = split(wkv_a(h), [512, 64])
        kv_cache.append(act_quant(kv_norm(kv)))     # fp8 latent cache
        # 4) 闪电索引器(DSA)
        idx = indexer(h, qr, freqs, mask)           # → topk_indices (b, s, 2048)
        index_mask = full((b, s, end), -inf).scatter_(-1, idx, 0)
        # 5) 稀疏注意力(MQA 解码)
        scores = einsum("bshc,btc->bsht", q_nope_absorb, kv_cache)  # O(L·k)
        scores = (scores + index_mask).softmax(-1)
        out = einsum("bsht,btc->bshc", scores, kv_cache)
        out = einsum("bshc,hdc->bshd", out, v_proj)  # 吸收 v 投影
        return wo(out)
```

---

## 5. V3.2 超参(真实 checkpoint config.json + DSA 算子维度)

> ⚠ `v32_model.py` 文件里的 `ModelArgs` 是**降采样 smoke-test 配置**(dim=2048 / 27 层 / 16 头),不代表真实模型。真实超参以官方 `config.json` 为准;其中 **DSA 索引器维度(dim=128 / 64 头 / topk=2048)在两者中一致**,是唯一"代码与 checkpoint 对齐"的部分。

### 5.1 主干超参(继承 V3/V3.1)

| 参数 | V3.2-Exp 值 | 说明 |
|---|---|---|
| hidden_size | 7168 | 模型维度 |
| num_hidden_layers | 61 | 层数 |
| num_attention_heads | 128 | MLA 头数 |
| q_lora_rank | 1536 | query 低秩维 |
| kv_lora_rank | 512 | KV 低秩维(MLA 压缩瓶颈) |
| qk_nope_head_dim / qk_rope_head_dim | 128 / 64 | 解耦 RoPE |
| v_head_dim | 128 | value 头维 |
| n_routed_experts / n_shared_experts | 256 / 1 | MoE |
| num_experts_per_tok (top-k) | 8 | 每 token 激活专家 |
| n_group / topk_group | 8 / 4 | 分组路由(noaux_tc) |
| scoring_func / routed_scaling_factor | sigmoid / 2.5 | 路由打分 |
| first_k_dense_replace | 3 | 前 3 层 dense |
| intermediate_size / moe_intermediate_size | 18432 / 2048 | FFN 隐维 |
| vocab_size | 129280 | 词表 |
| max_position_embeddings | 163840 | 长上下文(128K 扩展) |
| RoPE | theta=10000, YaRN(factor=40, β_fast=32, β_slow=1) | 解耦 RoPE + YaRN |

### 5.2 DSA 新增算子维度(V3 基础上引入的)

| 算子 | 形状/维度 | 值 | 说明 |
|---|---|---|---|
| 索引器头数 H_I | `index_n_heads` | **64** | 远小于主注意力 128 头 |
| 索引器头维 d_I | `index_head_dim` | **128** | 2 的幂,Hadamard 可旋转 |
| top-k | `index_topk` | **2048** | 每 query 选出的 KV 条目数 |
| 索引器 query 投影 `wq_b` | (1536 → 64×128=8192) | — | 复用主 qr latent |
| 索引器 key 投影 `wk` | (7168 → 128) | — | 独立,后接 LayerNorm(128) |
| 每头权重投影 `weights_proj` | (7168 → 64) | — | 输出每头标量 w^I |
| 索引器 RoPE 维 | 64(=qk_rope_head_dim) | 非交织 | 与 MLA 解耦 RoPE 同款 |
| KV 缓存 | k_cache 用 fp8 | — | 索引器专属 FP8 cache |

---

## 6. R1:没有新结构,是 V3 上的 RL 推理时扩展

**一句话:R1 的网络结构 = DeepSeek-V3 原样,不新增任何算子/层/注意力机制。**

- R1 论文(arXiv:2501.12948)明确:*"we use DeepSeek-V3-Base as the base model"*,用 **GRPO**(Group Relative Policy Optimization)做 RL。
- 三条线,全在后训练/推理时,不碰结构:
  1. **R1-Zero**:V3-Base 直接大规模 RL(无 SFT),涌现反思/验证等推理行为(但可读性差、语言混杂)。
  2. **R1**:先**冷启动**(少量高质量 CoT 数据 SFT)→ RL → 再 SFT → 再 RL 的多阶段流程,解决可读性与语言一致性问题。
  3. **蒸馏**:把 R1 的推理能力蒸馏到 Qwen/Llama 的 6 个 dense 模型(1.5B–70B)。
- 本质是**推理时扩展(inference-time scaling)**:延长 CoT 长度换推理能力,结构零改动。

> 与本篇主线关系:V3.2(结构创新,破注意力 O(L²))与 R1(训练/推理范式创新,不动结构)是 DeepSeek 演进的两条正交支线——**V3.2 改"网络怎么算",R1 改"怎么训、怎么推理"**。

---

## 7. 与 V3 代际 diff 总结 + 演进方向

### 7.1 代际 diff 表(V3 → V3.2)

| 维度 | V3 | V3.2-Exp | 变化 |
|---|---|---|---|
| 注意力 | MLA(全注意力) | MLA + **DSA 稀疏** | **新增 Indexer + top-k 选择** |
| 注意力复杂度 | O(L²) | **O(L·k),k=2048** | 破 O(L²) |
| 索引器 | 无 | 64 头 × 128 维,FP8,Hadamard 旋转 | **新增** |
| MoE | 256 专家 top-8 | 不变 | 0 |
| MTP | 有 | 不变 | 0 |
| 位置编码 | 解耦 RoPE + YaRN | 不变(+索引器非交织 RoPE) | 0 |
| 残差流 | post-norm 单流 | dual-residual pre-norm(参考实现) | 工程性 |
| 训练 | 常规预训练 | **dense 预热 indexer + 稀疏续训** | 新增两阶段 |

### 7.2 DSA 的两阶段训练(结构之外的关键配套)

1. **Dense 预热**:冻结除索引器外所有参数,只训索引器 1000 步(2.1B tokens),用 KL 损失让索引器输出对齐主注意力的(跨头求和 + L1 归一化的)分布。
2. **稀疏续训**:引入 top-k 选择,全参数训 15000 步(943.7B tokens),索引器只对齐被选集合 S_t 上的分布,且索引器输入 detach(索引器只吃 KL 损失,主模型只吃 LM 损失)。

### 7.3 演进方向定位

DeepSeek 网络结构演进的主线约束是**算力(FLOPs)与显存(KV cache)双重约束**:

- **省显存**:MLA(低秩 KV 压缩)已是极致;V3.2 的 DSA 进一步**省计算**——稀疏后 KV 读取/注意力计算都只发生在 top-k 条上。
- **破 O(L²)**:V3.2 是 DeepSeek 谱系里**第一次动注意力复杂度的结构创新**,方向对标 NSA(Native Sparse Attention,Yuan et al. 2025,论文引用的硬件对齐稀疏注意力)。
- **"训练质量-部署效率"双目标**:V3.2 用"轻索引器 + 稀疏选择"在长上下文性能不显著退化(Table 1 与 V3.1-Terminus 基本打平)的前提下,把端到端推理成本砍半(API 降价 50%+)。
- 一句话:**V2 用低秩(MLA)省 KV 显存,V3 用稀疏(MoE)省算力,V3.2 用稀疏(DSA)同时省长上下文的算力与 KV 读取——始终在同一主线:**以稀疏/低秩换效率,质量不倒退**。**

---

## 8. 溯源

- **V3.2-Exp 技术报告(一手)**:DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention — https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf (公式 (1)(2)、Figure 1 DSA 架构、两阶段训练、O(L·k) 复杂度)
- **V3.2 正式版论文**:arXiv:2512.02556 — https://arxiv.org/abs/2512.02556
- **V3.2 官方开源公告**:https://api-docs.deepseek.com/news/news250929 (API 降价 50%+、checkpoint 地址)
- **V3.2 一手源码**:inference/model.py(v32_model.py,922 行,含 Indexer/LayerNorm/rotate_activation/weight_dequant/fp8_index)— https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference ;本地 `/home/cambricon/zhanbin/obrain/.sources/deepseek/v32_model.py`
- **V3.2 真实超参**:config.json — https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/resolve/main/config.json
- **V3 对照源码**:本地 `/home/cambricon/zhanbin/obrain/.sources/deepseek/v3_model.py`(808 行);V3 论文 arXiv:2412.19437
- **R1**:arXiv:2501.12948 — https://arxiv.org/abs/2501.12948 (V3-Base + GRPO + 冷启动/蒸馏)
- **库内交叉**:deepseek-infra/concepts/deepselect.md(DSA 链路前置 TopK 算子)、deepseek-infra/concepts/flashmla.md(FlashMLA sparse 消费 indices 的后端)

---

## 附:V3.2 与 V3 代码 diff 速查(v32 相对 v3 的全部改动行)

| 位置 | v3_model.py | v32_model.py |
|---|---|---|
| import | `act_quant, weight_dequant, fp8_gemm` | `act_quant, fp8_gemm, fp8_index` |
| ModelArgs | 无 index 字段 | +`index_n_heads=64 / index_head_dim=128 / index_topk=2048` |
| RMSNorm | 仅 plain | +`residual` 融合分支 |
| LayerNorm | **无** | **新增**(索引器 k_norm 用) |
| apply_rotary_emb | 仅交织 | +`interleaved` 参数(索引器用 False) |
| rotate_activation | **无** | **新增**(哈达玛旋转) |
| weight_dequant | 从 kernel 导入 | 内联(新签名 `(weight, scale)`) |
| Indexer | **无** | **新增(整类,~53 行)** |
| MLA | 无 indexer | +`self.indexer`、+两条路径各插 index_mask、+`dequant_wkv_b` 懒缓存 |
| Block | post-norm 单流 | dual-residual pre-norm |
