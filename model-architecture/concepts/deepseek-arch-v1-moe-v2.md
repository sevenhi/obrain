---
title: DeepSeek 网络结构剖析 — V1(DeepSeek-LLM) + DeepSeekMoE + V2
type: concept
tags: [deepseek, architecture, transformer, mha, gqa, mla, moe, rope, kv-cache]
series: deepseek-model-evolution
source:
  - "paper: DeepSeek LLM / V1(arXiv:2401.02954)| https://arxiv.org/abs/2401.02954"
  - "paper: DeepSeekMoE(arXiv:2401.06066)| https://arxiv.org/abs/2401.06066"
  - "paper: DeepSeek-V2(arXiv:2405.04434)| https://arxiv.org/abs/2405.04434"
  - "code: HF modeling_deepseek.py(ds_hf_v2.py, 1906 行)| https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py"
  - "config: HF config.json(deepseek-llm-7b/67b-base, deepseek-moe-16b-base, DeepSeek-V2)| https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json"
date: 2026-10
---

# DeepSeek 网络结构剖析：V1(DeepSeek-LLM) → DeepSeekMoE → V2

> **主线定位**:本代是 DeepSeek 谱系的**奠基段**——注意力 `MHA → GQA → MLA`、FFN `dense → 稀疏 MoE` 两条主线的起点与第一次合流。完整演进总览(逐代对照 + 代际 diff 总表 + 演进方向分析)见主篇 [deepseek-model-evolution](deepseek-model-evolution.md);后续 V3(MLA 规模化 + MTP + aux-loss-free)与 V3.2+R1(DSA 破 O(n²))见 [deepseek-arch-v3](deepseek-arch-v3.md)、[deepseek-arch-v32-r1](deepseek-arch-v32-r1.md)。

> **TL;DR**：三代结构沿两条主线演进——**注意力**从 `MHA → GQA → MLA`（低秩压缩 KV cache），**FFN** 从 `dense → 稀疏 MoE（细粒度专家 + 共享专家）`。V1 是标准 dense decoder-only Transformer（7B 纯 MHA，67B 已用 GQA）；DeepSeekMoE 在 dense 骨架上把 FFN 换成「细粒度 + 共享专家」的稀疏 MoE；V2 同时引入 **MLA（Multi-head Latent Attention，KV 低秩压缩 + 解耦 RoPE）** 和 **DeepSeekMoE**，总参 236B / 激活 21B，KV cache 比 V1 67B 降 93.3%，训练成本省 42.5%。一句话主线：**算力(FLOPs)与显存(KV cache)双重约束下，用稀疏省算力、用低秩省显存，始终追求「训练质量-部署效率」双目标。**

---

## 0. 三代超参总表

下图把三代骨架并排摆开:V1 是 Llama 式 dense,V2 的两处改动正好落在两个可以「替换」的位置上——**注意力先换成 MLA(省显存),FFN 再换成 MoE(省算力)**:

<div class="archify-figure">
<iframe src="../../../diagrams/ma-ds-gen-stack.workflow.html"
        style="width:100%;height:680px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="V1 / DeepSeekMoE / V2 三代骨架对照"></iframe>
</div>

| 维度 | V1-7B | V1-67B | DeepSeekMoE-16B | DeepSeekMoE-145B | V2 |
|---|---|---|---|---|---|
| 类型 | dense | dense | 稀疏 MoE | 稀疏 MoE | MLA + 稀疏 MoE |
| hidden_size | 4096 | 8192 | 2048 | 4096 | 5120 |
| 层数 | 30 | 95 | 28 | 62 | 60 |
| 注意力头(查询头) | 32 | 64 | 16 | 32 | 128 |
| KV 头 | 32 (MHA) | **8 (GQA)** | 16 (MHA) | 32 (MHA) | **MLA**(低秩) |
| 头维 | 128 | 128 | 128 | 128 | qk_nope 128 + qk_rope 64 |
| 注意力机制 | MHA | GQA | MHA | MHA | **MLA** |
| intermediate_size(FFN) | 11008 | 22016 | 10944(dense) | — | 12288(dense) |
| MoE 路由专家 | — | — | 64 | 128 | 160 |
| MoE 共享专家 | — | — | 2 | 4 | 2 |
| 每 token 激活路由专家 | — | — | 6 | 12 | 6 |
| 专家相对 FFN 大小 | — | — | 0.25×(相对标准专家,论文 Table 口径) | 0.125× | ~0.125×(moe_inter=1536) |
| 总参数 | 7B | 67B | 16.4B | 144.6B | 236B |
| 激活参数 | 7B | 67B | 2.8B | 22.2B | 21B |
| 词表 | 102400 | 102400 | 102400 | 102400 | 102400 |
| 上下文 | 4096 | 4096 | 4096 | 4096 | 128K (163840) |
| 训练 tokens | 2T | 2T | 2T | 245B(初版) | 8.1T |
| 位置编码 | RoPE | RoPE | RoPE | RoPE | **解耦 RoPE** + YaRN |

> 注：V1 论文 arXiv ID 常被误写为 2302.09500（实为 DeepSeek LLM，正确 ID **2401.02954**）。V1-67B 的 GQA 是 DeepSeek 在稠密模型上对 KV cache 的第一次压缩尝试，是通往 MLA 的铺垫。

---

## 1. V1 — DeepSeek-LLM 7B / 67B（标准 dense decoder-only Transformer）

### 1.1 定位

DeepSeek 第一个开源 LLM 项目，目标是「用 scaling law 指导长期视角的开源模型」。结构上**无任何稀疏/低秩创新**，是 Llama 式的 dense decoder-only Transformer：Pre-Norm + RMSNorm + SwiGLU + RoPE。它的价值在于是后续所有代的**基座和对照系**（V2 论文里「相比 DeepSeek 67B」的参照物就是它）。

### 1.2 层叠结构

```
Embedding (vocab 102400 → hidden)
      │
      ├─ DecoderBlock × N
      │      │
      │      ├─ input_layernorm (RMSNorm)
      │      ├─ self_attn (MHA / GQA + RoPE)
      │      │      └─ residual +
      │      ├─ post_attention_layernorm (RMSNorm)
      │      └─ mlp (SwiGLU, dense)
      │             └─ residual +
      │
      └─ final RMSNorm
             └─ lm_head (hidden → vocab, bias=False)
```

### 1.3 DecoderBlock 内部（pytorch 结构）

```python
# V1-7B: 纯 MHA；V1-67B: GQA(num_key_value_heads=8)
class LlamaDecoderLayer(nn.Module):
    def __init__(self):
        self.input_layernorm       = LlamaRMSNorm(hidden)          # Pre-Norm
        self.self_attn             = LlamaAttention(...)           # MHA/GQA + RoPE
        self.post_attention_layernorm = LlamaRMSNorm(hidden)
        self.mlp                   = LlamaMLP(...)                 # SwiGLU dense

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))           # 注意子层
        x = x + self.mlp(self.post_attention_layernorm(x))        # FFN 子层
        return x

# SwiGLU FFN: gate + up 两路点乘,再 down
class LlamaMLP(nn.Module):
    def __init__(self, hidden, intermediate):
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

### 1.4 关键点

- **7B = 纯 MHA**（32 query 头 = 32 KV 头）；**67B = GQA**（64 query 头，8 KV 头，8:1 压缩）。这是 DeepSeek 首次在规模压力下对 KV cache 做压缩——KV 头从 64 压到 8，KV cache 直接降到 1/8。这是「省显存」主线的萌芽。
- **RoPE**（rope_theta=10000）作用在 query 和 key 上，每头 128 维。
- **Pre-Norm + RMSNorm**（eps=1e-6），与 Llama 一致。
- 词表 102400（大词表），无 tie_word_embeddings。
- 训练 2T tokens；7B/67B 超参完全由 scaling law 推导。

---

## 2. DeepSeekMoE — 细粒度专家 + 共享专家（FFN 稀疏化）

### 2.1 定位

第一个结构创新。**注意力仍是 MHA，骨架仍是 dense Transformer，唯一改动是：把除第一层外的所有 FFN 换成稀疏 MoE。** 目标是解决经典 MoE（GShard 式 top-K）的**专家专精不足**问题。

### 2.2 两大机制

1. **细粒度专家分割（fine-grained expert segmentation）**：把标准大小的专家再切小、数量增多，激活更多小专家。好处是激活专家组合更灵活，知识更聚焦，减少冗余。论文：把 N 个标准专家切成 `mN` 个小专家、激活 `mK` 个。
2. **共享专家隔离（shared expert isolation）**：单独设 `Ks` 个「共享专家」永远激活（不参与路由），负责捕获跨 token 的共性知识；路由专家专注差异化知识，从而减少路由专家间的冗余。

### 2.3 层叠 / MoE 层结构

下图把 MoE 层内部摊开:一个 token 同时走两条路——**恒激活的共享专家(2 个)和门控选出的路由专家(6 个)**,最后相加。注意共享专家完全不过门控,这是「隔离」二字的字面含义:

<div class="archify-figure">
<iframe src="../../../diagrams/ma-ds-moe-route.workflow.html"
        style="width:100%;height:660px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="MoE 层内部:共享专家 + 路由专家"></iframe>
</div>

```
Embedding → DecoderBlock × N → final Norm → lm_head
DecoderBlock:
   input_norm → self_attn(MHA+RoPE) → +resid
   post_norm  → MoE 层(除首层外)      → +resid
                │
                ├─ Router(Gate): 线性 + softmax → top-k 路由专家
                ├─ Shared Experts × Ks   (恒激活)
                └─ Routed Experts × mN   (激活 top-mK)
```

### 2.4 路由 + 输出（DeepSeekMoE 公式）

```
h^l_t = u^l_t + Σ_{i=1}^{Ks} FFN^l_i(u^l_t)                 # 共享专家(恒激活)
             + Σ_{i=1}^{mK} g_{i,t} · FFN^l_i(u^l_t)         # 路由专家(top-mK,加权)
g_{i,t} = softmax 后的 top-k 分数(或 s_{i,t})
```
论文里每个专家相对于标准 FFN 的缩放系数：16B 用 0.25×，145B 用 0.125×（口径注：该 0.25× 相对「标准专家」即论文 Table 口径，非相对 dense FFN——16B 的 moe_intermediate_size=1408 对 10944 的 dense FFN 约 0.129，1408 数值引自 DeepSeekMoE 论文 Table 口径，非 V3 推理 demo config）。

### 2.5 三代规模超参

| 规模 | 层数 | hidden | 头数 | 共享专家 | 路由专家 | 激活路由 | 专家大小 | 总参 | 激活参 |
|---|---|---|---|---|---|---|---|---|---|
| 2B | — | — | — | 1 | 63 | 6 | 0.25×(相对标准专家) | 2B | ~0.2B |
| 16B | 28 | 2048 | 16 | 2 | 64 | 6 | 0.25×(相对标准专家) | 16.4B | 2.8B |
| 145B | 62 | 4096 | 32 | 4 | 128 | 12 | 0.125× | 144.6B | 22.2B |

关键结论（论文实测）：
- DeepSeekMoE 2B ≈ GShard 2.9B 效果（后者专家参数/算力是其 1.5×）。
- DeepSeekMoE 16B ≈ LLaMA2 7B，**仅 ~40% 算力**；激活参数仅 2.8B。
- DeepSeekMoE 145B ≈ DeepSeek 67B，**仅 28.5%（甚至 18.2%）算力**。
- 共享专家 : 激活路由专家 = 1:3 是经验最优比。

### 2.6 首层不换 MoE 的原因

论文观察：**首层的负载均衡状态收敛特别慢**，所以第一层保持 dense FFN（`first_k_dense_replace=1`，V2 也继承了这个设计）。

---

## 3. V2 — MLA + DeepSeekMoE（注意力与 FFN 同时革命）

### 3.1 定位

236B 总参 / 21B 激活 / 128K 上下文。同时做两件事：**注意力换成 MLA**（省 KV cache），**FFN 换成 DeepSeekMoE**（省算力）。这是「双约束」主线第一次在同一代里合流。相比 V1-67B：训练成本 -42.5%，KV cache -93.3%，生成吞吐 ×5.76。

### 3.2 层叠结构（源码提炼）

```python
class DeepseekV2Model(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(vocab=102400, hidden=5120)   # 大词表
        self.layers = nn.ModuleList([
            DeepseekV2DecoderLayer(config, i)
            for i in range(config.num_hidden_layers)                  # 60 层
        ])
        self.norm = DeepseekV2RMSNorm(5120, eps=1e-6)                 # final norm

class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        self.self_attn = DeepseekV2Attention(config, layer_idx)       # MLA
        # 首层(first_k_dense_replace=1)用 dense MLP,其余全部 MoE(moe_layer_freq=1)
        self.mlp = (DeepseekV2MoE(config)
                    if (config.n_routed_experts is not None
                        and layer_idx >= config.first_k_dense_replace
                        and layer_idx % config.moe_layer_freq == 0)
                    else DeepseekV2MLP(config))
        self.input_layernorm = DeepseekV2RMSNorm(5120)
        self.post_attention_layernorm = DeepseekV2RMSNorm(5120)

    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, ...)            # MLA
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)                       # MoE/dense
        hidden_states = residual + hidden_states
        return hidden_states
```

### 3.3 MLA —— 核心机制（源码级还原）

MLA 的出发点：MHA 的 KV cache 是 `每层 2 × heads × head_dim`，长上下文下成为显存瓶颈。GQA 只是减少 KV **头数**，MLA 更进一步：把 K 和 V 整体压成一个**低秩隐向量 c^KV_t**（`kv_lora_rank=512` 维），KV cache 只需缓存这一个向量（外加 64 维的解耦 RoPE 位置分量），实现**维度级**压缩。

下图是 MLA 最该先看懂的一张:上排是计算路径,下排是真正驻留显存的东西。**升维出的 128 头 K/V 是每次 decode 临时算的,不进 cache**——这就是 576 vs 32768 的来源:

<div class="archify-figure">
<iframe src="../../../diagrams/ma-ds-mla-kv.dataflow.html"
        style="width:100%;height:430px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="MLA 的 KV cache:576 维 vs 32768 维"></iframe>
</div>

```python
class DeepseekV2Attention(nn.Module):
    def __init__(self, config):
        self.num_heads   = 128
        self.q_lora_rank = 1536              # Q 的低秩压缩维
        self.kv_lora_rank= 512               # KV 的低秩压缩维  d_c
        self.qk_rope_head_dim = 64           # 解耦 RoPE 的每头维 d^R
        self.qk_nope_head_dim = 128          # 不参与 RoPE 的每头维 d^N
        self.v_head_dim  = 128
        self.q_head_dim  = 128 + 64 = 192    # = nope + rope

        # --- Query: 先压缩再升维(低秩 Q,省训练参数) ---
        self.q_a_proj    = nn.Linear(5120, 1536)              # W^DQ
        self.q_a_layernorm= DeepseekV2RMSNorm(1536)
        self.q_b_proj    = nn.Linear(1536, 128*192)           # W^UQ

        # --- KV: 一次投影,同时得到 低秩KV(512) + 解耦RoPE分量(64) ---
        self.kv_a_proj_with_mqa = nn.Linear(5120, 512+64)     # W^DKV + W^KR
        self.kv_a_layernorm = DeepseekV2RMSNorm(512)
        self.kv_b_proj    = nn.Linear(512, 128*(128+128))     # W^UK + W^UV

        self.o_proj       = nn.Linear(128*128, 5120)          # W^O

    def forward(self, hidden_states, position_ids, past_key_value, ...):
        # 1) Query: 低秩压缩 → 解耦(无位置/有位置两部分)
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, 128, 192).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [128, 64], dim=-1)      # q_nope: 不旋转; q_pe: 旋转

        # 2) KV: 一次投影出 [低秩KV | 解耦RoPE分量]
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [512, 64], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, 64).transpose(1, 2)   # 共享的旋转分量(MQA式,仅1头)

        # 3) 低秩KV 升维成 每头 K_nope 和 V
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(bsz, q_len, 128, 128+128).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [128, 128], dim=-1)

        # 4) 只对 q_pe / k_pe 施加 RoPE(解耦:位置信息不经过低秩压缩,不被破坏)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        # 5) 拼回完整 q/k
        query_states = cat([q_nope, q_pe], dim=-1)            # [b,128,q_len,192]
        key_states   = cat([k_nope, k_pe], dim=-1)            # [b,128,q_len,192]

        # 6) KV cache 只存 key_states 和 value_states(注意:推理时其实只需缓存 c^KV + k_pe)
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, ...)

        # 7) 标准 attention(QK^T → softmax → AV),再 o_proj
        attn = softmax(Q @ K^T / sqrt(192)) @ V
        return self.o_proj(attn)
```

**MLA 的三个设计精髓：**

1. **KV 低秩压缩（省显存）**：`KV cache = c^KV_t(512维) + k_pe(64维) = 576 维/层(全头共享一份 latent)`，而不是 MHA 的 `2 × 128头 × 128维`。这是 KV cache -93.3% 的来源。
2. **解耦 RoPE（Decoupled RoPE）**：若把 RoPE 直接作用在低秩压缩后的 K 上，位置信息会与「被所有头共享的压缩表示」纠缠，导致**解压时位置信息失效**。解耦做法是——把每头的 K/Q 拆成「不旋转部分 q_nope/k_nope」和「旋转部分 q_pe/k_pe」，RoPE 只作用在后者，且 k_pe 是**所有头共享的 1 头 MQA 式分量**（`k_pe.view(bsz, q_len, 1, 64)`）。位置信息因此绕过低秩瓶颈、独立存储。

下图用一条反证链把第 2 点讲透:先看「在压缩表示上旋转」为什么必然坏,再看到底拆成了什么。**核心矛盾是「压缩要求所有头共享、RoPE 要求每头每位置不同」，两者不能共存于同一段向量**：

<div class="archify-figure">
<iframe src="../../../diagrams/ma-ds-decouple-rope.workflow.html"
        style="width:100%;height:640px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="解耦 RoPE 的反证链"></iframe>
</div>

3. **Q 也低秩**（省训练参数/显存）：`q_a_proj(5120→1536) → q_b_proj(1536→128×192)`，1536 << 5120，降低激活显存。

### 3.4 MLA 的 KV cache 量化

下图把这份表画成三档对照,关键是看清 **GQA 砍「头数」、MLA 砍「维度」是两条不同的路径**——MLA 并没有让多个 Q 共享一组 K/V,而是让 K/V 从一个小 latent 现算出来:

<div class="archify-figure">
<iframe src="../../../diagrams/ma-ds-kv-compare.architecture.html"
        style="width:100%;height:560px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="MHA / GQA / MLA 的 KV cache 对比"></iframe>
</div>

| 方案 | 每层每 token 的 KV 维度 |
|---|---|
| MHA（128 头，128 维） | 2 × 128 × 128 = 32768 |
| GQA（8 KV 头，128 维） | 2 × 8 × 128 = 2048 |
| **MLA**（c^KV 512 + k_pe 64） | 512 + 64 = **576** |

576 vs 32768 ≈ **1/57**，这就是 MLA 相对 MHA 的显存压缩比；相对 GQA 也再压 ~3.6×。

### 3.5 V2 的 MoE —— 细粒度 + 共享专家 + 分组路由（源码级）

V2 复用 DeepSeekMoE 哲学，但规模更大、路由更精细：

```python
class MoEGate(nn.Module):
    def __init__(self, config):
        self.top_k = 6                                    # 每 token 激活 6 个路由专家
        self.n_routed_experts = 160                       # 160 个路由专家
        self.routed_scaling_factor = 16.0                 # 路由分数放大系数
        self.norm_topk_prob = False                       # 用 scaling_factor 而非归一化
        self.topk_method = "group_limited_greedy"         # 分组受限贪心路由
        self.n_group = 8                                  # 8 组
        self.topk_group = 3                               # 先选 3 组,再组内 top-k
        self.scoring_func = "softmax"
        self.seq_aux = True                               # 序列级 aux loss
        self.alpha = 0.001                                # aux loss 权重
        self.weight = nn.Parameter(torch.empty(160, 5120))# 门控权重

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight)     # [*, 160]
        scores = logits.softmax(dim=-1)
        if self.topk_method == "group_limited_greedy":
            # 先把 160 专家分 8 组,取每组 max 得分 → 选 top-3 组
            group_scores = scores.view(-1, 8, 20).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=3).indices
            group_mask = ...                              # 只在选中的 3 组内继续
            tmp_scores = scores.masked_fill(~group_mask, 0.0)
            topk_weight, topk_idx = torch.topk(tmp_scores, k=6)
        # 非归一化:乘 routed_scaling_factor(16.0)
        topk_weight = topk_weight * self.routed_scaling_factor
        # 训练时附加 aux loss(序列级),用于负载均衡
        aux_loss = ... if self.training and self.alpha > 0.0 else None
        return topk_idx, topk_weight, aux_loss

class DeepseekV2MoE(nn.Module):
    """含共享专家的混合专家模块"""
    def __init__(self, config):
        self.experts = nn.ModuleList([
            DeepseekV2MLP(config, intermediate_size=1536)   # 每专家 SwiGLU, 1536 维
            for _ in range(160)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:             # 2 个共享专家
            self.shared_experts = DeepseekV2MLP(config, intermediate_size=1536*2)

    def forward(self, hidden_states):
        identity = hidden_states
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        # 路由专家:token 分派 → 各专家计算 → 按 topk_weight 加权求和
        y = Σ_i topk_weight_i * expert_i(x)
        # 共享专家恒激活,结果直接相加
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y
```

**V2 MoE 关键设计点：**

- **细粒度专家**：160 个路由专家，每专家 `moe_intermediate_size=1536`（相对 dense FFN 的 12288 是 0.125×，与 DeepSeekMoE-145B 同粒度）。
- **共享专家**：2 个（`intermediate_size=1536×2=3072`），恒激活。
- **每 token 激活 6 个路由专家**（top-6），激活参 ≈ 21B。
- **分组受限贪心路由**（group-limited greedy）：160 专家分 8 组 → 先按组内 max 得分选 top-3 组 → 再在组内选 top-6 专家。这比全量 top-6 更能保证「专家在组间均衡分布」，缓解负载不均。

下图把这个两级选择画出来:**第一级只在「组」这一层竞争,第二级才在选中的组内挑专家**——它的作用不是选得更准,而是把流量摊开:

<div class="archify-figure">
<iframe src="../../../diagrams/ma-ds-v2-group-route.workflow.html"
        style="width:100%;height:640px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="V2 分组受限贪心路由的两级选择"></iframe>
</div>

- **路由分数**：softmax 后**不归一化**，而是乘 `routed_scaling_factor=16.0`（放大信号）。
- **aux loss 负载均衡**：训练时用**序列级** aux loss（`seq_aux=True`，α=0.001），V2 仍是「aux loss」方案；到 V3 才换成 aux-loss-free。

### 3.6 V2 完整 config.json（一手，权威超参）

```json
{
  "hidden_size": 5120, "num_hidden_layers": 60,
  "num_attention_heads": 128, "num_key_value_heads": 128,
  "q_lora_rank": 1536, "kv_lora_rank": 512,
  "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
  "n_routed_experts": 160, "n_shared_experts": 2,
  "num_experts_per_tok": 6, "moe_intermediate_size": 1536,
  "intermediate_size": 12288, "n_group": 8, "topk_group": 3,
  "topk_method": "group_limited_greedy", "routed_scaling_factor": 16.0,
  "first_k_dense_replace": 1, "moe_layer_freq": 1,
  "hidden_act": "silu", "vocab_size": 102400,
  "max_position_embeddings": 163840, "rope_theta": 10000,
  "rope_scaling": {"type": "yarn", "factor": 40,
                   "original_max_position_embeddings": 4096,
                   "mscale": 0.707, "mscale_all_dim": 0.707,
                   "beta_fast": 32, "beta_slow": 1},
  "aux_loss_alpha": 0.001, "seq_aux": true,
  "rms_norm_eps": 1e-06, "attention_bias": false,
  "tie_word_embeddings": false, "torch_dtype": "bfloat16"
}
```

**关键超参解读：**
- 128K 上下文靠 **YaRN** 外推（RoPE base 10000，factor=40，训练时 4096 → 推理 163840），`mscale=0.707` 是 YaRN 的注意力缩放。
- `num_key_value_heads=128` 在 config 里与 query 头相同，是因为 MLA 的 KV 压缩维度由 `kv_lora_rank` 独立决定，此字段已失去传统「GQA KV 头数」的含义。

---

## 4. 代际 diff 与演进动因

| 代 | 注意力 | FFN | 核心创新 | 动因 |
|---|---|---|---|---|
| V1-7B | MHA | dense | 无（scaling law 基座） | 打底、验证 scaling law |
| V1-67B | **GQA**(8 KV头) | dense | KV 头数压缩 8:1 | 规模大了 KV cache 顶不住 → 省显存 |
| DeepSeekMoE | MHA | **稀疏 MoE**(细粒度+共享) | 专家专精 + 稀疏省算力 | 容量不变、算力大降 → 省算力 |
| V2 | **MLA**(低秩 KV + 解耦 RoPE) | **稀疏 MoE**(160+2,top-6,分组路由) | 两条主线合流 | 双约束：省算力 + 省显存 |

**为什么这么演进（对齐简报「演进方向框架」）：**

1. **注意力 MHA→GQA→MLA**：本质是 **KV cache 显存约束**驱动的。KV cache ∝ 层数 × KV头数 × 头维 × 序列长。GQA 砍「头数」，MLA 砍「维度」——用 `c^KV`(512维) 这个低秩隐向量取代完整的 K/V，把每 token 的 KV 从 32768 维压到 576 维。**解耦 RoPE** 是为了让位置信息绕过低秩压缩这个「多头共享瓶颈」：若 RoPE 直接旋在压缩后的 K 上，解压到多头时位置信息会失效，所以把位置相关的部分(q_pe/k_pe)独立出来单独旋转、单独缓存。

2. **FFN dense→稀疏 MoE**：本质是 **算力(FLOPs)约束**驱动的。dense 模型每个 token 都要过完整 FFN；MoE 只激活 6/160 个专家，激活参数 21B/236B。**细粒度专家**让激活组合更灵活、知识更聚焦，**共享专家**捕获共性知识减少路由专家冗余，两者合力「用更少算力逼近同容量 dense 上限」。

3. **位置编码**：全程 RoPE；V2 在 MLA 里做**解耦 RoPE**，并用 YaRN 外推到 128K。

4. **一句话主线**：在算力(FLOPs)与显存(KV cache)双重约束下，用**稀疏(MoE)省算力**、用**低秩(MLA)省显存**，始终「训练质量-部署效率」双目标。V2 是这两条主线的第一次合流，V3 在其上叠加 MTP 与 aux-loss-free，V3.2 再叠加 DSA 稀疏注意力（破 O(n²)）。

---

## 5. 溯源

- DeepSeek LLM (V1)：https://arxiv.org/abs/2401.02954 （注：任务简报中的 2302.09500 为错误 ID，指向一篇数学论文）
- DeepSeekMoE：https://arxiv.org/abs/2401.06066
- DeepSeek-V2：https://arxiv.org/abs/2405.04434
- V2 一手源码：HF modeling_deepseek.py 在线版（约 1900 行）— https://github.com/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py
- 一手 config：HF `deepseek-ai/deepseek-llm-7b-base`、`deepseek-ai/deepseek-llm-67b-base`、`deepseek-ai/deepseek-moe-16b-base`、`deepseek-ai/DeepSeek-V2` 的 config.json

> 本文所有代码块结构均来自 V2 一手源码（HF modeling_deepseek.py 在线版，见上溯源链接）与各代 HF config.json 的真实结构，超参均为一手数值，无杜撰。

## 相关条目(库内互链)

- DeepSeekMoE 的**数据规模与评测**(40% 算力打平 DeepSeek 7B/LLaMA2 7B 等卖点,本篇只讲结构)见 [deepseek-models/concepts/deepseek-moe.md](../../deepseek-models/concepts/deepseek-moe.md)。
- 领域模型谱系(数学/定理证明/代码,复用本系列结构的底座)见 [deepseek-models/index.md](../../deepseek-models/index.md)。
