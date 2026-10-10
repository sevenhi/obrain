---
title: DeepSeek-V3 网络结构剖析(MLA + DeepSeekMoE + MTP)
type: concept
tags: [deepseek, v3, mla, moe, mtp, architecture, llm]
series: deepseek-model-evolution
source:
  - "paper: DeepSeek-V3(arXiv:2412.19437)| https://arxiv.org/abs/2412.19437"
  - "code: inference/model.py| https://github.com/deepseek-ai/DeepSeek-V3/tree/main/inference"
  - "code: HF modeling_deepseek.py| https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/blob/main/modeling_deepseek.py"
date: 2026-10
---

# DeepSeek-V3 网络结构剖析

> **主线定位**:本代是 DeepSeek 谱系的**规模化 + 训练工程成熟化**段——沿用 V2 的 MLA（省显存）+ DeepSeekMoE（省算力），把优化重心转向「训练质量与效率」（aux-loss-free、MTP、FP8）。完整演进总览见主篇 [deepseek-model-evolution](deepseek-model-evolution.md);奠基段 V1/DeepSeekMoE/V2 见 [deepseek-arch-v1-moe-v2](deepseek-arch-v1-moe-v2.md);后续 V3.2(DSA 破 O(n²))+ R1 见 [deepseek-arch-v32-r1](deepseek-arch-v32-r1.md)。

> **TL;DR** DeepSeek-V3(671B 总参数 / 37B 激活)把 V2 验证过的两大利器——**MLA(低秩 KV 压缩 + 解耦 RoPE,省 KV cache 显存)** 和 **DeepSeekMoE(1 共享 + 256 细粒度路由专家,top-8,省算力)**——继续做大做强,并在此之上新增两项:**aux-loss-free 免辅助损失负载均衡**(bias 项动态调节,消除负载均衡对模型质量的拉扯)和 **MTP 多 token 预测**(预测额外 1 个 token,密化训练信号 + 让模型"预规划"表示)。三大件组合 = "用稀疏 MoE 省 FLOPs、用低秩 MLA 省显存、用 aux-loss-free + MTP 保训练质量",最终用 2.788M H800 GPU 小时(≈$5.6M)训出超越当时所有开源模型的基座。

---

## 1. 结构总览(层叠结构)

DeepSeek-V3 仍是标准 decoder-only Transformer,顶层结构为 `Embedding → N×DecoderBlock → RMSNorm → LM Head`。

```
                    tokens [B, T]  (Byte-level BPE, vocab=128K)
                              │
                   ┌──────────▼──────────┐
                   │  Embedding (128K,7168)│
                   └──────────┬──────────┘
                              │  h ∈ R^7168
              ┌───────────────▼────────────────┐
              │  × 61  DecoderBlock            │  前 3 层 FFN 为 dense MLP,
              │   ┌────────────────────────┐   │  后 58 层为 DeepSeekMoE
              │   │ RMSNorm → MLA(attention)│──┼──► + residual
              │   │ RMSNorm → MLP/MoE(FFN) │──┼──► + residual
              │   └────────────────────────┘   │
              └───────────────┬────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  RMSNorm (final)    │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  LM Head (7168,128K)│  (独立 lm_head(tie_word_embeddings=false))
                    └────────────────────┘
                              │
                         logits [B, T, 128K]
```

与 V2 结构完全一致的地方:MLA + DeepSeekMoE、RMSNorm、RoPE、YaRN 长上下文扩展。V3 的两处结构级增量:(a) MoE 路由改用 **sigmoid 亲和度 + top-k 归一化**,并引入 **aux-loss-free 负载均衡**;(b) 训练时挂 **MTP 模块**(预测额外 1 个 token),推理时可整体丢弃。

---

## 2. DecoderBlock 内部(代码级结构)

来自仓库 `inference/model.py` 的 `Block`(注意:该文件的 `ModelArgs` 默认值是小型示例,真实 671B 配置见 §5 超参表;结构不变):

```python
class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        self.attn = MLA(args)
        # 前 n_dense_layers 层用 dense MLP,其余用 MoE
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm  = RMSNorm(args.dim)

    def forward(self, x, start_pos, freqs_cis, mask):
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)  # Pre-Norm + residual
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

要点:
- **Pre-Norm(RMSNorm)+ 双残差**;归一化 eps=1e-6。
- 每层归一化后、进 MLA/FFN 前的残差分支,即 standard pre-Norm residual 结构(x = x + attn(attn_norm(x)))(V3 未用 parallel attention/FFN)。
- **dense 层:前 3 层** FFN 是普通 MLP(`SwiGLU` 门控: `down(silu(gate(x)) * up(x))`,inter_dim=18432),后 58 层是 MoE。
- MoE 层的每个专家是一个 `SwiGLU` MLP(moe_inter_dim=2048),共享专家是合并为一个大的 MLP。

---

## 3. MLA(Multi-head Latent Attention)

### 3.1 设计动机

KV cache 是长序列推理的最大显存瓶颈。MHA 下每 token 每层的 KV cache = `2 · n_h · d_h`;GQA/MQA 靠"共享 KV 头"省一点;MLA 更进一步,把 K、V 联合**低秩压缩**成一个小的 latent 向量 `c^KV`,只缓存它和解耦的 RoPE key。同时把 query 也低秩压缩,省训练时的激活显存。

### 3.2 公式(论文 §2.1.1)

KV 联合低秩压缩 + 解耦 RoPE:

```
c_t^KV = W^DKV · h_t                          (1)   down-projection: d → d_c
[k_{t,1}^C;...;k_{t,n_h}^C] = W^UK · c_t^KV  (2)   up-projection: d_c → n_h·d_h
k_t^R  = RoPE( W^KR · h_t )                   (3)   解耦的位置 key,d_h^R 维
k_{t,i} = [ k_{t,i}^C ; k_t^R ]               (4)   拼接(位置信息不被压缩破坏)
[v_{t,1}^C;...;v_{t,n_h}^C] = W^UV · c_t^KV  (5)
```

Query 低秩压缩(省激活内存,训练用):

```
c_t^Q = W^DQ · h_t                            (6)
[q_{t,1}^C;...;q_{t,n_h}^C] = W^UQ · c_t^Q    (7)
q_{t,i}^R = RoPE( W^QR · c_t^Q )              (8)
q_{t,i} = [ q_{t,i}^C ; q_{t,i}^R ]           (9)
```

注意力输出(注意 value 只用 `v^C`,head_dim 不同):

```
o_{t,i} = Σ_j Softmax( q_{t,i}ᵀ k_{j,i} / √(d_h + d_h^R) ) v_{j,i}^C
u_t = W^O [o_{t,1};...;o_{t,n_h}]
```

### 3.3 关键点:只缓存蓝色框向量

推理时**只需缓存 `c_t^KV`(d_c=512)和 `k_t^R`(d_h^R=64)**,共 **576 元素**;而标准 MHA 要缓存 `2·128·128 = 32768` 元素 → **KV cache 缩小约 57×**。这是 MLA 相对 GQA 的本质优势:不是"共享头",而是"低秩重构 + RoPE 解耦"。

### 3.4 源码实现(`inference/model.py` 的 MLA,含 naive 与 absorb 两种注意力实现)

```python
class MLA(nn.Module):
    def __init__(self, args):
        # q 压缩(d_c' = q_lora_rank = 1536)
        self.wq_a = Linear(args.dim, args.q_lora_rank)          # W^DQ
        self.q_norm = RMSNorm(args.q_lora_rank)
        self.wq_b = ColumnParallelLinear(args.q_lora_rank, n_heads * (qk_nope_head_dim + qk_rope_head_dim))  # W^UQ (+W^QR 合并)
        # kv 压缩 + 解耦 RoPE key
        self.wkv_a = Linear(args.dim, args.kv_lora_rank + args.qk_rope_head_dim)  # [W^DKV; W^KR]
        self.kv_norm = RMSNorm(args.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(args.kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim))  # [W^UK; W^UV]
        self.wo = RowParallelLinear(n_heads * v_head_dim, args.dim)                 # W^O

    def forward(self, x, start_pos, freqs_cis, mask):
        # query 低秩路径
        q = self.wq_b(self.q_norm(self.wq_a(x)))     # [bs, seq, n_heads, d_h + d_h^R]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        # kv 压缩 + 解耦 RoPE key
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        # absorb 实现:把 W^UK/W^UV 吸收进 q_nope 与输出投影,直接对 latent 做注意力
        wkv_b = self.wkv_b.weight.view(n_heads, -1, kv_lora_rank)
        q_nope = einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope_head_dim])
        self.kv_cache[...] = self.kv_norm(kv)   # 只缓存 latent c^KV
        self.pe_cache[...] = k_pe.squeeze(2)    # 只缓存解耦 RoPE key
        scores = (einsum("bshc,btc->bsht", q_nope, kv_cache) +
                  einsum("bshr,btr->bsht", q_pe,   pe_cache)) * softmax_scale
        scores = softmax(scores + mask)
        x = einsum("bsht,btc->bshc", scores, kv_cache)
        x = einsum("bshc,hdc->bshd", x, wkv_b[:, -v_head_dim:])  # W^UV 吸收到输出
        return self.wo(x.flatten(2))
```

`absorb` 优化的意义:通过把 `W^UK`、`W^UV` 吸收进 `q_nope` 和输出投影,注意力分数可直接在 `d_c=512` 的 latent 空间计算,无需显式展开到 `n_h·d_h` 的 K/V,进一步省显存和带宽(这也是 FlashMLA 内核优化点)。

---

## 4. DeepSeekMoE(1 共享 + 256 路由,top-8)

### 4.1 FFN 输出公式(论文 §2.1.2)

```
h_t' = u_t + Σ_{i=1}^{N_s} FFN_i^(s)(u_t) + Σ_{i=1}^{N_r} g_{i,t} · FFN_i^(r)(u_t)   (12)
```

- **共享专家** `N_s=1`(所有 token 都过,捕获通用知识/共性);
- **路由专家** `N_r=256`,每 token 激活 `K_r=8` 个;
- **细粒度专家**:每专家 intermediate dim 仅 2048(比传统 MoE 每个专家大得多、数量少的做法更细),专家数多 + 每个小 → 更容易学出专门化。

### 4.2 路由(V3 相对 V2 的改动:sigmoid + 归一化)

```
s_{i,t} = Sigmoid( u_tᵀ e_i )                     (15)  亲和度用 sigmoid(非 softmax)
g'_{i,t} = { s_{i,t}, 若 s_{i,t} ∈ Topk({s_{j,t}}, K_r); 否则 0 }   (14)
g_{i,t}  = g'_{i,t} / Σ_j g'_{j,t}               (13)  归一化(softmax 变体)
```

源码对应 `Gate.forward`:`scores=sigmoid(...) → topk 选 8 个 → weights = 原始分 / Σ(选中分) → 乘 route_scale`。

### 4.3 aux-loss-free 负载均衡(核心创新)

传统 MoE 靠 **auxiliary loss** 强制专家负载均衡,但 aux loss 太大会拉扯模型质量(为均衡牺牲表达)。V3 改为**免辅助损失**方案:

- 给每个专家一个 **bias 项 `b_i`**,加到亲和度上决定 top-k 路由:
  ```
  g'_{i,t} = { s_{i,t},  若 s_{i,t} + b_i ∈ Topk({s_{j,t} + b_j}, K_r); 否则 0 }   (16)
  ```
- **bias 只用于选路,不参与 gating 值的计算**(gating 仍用原始 `s_{i,t}`),所以不扰动梯度/表示;
- 训练时监控整 batch 的专家负载,**每步结束**:过载专家 `b_i -= γ`,欠载专家 `b_i += γ`(`γ` = bias update speed,前 14.3T token 取 0.001,后 500B 取 0.0 冻结);
- 动态调整让负载长期均衡,且**不掉 token**(no token-dropping)。

辅助的 **互补 sequence-wise balance loss**(只在极端失衡时兜底,权重 α=0.0001 极小):

```
L_Bal = α Σ_i f_i · P_i
f_i = (N_r / (K_r T)) Σ_t 𝟙(s_{i,t} ∈ Topk(...))     # 专家 i 被选中的频率
P_i = (1/T) Σ_t s'_{i,t} ,  s'_{i,t} = s_{i,t}/Σ_j s_{j,t}
```

源码对应(HF `MoEGate`, `topk_method="noaux_tc"`):`e_score_correction_bias` 就是 bias 项 `b_i`;选择时用 `scores + bias` 做 topk,权重回落到原始 `scores`。

### 4.4 专家分组 + 节点受限路由(Node-Limited Routing)

为降低训练通信成本,把 256 路由专家按 8 个专家一组分成组,每 token **最多路由到 4 个节点(M=4)**,节点按其上专家的 top-(K_r/M) 亲和度之和选取。训练框架据此实现近完全的计算-通信 overlap。源码 `Gate` 的 `n_groups/n_limited_groups/topk_groups` 即此机制(`group_scores = topk(2).sum(-1)` 再选组)。

### 4.5 MoE 模块结构(源码)

```python
class MoE(nn.Module):
    def __init__(self, args):
        self.gate = Gate(args)                 # 路由(256 个权重向量 + bias)
        self.experts = ModuleList([Expert(dim, moe_inter_dim) for _ in range(256)])  # 每个 SwiGLU MLP
        self.shared_experts = MLP(dim, n_shared_experts * moe_inter_dim)  # 1 个共享专家(合并成一个 MLP)
    def forward(self, x):
        weights, indices = self.gate(x)        # [n, 8] 权重与专家索引
        y = 0
        for i in local_experts:                # 对每个被路由到的专家
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)             # 共享专家恒过
        return (y + z)                          # 路由专家加权和 + 共享专家输出
```

---

## 5. MTP(Multi-Token Prediction,多 token 预测)

### 5.1 动机

预测额外的未来 token,一方面**密化训练信号**提高数据效率,另一方面让模型学会**"预规划"表示**,从而提升主模型在下游 benchmark 上的表现。

### 5.2 模块结构(D=1,即只预测额外 1 个 token)

```
主模型 last hidden: h_i^0  ────────────────┐
                                           ├─► concat [RMSNorm(h_i^0); RMSNorm(Emb(t_{i+1}))]
下个 token 的 embedding: Emb(t_{i+1}) ─────┘         │  (共享 embedding 层)
                                                    ▼
                                    M_k ∈ R^{d × 2d} 线性投影
                                                    ▼
                                        h_i'^k = M_k[·;·]  (k=1 时 h_i^0 来自主模型)
                                                    ▼
                                        TRM_k (一个 Transformer Block)
                                                    ▼
                                        h_i^k ──► OutHead (共享输出头) ──► P_{i+1+k}^k
```

论文公式:

```
h_i'^k = M_k [ RMSNorm(h_i^{k-1}) ; RMSNorm(Emb(t_{i+k})) ]   (21)
h_{1:T-k}^k = TRM_k( h_{1:T-k}'^k )                           (22)
P_{i+k+1}^k = OutHead( h_i^k )                                (23)
L_MTP^k = CrossEntropy( P_{2+k:T+1}^k, t_{2+k:T+1} )          (24)
L_MTP = (λ / D) Σ_{k=1}^D L_MTP^k                              (25)
```

关键设计:
- **顺序预测**(非 Gloeckle 的并行独立头):每个深度保留完整因果链,`h_i^k` 依赖 `h_i^{k-1}` 与 `Emb(t_{i+k})`;原理类似 EAGLE,但 EAGLE 目标是投机解码,V3 用它**提升训练**。
- **共享参数**:MTP 模块的 embedding 层与 output head 与主模型**物理共享**(共享参数 + 梯度)。
- **推理丢弃**:MTP 只服务训练,推理时直接丢弃,主模型独立运行;或反手把它当 **speculative decoding(投机解码)** 加速器用。
- 权重 `λ`:前 10T token 取 0.3,后 4.8T 取 0.1。

---

## 6. 关键超参表(真实 671B 配置)

| 项 | 值 | 说明 |
|---|---|---|
| 总参数量 | **671B** | 每 token 激活 **37B** |
| 层数 `n_layers` | **61** | 前 3 层 dense MLP,后 58 层 MoE |
| hidden dim `d` | **7168** | |
| 注意力头 `n_h` | **128** | |
| 每头维度 `d_h` | 128 | |
| KV 压缩维 `d_c` | **512** | latent c^KV |
| Query 压缩维 `d_c'` | **1536** | latent c^Q |
| 解耦 RoPE 维 `d_h^R` | **64** | q^R / k^R |
| 共享专家 `N_s` | **1** | |
| 路由专家 `N_r` | **256** | 每专家 intermediate=2048 |
| 激活专家 `K_r` | **8** | top-8 |
| 节点受限 `M` | 4 | 每 token 最多路由到 4 节点 |
| dense MLP inter_dim | 18432 | 前 3 层 |
| 词表 | 128K | Byte-level BPE |
| RoPE θ | 10000 | YaRN scale s=40,β_fast=32,β_slow=1 |

> ⚠ 注意:`inference/model.py` 里 `ModelArgs` 的默认值(27 层、dim 2048、64 专家等)是**可运行的小型示例**(`__main__` 自测用),不是 671B 真实配置。真实值见上表与 `config.json`/论文 §4.2。一个佐证:`Gate.__init__` 里 `self.bias = ... if self.dim == 7168 else None`,即只有 dim=7168(真实 V3)才启用 aux-loss-free 的 bias 项。

---

## 7. 数据与训练

| 项 | 值 |
|---|---|
| 预训练 tokens | **14.8T**(多样化高质量语料) |
| 训练集群 | 2048 × NVIDIA H800 |
| 全训练 GPU 小时 | **2.788M H800 GPU 小时** |
| 训练成本(按 $2/GPU·h) | ≈ **$5.576M** |
| 每 1T tokens 成本 | 180K H800 GPU 小时(3.7 天) |
| 优化器 | AdamW,β1=0.9,β2=0.95,wd=0.1 |
| 学习率 | 峰值 2.2e-4(2K 步 warmup)→ 10T 后 cosine 衰减到 2.2e-5 → 末段 7.3e-6 |
| batch size | 3072 → 15360 |
| 梯度裁剪 | 1.0 |
| 精度 | **FP8 混合精度训练**(首例超大规模验证) |
| 上下文 | 4K 预训练 → YaRN 两阶段扩到 32K → 128K |
| 稳定性 | 全程无不可恢复 loss spike、无回滚 |
| 数据技巧 | FIM(PSM 框架,比例 0.1)、document packing(无跨样本 mask) |

---

## 8. 关键 benchmark(指令版/基座)

| Benchmark | DeepSeek-V3 | 对照(开源最优级) |
|---|---|---|
| MMLU | **88.5** | 超越所有开源 |
| MMLU-Redux | 89.1 | |
| MMLU-Pro | **75.9** | |
| GPQA-Diamond | **59.1** | 仅次 Claude 3.5 Sonnet |
| DROP (3-shot F1) | 91.6 | |
| MATH-500 (EM) | **90.2** | 超 o1-preview |
| AIME 2024 (Pass@1) | **39.2** | 领先开源 |
| HumanEval (Pass@1) | 65.2 | |
| LiveCodeBench | 领先开源代码竞赛榜 | |

---

## 9. 与 V2 的代际 diff + 演进动因

| 维度 | V2 | **V3** | 动因 |
|---|---|---|---|
| 注意力 | MLA(d_c=512,d_h^R=64) | MLA **同** | 沿用已验证的省 KV 方案 |
| FFN | 2 共享 + 160 路由,激活 6 | **1 共享 + 256 路由,top-8**,专家更细(inter=2048) | 专家更细 + 更多,增强专门化 |
| 路由亲和度 | softmax | **sigmoid + top-k 归一化** | sigmoid 亲和度更适合 top-k 选择 + 归一化更稳 |
| 负载均衡 | 纯 auxiliary loss | **aux-loss-free(bias 项动态调)** + 极小 sequence-wise loss | 消除 aux loss 对质量的拉扯,兼顾均衡与性能 |
| 训练目标 | 单 token 预测 | **+ MTP(预测额外 1 token)** | 密化信号 + 预规划表示,提升数据效率与下游表现 |
| 训练精度 | BF16 | **FP8 混合精度** | 大规模训练降本提速 |
| 训练 tokens | 8.1T | **14.8T** | 规模扩张 |
| 上下文 | 128K | 128K(YaRN 同) | 沿用 |
| 参数量 | 236B/21B 激活 | **671B/37B 激活** | 容量大幅扩展 |

**演进主线定位**(对应共享简报框架):V3 在"算力(FLOPs)与显存(KV cache)双重约束下"继续走 **稀疏 MoE 省算力 + 低秩 MLA 省显存** 的老路,并把优化重心从"结构省资源"转向"**训练质量与效率**"——用 aux-loss-free 卸掉负载均衡的负作用、用 MTP 强化信号、用 FP8 压成本,是 V2 架构的"规模化 + 训练工程成熟化"一代。后续 R1 在 V3 基座上做 RL 推理时扩展(无新结构),V3.2 再加 DSA 稀疏注意力破 O(n²)。

---

## 10. 溯源

- 论文:DeepSeek-V3 Technical Report,arXiv:2412.19437 — https://arxiv.org/abs/2412.19437 (v1 2024-12-27, v2 2025-02-18)
- 一手源码:
  - `inference/model.py`(仓库 DeepSeek-V3;808 行,含 MLA/MLP/Gate/Expert/MoE/Block/Transformer,MLA 的 naive+absorb 两种实现)
  - HF `modeling_deepseek.py`(1848 行;MoEGate 含 `noaux_tc` aux-loss-free bias、MoE 专家并行、FlashAttention2)
- 模型权重:https://github.com/deepseek-ai/DeepSeek-V3

---

## 附录:一行看懂 V3 三大件怎么组合

> **MLA** 把 KV 压成 512 维 latent + 64 维解耦 RoPE(省 ~57× KV cache)→ **DeepSeekMoE** 用 1 共享 + 256 细粒度路由专家、top-8 + 免辅助损失 bias 均衡(省 ~18× 算力,671B 只激活 37B)→ **MTP** 额外预测 1 个 token 强化训练信号(推理时丢弃/转投机解码)。三者协同 = 显存、算力、训练质量三线同时最优,以最小成本训出最强开源基座。
