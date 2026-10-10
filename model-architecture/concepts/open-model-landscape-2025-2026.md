---
title: 开源模型谱系 2025-2026:DeepSeek V3.2(DSA)/V4 与 Qwen3.x / Llama 4 续篇
type: concept
tags: [open-llm, model-landscape, deepseek, v3.2, v4, dsa, sparse-attention, qwen3, qwen3.8, llama4, moe, mtp, hybrid-thinking]
source:
  - "web: DeepSeek-V3.2 Hugging Face README(含 DSA 介绍与 config) | https://huggingface.co/deepseek-ai/DeepSeek-V3.2 ↑primary"
  - "web: DeepSeek-V3.2-Exp Hugging Face | https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp"
  - "web: DeepSeek-V4-Flash / V4-Pro / V4.1-Flash Hugging Face | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash"
  - "web: Llama 4 Scout / Maverick Hugging Face | https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E"
  - "web: Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3.8 系列 Hugging Face | https://huggingface.co/Qwen"
  - "code: deepseek-ai/FlashMLA(昇腾版,V4.1 服务对象) | https://github.com/deepseek-ai/FlashMLA"
  - "code: deepseek-ai/DeepSelect(DSA 的 TopK 算子) | https://github.com/deepseek-ai/DeepSelect"
date: 2026-10
---

# 开源模型谱系 2025-2026:DeepSeek V3.2(DSA)/V4 与 Qwen3.x / Llama 4 续篇

## TL;DR

本篇是 [open-model-landscape.md](open-model-landscape.md)(2024-12 快照)的**续篇/增量**,覆盖 2025-2026 主线,不替代旧篇。三条主线:

- **DeepSeek 谱系(2025-26)**:R1(2025-01)→ V3-0324 → **V3.1(2025-08-21)** → **V3.2-Exp(2025-09-29)** → **V3.2(2025-12-01)** → V4-Flash/Pro(2026-04-22)→ **V4.1-Flash(2026-09-10)**。**V3.2 的核心增量是 DSA(DeepSeek Sparse Attention)稀疏注意力**——用 indexer 选出每个 token 该 attend 的 top-k KV,长上下文下计算/显存大幅省;配套开源了 DeepSelect(TopK)与 FlashMLA sparse(见 [../../deepseek-infra/index.md](../../deepseek-infra/index.md))。
- **Qwen3.x 谱系**:Qwen3(0.6B~32B,含 **Hybrid 思考模式**)→ **Qwen3.5**(dense 0.8B~27B + MoE 35B-A3B/122B-A10B/397B-A17B)→ **Qwen3.6**(27B / 35B-A3B)→ **Qwen3.8**(27B / **Flash-Next** / **2.4T-A95B 大 MoE**)、Qwen3-Coder-Next。
- **Llama 4**:Llama-4-Scout-17B-16E(109B 总/17B 激活)与 Llama-4-Maverick-17B-128E(400B/17B),原生 MoE、Community License。

**选型速记**:追开源最强推理/长上下文 → DeepSeek-V4/V4.1(原生 MLA + DSA);要中等规模高质量基座 → Qwen3.5/3.6;要原生 MoE 可微调 → Llama 4 Scout/Maverick;要超大 MoE(2.4T)→ Qwen3.8-2.4T-A95B。

> ⚠ **时效警示(与旧篇同款)**:以下数字/日期是 2026-10 抓取 HF API/config 的一手快照,**会过时**;选型务必以 HF 当日 config 为准。

---

## 1. DeepSeek:从 R1 到 V4.1,以及 V3.2 的 DSA

### 1.1 谱系时间线(一手 HF 创建时间 + 官方 README)

| 模型 | 发布时间 | 要点 | 激活参数(HF safetensors 口径) |
|---|---|---|---|
| R1 | 2025-01 | 推理模型(RL 后训练) | ~684.5B 总 |
| V3-0324 | 2025-03 | V3 强化版 | — |
| V3.1 | 2025-08-21 | MoE,`num_experts_per_tok=8` | — |
| **V3.2-Exp** | 2025-09-29 | **DSA 首次亮相**,`index_topk=2048`/`index_n_heads=64`/`index_head_dim=128` | 685.4B |
| **V3.2** | 2025-12-01 | **DSA 正式版**,61 层/hidden 7168/128 Q 头,`max_position_embeddings=163840`,top-8 MoE | 685.4B |
| V4-Flash | 2026-04-22 | 43 层/hidden 4096/64 头/**KV 头=1(MLA)**/vocab 129280/1M ctx/top-6 | 290.9B |
| V4-Pro | 2026-04-22 | 61 层/hidden 7168/128 头/KV 头=1(MLA)/1M ctx/top-6 | ~1598.8B(1.6T) |
| V4.1-Flash | 2026-09-10 | FlashMLA 昇腾版唯一服务对象 | 763.2B |

> 注:V3.2 的 config 同时含 `num_key_value_heads=128`(MHA 形状)与 `index_*` 字段(DSA indexer),`max_position_embeddings=163840`;V4 系转向 `num_key_value_heads=1`(MLA 低秩压缩)。

### 1.2 DSA(DeepSeek Sparse Attention)—— V3.2 的核心增量

DSA 是 V3.2 标题级贡献(README「substantially reduces computational complexity while preserving model performance, specifically optimized for long-context scenarios」)。机制上:

1. **indexer(稀疏索引器)**:为每个 query token 学习一个轻量 indexer,算出该 token 最该 attend 的 top-k KV 位置(config 里 `index_topk=2048`、`index_n_heads=64`、`index_head_dim=128`);
2. **只算稀疏注意力**:query 只对选中的 top-k KV 算 softmax attention,复杂度从 O(N²) 降到 O(N·topk),长上下文收益最大;
3. **配套算子开源**:TopK 用 [DeepSelect](../../deepseek-infra/concepts/deepselect.md)(2~20× vs torch.topk),稀疏注意力内核用 [FlashMLA](../../deepseek-infra/concepts/flashmla.md) sparse(昇腾/NVIDIA 双平台)——**FlashMLA 是实现 DSA 的内核库,DeepSelect 出 indices、FlashMLA 消费 indices,三者是「机制→索引→内核」链路**(详见 deepseek-infra 域)。

V3.2 另两项:可扩展 RL 后训练(对标 GPT-5,高算力变体 V3.2-Speciale 超 GPT-5、2025 IMO/IOI 金牌)与大规模 agentic 任务合成管线(推理进 tool-use)。

### 1.3 V4 系:MoE + MLA 的延续,规模分层

V4-Flash(轻量,290B)与 V4-Pro(旗舰,1.6T)都是 MoE(`num_experts_per_tok=6`)+ MLA(`num_key_value_heads=1`)+ 1M 上下文;V4.1-Flash(2026-09)是 FlashMLA 昇腾版唯一服务对象(该版本移除了 V3/V3.2/V4.0 旧模型支持,breaking change,见 [flashmla.md](../../deepseek-infra/concepts/flashmla.md))。

---

## 2. Qwen3.x:Hybrid 思考模式与 2026 年的三档扩张

### 2.1 谱系(一手 HF 创建时间)

| 模型 | 时间 | 定位 | 规模(HF safetensors) |
|---|---|---|---|
| Qwen3(0.6B~32B) | 2025-04-27 起 | dense,**Hybrid 思考模式**(可开关思考) | 32B ≈ 36.5B |
| Qwen3.5 | 2026-02-24 起 | dense 0.8B/2B/4B/9B/27B + MoE 35B-A3B/122B-A10B/397B-A17B | 122B-A10B ≈ 125B 总 |
| Qwen3.6 | 2026-04-21 起 | 27B / 35B-A3B | 27B ≈ 27.8B |
| Qwen3.8 | 2026-08-05 起 | **27B / Flash-Next / 2.4T-A95B** | 27B ≈ 27.8B;2.4T-A95B = 512 专家 top-10 |
| Qwen3-Coder-Next | 2026-01-30 | 代码专精,Apache 2.0 | — |

### 2.2 要点

- **Qwen3 "Hybrid 思考模式"是特性**:同一模型可在「思考」与「不思考」之间切换(非思考模式跳过 thinking token),是 Qwen3 相对 Qwen2.5 的标志性变化;
- **Qwen3.8-2.4T-A95B**:总参 2.4T 的大 MoE(config `num_experts=512`、`num_experts_per_tok=10`,架构 `Qwen3_5MoeForCausalLM`,license `other` 即 Qwen 许可)——「2.4T 总参大 MoE」定位,不是 dense 2.4T;
- **Qwen3.8-Flash-Next**:架构 `Qwen4ExpForConditionalGeneration`,是 Qwen4 的预览/实验版;
- **授权**:Qwen3-Coder-Next 是 Apache 2.0;大尺寸档(如 2.4T-A95B)仍是 Qwen 许可(`license:other`)——沿用旧篇「≤7B 档 Apache 2.0、大档 Qwen 许可」的许可分层(以 HF tag 为准)。

---

## 3. Llama 4:原生 MoE

| 模型 | 总参 | 激活 | 时间 | 说明 |
|---|---|---|---|---|
| **Llama-4-Scout-17B-16E** | 108.6B | 17B | 2025-04-02 | 16 专家,`Llama4ForConditionalGeneration` |
| **Llama-4-Maverick-17B-128E** | 401.6B | 17B | 2025-04-02 | 128 专家,同架构 |

- **原生 MoE**:Scout 与 Maverick 都是 MoE,区别在专家数(16 vs 128)与总参(109B vs 400B),激活参数都是 17B;
- **Community License**(`license:other`,即 Llama Community License,非 OSI)——沿用旧篇许可结论;
- 命名记号「17B-16E」= 17B 激活 / 16 专家,与 MoE 记号 `A{N}B` 不同,别混。

---

## 4. 横向对比表(2025-2026 增量,与旧篇 §3 表互补)

| 模型 | 年份 | 参数(总/激活) | 上下文 | attention | MoE | 许可 |
|---|---|---|---|---|---|---|
| DeepSeek-R1 | 2025-01 | 684.5B/— | — | MLA(继承 V3) | 是 | MIT |
| DeepSeek-V3.1 | 2025-08 | 685B/— | — | MLA | top-8 | MIT |
| **DeepSeek-V3.2** | 2025-12 | 685.4B/— | 163840 | MLA + **DSA 稀疏**(index_topk 2048) | top-8 | MIT |
| DeepSeek-V4-Flash | 2026-04 | 290.9B/— | 1048576 | MLA(KV 头=1) | top-6 | MIT |
| DeepSeek-V4-Pro | 2026-04 | 1.6T/— | 1048576 | MLA(KV 头=1) | top-6 | MIT |
| Qwen3-32B | 2025-04 | 32.8B dense | — | GQA | — | apache-2.0 |
| Qwen3.5-122B-A10B | 2026-02 | 125B/10B | — | GQA | MoE | Qwen 许可 |
| Qwen3.8-2.4T-A95B | 2026-08 | 2.4T/95B | — | GQA | 512 专家 top-10 | Qwen 许可 |
| Qwen3-Coder-Next | 2026-01 | — | — | GQA | — | Apache 2.0 |
| Llama-4-Scout-17B-16E | 2025-04 | 108.6B/17B | — | GQA | 16 专家 | Community |
| Llama-4-Maverick-17B-128E | 2025-04 | 401.6B/17B | — | GQA | 128 专家 | Community |

> 数字口径:参数 = HF API safetensors.total 实测(2026-10);日期 = HF API createdAt;「—」= 未在本篇抓取到的字段。旧篇 open-model-landscape.md §3 表含 GPT-3→DeepSeek-V3 的完整 2024 快照,两表并读才是完整谱系。

---

## 5. 实践要点(选型准则,续旧篇)

1. **DSA 系模型的长上下文收益最大**:DeepSeek V3.2/V4 的稀疏注意力在长上下文才省钱;短上下文下 indexer 开销可能得不偿失——选型看你的平均序列长度。
2. **DSA 有配套算子依赖**:要完整跑 DSA,需要 DeepSelect(TopK)+ FlashMLA sparse(昇腾仅 bfloat16、indices_type 仅 int32),部署栈见 [../../deepseek-infra/practices/ascend-deployment-checklist.md](../../deepseek-infra/practices/ascend-deployment-checklist.md)。
3. **「2.4T」是总参不是激活**:Qwen3.8-2.4T-A95B 激活 95B,显存按 2.4T 估、速度按 95B 估(与旧篇 MoE 记账一致)。
4. **许可照旧**:Llama Community(非 OSI)、Qwen 大档 Qwen 许可、DeepSeek MIT、Qwen3-Coder-Next Apache 2.0。
5. **与旧篇互链**:本文只补 2025-26 增量,GPT→LLaMA→Mistral→DeepSeek-V3 的完整推导、MoE router/top-k 机制、aux-loss-free 负载均衡、MTP 介绍仍在 [open-model-landscape.md](open-model-landscape.md)。

---

## 6. 溯源

- **DeepSeek-V3.2**(README + config:`index_topk=2048`/`index_n_heads=64`/`index_head_dim=128`/`max_position_embeddings=163840`;DSA 介绍):https://huggingface.co/deepseek-ai/DeepSeek-V3.2 ;Exp 版 https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- **DeepSeek-V4 系**:https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash / V4-Pro / V4.1-Flash(config 均 `num_key_value_heads=1`、V4-Flash 43 层/4096 hidden、V4-Pro 61 层/7168 hidden)
- **Llama 4**:https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E / Llama-4-Maverick-17B-128E(safetensors.total 108.6B / 401.6B)
- **Qwen3 系**:https://huggingface.co/Qwen (Qwen3/Qwen3.5/Qwen3.6/Qwen3.8/Qwen3-Coder-Next,HF API 实测)
- **DSA 算子链**:[../../deepseek-infra/concepts/deepselect.md](../../deepseek-infra/concepts/deepselect.md)、[../../deepseek-infra/concepts/flashmla.md](../../deepseek-infra/concepts/flashmla.md)
- **库内交叉**:[open-model-landscape.md](open-model-landscape.md)(2024-12 前篇)、[llm-architecture.md](llm-architecture.md)(组件演进)、[attention-heads-mha-mqa-gqa.md](attention-heads-mha-mqa-gqa.md)(MLA/GQA 机制)。
