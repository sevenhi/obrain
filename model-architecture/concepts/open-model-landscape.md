---
title: 开源模型架构谱系:GPT→LLaMA→Mistral→DeepSeek/Qwen 与 MoE 路线
type: concept
tags: [open-llm, model-landscape, llama, mistral, mixtral, deepseek, qwen, rwkv, moe, mixture-of-experts, sparse-activation, gqa, mla, sliding-window, decoder-only, apache-2.0]
source:
  - "pdf: Mixtral of Experts (primary, MoE 结构全表) | arXiv:2401.04088 (Mistral AI, 2024-01) | https://arxiv.org/abs/2401.04088"
  - "pdf: DeepSeek-V3 Technical Report (primary, MoE+MLA 当前开源旗舰) | arXiv:2412.19437 (2024-12) | https://arxiv.org/abs/2412.19437"
  - "pdf: DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model | arXiv:2405.04434 | https://arxiv.org/abs/2405.04434"
  - "pdf: DeepSeekMoE: Towards Ultimate Expert Specialization | arXiv:2401.06066 (Dai et al. 2024-01) | https://arxiv.org/abs/2401.06066"
  - "pdf: Mistral 7B | arXiv:2310.06825 | https://arxiv.org/abs/2310.06825"
  - "pdf: Qwen2 Technical Report (GQA/DCA+YaRN/MoE 配置表) | arXiv:2407.10671 | https://arxiv.org/abs/2407.10671"
  - "pdf: Qwen2.5 Technical Report (18T tokens) | arXiv:2412.15115 | https://arxiv.org/abs/2412.15115"
  - "pdf: Qwen Technical Report (初代) | arXiv:2309.16609 | https://arxiv.org/abs/2309.16609"
  - "pdf: RWKV: Reinventing RNNs for the Transformer Era | arXiv:2305.13048 (Peng et al. 2023) | https://arxiv.org/abs/2305.13048"
  - "pdf: The Llama 3 Herd of Models | arXiv:2407.21783 | https://arxiv.org/abs/2407.21783"
  - "pdf: Language Models are Few-Shot Learners (GPT-3) | arXiv:2005.14165 | https://arxiv.org/abs/2005.14165"
date: 2026-09
---

# 开源模型架构谱系:GPT→LLaMA→Mistral→DeepSeek/Qwen 与 MoE 路线

## TL;DR

开源 LLM 的架构谱系是一棵**单主干 + 两条分叉**的树:主干是 decoder-only Transformer,配方在 2023 年被 LLaMA 定型(pre-RMSNorm + SwiGLU + RoPE + GPT 式 BPE),此后 Llama 2/3、Mistral、Qwen、DeepSeek(早期)全部沿用,**架构差异收敛到四个变量:GQA 头数、上下文长度、tokenizer、以及 FFN 是否换成 MoE**。分叉一:**MoE(稀疏激活)**——Mixtral 8x7B(8 专家选 2,47B 总/13B 激活)证明「用 1/5 激活参数打平 Llama-2-70B」;DeepSeekMoE(细粒度专家+共享专家)→ DeepSeek-V2(236B/21B,MLA)→ V3(671B/37B,256 路由专家+1 共享,top-8,**aux-loss-free 负载均衡 + 多 token 预测**,2.788M H800 GPU 时,14.8T tokens)把 MoE 推到开源最强。分叉二:**线性注意力/RNN**——RWKV 用「可并行训练的 RNN」换取推理期常数显存,但生态与质量未成主流。**选型准则**:同显存下要最强质量 → 稀疏 MoE(V3/Mixtral,注意激活参数才决定速度、总参数决定显存);要稠密稳态/微调基座 → Llama 3/Qwen2.5;超长上下文 → Llama 3.1/Qwen2.5(原生 128k);极限吞吐小显存 → 小 dense(Mistral 7B/Qwen2.5-7B)。许可要看清:Llama 系是 Community License(非 OSI),Mistral/Qwen/DeepSeek 主力是 Apache 2.0。

> ⚠ **快照框定(时效警示)**:本谱系截止 **2024-12**(最新收录 DeepSeek-V3、Qwen2.5、Llama 3.1)。后续的 Qwen3、DeepSeek-R1/V4 等未纳入;**2025-26 谱系(DeepSeek V3.2(DSA)/V4、Qwen3.x、Llama 4)见续篇 [open-model-landscape-2025-2026.md](open-model-landscape-2025-2026.md)**。选型务必以 HF 当日 config 为准(见 §4「看 config 四件套」),别把本页当作当前最新清单。

---

## 1. 演进主线:一棵树的四次分枝

### 1.1 时间线(架构视角)

```
GPT-2 (2019, open-weight) / GPT-3 (2020, 闭源 API)          ← decoder-only + learned PE + GeLU + pre-LN(相对原 Transformer post-LN 的代际切换)
   └─ LLaMA (Meta, 2023-02, 权重公开)       ← 三件套定型: pre-RMSNorm + SwiGLU + RoPE;公开数据
        ├─ Llama 2 (2023-07)               ← 2T tokens,4k ctx,70B 首用 GQA-8,可商用许可
        │    └─ Llama 3/3.1 (2024)         ← 15T tokens,128K vocab(tiktoken),全系 GQA-8,
        │                                     θ=500k,8k→128K 渐进;405B dense 旗舰
        ├─ Mistral 7B (2023-10, Apache 2.0) ← LLaMA 配方 + GQA-8 + SWA(4096 滑窗),7B 打 Llama2-13B
        │    └─ Mixtral 8x7B (2024-01)      ← 同 attention,FFN → 8 专家选 2 的 MoE;47B/13B 激活,32k
        ├─ Qwen (Alibaba, 2023-09 起)        ← 初代近 LLaMA 配方;Qwen2 起全系 GQA + DCA/YaRN 长上下文,
        │                                     12T(Qwen2)/18T(Qwen2.5) tokens;dense 0.5B~72B + MoE 57B-A14B
        └─ DeepSeek (2023-12 起)             ← 67B dense(MHA)→ V2(236B/21B MoE + MLA)
                                              → V3(671B/37B,开源 MoE 旗舰)
旁支:RWKV (2023)                            ← 线性注意力 RNN 化,常数推理显存,非 Transformer 主干
```

组件级演进(归一化/激活/位置编码三线的完整推导)见 [llm-architecture](llm-architecture.md);注意力头共享(MHA/MQA/GQA/MLA)见 [attention-heads-mha-mqa-gqa](attention-heads-mha-mqa-gqa.md);上下文扩展机制(PI/YaRN/LongRoPE/θ)见 [context-extension](context-extension.md)。本篇聚焦**谱系、MoE 与选型**。

### 1.2 谱系的共识与分化点

| 维度 | 共识(2023 后几乎全员) | 分化点(各家选择) |
|---|---|---|
| 骨架 | decoder-only causal Transformer | — |
| 归一化 | pre-RMSNorm | — |
| 激活 | SwiGLU((2/3)·4d 内维) | MoE 派的专家内维另算 |
| 位置编码 | RoPE | base θ(10k→500k)、扩展方式(见 context-extension) |
| attention 头 | GQA(8 KV 头最常见) | DeepSeek 用 MLA(低秩压缩) |
| FFN | dense | Mixtral/DeepSeek/Qwen-MoE 用稀疏专家 |
| 局部性 | 全局 attention | Mistral SWA(滑窗 4096);Qwen2 长文 DCA 分块 |
| tokenizer | BPE | Llama≤2 用 SentencePiece 32k;Llama3 tiktoken 128K;Qwen 151,646 |
| 许可 | — | Llama: Community License(非 OSI);Mistral/Qwen/DeepSeek 主力: Apache 2.0 |

---

## 2. MoE 专讲:稀疏激活怎么算账、怎么路由

### 2.1 机制:router + top-k 专家

MoE 把每层(或部分层)的 FFN 换成 N 个并行「专家」(每个专家就是一套独立 SwiGLU FFN 权重),外加一个 router(门控网络)对每个 token 选 top-K 个专家:

```
p = softmax(G(x))                    # G: router 线性层,x 为 token 表示
y = Σ_{i ∈ top-K(p)} p_i · E_i(x)    # 只有被选中的 K 个专家参与计算
```

(Mixtral 论文 §2.1 公式;Qwen2 技术报告式 (1)(2) 同构。)

- **总参数 vs 激活参数**:Mixtral 8x7B = 8 专家 × 7B 级 → 47B 总参数,**每 token 只走 2 个专家 + attention ≈ 13B 激活**(论文摘要口径;「8x7B」是营销记号,不是 56B 也不是每 token 用 14B)。显存要装下**全部** 47B(权重常驻),速度/算力只花 13B 的份——**MoE 是「显存换计算」,不是省显存**;
- **为什么有效**:专家可各自专业化(不同语言/代码/领域知识),同等激活算力下总容量更大。Mixtral 论文口径:13B 激活打平或超过 Llama 2 70B(MMLU 70.6% vs 69.9%,GSM8K 74.4% vs 69.6%,HumanEval 40.2% vs 29.3%),5× 更少激活参数;
- **负载均衡问题**:router 天然倾向把 token 都送进少数「热门专家」,导致专家饿死/过载。传统解法是辅助损失(aux loss,鼓励均匀);DeepSeek-V3 改用 **aux-loss-free 策略**:给每个专家一个可学习的 bias 项加进路由打分(bias 只影响选谁,不影响门控值),训练期监控负载在线调 bias——免去 aux loss 与主损失的权重拉扯(§2.1.2,arXiv:2412.19437)。

### 2.2 DeepSeekMoE 的两个改进(细粒度 + 共享专家)

DeepSeekMoE 论文(arXiv:2401.06066)针对 GShard 式「粗专家 top-K」的专家专业化不足:

1. **细粒度分割**:把 N 个专家拆成 mN 个更小的专家、激活 mK 个——组合更灵活,专业化更纯;
2. **共享专家**:隔离 K_s 个专家恒定激活,承接公共知识,减少路由专家里的冗余。

数据(论文摘要):DeepSeekMoE 2B ≈ GShard 2.9B(1.5× 专家参数与计算);16B ≈ LLaMA2-7B 但只用 ~40% 计算;145B ≈ DeepSeek 67B 但只用 28.5%(或更低 18.2%)计算。**V2/V3 的 MoE 层 = 1 共享专家 + 大量细粒度路由专家**(V3:1+256,每 token 激活 8 路由 + 1 共享)。

### 2.3 DeepSeek-V2/V3:MoE + MLA 的组合拳

- **V2**(2024-05):236B 总 / 21B 激活,128K 上下文,8.1T tokens;attention 用 **MLA**(KV 低秩压缩,cache 省 93.3%、吞吐 5.76×,对比同系 dense 67B),训练成本较 67B dense 省 42.5%;
- **V3**(2024-12):**671B 总 / 37B 激活**;61 层、hidden 7168;每 MoE 层 256 路由专家(每专家内维 2048)+ 1 共享,top-8,且**每 token 最多发往 4 个节点**(推理通信约束);14.8T tokens;**全程 2.788M H800 GPU 时**(≈$5.5M 论文口径),无一次不可恢复 loss spike;上下文两阶段扩展 32K→128K;另加**多 token 预测**(MTP,深度 1:每 token 同时预测下下个 token,可作投机解码草稿头);
- 谱系意义:V3 证明「开源 MoE + 工程化训练」能逼近闭源前沿(论文自评:强于其他开源、可比领先闭源),并确立 **MoE(省算力)+ MLA(省 KV cache)** 为 DeepSeek 路线的双支柱。

### 2.4 MoE 与推理引擎的关系

MoE 推理的痛点不是算力而是**权重搬运与专家分布**:47B/671B 权重全要常驻,单卡放不下 → 专家并行(EP)跨节点分片;router 动态性 → batch 内 token 发往不同专家,需要 all-to-all 通信;V3 的「每 token ≤4 节点」约束就是为限制 all-to-all 爆炸。vLLM 对 Mixtral 的支持走 Megablocks CUDA 内核(Mixtral 论文 §部署节自述其向 vLLM 贡献);引擎侧细节见 [`vllm/index.md`](../../vllm/index.md)。

---

## 3. 数据对比:代表模型总表

(数字口径:各模型技术报告/官方 config,2026-09-21 arXiv HTML 与 HF config.json 交叉核对;ctx=最大上下文,激活=每 token 激活参数)

| 模型 | 年份 | 参数(总/激活) | ctx | attention | 归一化 | 激活 | PE | tokenizer | 许可 |
|---|---|---|---|---|---|---|---|---|---|
| GPT-3 | 2020 | 175B dense | 2048 | MHA 96头 | pre-LN | GeLU | learned 绝对 | BPE 50k | 闭源 API |
| LLaMA-65B | 2023-02 | 65B dense | 2048 | MHA | pre-RMSNorm | SwiGLU | RoPE θ=10k | SP-BPE 32k | 研究许可 |
| Llama 2 70B | 2023-07 | 70B dense | 4096 | **GQA-8** | pre-RMSNorm | SwiGLU | RoPE | SP-BPE 32k | Community(可商用) |
| Mistral 7B | 2023-10 | 7B dense | 32768 | **GQA-8 + SWA 4096** | pre-RMSNorm | SwiGLU | RoPE | SP-BPE 32k | Apache 2.0 |
| Mixtral 8x7B | 2024-01 | **47B/13B** | 32768 | GQA-8(32Q/8KV) | pre-RMSNorm | SwiGLU×**8专家选2** | RoPE | SP-BPE 32k | Apache 2.0 |
| Llama 3.1 405B | 2024-07 | 405B dense | **131072** | GQA-8(128Q) | pre-RMSNorm | SwiGLU | RoPE **θ=500k** | tiktoken 128K | Community |
| Qwen2 72B | 2024-06 | 72B dense | 32k(YaRN 可到 128k) | GQA(64Q/8KV) | pre-RMSNorm | SwiGLU | RoPE + DCA/YaRN | BPE 151,646 | Qwen 许可(≤7B 档 Apache 2.0) |
| Qwen2.5 72B | 2024-09 | 72B dense | 128k | GQA | pre-RMSNorm | SwiGLU | RoPE | 151,646 | Qwen 许可(≤7B 档 Apache 2.0) |
| DeepSeek-V2 | 2024-05 | **236B/21B** | 131072 | **MLA** | pre-RMSNorm | SwiGLU MoE(2共享+160路由,激活6路由,组路由8选3组) | RoPE θ=10k(原生4096,YaRN扩128K) | 128k vocab | DeepSeek License(HF 实测) |
| DeepSeek-V3(-0324) | 2024-12 | **671B/37B** | 4k预训→128k | **MLA**(128头,d_c=512) | pre-RMSNorm | SwiGLU MoE(1+256,top-8) | RoPE | 128K | MIT |
| RWKV 14B | 2023 | 14B dense | 流式(理论无限) | **线性注意力(RNN 态)** | LN 变体 | 门控时混(time-mixing) | **时间衰减替代 PE** | 多语 BPE | Apache 2.0 |

要点交叉说明:

- **Mixtral 架构全表**(论文 Table 1 实测核对):dim 4096、32 层、32 Q 头/8 KV 头、head_dim 128、hidden_dim(每专家)14336、ctx 32768、vocab 32000、8 专家 top-2——**attention 部分与 Mistral 7B 完全一致,只换 FFN**;
- **Qwen2 配置表**(技术报告 Table 1):0.5B/1.5B/7B/72B 的 KV 头 2/2/4/8,72B 是 64Q/8KV;MoE 57B-A14B:64 路由专家+8 共享,激活 8 路由;
- **DeepSeek-V3 超参**(技术报告 §模型超参核对):61 层、hidden 7168、n_h=128、d_h=128、KV 压缩维 d_c=512、query 压缩维 1536、解耦 RoPE 维 64、256 路由专家(内维 2048)+1 共享、top-8、MTP 深度 1;
- **RWKV**(arXiv:2305.13048 摘要):线性注意力,训练时可写成 Transformer 形态(并行),推理时是 RNN(常数显存/计算);14B 规模验证。是「不要 softmax attention」路线里生态最大的一支,但下游质量与生态仍显著小于 Transformer 主干;
- LLaMA/Llama/Mistral 行与 [llm-architecture](llm-architecture.md) §5 的表一致(该表更细,含训练 tokens 与训练哲学);**训练 tokens 对比**:LLaMA 1.4T → Llama2 2T → Llama3 15T;Qwen2 12T → Qwen2.5 18T;DeepSeek-V2 8.1T → V3 14.8T。数据规模军备竞赛是 2023-2024 的主线,架构反而稳定。

---

## 4. 实践要点(选型准则)

> ⚠ **快照框定**:以下选型准则基于本谱系收录到 2024-12 的模型;Qwen3、DeepSeek-R1/V4 等未纳入,**2025-26 谱系见续篇 [open-model-landscape-2025-2026.md](open-model-landscape-2025-2026.md)**,选型以 HF 当日 config 为准(见下方「看 config 四件套」)。

- **先定约束再选模型**:
  - 显存紧、要快 → 小 dense(Mistral 7B、Qwen2.5 7B)或量化版(见 [`quantization/`](../../quantization/index.md));
  - 单卡/双卡放得下大权重、要高智能 → 稀疏 MoE:Mixtral 47B(2×24G 可跑)或 Qwen-MoE;**记住 MoE 显存按总参数、速度按激活参数**;
  - 多卡集群、追开源最强 → DeepSeek-V3 671B(需 EP 分布式,vLLM/SGLang 支持);
  - 微调基座 → 稠密 Llama 3/Qwen2.5(生态最全,LoRA/QLoRA 成熟);
  - 长文档 → 原生 128k(Llama 3.1、Qwen2.5、DeepSeek-V2/V3),别拿 4k 模型自己 YaRN 硬扩到生产(社区扩展版先过下游回归);
- **许可逐条看**:Llama Community License 有月活阈值与命名要求,非 OSI;Apache 2.0(Mistral/Qwen 多数档/DeepSeek/RWKV)可自由商用;Qwen 部分尺寸档用 Qwen 许可(注册制)。「开源」在 LLM 语境默认指 open-weight,权重开放 ≠ OSI 开源;
- **看 config 四件套**:任何新模型先读 HF config.json 的 `num_key_value_heads`(GQA?)、`num_local_experts`/`num_experts_per_tok`(MoE?)、`max_position_embeddings` + `rope_scaling`(上下文怎么来的)、`sliding_window`(SWA?)——30 秒定位它在谱系树上的位置;
- **MoE 部署三坑**:① 权重常驻显存(47B bf16 ≈ 94GB,别按 13B 估);② 专家并行需要高速互联(all-to-all),NVLink/IB 之外性能崩;③ router 负载不均时长尾专家拖慢整个 batch——V3 的 aux-loss-free bias 与 ≤4 节点约束是工程答案,自训 MoE 别忽略;
- **「x7B」「8x7B」等记号别当真**:Mixtral 8x7B = 47B 总/13B 激活;Qwen 57B-A14B = 57B 总/14B 激活。A{N}B 记号里 N 才是激活参数;
- **RWKV/线性注意力何时考虑**:超长流式、端侧常数显存、极端成本敏感——但质量/生态折价明显,通用场景仍选 Transformer 主干;
- **谱系思维**:读新模型公告先归类「LLaMA 配方 + 哪些变量改了」——2023 年后几乎所有开源模型都是这个模式的变体,真正的新结构(MLA、MTP、线性注意力)才值得细读论文。

## 溯源

- **Mixtral**(arXiv:2401.04088 HTML v1,2026-09-21 核对):Table 1 架构全参数(dim 4096/32层/32Q/8KV/head_dim 128/hidden 14336/ctx 32768/vocab 32000/8专家top-2)、47B总/13B激活、32k 训练、Table 2 对 Llama 2 70B 基准(MMLU 70.6 vs 69.9、GSM8K 74.4 vs 69.6、HumanEval 40.2 vs 29.3)、Apache 2.0、向 vLLM 贡献 Megablocks 内核部署;
- **DeepSeek-V3**(arXiv:2412.19437 HTML v1 核对):671B/37B、61层/7168 hidden、MLA(d_c=512/1536、解耦维64、n_h=128)、1+256 专家 top-8 ≤4节点、aux-loss-free bias 路由(式14-16)、MTP 深度1、14.8T tokens、2.788M H800 GPU时、无 loss spike、两阶段 32K→128K;
- **DeepSeek-V2**(arXiv:2405.04434 摘要核对):236B/21B、128K、8.1T tokens、MLA+DeepSeekMoE、KV cache −93.3%、吞吐 5.76×、训练成本 −42.5%;专家配置以 HF config.json 实测为准(2026-09-21 抓取:n_shared_experts=2、n_routed_experts=160、num_experts_per_tok=6、n_group=8/topk_group=3、q_lora_rank=1536、kv_lora_rank=512、rope_theta=10000、original_max_position_embeddings=4096→YaRN 扩 128K、first_k_dense_replace=1);
- **DeepSeekMoE**(arXiv:2401.06066 摘要核对):细粒度 mN/mK + 共享专家 K_s 两策略、2B≈GShard2.9B、16B≈LLaMA2-7B@40%计算、145B≈67B@28.5%;
- **Mistral 7B**(arXiv:2310.06825 HTML v1 + HF config 核对):GQA+SWA、W=4096、32k、Apache 2.0;
- **Qwen**(arXiv:2309.16609 / 2407.10671 HTML v1 / 2412.15115 摘要核对):Qwen2 Table 1 配置与 KV 头数、GQA 动机原话、DCA+YaRN、MoE 57B-A14B(64+8 共享选 8)、Qwen2 12T / Qwen2.5 18T tokens、vocab 151,646;
- **RWKV**(arXiv:2305.13048 摘要核对):线性注意力、Transformer/RNN 双形态、常数推理复杂度、14B;
- **Llama 3**(arXiv:2407.21783)与 GPT-3(arXiv:2005.14165):与 [llm-architecture](llm-architecture.md) 共享来源,该篇已做全文核对;
- GPT-2 未单列 source(架构细节以 GPT-3 论文与 llm-architecture 篇为准);许可信息 2026-09-21 实测核对:DeepSeek-V3 = MIT(GitHub API license 端点)、DeepSeek-V2 = DeepSeek License Agreement(HF LICENSE 文件首行)、Qwen2/2.5 = Qwen/Tongyi Qianwen License(72B 档;≤7B 档 HF tag 为 Apache 2.0)、Mixtral = Apache 2.0(HF tag);
- mistral.ai 官方博客在本网络环境不可达(连接超时,非 404),Mistral 系全部以 arXiv 论文+HF config 为准;
- 全部 arXiv URL curl 200 实测(2026-09-21)。
- 关联条目:[llm-architecture](llm-architecture.md)(组件演进线全文)、[attention-heads-mha-mqa-gqa](attention-heads-mha-mqa-gqa.md)(GQA/MLA 机制)、[context-extension](context-extension.md)(长上下文机制)、[attention-is-all-you-need](../../attention/algorithm/attention-is-all-you-need.md)(原始 Transformer)、[`vllm/`](../../vllm/index.md)(MoE/GQA 模型的推理引擎)。
