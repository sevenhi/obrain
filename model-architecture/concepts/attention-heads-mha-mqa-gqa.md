---
title: 注意力头结构 MHA/MQA/GQA:KV cache 驱动的共享谱系
type: concept
tags: [mha, mqa, gqa, kv-cache, attention-heads, multi-head-attention, multi-query-attention, grouped-query-attention, memory-bandwidth, t5, llama, mistral, palm, uptraining, mla]
source:
  - "pdf: GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (primary) | arXiv:2305.13245 (Ainslie et al., Google Research, 2023-05) | https://arxiv.org/abs/2305.13245"
  - "pdf: Fast Transformer Decoding: One Write-Head is All You Need (MQA, Shazeer 2019) | arXiv:1911.02150 | https://arxiv.org/abs/1911.02150"
  - "pdf: Mistral 7B (GQA+SWA 工程采用) | arXiv:2310.06825 (2023-10) | https://arxiv.org/abs/2310.06825"
  - "pdf: The Llama 3 Herd of Models (全系 GQA 8 KV 头) | arXiv:2407.21783 | https://arxiv.org/abs/2407.21783"
  - "pdf: DeepSeek-V3 Technical Report (MLA:KV 压缩的下一步) | arXiv:2412.19437 | https://arxiv.org/abs/2412.19437"
  - "web: HF Mistral-7B-v0.1 config.json (32 Q 头/8 KV 头实测) | https://huggingface.co/mistralai/Mistral-7B-v0.1 (2026-09-21 抓取核对)"
date: 2026-09
---

# 注意力头结构 MHA/MQA/GQA:KV cache 驱动的共享谱系

## TL;DR

三种注意力头结构的差别只在**「K/V 投影头的数量」**:MHA(原始 Transformer)每层 H 个 Q 头配 H 个 K/V 头;MQA(Shazeer 2019)所有 Q 头共享**1 组** K/V;GQA(Ainslie 2023)把 Q 头分成 G 组、每组共享 1 组 K/V——G 在 1 和 H 之间插值,**GQA-1=MQA,GQA-H=MHA**。动机全部来自推理侧:自回归解码每步要把**全部历史 token 的 K/V(KV cache)从显存重新加载一遍**,解码是内存带宽瓶颈而非算力瓶颈;KV cache 大小正比于 K/V 头数,砍头=直接砍带宽与显存。**代价是模型容量**:MQA 质量掉得明显且训练不稳,GQA 用「组」保住大部分容量——T5-XXL 上 GQA-8 以 5% 原预训练算力从 MHA checkpoint uptrain 而来,平均质量 47.1 vs MHA 47.2(vs MQA 46.5),推理时间 0.28s vs 1.51s(接近 MQA 的 0.24s)。2023 年后 GQA 成为开源 LLM 事实标准:Llama 2 70B、Llama 3 全系(8 KV 头)、Mistral 7B(32 Q/8 KV)、Mixtral、Qwen 全系。DeepSeek 走得更远:MLA 把 KV 低秩压缩成单个 latent 向量(KV cache 再省 ~93%)。**读模型 config 时 `num_key_value_heads` < `num_attention_heads` 即是 GQA;比值 H/G 就是 KV cache 缩减倍数。**

---

## 1. 机制:为什么解码瓶颈是「加载 K/V」

### 1.1 KV cache 与自回归解码的带宽账单

自回归解码每步只算 1 个新 token 的 Q,但 attention 要对**全部历史位置**的 K/V 做内积与加权求和。历史 K/V 已算过,存下来复用即 **KV cache**;每步解码都要把整份 cache 从 HBM 读进 SM 一遍。于是:

```
每步解码内存流量 ≈ 模型权重 + KV cache
KV cache 大小 = 2(K,V) × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes/元素
```

- 权重流量固定,**KV cache 流量随 seq_len×batch 线性涨**——长上下文/大 batch 服务时 KV 读取主导延迟(Shazeer 2019 §1:「增量推理速度被重新加载 keys/values 大张量的内存带宽限制」);
- 解码每步 FLOPs 极小(1 token),GPU 算力大量闲置——这是典型的**带宽受限**场景,砍 KV 头数=线性砍这块流量,几乎不损算力吞吐。

### 1.2 三种结构(数学定义)

设每层 Q 头数 H、K/V 头数 H_kv、head_dim = d/H。三种结构只是 H_kv 的取值不同:

```
MHA:  H_kv = H     每个 Q 头 h 配专属 K_h, V_h:
      o_h = softmax(q_h K_hᵀ / √d) V_h ,  H 个头拼接后过 W_O
MQA:  H_kv = 1     所有 Q 头共享同一份 K, V:
      o_h = softmax(q_h Kᵀ / √d) V ,  ∀h ∈ [1,H]
GQA-g: H_kv = g    Q 头均分 g 组,组内共享 K^(i), V^(i):
      o_h = softmax(q_h K^(g(h))ᵀ / √d) V^(g(h)) ,  g(h) = ⌈h·g/H⌉
```

- **GQA-1 ≡ MQA,GQA-H ≡ MHA**(GQA 论文 §2.2 原文:「GQA-1, with a single group … is equivalent to MQA, while GQA-h … is equivalent to MHA」)——这是一条**连续插值谱系**,不是三种孤立设计;
- KV cache 缩减倍数 = H/H_kv。MHA→MQA 缩 H 倍(H=64 时缩 64 倍);MHA→GQA-8 缩 8 倍;
- **为什么 MQA 掉质量**:K/V 是「被检索的内容」,1 组 K/V 要服务 H 个不同视角的 Q,表达容量骤降;GQA 论文还指出 MQA 有**训练不稳定**问题(§1:quality degradation and training instability)。GQA 保留 g 组,给「不同 Q 看不同 K/V 视角」留了余量;
- **为什么大模型更偏爱 GQA 而非 MQA**(GQA 论文 §2.2):① 模型越大 H 越多,MQA 的砍头比例越激进(容量损失随规模放大),GQA 可按比例温和地砍;② KV cache 随模型维度线性涨、FLOPs 随维度平方涨,大模型里 attention 带宽占比相对小,GQA 够用;③ **张量并行浪费**:标准 sharding 下单 KV 头要被复制到每个分片(Pope et al. 2022),GQA 的 g≥并行度时每片分到自己的 KV 头,消除复制浪费。

### 1.3 Uptraining:5% 算力把 MHA 改成 GQA

GQA 论文的第二个贡献是**改造存量 checkpoint 的配方**(不用从头训):

1. **checkpoint 转换**:每组新 K/V 投影矩阵 = 组内原 MHA 各头 K/V 投影矩阵的**均值池化**(mean pooling;实测优于「只留一头」或「随机重新初始化」);
2. **少量续训(uptrain)**:转换后按原预训练配方继续训 α=5% 的原步数,让模型适应新结构。

T5-XXL 上 5% 算力即恢复接近 MHA 的质量(见 §3 表)。这条路线后来被工业界反复使用(如 Llama 3 长上下文继续预训练同属「checkpoint 改造+续训」范式)。

---

## 2. 可复现细节

### 2.1 config 层面的识别(HF Transformers 惯例)

```jsonc
// Mistral-7B-v0.1 config.json (2026-09-21 HF raw 实测)
{
  "hidden_size": 4096,
  "num_attention_heads": 32,      // H = 32 个 Q 头
  "num_key_value_heads": 8,       // H_kv = 8 → GQA-8,KV cache 缩 4 倍
  "num_hidden_layers": 32,
  "max_position_embeddings": 32768,
  "sliding_window": 4096
}
```

判定规则:`num_key_value_heads == num_attention_heads` → MHA;`== 1` → MQA;介于两者 → GQA-g(g = num_key_value_heads)。head_dim = hidden_size / num_attention_heads(注意是除以 **Q 头数**;Mistral 7B: 4096/32 = 128,K/V 每头也是 128 维)。

### 2.2 推理侧实现:K/V 广播扩展

GQA 的 attention 内核里 K/V 头需要「repeat」到 Q 头数再算,常见两种等价做法:

```python
# 做法一:显式 repeat_interleave(教学版)
K = K.repeat_interleave(H // H_kv, dim=1)   # [B, H_kv, S, d] -> [B, H, S, d]
# 做法二:FlashAttention/FA2 内核原生支持 GQA,
# 按 H//H_kv 的组直接寻址共享的 K/V,不做物化拷贝(省显存与带宽)
```

FlashAttention 2/3、xFormers、vLLM 各后端均原生支持 GQA(把 MQA 当 H_kv=1 特例)——推理引擎不需要为 GQA 写新调度,只是 KV cache block 变小,详见 [`vllm/concepts/paged-attention.md`](../../vllm/concepts/paged-attention.md)。

### 2.3 KV cache 大小手算(核对用)

以 bf16(2 bytes/元素)、seq_len = S、batch = B:

| 模型 | 层数 | H_kv | head_dim | 每 token KV | 每序列 4K token | 每序列 128K token |
|---|---|---|---|---|---|---|
| Llama-2-7B(MHA) | 32 | 32 | 128 | 2×32×32×128×2B = 512 KB | 2 GB | — |
| Llama-3.1-8B(GQA-8) | 32 | 8 | 128 | 128 KB | 512 MB | 16 GB |
| Llama-2-70B(GQA-8) | 80 | 8 | 128 | 320 KB | 1.25 GB | — |

公式 `2 × L × H_kv × d_h × bytes`;Llama-3.1-8B 在 128K 满上下文时单序列 KV cache 就要 ~16 GB(bf16)——这就是为什么 128K 长上下文模型必须 GQA 起步、并进一步配合 KV 量化(FP8/INT4,见 [`vllm/concepts/quantization-gptq-awq-fp8.md`](../../vllm/concepts/quantization-gptq-awq-fp8.md))或 MLA。

---

## 3. 数据对比

### 3.1 GQA 论文主表(T5.1.1,摘要+翻译+WMT+TriviaQA 平均质量)

来源:arXiv:2305.13245 Table 1(2026-09-21 arXiv HTML 原文核对;Average 列为 8 个 dev 集平均,时间列为 TPUv4 每芯片每样本推理时间):

| 模型 | 推理时间(s/样本) | Average | 备注 |
|---|---|---|---|
| MHA-Large | 0.37 | 46.0 | 基线小模型 |
| MHA-XXL | 1.51 | **47.2** | 质量上限 |
| MQA-XXL | **0.24** | 46.5 | 最快,但质量掉 0.7 |
| GQA-8-XXL | 0.28 | **47.1** | 5% uptrain;质量差 0.1,速度快 5.4× |

结论:**GQA-8 拿到 MQA 几乎全部的速度收益(0.28 vs 0.24),质量只差 MHA 0.1**(MQA 差 0.7)。消融(§3.3)还显示:组数 g 从 1→8 质量快速回升,8→16 之后收益趋平;uptrain 步数 5% 是质量/成本的甜点。

### 3.2 采用情况(架构 config 口径)

| 模型 | 年份 | Q 头 / KV 头 | 结构 | 备注 |
|---|---|---|---|---|
| GPT-3 175B | 2020 | 96/96 | MHA | 前 GQA 时代 |
| PaLM 540B | 2022 | 48/1(等效) | MQA | MQA 的大规模首次验证 |
| LLaMA 7/13/33B | 2023-02 | H/H | MHA | 65B 亦 MHA |
| Llama 2 70B | 2023-07 | 64/8 | GQA-8 | 7/13B 仍 MHA |
| Mistral 7B | 2023-10 | 32/8 | GQA-8 | +SWA(sliding_window=4096) |
| Mixtral 8x7B | 2024-01 | 32/8 | GQA-8 | 同 Mistral 7B attention,FFN 换 MoE |
| Llama 3 全系(8/70/405B) | 2024 | H/8 | GQA-8 | 论文自述:「improve inference speed and reduce the size of key-value caches during decoding」 |
| Qwen2/2.5 全系 | 2024 | 如 72B: 64/8 | GQA | 技术报告 §2.2.1:「GQA optimizes KV cache usage during inference, significantly enhancing throughput」 |
| DeepSeek-V2/V3 | 2024 | 128/(latent) | MLA | KV 低秩压缩,较 DeepSeek 67B(MHA)KV cache 省 93.3%(V2 论文口径) |

演进逻辑一句话:**MHA(容量)→ MQA(极致省带宽,质量受损)→ GQA(工程折中,2023 后主流)→ MLA(低秩压缩,把「共享」推到「压缩」)**。

### 3.3 MLA:GQA 思路的延伸(DeepSeek-V2/V3)

GQA 是「头级共享」,MLA(Multi-head Latent Attention)是「低秩压缩」:把每 token 的 K/V 联合压缩成一个 latent 向量 `c^KV = W^{DKV} h`(V3:d_c=512 维),解码时 cache 只存 latent,用时再上投影还原各头 K/V;Q 也做类似压缩。V3 配置:61 层、128 头、head_dim 128、KV 压缩维 512——**每 token cache ≈ 512+64(RoPE 部分)维,而非 128×128×2**;DeepSeek-V2 论文口径:相比同系 DeepSeek 67B(dense/MHA)KV cache 省 93.3%、最大生成吞吐 5.76×。MLA 与 GQA 不在一条互斥路线上,但目标同源:**解码期 KV 带宽/显存最小化**。

---

## 4. 实践要点

- **选型先看 KV 头数**:HF config `num_key_value_heads`;长上下文/大 batch 服务选 GQA/MLA 模型,H/H_kv 比值直接决定同显存下能塞的并发数(vLLM 的 KV block 数、`gpu_memory_utilization` 预算都受它影响,见 [`vllm/concepts/paged-attention.md`](../../vllm/concepts/paged-attention.md));
- **GQA 对推理引擎透明**:内核只是按组寻址 K/V,调度/分页逻辑不变——GQA 的收益体现在 **KV cache 变小 → 同样显存更多 block → 更大 batch/更长序列**,与连续批处理([`vllm/concepts/continuous-batching.md`](../../vllm/concepts/continuous-batching.md))相乘;
- **张量并行度 ≤ H_kv**:TP 切分按 KV 头分片,H_kv=8 的模型 TP>8 时 KV 头必须复制,浪费显存与带宽(GQA 论文 §2.2 指出的 MQA 复制问题的残余形态);部署 Llama-3-8B(KV 头 8)时 TP=8 是上限的常见原因之一;
- **uptrain 配方可迁移**:想把存量 MHA checkpoint 改 GQA:mean-pool 组内 K/V 投影 → 原配方续训 ~5% 步数(GQA 论文 §2.1);直接硬转不续训质量损失大;
- **质量-速度不是单调折中**:GQA-8 在 T5-XXL 上质量 47.1 vs MHA 47.2——差距在测量噪声级;不要默认「GQA 一定明显掉点」,那是 MQA 的问题;
- **MQA 今天基本只用于存量**(PaLM/Falcon 等),新模型没有理由选 MQA 而不是 GQA-8;
- **与 attention 内核的关系**:FA2/FA3 对 GQA 有专门优化(K/V 复用减少非 matmul FLOPs 与共享内存读取),MHA 内核路径与 GQA 不同;内核细节见 [flash-attention-implementation](../../attention/implementation/flash-attention-implementation.md);
- **多头机制本身的数学**(为何多头、softmax 性质)见 [attention-is-all-you-need](../../attention/algorithm/attention-is-all-you-need.md);本篇只讲「K/V 头怎么共享」。

## 溯源

- **GQA 主源**(arXiv:2305.13245,2026-09-21 arXiv HTML v1 全文核对):Table 1 全部数字(MHA-XXL 1.51s/47.2、MQA-XXL 0.24s/46.5、GQA-8-XXL 0.28s/47.1、MHA-Large 0.37s/46.0)、mean-pooling 转换、α=5% uptrain、GQA-1≡MQA/GQA-h≡MHA 等价性、§2.2 大模型偏爱 GQA 的三条理由(比例砍头/带宽占比/TP 复制)、T5.1.1+JAX/Flaxformer 实验设置;
- **MQA 主源**(arXiv:1911.02150,ar5iv 全文核对):带宽受限解码的动机表述、「minor quality degradation」结论、PaLM 采用(GQA 论文 §1 交叉引用);
- **Mistral 7B**(arXiv:2310.06825 HTML v1 + HF config.json 实测):GQA+SWA 组合、sliding_window=4096、32 Q/8 KV 头、32k 上下文;
- **Llama 3**(arXiv:2407.21783):全系 GQA 8 KV 头、动机原话(与 [llm-architecture.md](llm-architecture.md) §5 共享该源);
- **DeepSeek-V3**(arXiv:2412.19437 HTML v1 核对):MLA 公式与 d_c=512/128 头配置;DeepSeek-V2(arXiv:2405.04434):93.3% KV 省、5.76× 吞吐、236B/21B;
- **Qwen2 技术报告**(arXiv:2407.10671 HTML v1 核对):Table 1 KV 头配置(72B: 64 Q/8 KV)、GQA 动机原话;
- URL 全部 curl 200 实测(2026-09-21);mistral.ai 官方博客在本网络环境连接超时(非 404),未列入来源,以 arXiv 原文替代。
- 关联条目:[llm-architecture](llm-architecture.md)(GQA 在 Llama 2/3 演进线中的位置)、[open-model-landscape](open-model-landscape.md)(各模型采用谱系全景)、[context-extension](context-extension.md)(长上下文让 KV cache 问题放大 GQA 的价值)。
