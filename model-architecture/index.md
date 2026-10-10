---
title: Model Architecture(通用 LLM 架构)
type: domain
tags: [llm-architecture, llama, rmsnorm, swiglu, rope, alibi, gqa, mqa, mla, context-extension, yarn, moe, mixtral, deepseek, qwen, rwkv, chinchilla, scaling-laws, gpt]
source:
  - "pdf: LLaMA | arXiv:2302.13971"
  - "pdf: Llama 2 | arXiv:2307.09288"
  - "pdf: The Llama 3 Herd of Models | arXiv:2407.21783"
  - "pdf: Chinchilla (Training Compute-Optimal LLMs) | arXiv:2203.15556"
  - "pdf: GQA (Ainslie et al.) | arXiv:2305.13245"
  - "pdf: MQA (Shazeer) | arXiv:1911.02150"
  - "pdf: Position Interpolation (Chen et al.) | arXiv:2306.15595"
  - "pdf: YaRN (Peng et al.) | arXiv:2309.00071"
  - "pdf: LongRoPE (Ding et al.) | arXiv:2402.13753"
  - "pdf: Mixtral of Experts | arXiv:2401.04088"
  - "pdf: DeepSeek-V3 Technical Report | arXiv:2412.19437"
date: 2026-10
---

# Model Architecture(通用 LLM 架构)

> 领域索引(导航起始页)。领域知识本体在 `concepts/` 子目录,每篇一个专题、可独立打开读懂;本页放定位与导航。外部链接索引见 [references.md](references.md)。
> 本域 2026-09 新建:原 attention 域的 `llama-architecture.md` 经 git mv 移入并扩展为**通用 LLM 架构篇**;同月补充三篇专题(注意力头结构 GQA、上下文扩展、开源模型谱系),2026-10 补开源谱系 2025-26 续篇;同月落 DeepSeek 网络结构演进四篇(3 子篇 + 1 主篇),现共 9 篇 concept。

## 一句话定位

回答"一个现代 decoder-only LLM 长什么样、为什么长这样、有哪些变体可选":以 LLaMA 三件套(pre-RMSNorm + SwiGLU + RoPE)为定型锚点,覆盖组件演进线(归一化/激活/位置编码)、训练哲学(Kaplan→Chinchilla→LLaMA)、注意力头共享谱系(MHA→MQA→GQA→MLA)、长上下文扩展机制(PI/NTK/YaRN/LongRoPE),以及开源模型谱系与 MoE 路线(Mistral/DeepSeek/Qwen/RWKV)。

## 条目索引

| 条目 | 类型 | 一句话 | 状态 |
|---|---|---|---|
| [llm-architecture](concepts/llm-architecture.md) | concept | 通用 LLM 架构:post/pre-norm→RMSNorm、ReLU→GeLU→SwiGLU((2/3)4d)、正弦→learned→RoPE→ALiBi 三条演进线 + Chinchilla/推理预算训练哲学 + GPT→LLaMA→Llama 2/3 演进表(GQA/θ=500k/128K vocab/15T tokens) | ✅ 已入库 |
| [attention-heads-mha-mqa-gqa](concepts/attention-heads-mha-mqa-gqa.md) | concept | 注意力头结构谱系:KV cache 带宽账单为何驱动 K/V 头共享(MHA→MQA→GQA-g 插值,mean-pool uptrain 5% 算力),T5-XXL 质量/速度数据,GQA-8 成开源标配,延伸到 DeepSeek MLA 低秩压缩 | ✅ 已入库(2026-09) |
| [context-extension](concepts/context-extension.md) | concept | RoPE 难外推的频率机制(陌生相位→灾难 logit,理论上界差 ~600×)+ 四代扩展法:PI 线性插值→NTK-aware 改 base→YaRN 分段插值+注意力温度→LongRoPE 逐维搜索到 2M;工业路线 θ=500k 渐进预训练;选型决策树 | ✅ 已入库(2026-09) |
| [open-model-landscape](concepts/open-model-landscape.md) | concept | 开源谱系树:GPT→LLaMA 定型→Llama2/3、Mistral(GQA+SWA)、Qwen、DeepSeek;MoE 专讲(router/top-k、总参数 vs 激活参数、DeepSeekMoE 细粒度+共享专家、V3 671B/37B aux-loss-free+MTP);11 模型对比表 + 许可实测 + 选型准则 + RWKV 旁支(2024-12 快照) | ✅ 已入库(2026-09) |
| [open-model-landscape-2025-2026](concepts/open-model-landscape-2025-2026.md) | concept | 2025-26 谱系续篇:DeepSeek R1→V3.1→V3.2(DSA 稀疏注意力)→V4/V4.1;Qwen3(Hybrid 思考)/3.5/3.6/3.8(2.4T-A95B 大 MoE);Llama 4 Scout/Maverick 原生 MoE;横向对比表 | ✅ 已入库(2026-10) |
| [deepseek-model-evolution](concepts/deepseek-model-evolution.md) | concept(主篇) | DeepSeek 网络结构演进总览:TL;DR + 逐代概览(V1→DeepSeekMoE→V2→V3→V3.1→V3.2)+ 代际 diff 总表(注意力 MHA→GQA→MLA→MLA+DSA;FFN dense→细粒度→256 专家;位置 RoPE→解耦→YaRN)+ 双约束演进方向分析(稀疏省算力/低秩+稀疏省显存/训练机制保质量);3 子篇导航 | ✅ 已入库(2026-10) |
| [deepseek-arch-v1-moe-v2](concepts/deepseek-arch-v1-moe-v2.md) | concept(子篇) | 奠基段:V1(7B MHA/67B GQA)+ DeepSeekMoE(细粒度+共享专家)+ V2(MLA 低秩 KV + 解耦 RoPE + 160 专家 MoE,双主线首次合流);MHA/GQA/MLA KV cache 量化 | ✅ 已入库(2026-10) |
| [deepseek-arch-v3](concepts/deepseek-arch-v3.md) | concept(子篇) | V3:MLA 规模化 + 256 专家 MoE(top-8,sigmoid 路由)+ MTP 多 token 预测 + aux-loss-free 负载均衡 + FP8;671B/37B、2.788M H800 小时训练 | ✅ 已入库(2026-10) |
| [deepseek-arch-v32-r1](concepts/deepseek-arch-v32-r1.md) | concept(子篇) | V3.2(DSA 闪电索引器 + top-2048 稀疏选择,破 O(L²)→O(L·k))+ R1(无新结构,GRPO RL 推理时扩展);V3→V3.2 代码级 diff | ✅ 已入库(2026-10) |
| [hybrid-linear-attention](concepts/hybrid-linear-attention.md) | concept | 混合线性注意力/SSM 第三主轴:核化线性注意力、RWKV 时间混合、Mamba/S6 选择性与 Mamba-2/SSD 对偶、Griffin/Jamba 混合层比、Gated DeltaNet→Qwen3-Next;训练并行 vs O(1) 状态两难与 hybrid 甜点 | ✅ 已入库(2026-10) |

## 关键事实速记

- **定型配方**:pre-RMSNorm + SwiGLU + RoPE 是 2023 年后开源 LLM 事实标准(LLaMA/Qwen/Mistral 全沿用);
- **RMSNorm** = LayerNorm 去掉减均值,只按均方根缩放——少一次统计量、更快,质量无损;pre-norm 让残差主干恒等直通、深层稳训(可不用 warmup);
- **SwiGLU** 三个权重矩阵 → 内维取 (2/3)·4d 对齐参数量;更优是经验性的(T5 规模系统评测),无干净理论;
- **RoPE**:旋转矩阵把绝对位置编进 Q/K,内积只依赖相对距离;改 base θ 即可扩上下文(Llama 3: 10k→500k);ALiBi 是另一条零参数外推路线(BLOOM/MPT);
- **训练哲学**:Chinchilla 算力最优 ~20 tokens/参数;LLaMA 把推理预算写进目标函数 → 7B×1T(143 tokens/参数);13B 多数基准超 GPT-3(175B);
- **Llama 2→3**:2T→15T tokens、4k→128K context、70B GQA→全系 8 KV 头、32k SentencePiece→128K tiktoken vocab(压缩率 3.17→3.94)——架构不动,收益来自数据;
- **GQA**:K/V 头分组共享(KV cache 缩 H/G 倍),T5-XXL 上 5% uptrain 质量 47.1 vs MHA 47.2、速度接近 MQA——2023 后开源标配;DeepSeek MLA 用低秩压缩再省 ~93% KV cache;
- **上下文扩展**:直接外推崩坏(分布外相位);PI(1000 步微调→32k)→NTK-aware(免微调改 base)→YaRN(分段插值+温度,128k,比 PI 省 10× tokens)→LongRoPE(逐维搜索,2M);
- **MoE 账**:显存按总参数、速度按激活参数(Mixtral 47B/13B 打平 Llama2-70B;DeepSeek-V3 671B/37B);
- **许可**:Llama=Community License(非 OSI);Mistral/Mixtral=Apache 2.0;DeepSeek-V3=MIT、V2=DeepSeek License;Qwen 大尺寸档=Qwen License(≤7B 档 Apache 2.0)。

## 相关条目(跨域互链)

- **[`attention/`](../attention/index.md)**:本域架构篇的 attention 机制基础(softmax 数学/O(n²) 账单/多头等)在 attention 域 algorithm 层;高效内核(FA 四代)在 [attention/implementation/flash-attention-implementation.md](../attention/implementation/flash-attention-implementation.md)——LLaMA 训练用的 xformers attention = Rabe & Staats 前向 + FA1 反向;GQA 缩小的 KV cache 由 [`vllm/`](../vllm/index.md) PagedAttention 管理([paged-attention](../vllm/concepts/paged-attention.md))。
- **[`vllm/`](../vllm/index.md)**:GQA/MoE/长上下文的收益都在推理引擎兑现——KV cache 分页([paged-attention](../vllm/concepts/paged-attention.md))、大 batch 吞吐([continuous-batching](../vllm/concepts/continuous-batching.md))、KV/权重量化([quantization-gptq-awq-fp8](../vllm/concepts/quantization-gptq-awq-fp8.md))。
- **[`quantization/`](../quantization/index.md)**:LLaMA 权重是 GPTQ/AWQ/SpQR 等量化的最常用测试床。
- **[`deepseek-models/`](../deepseek-models/index.md)**:本域 DeepSeek 网络结构四篇的**领域侧互补**——结构篇讲 MHA→MLA、dense→MoE「怎么变」,领域篇讲数学(GRPO/自验证)、定理证明(Lean 4)、代码(FIM/MoE)、MoE 数据评测「怎么做出来」。DeepSeekMoE 结构在本域 [deepseek-arch-v1-moe-v2](concepts/deepseek-arch-v1-moe-v2.md),数据/评测侧在 [deepseek-models/concepts/deepseek-moe.md](../deepseek-models/concepts/deepseek-moe.md)。

## 扩展位(2026-10 全库审查待补)

- ~~hybrid-linear-attention~~(已入库)
- MoE 路由对照(expert-choice/capacity/token dropping vs aux-loss-free)
- 训练配方 2025-26(inference-aware scaling ~40+ tokens/param、WSD/muon、FP8 recipe)
- context-extension 增量(ABF/相位重置、Llama 3.1/4 原生长上下文、10M 窗口)
- 2025-26 谱系补漏(Kimi-K2/GLM-4.5/MiniMax-M2/gpt-oss/Gemma3/Qwen3-Next)
- config 四件套速查卡(kv_lora_rank/index_*/noaux_tc 等新字段)

## 溯源

- 外部链接索引见 [references.md](references.md)。
