---
title: DeepSeek 网络结构演进总览(V1→V2→V3→V3.2,MLA/MoE/MTP/DSA)
type: concept
tags: [deepseek, evolution, mla, moe, mtp, dsa, gqa, yarn, sparse-attention, architecture]
series: deepseek-model-evolution
source:
  - "paper: DeepSeek LLM / V1(arXiv:2401.02954)| https://arxiv.org/abs/2401.02954"
  - "paper: DeepSeekMoE(arXiv:2401.06066)| https://arxiv.org/abs/2401.06066"
  - "paper: DeepSeek-V2(arXiv:2405.04434)| https://arxiv.org/abs/2405.04434"
  - "paper: DeepSeek-V3(arXiv:2412.19437)| https://arxiv.org/abs/2412.19437"
  - "paper: DeepSeek-V3.2(arXiv:2512.02556)| https://arxiv.org/abs/2512.02556"
  - "paper: DeepSeek-R1(arXiv:2501.12948)| https://arxiv.org/abs/2501.12948"
  - "pdf: DeepSeek_V3_2.pdf(V3.2-Exp 技术报告)| https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf"
date: 2026-10
---

# DeepSeek 网络结构演进总览(V1 → V2 → V3 → V3.2)

> 本页是 DeepSeek 网络结构演进的**导航主篇**:30 秒 TL;DR、逐代概览、代际 diff 总表、演进方向分析、溯源互链。三份逐代深挖见子篇:
> - 奠基段(注意力 MHA→GQA→MLA、FFN dense→稀疏 MoE、V2 双线合流):[deepseek-arch-v1-moe-v2](deepseek-arch-v1-moe-v2.md)
> - V3(MLA 规模化 + 256 专家 MoE + MTP + aux-loss-free + FP8):[deepseek-arch-v3](deepseek-arch-v3.md)
> - V3.2 + R1(DSA 稀疏注意力破 O(n²) + RL 推理时扩展):[deepseek-arch-v32-r1](deepseek-arch-v32-r1.md)

## TL;DR(30 秒结论)

**DeepSeek 六代结构(DeepSeek-LLM → DeepSeekMoE → V2 → V3 → V3.1 → V3.2)沿着一条不变的工程主线演进:在「算力(FLOPs)」与「显存(KV cache)」双重约束下——用稀疏省算力、用低秩省显存、用训练质量机制保住效果,始终把「训练质量-部署效率」当作双目标。** 具体分三条并行的机制线:

1. **省算力(FLOPs)→ FFN 稀疏化**:dense → DeepSeekMoE 细粒度+共享专家 → V2 160 专家 top-6 → V3 256 专家 top-8。671B 只激活 37B,算力省 ~18×。
2. **省显存(KV cache)→ 注意力低秩化 + 稀疏化**:MHA → GQA(砍头数)→ MLA(砍维度,低秩 latent 512 维 + 解耦 RoPE,比 MHA 省 ~57×)→ V3.2 DSA(再稀疏化,把注意力复杂度从 O(L²) 破到 O(L·2048),同时省计算与 KV 读取)。
3. **保质量 / 提效率 → 训练机制升级**:aux-loss-free 负载均衡(卸掉 aux loss 对质量的拉扯)、MTP 多 token 预测(密化训练信号 + 预规划表示)、FP8 混合精度(压成本)、R1 GRPO + 冷启动/蒸馏(推理时扩展,不动结构)。

一句话:**V2 用低秩(MLA)省显存、V3 用稀疏(MoE)+ 训练工程省算力保质量、V3.2 用稀疏(DSA)破注意力 O(n²)——每一代都在「以稀疏/低秩换效率,质量不倒退」这条轴上往前推一格。**

---

## 逐代概览表

| 代 | 结构名 | 核心增量 | 总参 / 激活 | 上下文 | arXiv |
|---|---|---|---|---|---|
| V1(DeepSeek-LLM) | dense Transformer | 基座;7B 纯 MHA / **67B 用 GQA**(首压 KV cache) | 7B / 67B | 4K | [2401.02954](https://arxiv.org/abs/2401.02954) |
| DeepSeekMoE | 稀疏 MoE(独立篇) | FFN 换**细粒度专家 + 共享专家**;2B≈GShard2.9B、16B≈LLaMA2 7B(仅 40% 算力) | 16.4B / 2.8B | 4K | [2401.06066](https://arxiv.org/abs/2401.06066) |
| **V2** | MLA + 稀疏 MoE | **MLA(低秩 KV + 解耦 RoPE)** + 160+2 专家 top-6 分组路由;双主线首次合流 | 236B / 21B | 128K | [2405.04434](https://arxiv.org/abs/2405.04434) |
| **V3** | MLA + 256 专家 MoE + MTP | **aux-loss-free 负载均衡 + MTP 多 token 预测 + FP8**;256 专家 top-8 | 671B / 37B | 128K | [2412.19437](https://arxiv.org/abs/2412.19437) |
| V3.1-Terminus | 结构同 V3 | 长上下文 128K + RL/agent 后训练(结构不变) | 671B / 37B | 128K | — |
| **V3.2-Exp/正式** | MLA + DSA 稀疏注意力 | **DSA 闪电索引器 + top-2048 选择**,破 O(L²)→O(L·k);MLA/MoE/MTP 不动 | 671B / 37B | 128K(训练)/163840(config) | [2512.02556](https://arxiv.org/abs/2512.02556) |
| **R1**(旁支) | 结构 = V3 | **无新结构**;V3-Base 上 GRPO + 冷启动/蒸馏(RL 推理时扩展) | 671B / 37B | — | [2501.12948](https://arxiv.org/abs/2501.12948) |

> 深挖各代:V1/DeepSeekMoE/V2 → [deepseek-arch-v1-moe-v2](deepseek-arch-v1-moe-v2.md);V3 → [deepseek-arch-v3](deepseek-arch-v3.md);V3.2+R1 → [deepseek-arch-v32-r1](deepseek-arch-v32-r1.md)。

---

## 代际 diff 总表(三条机制线)

### 注意力:MHA → GQA → MLA → MLA+DSA

| 代 | 机制 | 每 token 每层 KV 维度 | 核心增量 |
|---|---|---|---|
| V1-7B | MHA(32 头·128 维) | 2×32×128 = 8192 | 基线 |
| V1-67B | GQA(8 KV 头) | 2×8×128 = 2048 | 砍**头数**(8:1) |
| V2/V3 | **MLA**(低秩 latent c^KV 512 + 解耦 RoPE 64) | 512+64 = **576** | 砍**维度**(低秩压缩;比 MHA 省 ~57×,比 GQA 再省 ~3.6×) |
| V3.2 | MLA + **DSA 稀疏** | 576(缓存)+ 只 attend top-2048 | 破复杂度 O(L²)→O(L·k),稀疏读 KV |

### FFN:dense → 细粒度 MoE → 256 专家 MoE

| 代 | 结构 | 路由/共享 | 每 token 激活 |
|---|---|---|---|
| V1 | dense | — | 全部 |
| DeepSeekMoE | 细粒度 MoE(首层 dense) | N 路由 + Ks 共享 | 2B:6/64 路由 + 1 共享 |
| V2 | 细粒度 MoE | 160 路由 + 2 共享 | 6 路由(top-6,分组受限贪心) |
| V3 | 细粒度 MoE | **256 路由 + 1 共享**,每专家 inter=2048 | 8 路由(top-8,节点受限) |
| V3.2 | 同 V3(0 改动) | 同上 | 同上 |

### 位置编码:RoPE → 解耦 RoPE → YaRN

| 代 | 机制 |
|---|---|
| V1/DeepSeekMoE | 标准 RoPE(θ=10000) |
| V2/V3/V3.2 | **解耦 RoPE**(q_nope/k_nope 不旋转 + q_pe/k_pe 旋转,位置绕过低秩瓶颈)+ **YaRN 外推**(base 10000,factor=40,4K→128K) |

### 新增机制累计(每代引入的关键项)

- **GQA**(V1-67B):KV 头数压缩。
- **细粒度专家 + 共享专家**(DeepSeekMoE):MoE 稀疏化的两大支柱。
- **MLA + 解耦 RoPE**(V2):KV 低秩压缩 + 位置解耦。
- **aux-loss-free 负载均衡**(V3):bias 项动态调,卸掉 aux loss 对质量的拉扯。
- **MTP 多 token 预测**(V3):密化信号 + 预规划表示,推理可丢弃/当投机解码。
- **DSA / 闪电索引器**(V3.2):GPU 对齐的稀疏注意力,破 O(L²)。

---

## 演进方向分析(核心)

### 双约束下的「三件套」主线

DeepSeek 的所有结构决策都能归到两个硬约束与一个质量护栏:

- **约束 1 — 算力(FLOPs)**:训练/推理每 token 的浮点运算。dense 模型每 token 过完整 FFN,成本随容量线性涨 → 用**稀疏 MoE** 只激活一小部分专家(671B 激活 37B)。
- **约束 2 — 显存(KV cache)**:长上下文下 KV cache 随 层数×头数×头维×序列长 膨胀 → 用**低秩 MLA**(砍维度)+ **稀疏 DSA**(只读 top-k 个 KV)省显存。
- **质量护栏 — 训练机制**:省资源不能牺牲效果,所以每代都配「保质量」机制(aux-loss-free、MTP、FP8、RL),把「部署效率」的收益兑现成实打实的能力。

### 三条机制线的演进逻辑

1. **省算力线**:dense → DeepSeekMoE(细粒度+共享专家,解决 GShard 专家专精不足)→ 规模化(160→256 专家);核心是「用更少算力逼近同容量 dense 上限」。
2. **省显存线**:MHA → GQA(砍头)→ MLA(砍维度,低秩重构 + RoPE 解耦,这是相对 GQA 的本质优势,不是共享头而是低秩重构)→ DSA(稀疏选择,同时省长上下文算力与 KV 读取)。
3. **保质量线**:V2 用 aux loss → V3 换 **aux-loss-free**(bias 项,消除 aux loss 对质量的拉扯)+ MTP(密化信号);V3.2 用「dense 预热索引器 + 稀疏续训」两阶段,保证稀疏选择不显著退化长上下文性能(Table 1 与 V3.1-Terminus 基本打平)。

### 结构化演进结论(3-5 条)

1. **双目标是贯穿性框架**:从 V1-67B 的 GQA 到 V3.2 的 DSA,每一步都在「训练质量-部署效率」这对目标上同时做文章——用稀疏/低秩换效率,同时用训练机制/两阶段训练把关质量,质量不倒退是硬的验收线。
2. **注意力走「先砍头数、再砍维度、再稀疏化」三步**:GQA 是对 KV cache 的第一次压缩(铺垫),MLA 用低秩把压缩推到极致(57×),DSA 再进一步从「缓存多少」推进到「attend 多少」(O(L²)→O(L·k))——省显存到最后变成省计算 + 省 KV 读取。
3. **MoE 是「省算力」的固定解法,但负载均衡手段在升级**:细粒度+共享专家(解决专精)→ 分组/节点受限路由(优化通信)→ aux-loss-free bias 均衡(消除 aux loss 对质量的负面拉扯)。MoE 本身从 V2 到 V3.2 结构几乎不再演进,演进重心转向训练工程。
4. **训练工程是 V3/V3.2 的隐形主角**:V3 用 aux-loss-free + MTP + FP8 把「规模化 + 训练工程成熟化」具象化(2.788M H800 小时训出当时最强开源基座);V3.2 用 DSA 两阶段训练把稀疏注意力安全地长出来——结构创新与训练策略深度绑定。
5. **R1 是结构主线外的正交支线**:V3.2 改「网络怎么算」,R1 改「怎么训、怎么推理」(推理时扩展,延长 CoT 换能力),结构零改动——证明 DeepSeek 的能力跃迁来自结构效率与训练/推理范式的双轮驱动。

---

## 溯源

- DeepSeek LLM (V1):https://arxiv.org/abs/2401.02954
- DeepSeekMoE:https://arxiv.org/abs/2401.06066
- DeepSeek-V2:https://arxiv.org/abs/2405.04434
- DeepSeek-V3 Technical Report:https://arxiv.org/abs/2412.19437
- DeepSeek-V3.2:https://arxiv.org/abs/2512.02556
- DeepSeek-V3.2-Exp 技术报告:https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
- DeepSeek-R1:https://arxiv.org/abs/2501.12948

## 相关条目(库内互链)

- 三子篇(本主篇的逐代深挖):[deepseek-arch-v1-moe-v2](deepseek-arch-v1-moe-v2.md)、[deepseek-arch-v3](deepseek-arch-v3.md)、[deepseek-arch-v32-r1](deepseek-arch-v32-r1.md)
- [attention-heads-mha-mqa-gqa](attention-heads-mha-mqa-gqa.md):注意力头共享谱系,MLA 低秩压缩的上位视角。
- [context-extension](context-extension.md):RoPE 外推机制(YaRN/PI/NTK/LongRoPE)的通用讲解。
- [open-model-landscape](open-model-landscape.md) 与 [open-model-landscape-2025-2026](open-model-landscape-2025-2026.md):DeepSeek 在开源谱系中的位置(含 V3.2 DSA、R1 的横向对比)。
- 架构通用底座:[llm-architecture](llm-architecture.md)(pre-RMSNorm + SwiGLU + RoPE 定型配方)。
- 库内 infra 交叉(V3.2 依赖算子):`deepseek-infra/concepts/deepselect.md`(DSA 链路前置 TopK 算子)、`deepseek-infra/concepts/flashmla.md`(FlashMLA sparse 消费 indices 的后端)。
- **领域模型谱系(结构与领域互补)**:本系列讲「网络结构怎么变」;「领域能力(数学/编码/定理证明)怎么做出来」——GRPO RL、Lean 4 形式化证明、FIM 代码填充、MoE 数据评测——见 [`deepseek-models/`](../../deepseek-models/index.md)。其中 DeepSeekMoE 的数据/评测侧在 [deepseek-models/concepts/deepseek-moe.md](../../deepseek-models/concepts/deepseek-moe.md)(结构在本系列 v1-moe-v2 篇,两篇互链不重复)。
