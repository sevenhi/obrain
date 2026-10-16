---
title: deepseek-models — DeepSeek 领域模型谱系(数学/定理证明/编码)
type: domain
tags: [deepseek, math, theorem-proving, coding, grpo, rl, moe, lean4, fim, code-model, r1, reasoning, esft, domain-model]
date: 2026-10
---

# deepseek-models — DeepSeek 领域模型谱系(数学/定理证明/编码)

> 本页是 deepseek-models 领域的**导航起始页**。真正的可读正文在 `concepts/`(领域模型谱系)与 `frameworks/`(工具)子目录,每篇一个专题、可独立打开读懂。本页只做定位与导航。

## 一句话定位

**deepseek-models** 记录 DeepSeek 面向特定能力的**领域模型谱系**——数学推理(Math/Math-V2)、形式化定理证明(Prover-V1/V1.5/V2)、代码智能(Coder/Coder-V2)、MoE 基座(MoE),以及两条训练策略主线:推理 RL(R1-Zero→R1→Distill,GRPO 规模化)与 MoE 专家专精微调(ESFT)。与 [`model-architecture/`](../model-architecture/index.md) 里偏「网络结构」的 DeepSeek 演进四篇**互补**:那四篇讲「注意力 MHA→GQA→MLA、FFN dense→MoE 的**结构**怎么变」,本域讲「**领域能力怎么做出来**」——核心卖点是**数据工程、训练策略(GRPO/RLPAF/RL)、评测**,结构多复用 V1/V2/V3 家族(详见各篇与网络结构篇互链)。

## 谱系总览(五条)

| 谱系 | 仓库 | 底座 | 核心增量 | 概念篇 |
|---|---|---|---|---|
| 数学推理 | DeepSeek-Math → Math-V2 | Coder-v1.5 7B → V3.2-Exp-Base | GRPO RL / 自验证 | [deepseek-math](concepts/deepseek-math.md) |
| 定理证明(Lean 4) | Prover-V1 → V1.5 → V2 | Math 7B → V1.5 → V3-Base | 合成数据 / RLPAF / 子目标分解 | [deepseek-prover](concepts/deepseek-prover.md) |
| 代码智能 | Coder → Coder-V2 | 从零 → DeepSeek-V2 | FIM 填充 / MoE 代码模型 | [deepseek-coder](concepts/deepseek-coder.md) |
| MoE 基座 | DeepSeek-MoE | 从零 | 细粒度+共享专家(数据/评测侧) | [deepseek-moe](concepts/deepseek-moe.md) |
| 推理 RL | DeepSeek-R1-Zero → R1 → Distill | V3-Base → Qwen/Llama dense | 纯 RL 推理涌现 / 冷启动四阶段 / 800k 蒸馏 | [deepseek-r1-reasoning](concepts/deepseek-r1-reasoning.md) |

## 条目索引

| 条目 | 类型 | 一句话 | 状态 |
|---|---|---|---|
| [deepseek-math](concepts/deepseek-math.md) | concept | 数学推理:DeepSeekMath 7B(120B 数学 token 数据管道 + GRPO RL)51.7% MATH → Math-V2(自验证:训练 verifier→用其做 reward 训 generator,IMO 2025 金牌/Putnam 118/120) | ✅ 已入库(2026-10) |
| [deepseek-prover](concepts/deepseek-prover.md) | concept | 形式化定理证明(Lean 4):Prover-V1(8M 合成证明数据,miniF2F 46.3%)→ V1.5(RLPAF + RMaxTS,63.5%)→ V2(DeepSeek-V3 递归子目标分解 + RL,miniF2F 88.9%) | ✅ 已入库(2026-10) |
| [deepseek-coder](concepts/deepseek-coder.md) | concept | 代码模型:Coder(从零 2T tokens,87% 代码,FIM 填充,1B~33B)→ Coder-V2(DeepSeekMoE 底座 16B/236B,338 语言,HumanEval 90.2 打平 GPT-4-Turbo) | ✅ 已入库(2026-10) |
| [deepseek-moe](concepts/deepseek-moe.md) | concept | DeepSeekMoE 16B:细粒度专家+共享专家,40% 算力打平 DeepSeek 7B/LLaMA2 7B;数据/评测侧重(结构细节引网络结构篇) | ✅ 已入库(2026-10) |
| [deepseek-r1-reasoning](concepts/deepseek-r1-reasoning.md) | concept | R1 推理谱系(训练策略视角):R1-Zero(V3-Base 纯 RL,AIME 15.6→71.0%,反思/aha moment 涌现)→ R1(冷启动+四阶段流水线,AIME 79.8 对标 o1-1217)→ Distill(800k 样本只 SFT,32B 超 o1-mini,蒸馏打赢同规模 RL 72.6 vs 47.0);GRPO 公式/reward 设计/失败尝试(PRM/MCTS)核自本地论文 PDF+README | ✅ 已入库(2026-10-14) |
| [deepseek-esft](concepts/deepseek-esft.md) | concept | ESFT 专家专精微调(模型领域视角):打分(gate/token 两口径)→ top_p 选任务相关专家 → parameter↔buffer 互换只训这批专家(esft.py 源码机制);MoE 版 PEFT,adapter 即被训专家权重包;操作流程见 deepseek-infra 篇,本篇讲机制与谱系定位 | ✅ 已入库(2026-10-14) |
| [deepseek-harness](frameworks/deepseek-harness.md) | howto | DeepSeek Harness(dsh):everything-is-a-plugin 架构的 agent 框架,基于 Cordis,TS/npm,developer preview | ✅ 已入库(2026-10,含 0.2.1-alpha.5) |
| [deepspec](frameworks/deepspec.md) | howto | DeepSpec:投机解码草稿模型训练/评测全栈(DSpark/DFlash/Eagle3),数据→训练→评测,可为 Qwen3/Gemma 等目标模型练 draft(arXiv:2607.05147) | ✅ 已入库(2026-10) |
| [dsh-node-addon-require-builtin](frameworks/dsh-node-addon-require-builtin.md) | howto | dsh 的内部 Node 原生插件:不亮 `--expose-internals` 拿到私有 requireBuiltin(探 Realm 指针/机器码偏移)、internal-loader 白名单 vs require-builtin 无限制、fail-closed,Node 20/22/24/26+Electron(2026-10-12 新仓) | ✅ 已入库(2026-10-12) |

## 关键事实速记

- **领域模型 = 底座复用 + 领域数据/训练策略**:数学/编码模型几乎不发明新网络结构,而是「选一个底座(通用/代码基座)→ 领域数据续训 → 强化学习/指令微调」。真正的工作量在**数据工程**(DeepSeekMath 的 CC 数学语料管道、Prover 的 Lean 合成证明、Coder 的依赖重排+repo 去重)和**训练策略**(GRPO/RLPAF/RMaxTS/RL)。
- **GRPO(Group Relative Policy Optimization)**:DeepSeekMath 提出,PPO 的变体——去掉价值函数 critic,用「组内相对奖励」做优势估计,省 PPO 的内存;后被 DeepSeek-R1 复用,成为推理 RL 事实标准。
- **定理证明分两条路**:Prover 系列走 **Lean 4 形式化**(formal,编译器级验证),Math-V2 走 **自然语言非形式证明**(informal,靠训练出来的 verifier 自验证)。V2 用「V3 拆子目标 + 7B 证子目标」把两条路的推理能力合流。
- **代码模型的杀手锏是 FIM(fill-in-the-middle)**:Coder 用 `<|fim_begin|>…<|fim_hole|>…<|fim_end|>` 特殊 token 训练填充任务,支持项目级补全/插入,这是它相对纯自回归代码模型的核心差异。
- **底座谱系**:Math 起于 Coder-v1.5 7B;Coder-V2/Prover-V2-671B 起于 DeepSeek-V2/V3 的 MoE 底座;Math-V2 起于 V3.2-Exp-Base——领域模型跟着主模型代际走。
- **推理 RL 主线(R1)**:GRPO 从 DeepSeekMath 的「RL 收尾步骤」放大为 R1 的核心范式——纯 RL 可激励推理(无 SFT),冷启动+迭代 RL/SFT 交替更稳;小模型学推理「蒸馏 > 自己 RL」。详见 [deepseek-r1-reasoning](concepts/deepseek-r1-reasoning.md);R1 无新网络结构,结构侧见 model-architecture 域。
- **MoE 微调主线(ESFT)**:稀疏结构的微调端兑现——「任务只用一小撮专家,就只训这一小撮」,省算力省存储;与 R1 的 RL 路线正交(一个管领域知识注入,一个管推理行为涌现)。详见 [deepseek-esft](concepts/deepseek-esft.md)。

## 相关条目(跨域互链)

- **[`model-architecture/`](../model-architecture/index.md)**:本域各模型复用的**网络结构**在这——DeepSeek 演进主篇 [deepseek-model-evolution](../model-architecture/concepts/deepseek-model-evolution.md)、奠基段 [deepseek-arch-v1-moe-v2](../model-architecture/concepts/deepseek-arch-v1-moe-v2.md)(DeepSeekMoE 细粒度+共享专家结构)、V3 [deepseek-arch-v3](../model-architecture/concepts/deepseek-arch-v3.md)、V3.2+R1 [deepseek-arch-v32-r1](../model-architecture/concepts/deepseek-arch-v32-r1.md)(DSA 稀疏注意力;**R1 结构侧解读**——与本域 [deepseek-r1-reasoning](concepts/deepseek-r1-reasoning.md) 的训练策略视角明确分工)。本域 [deepseek-moe](concepts/deepseek-moe.md) 只补数据/评测,结构不重复。
- **[`deepseek-infra/`](../deepseek-infra/index.md)**:本域 deepseek-harness 是 DeepSeek 的 agent 工具,与 infra 域(算子/昇腾栈)同属「非模型」工具侧,但按简报归入本域工具节。**ESFT 双视角**:操作流程/脚本/实践坑在 [deepseek-infra/concepts/esft.md](../deepseek-infra/concepts/esft.md),机制/源码/谱系定位在本域 [deepseek-esft](concepts/deepseek-esft.md),两篇互链不重复;EPLB/LPLB(部署侧动专家)对照见 infra 域。
- **[`vllm/`](../vllm/index.md)**:Coder/Coder-V2 官方推荐 vLLM/SGLang 高吞吐推理;MoE 代码模型的专家并行/MLA 在引擎侧兑现;R1-Distill 系列的本地服务(vllm serve)同样走 vLLM。

## 溯源

- 外部链接索引见 [references.md](references.md)。
