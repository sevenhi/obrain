---
title: deepseek-models — DeepSeek 领域模型谱系(数学/定理证明/编码)
type: domain
tags: [deepseek, math, theorem-proving, coding, grpo, rl, moe, lean4, fim, code-model, domain-model]
date: 2026-10-09
---

# deepseek-models — DeepSeek 领域模型谱系(数学/定理证明/编码)

> 本页是 deepseek-models 领域的**导航起始页**。真正的可读正文在 `concepts/`(领域模型谱系)与 `frameworks/`(工具)子目录,每篇一个专题、可独立打开读懂。本页只做定位与导航。

## 一句话定位

**deepseek-models** 记录 DeepSeek 面向特定能力的**领域模型谱系**——数学推理(Math/Math-V2)、形式化定理证明(Prover-V1/V1.5/V2)、代码智能(Coder/Coder-V2)以及 MoE 基座(MoE)。与 [`model-architecture/`](../model-architecture/index.md) 里偏「网络结构」的 DeepSeek 演进四篇**互补**:那四篇讲「注意力 MHA→GQA→MLA、FFN dense→MoE 的**结构**怎么变」,本域讲「**领域能力怎么做出来**」——核心卖点是**数据工程、训练策略(GRPO/RLPAF/RL)、评测**,结构多复用 V1/V2/V3 家族(详见各篇与网络结构篇互链)。

## 三条谱系

| 谱系 | 仓库 | 底座 | 核心增量 | 概念篇 |
|---|---|---|---|---|
| 数学推理 | DeepSeek-Math → Math-V2 | Coder-v1.5 7B → V3.2-Exp-Base | GRPO RL / 自验证 | [deepseek-math](concepts/deepseek-math.md) |
| 定理证明(Lean 4) | Prover-V1 → V1.5 → V2 | Math 7B → V1.5 → V3-Base | 合成数据 / RLPAF / 子目标分解 | [deepseek-prover](concepts/deepseek-prover.md) |
| 代码智能 | Coder → Coder-V2 | 从零 → DeepSeek-V2 | FIM 填充 / MoE 代码模型 | [deepseek-coder](concepts/deepseek-coder.md) |
| MoE 基座 | DeepSeek-MoE | 从零 | 细粒度+共享专家(数据/评测侧) | [deepseek-moe](concepts/deepseek-moe.md) |

## 条目索引

| 条目 | 类型 | 一句话 | 状态 |
|---|---|---|---|
| [deepseek-math](concepts/deepseek-math.md) | concept | 数学推理:DeepSeekMath 7B(120B 数学 token 数据管道 + GRPO RL)51.7% MATH → Math-V2(自验证:训练 verifier→用其做 reward 训 generator,IMO 2025 金牌/Putnam 118/120) | ✅ 已入库(2026-10) |
| [deepseek-prover](concepts/deepseek-prover.md) | concept | 形式化定理证明(Lean 4):Prover-V1(8M 合成证明数据,miniF2F 46.3%)→ V1.5(RLPAF + RMaxTS,63.5%)→ V2(DeepSeek-V3 递归子目标分解 + RL,miniF2F 88.9%) | ✅ 已入库(2026-10) |
| [deepseek-coder](concepts/deepseek-coder.md) | concept | 代码模型:Coder(从零 2T tokens,87% 代码,FIM 填充,1B~33B)→ Coder-V2(DeepSeekMoE 底座 16B/236B,338 语言,HumanEval 90.2 打平 GPT-4-Turbo) | ✅ 已入库(2026-10) |
| [deepseek-moe](concepts/deepseek-moe.md) | concept | DeepSeekMoE 16B:细粒度专家+共享专家,40% 算力打平 DeepSeek 7B/LLaMA2 7B;数据/评测侧重(结构细节引网络结构篇) | ✅ 已入库(2026-10) |
| [deepseek-harness](frameworks/deepseek-harness.md) | howto | DeepSeek Harness(dsh):everything-is-a-plugin 架构的 agent 框架,基于 Cordis,TS/npm,developer preview | ✅ 已入库(2026-10) |

## 关键事实速记

- **领域模型 = 底座复用 + 领域数据/训练策略**:数学/编码模型几乎不发明新网络结构,而是「选一个底座(通用/代码基座)→ 领域数据续训 → 强化学习/指令微调」。真正的工作量在**数据工程**(DeepSeekMath 的 CC 数学语料管道、Prover 的 Lean 合成证明、Coder 的依赖重排+repo 去重)和**训练策略**(GRPO/RLPAF/RMaxTS/RL)。
- **GRPO(Group Relative Policy Optimization)**:DeepSeekMath 提出,PPO 的变体——去掉价值函数 critic,用「组内相对奖励」做优势估计,省 PPO 的内存;后被 DeepSeek-R1 复用,成为推理 RL 事实标准。
- **定理证明分两条路**:Prover 系列走 **Lean 4 形式化**(formal,编译器级验证),Math-V2 走 **自然语言非形式证明**(informal,靠训练出来的 verifier 自验证)。V2 用「V3 拆子目标 + 7B 证子目标」把两条路的推理能力合流。
- **代码模型的杀手锏是 FIM(fill-in-the-middle)**:Coder 用 `<|fim_begin|>…<|fim_hole|>…<|fim_end|>` 特殊 token 训练填充任务,支持项目级补全/插入,这是它相对纯自回归代码模型的核心差异。
- **底座谱系**:Math 起于 Coder-v1.5 7B;Coder-V2/Prover-V2-671B 起于 DeepSeek-V2/V3 的 MoE 底座;Math-V2 起于 V3.2-Exp-Base——领域模型跟着主模型代际走。

## 相关条目(跨域互链)

- **[`model-architecture/`](../model-architecture/index.md)**:本域各模型复用的**网络结构**在这——DeepSeek 演进主篇 [deepseek-model-evolution](../model-architecture/concepts/deepseek-model-evolution.md)、奠基段 [deepseek-arch-v1-moe-v2](../model-architecture/concepts/deepseek-arch-v1-moe-v2.md)(DeepSeekMoE 细粒度+共享专家结构)、V3 [deepseek-arch-v3](../model-architecture/concepts/deepseek-arch-v3.md)。本域 [deepseek-moe](concepts/deepseek-moe.md) 只补数据/评测,结构不重复。
- **[`vllm/`](../vllm/index.md)**:Coder/Coder-V2 官方推荐 vLLM/SGLang 高吞吐推理;MoE 代码模型的专家并行/MLA 在引擎侧兑现。
- **[`deepseek-infra/`](../deepseek-infra/index.md)**:本域 deepseek-harness 是 DeepSeek 的 agent 工具,与 infra 域(算子/昇腾栈)同属「非模型」工具侧,但按简报归入本域工具节。

## 溯源

- 外部链接索引见 [references.md](references.md)。
