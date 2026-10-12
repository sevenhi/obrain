---
title: deepseek-models 外部链接索引
type: reference
tags: [deepseek, math, theorem-proving, coding, grpo, rl, moe, references]
date: 2026-10
---

# References — deepseek-models

> 只放链接索引,不复制全文(库规约)。全部 URL 于 2026-10-09 curl 实测存活(仓库返回 200,arXiv 返回 200)。论文结论以 arXiv 摘要为准;README 与摘要数字冲突处已在各概念篇标注。

## 仓库(9 个,均 deepseek-ai 官方,2026-10-09 实测 200)

- DeepSeek-Math — https://github.com/deepseek-ai/DeepSeek-Math
- DeepSeek-Math-V2 — https://github.com/deepseek-ai/DeepSeek-Math-V2
- DeepSeek-Prover-V1.5 — https://github.com/deepseek-ai/DeepSeek-Prover-V1.5
- DeepSeek-Prover-V2 — https://github.com/deepseek-ai/DeepSeek-Prover-V2
- DeepSeek-Coder — https://github.com/deepseek-ai/DeepSeek-Coder
- DeepSeek-Coder-V2 — https://github.com/deepseek-ai/DeepSeek-Coder-V2
- deepseek-harness — https://github.com/deepseek-ai/deepseek-harness
- DeepSeek-MoE — https://github.com/deepseek-ai/DeepSeek-MoE
- dsh-node-addon-require-builtin(dsh 内部 Node 插件,2026-10-12 实测 200)— https://github.com/deepseek-ai/dsh-node-addon-require-builtin

## 原论文(arXiv)

- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models — https://arxiv.org/abs/2402.03300
- DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning — https://arxiv.org/abs/2511.22570
- DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data(V1)— https://arxiv.org/abs/2405.14333
- DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search — https://arxiv.org/abs/2408.08152
- DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition — https://arxiv.org/abs/2504.21801
- DeepSeek-Coder: When the Large Language Model Meets Programming — The Rise of Code Intelligence — https://arxiv.org/abs/2401.14196
- DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence — https://arxiv.org/abs/2406.11931
- DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models — https://arxiv.org/abs/2401.06066

## 工具/框架

- DeepSeek Harness 文档 — https://deepseek-harness.github.io/deepseek-harness/
- Cordis(deepseek-harness 底层)— https://github.com/cordiverse/cordis
- A Programming Paradigm for Spatiotemporal Composability(Cordis 设计)— https://arxiv.org/abs/2608.25512

## 模型下载(HuggingFace,deepseek-ai)

- deepseek-math-7b-base / -instruct / -rl — https://huggingface.co/deepseek-ai/deepseek-math-7b-base
- DeepSeek-Math-V2 — https://huggingface.co/deepseek-ai/DeepSeek-Math-V2
- DeepSeek-Prover-V1.5-Base / -SFT / -RL — https://huggingface.co/deepseek-ai/DeepSeek-Prover-V1.5-Base
- DeepSeek-Prover-V2-7B / -671B — https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B
- DeepSeek-Coder-V2-Lite-Base / -Instruct、-Base / -Instruct — https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base
- deepseek-moe-16b-base / -chat — https://huggingface.co/deepseek-ai/deepseek-moe-16b-base

## 相关领域(跨域)

- DeepSeek 网络结构演进(本域模型的结构底座)— 见 [model-architecture/references.md](../model-architecture/references.md)

### 投机解码草稿训练(2026-10 补)

- DeepSpec — https://github.com/deepseek-ai/DeepSpec
- DSpark — https://arxiv.org/abs/2607.05147
- DFlash — https://arxiv.org/abs/2602.06036
- EAGLE-3 — https://arxiv.org/abs/2503.01840
