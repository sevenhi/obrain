---
title: DeepSpec — 投机解码草稿模型的训练与评测全栈
type: howto
tags: [deepspec, speculative-decoding, eagle3, dspark, dflash, draft-model, mtp, vllm, deepseek]
date: 2026-10
source:
  - "code: deepseek-ai/DeepSpec | https://github.com/deepseek-ai/DeepSpec | 2026-10-11 | ↑primary"
  - "paper: DSpark(Confidence-Scheduled Speculative Decoding)| arXiv:2607.05147 | https://arxiv.org/abs/2607.05147"
  - "paper: DFlash | arXiv:2602.06036 | https://arxiv.org/abs/2602.06036"
  - "paper: EAGLE-3 | arXiv:2503.01840 | https://arxiv.org/abs/2503.01840"
---

# DeepSpec — 投机解码草稿模型的训练与评测全栈

> **TL;DR**:DeepSpec 是 DeepSeek 开源的**投机解码草稿模型(draft model)训练与评测全栈**,覆盖「数据准备 → 训练 → 评测」三阶段。内置三种草稿模型算法:**DSpark**(自研,confidence-scheduled + 半自回归)、**DFlash**、**Eagle3**。可对任意目标模型(Qwen3 / Gemma 等)训练草稿模型,在 gsm8k/math500/aime25/humaneval 等基准上量 acceptance。与库内 V3 的 MTP、[`vllm` 投机解码](../../vllm/concepts/speculative-decoding.md)同属「多 token / 草稿加速」谱系——DeepSpec 负责**把草稿模型练出来**,vLLM 负责**把投机解码跑起来**。

## 1. 是什么

- **不是模型,是训练框架**:数据管道 + 草稿模型实现 + 训练脚本 + 评测脚本。
- 解决的问题:投机解码要一个「跟得上目标模型分布、又足够小」的草稿模型;DeepSpec 把这类模型的**可复现训练配方**开源。
- 默认硬件假设:单机 8 卡;少卡需改 `CUDA_VISIBLE_DEVICES`。

## 2. 三种草稿算法

| 算法 | 论文 | 要点 | 备注 |
|---|---|---|---|
| **DSpark** | [arXiv:2607.05147](https://arxiv.org/abs/2607.05147) | Confidence-Scheduled Speculative Decoding + 半自回归生成 | DeepSeek 自研,配置在 `config/dspark/` |
| **DFlash** | [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) | 块扩散式草稿模型(block diffusion)+ 训练配方 | 配置在 `config/dflash/` |
| **Eagle3** | [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) | 特征层对齐的自回归草稿 | 实现部分改编自 [SpecForge](https://github.com/sgl-project/SpecForge)(Apache-2.0,见 `NOTICE`) |

已发布的 checkpoint 覆盖目标模型:`Qwen/Qwen3-4B` / `8B` / `14B`、`google/gemma-4-12B-it`;HF 命名如 `deepseek-ai/dspark_qwen3_4b_block7`、`deepseek-ai/eagle3_qwen3_4b_ttt7`。

## 3. 工作流(howto)

```sh
# 1) 数据准备 —— 下载 prompt、用目标模型重生成答案、构建 target cache
#    ⚠ 存储警告:默认 Qwen/Qwen3-4B 设置下 target cache 约 38 TB
#    细节见 scripts/data/README.md

# 2) 训练
bash scripts/train/train.sh
# train.sh 按 config_path 选算法与目标模型,例如 config/dspark/dspark_qwen3_4b.py
# checkpoint 写到 ~/checkpoints/<project_name>/<exp_name>/step_*

# 3) 评测 —— 在 gsm8k / math500 / aime25 / humaneval / mbpp /
#    livecodebench / mt-bench / alpaca / arena-hard-v2 上量 acceptance
bash scripts/eval/eval.sh
# 指定 target_name_or_path 与 draft_name_or_path(本地 step_* 或 HF checkpoint)
```

## 4. 代码结构

```
DeepSpec/
├── deepspec/           # 核心库
│   ├── data/           # 数据管道
│   ├── modeling/       # 草稿模型实现
│   ├── trainer/        # 训练循环
│   └── eval/           # 评测
├── config/             # 训练配置(dspark / dflash / eagle3)
├── scripts/            # train.sh / eval.sh / data/
├── eval_datasets/      # 评测集
├── train.py  eval.py
└── NOTICE              # SpecForge / DFlash 上游归属
```

## 5. 在 DeepSeek 谱系中的位置

- **与 V3 MTP 的关系**:MTP 是**训练目标**(预测额外 token,推理可丢弃/转投机);DeepSpec 是**独立的草稿模型训练**,两者都是「一次前向多出 token」路线,但 MTP 嵌在主模型训练里,DeepSpec 是事后另训一个小模型。
- **与 vLLM 的关系**:vLLM 侧有 `Eagle3{...}ForCausalLM` 等草稿模型类与投机解码调度(见 [`vllm/concepts/speculative-decoding.md`](../../vllm/concepts/speculative-decoding.md));DeepSpec 产出的 checkpoint 可作为这类引擎的 draft 输入。
- **适用场景**:要给自己的目标模型配一个可验证 acceptance 的草稿模型;或复现 DSpark/DFlash/Eagle3 对比实验。

## 6. 溯源

- 仓库:https://github.com/deepseek-ai/DeepSpec
- DSpark 论文:https://arxiv.org/abs/2607.05147
- DFlash:https://arxiv.org/abs/2602.06036 · 原实现 https://github.com/z-lab/dflash
- EAGLE-3:https://arxiv.org/abs/2503.01840 · 训练框架参考 SpecForge
- 库内互链:[deepseek-models/index.md](../index.md)、[vllm/concepts/speculative-decoding.md](../../vllm/concepts/speculative-decoding.md)、[model-architecture/concepts/deepseek-arch-v3.md](../../model-architecture/concepts/deepseek-arch-v3.md)(MTP)
