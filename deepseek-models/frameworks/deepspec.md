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

## 5. 训练侧工程优化:草稿模型训练也能快 3.5×(NVIDIA ModelOpt PR #2279)

DeepSpec 产出的草稿模型要反复试配方,单次训练成本直接决定迭代速度。NVIDIA 在自家 ModelOpt 的 DFlash 族 drafter 训练上做的一组优化,把 **Gemma-4-E4B DSpark streaming 的 step time 从 1.690s 压到 0.484s(3.49×)**,loss 在噪声内不变(PDX B300、3 节点 = 1 vLLM serve + 2 trainer、16 DP rank、batch 4×4096,取 steps 21-40 中位数;数字照抄 PR,未独立复现)。四步贡献拆得很干净,**每一步都是投机解码训练的通病**:

| 改动 | step time | 为什么有效 |
|---|---|---|
| 基线段(eager) | 1.690s | — |
| FlexAttention + block-sparse draft mask(可选 `dflash_use_flex_attention`) | → 0.711s(−58%) | 原来传的是 dense `[B,1,Q,KV]` mask,SDPA 被迫回落到 **sm80-only 的 memory-efficient kernel**;块稀疏 mask 让 FlexAttention 走专用路径 |
| 只投影 loss 真正读到的 teacher 行;DSpark 指标合并成一次 device sync | → 0.629s | 原来要为整段 `[B, seq, vocab]` 算 logits——vocab 大时这是主要显存/带宽开销;5 次 sync 变 1 次 |
| DSpark TVD 分块用 `Tensor.split` 而非切片 | → 0.541s | **切片的 backward 会给每个 chunk 零填充一整块 `[N, vocab]`**——一个纯反向图的隐性分配 |
| 对 draft stack 与 TVD chunk 上 `torch.compile`(可选 `dflash_use_torch_compile`) | → 0.484s | 编译或 FlexAttention 开启时 anchor 数由 config+seq_len 定死 → shape 静态,40 步内 recompile 次数 4→1 |

**踩坑清单(移植到自己的 draft 训练时直接照抄)**:

- **别用 HF 的 `flex_attention_forward` 派发**:它只要 query head 数是 2 的幂就传 `enable_gqa=True`,那条 backward 在 Gemma-4-E4B drafter 上要 **568ms**(是该 PR 自建路径的 10×、SDPA baseline 的 2.5×慢)——**mask 构造和 kernel 调用要自己控**;
- `head_dim > 256` 时 FlexAttention 默认 tile 在 **sm103** 上会直接挂(misaligned address / launch failure),需把 kernel tile 钉到 32×32,且 **BlockMask 的 block size 必须能整除 tile**;
- FlexAttention 与 **attention sink、dropout 冲突**,要在 convert 阶段就报错拒绝,别留到训练中途;
- 多节点训练必须把 `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` 指到**节点本地盘**——共享文件系统上并发 rank 会争抢编译缓存。

## 6. 在 DeepSeek 谱系中的位置

- **与 V3 MTP 的关系**:MTP 是**训练目标**(预测额外 token,推理可丢弃/转投机);DeepSpec 是**独立的草稿模型训练**,两者都是「一次前向多出 token」路线,但 MTP 嵌在主模型训练里,DeepSpec 是事后另训一个小模型。
- **与 vLLM 的关系**:vLLM 侧有 `Eagle3{...}ForCausalLM` 等草稿模型类与投机解码调度(见 [`vllm/concepts/speculative-decoding.md`](../../vllm/concepts/speculative-decoding.md));DeepSpec 产出的 checkpoint 可作为这类引擎的 draft 输入。
- **适用场景**:要给自己的目标模型配一个可验证 acceptance 的草稿模型;或复现 DSpark/DFlash/Eagle3 对比实验。

## 7. 溯源

- 仓库:https://github.com/deepseek-ai/DeepSpec
- DSpark 论文:https://arxiv.org/abs/2607.05147
- DFlash:https://arxiv.org/abs/2602.06036 · 原实现 https://github.com/z-lab/dflash
- EAGLE-3:https://arxiv.org/abs/2503.01840 · 训练框架参考 SpecForge
- §5 训练优化数据与踩坑:ModelOpt PR #2279 *perf(speculative): speed up DFlash/DSpark drafter training 3.49x*(2026-10-23 合并)— https://github.com/NVIDIA/Model-Optimizer/pull/2279
- 库内互链:[deepseek-models/index.md](../index.md)、[vllm/concepts/speculative-decoding.md](../../vllm/concepts/speculative-decoding.md)、[model-architecture/concepts/deepseek-arch-v3.md](../../model-architecture/concepts/deepseek-arch-v3.md)(MTP)、[model-architecture/concepts/hybrid-linear-attention.md](../../model-architecture/concepts/hybrid-linear-attention.md)(Gated DeltaNet 与 FLA 版本坑)
