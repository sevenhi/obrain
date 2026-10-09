---
title: ESFT — 专家特化微调(Expert-Specialized Fine-Tuning)
type: concept
tags: [esft, fine-tuning, moe, expert-specialization, sparse-architecture, adapters, emnlp2024, peft]
source:
  - "code: deepseek-ai/ESFT | https://github.com/deepseek-ai/ESFT ↑primary"
  - "web: Let the Expert Stick to His Last(ESFT 论文,arXiv:2407.01906)| https://arxiv.org/abs/2407.01906"
date: 2026-10
---

# ESFT — 专家特化微调(Expert-Specialized Fine-Tuning)

## TL;DR

**ESFT(Expert-Specialized Fine-Tuning)** 是 DeepSeek 的 MoE 高效微调方法(EMNLP 2024 Main Conference):**只微调与目标任务相关的 expert,冻结其余部分**,从而用更少的资源和存储高效定制 LLM。

**核心思想**:MoE 的不同 expert 天然对不同类型的输入敏感。先**打分**找出「这个任务主要激活哪些 expert」,再**只训练这批任务相关的 expert + 插 adapter**,其余 expert 和骨干冻结。效果:少训参数、省存储(只需存任务相关的 expert 差异/adapter)、性能不打折甚至更好——「让 expert 干它擅长的活(Stick to His Last)」。

---

## 1. 机制:三阶段流水线

ESFT 不是简单「冻几个 expert」,而是一条**「打分 → 选 expert → 特化训练」**的流水线:

### 1.1 阶段一:expert 打分(get_expert_scores.py)

在评估数据集上跑模型,统计**每个 expert 的激活分数/贡献**,量化「哪个 expert 对当前任务重要」。

```bash
python scripts/expert/get_expert_scores.py \
    --eval_dataset=intent --base_model_path=deepseek-ai/ESFT-vanilla-lite \
    --output_dir=results/expert_scores/intent \
    --n_sample_tokens=131072 --world_size=4 --gpus_per_rank=2
```

### 1.2 阶段二:生成 expert 配置(generate_expert_config.py)

按分数选出「任务相关 expert」,生成一个 expert 配置(哪些 expert 训、哪些冻),并按 score_function 和 top_p 超参裁剪。

```bash
python scripts/expert/generate_expert_config.py \
    --eval_dataset=intent --expert_scores_dir=results/expert_scores/intent \
    --output_path=results/expert_configs/intent.json \
    --score_function=token --top_p=0.2
```

### 1.3 阶段三:特化训练(train.py / train_ep.py)

按 expert 配置训练:只更新任务相关 expert(可能配 adapter),其余冻结。`train_ep.py` 是专家并行版,为多 GPU 优化。

```bash
python train.py --base_model_path=deepseek-ai/ESFT-vanilla-lite \
    --expert_config=results/expert_configs/intent.json \
    --train_dataset=intent --train_config=configs/base.yaml \
    --output_dir=results/checkpoints/intent

torchrun --nproc-per-node=8 train_ep.py \
    --base_model_path=deepseek-ai/ESFT-vanilla-lite \
    --expert_config=results/expert_configs/intent.json \
    --train_dataset=intent --save_opt_states \
    --train_config=configs/base.yaml --output_dir=results/checkpoints/test/eval_intent
```

### 1.4 为什么有效

MoE 的稀疏激活意味着**一个任务只激活一小撮 expert**。既然大部分 expert 对当前任务没贡献,就不该为它们付训练/存储成本。ESFT 把微调范围收窄到「任务真正用到的那批 expert」,实现参数高效微调(PEFT)的 MoE 版——比全量微调省资源,比只训 adapter 的通用 PEFT 更贴合 MoE 的稀疏结构。

---

## 2. 可复现细节

- 依赖:`transformers torch safetensors accelerate`;
- 模型:deepseek-ai/ESFT-vanilla-lite(基础 MoE-lite);adapter 预下载 `bash scripts/download_adapters.sh`;
- 评估:`eval_multigpu.py`(配 `scripts/eval.sh`,数据集如 translation/intent);
- 仓库另含训练/评估/选 expert 脚本全套;`score_function=token`、`top_p=0.2` 是选 expert 的两个超参。

---

## 3. 数据 / 定位

- **会议**:EMNLP 2024 Main Conference(2024-09-20 接收);
- **论文**:arXiv:2407.01906 《Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models》;
- **定位**:这是本批 11 仓里**唯一的「训练方法/论文仓库」**(其余多为算子/通信/存储/调度库),时间也最早(2024),是 DeepSeek 早期 MoE 高效微调方向的沉淀。

---

## 4. 实践要点 / 坑

- **ESFT 是「选专家 + 特化训练」方法,不是通用 PEFT 的替代**:它利用 MoE 稀疏结构,把「只训相关部分」做到 expert 粒度,与 LoRA/adapter 可组合(仓库即用 adapter)。
- **先打分再训练**:三阶段顺序不能省——没有 expert 分数就没有「该训谁」的依据;`top_p`/`score_function` 决定裁剪激进程度,是主要超参。
- **train_ep.py 才吃多 GPU**:单机 `train.py`、多 GPU/专家并行用 `torchrun ... train_ep.py`,别在单卡场景硬上 EP。
- **vanilla-lite 是基座**:ESFT 的评估/训练脚本围绕 deepseek-ai/ESFT-vanilla-lite(轻量 MoE)设计,换基座需核对 expert 打分脚本的兼容性。
- **别与 EPLB/LPLB 的「expert 负载均衡」混淆**:ESFT 的「expert 特化」是**微调时挑专家训练**;EPLB/LPLB 的「expert 复制」是**部署时摊负载**(见 [lplb-epb-load-balancing.md](lplb-epb-load-balancing.md))。一个管训练参数,一个管运行时算力。

---

## 溯源

- **ESFT 仓库**(主源:三阶段脚本/接口/依赖/EMNLP 信息):https://github.com/deepseek-ai/ESFT
- **ESFT 论文**(arXiv:2407.01906):https://arxiv.org/abs/2407.01906
- **库内交叉**:[lplb-epb-load-balancing.md](lplb-epb-load-balancing.md)(expert 负载均衡的对照)、[../../model-architecture/concepts/open-model-landscape.md](../../model-architecture/concepts/open-model-landscape.md)(MoE 架构背景)、[../../vllm/index.md](../../vllm/index.md)(推理侧,ESFT 微调产物的落地)。
