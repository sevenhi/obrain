---
title: DeepSeek ESFT — 专家专精微调(模型领域视角:机制与源码)
type: concept
tags: [deepseek, esft, moe, expert-specialization, peft, fine-tuning, sparse-moe, emnlp2024, domain-model]
series: deepseek-esft
source:
  - "code: deepseek-ai/ESFT(esft.py / scripts/expert/ / configs/base.yaml)| https://github.com/deepseek-ai/ESFT | 2026-10-14 | ↑primary"
  - "pdf: Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models | arXiv:2407.01906 | https://arxiv.org/abs/2407.01906 | 2026-10-14"
date: 2026-10
---

# DeepSeek ESFT — 专家专精微调(模型领域视角:机制与源码)

> **TL;DR**:ESFT(Expert-Specialized Fine-Tuning,EMNLP 2024)回答的问题是「**MoE 大模型做领域微调,能不能只动一小部分参数?**」——答案是:MoE 每层只激活少数专家,一个任务其实只重度依赖「一小撮专家」,**先统计任务数据上每个专家的激活分,再只解冻这批专家训练,其余全部冻结**。实现上极其轻巧:esft.py 用 PyTorch 的 `parameter ↔ buffer` 互换来开关「可训练」状态(冻结 = 把 nn.Parameter 降级成 buffer,优化器就看不见它),按层按专家 ID 查配置表决定谁升回 parameter。效果:少训参数、少存 checkpoint(adapter 只含被训专家的权重差异)、性能不打折。**一句话:把 PEFT「低秩旁路」的思路换成「按专家挑选」,让稀疏结构自己告诉你该训谁。**
>
> **归属说明(先读这条)**:ESFT 的**操作流程/脚本用法/实践坑**已入库在 [deepseek-infra/concepts/esft.md](../../deepseek-infra/concepts/esft.md)(infra 域,「微调怎么省」环节)。本篇是**模型领域视角**:讲清它作为 DeepSeek MoE 谱系一环的「专家专精」思想、esft.py 的源码机制(parameter↔buffer 开关)、专家打分与 top_p 选专家算法。两篇互链、不重复:想上手跑脚本看 infra 篇,想懂「为什么这么设计、代码里发生了什么」看本篇。

---

## 1. 为什么:MoE 微调的浪费在哪(新手向)

先回忆 MoE 的基本盘(见 [deepseek-moe.md](deepseek-moe.md);注意 ESFT 实际基座是 **ESFT-vanilla-lite**,27 层/其中 26 层 MoE,结构细节见 §5):每层有很多专家(如 64 个),每个 token 只路由给 top-K 个(如 6 个)。全量微调一个 MoE 模型意味着:

- **算力浪费**:反向传播要更新所有专家的权重,但你的任务数据可能只重度激活其中 10–20%;
- **存储浪费**:每个下游任务都得存一份完整模型 checkpoint——671B 级模型每任务存一份根本不可行;
- **能力干扰**:动了与任务无关的专家,可能破坏模型在其它领域的既有能力(灾难性遗忘)。

**直觉例子**:一家医院有 64 个科室(MoE 专家),你要提升「骨科手术」水平(领域微调)。全量微调 = 给全院所有科室重新培训;ESFT = 先统计骨科病人实际经过哪些科室(挂号→影像→骨科→麻醉→康复),只给这几个科室做专项培训,其余科室照常运转。论文标题 *"Let the Expert Stick to His Last"*(让专家安守本行)说的就是这个:**每个专家只深耕它擅长的任务**。

下图把「三重浪费」和 ESFT 的三步解法摆成上下对照——上排解释为什么存在省的可能(稀疏性),中排是 ESFT 实际做的三步,下排是每一步省掉了什么:

<div class="archify-figure">
<iframe src="../../../diagrams/dm-esft-why-waste.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="MoE 全量微调的三重浪费与 ESFT 的三步解法"></iframe>
</div>

**谱系位置**:DeepSeekMoE(见 [deepseek-moe.md](deepseek-moe.md))提出「细粒度专家分割 + 共享专家隔离」,论证了专家会自然专精;ESFT 是这个思想的**微调端兑现**——既然专家已专精,微调就只训任务相关的专家。这是 DeepSeek MoE 谱系里「训练方法」一支的代表作(2024-07,arXiv:2407.01906),早于 R1 的 RL 路线。

---

## 2. 机制总览:打分 → 选专家 → 特化训练

三阶段流水线(脚本命令详见 infra 篇,这里讲**每步内部发生了什么**,全部核自本地源码):

下图按「阶段 / 内部动作 / 留下什么」三行展开这条流水线——注意每一行是同一列的三层视角,不是三个独立流程:

<div class="archify-figure">
<iframe src="../../../diagrams/dm-esft-three-stages.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="ESFT 三阶段:打分 → 选专家 → 特化训练"></iframe>
</div>

```
① get_expert_scores.py      在任务评估集上前向,记录每层每个专家的激活
        │                    (模型打开 config.log_expert_weights=True,
        │                     逐层落盘 expert_ids + expert_weights)
        ▼
② generate_expert_config.py  聚合分数 → 每层按分数排序 → top_p 累积裁剪
        │                    → 输出 expert_cfg.json(哪些层哪些专家可训)
        ▼
③ train.py / train_ep.py     esft.to_esft(model, expert_config)
                             把「选中专家」升为可训练参数,其余冻成 buffer
```

### 2.1 两种打分口径(核自 generate_expert_config.py)

对每层 l、每个专家 e,累计两种分数后归一化:

| 分数 | 计算方式 | 直觉 |
|---|---|---|
| **gate_scores** | 累加路由门控给该专家的权重值 `expert_weights` | 「门控认为这个专家多重要」(软权重) |
| **token_scores** | 每次被选中 +1/TOP_K | 「多少比例的 token 路过这个专家」(硬计数) |

`--score_function=token` 即用 token_scores。**选专家是 top_p 累积裁剪**(类似 nucleus sampling 的思路):

下图是这个循环的完整形状——两种打分口径进同一层,排序后逐专家累加,累计到 top_p 就停,产物是每层一张专家列表:

<div class="archify-figure">
<iframe src="../../../diagrams/dm-esft-top-p.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="top_p 累积裁剪:每层选出承担 20% 激活分数的头部专家"></iframe>
</div>

```python
# generate_expert_config.py 核心逻辑(简化,核自本地源码)
for layer, l_score in scores.items():
    l_score = sorted(l_score, key=lambda x: x[1], reverse=True)  # 按分数降序
    selected, current = [], 0.0
    for expert, score in l_score:
        if current >= top_p:      # 累计分数够 top_p(如 0.2)就停
            break
        selected.append(expert)
        current += score
    expert_cfg["experts"][layer] = selected
```

top_p=0.2 意思是「每层选出累计承担 20% 激活分数的头部专家」。配置还带两个开关:`shared_experts`(共享专家是否也训)、`non_expert_modules`(注意力/词表等非专家模块是否也训)。

### 2.2 冻结的实现:parameter ↔ buffer 互换(esft.py 核心,新手最值得学的技巧)

PyTorch 优化器只收集 `nn.Parameter`;`buffer` 是随模型保存但不参与梯度的张量。esft.py 的 `to_buffer()` 把模块所有 parameter **删掉注册、重新登记为 buffer**(并把参数名记进 `module.param_list` 以便还原),`to_param()` 反向还原。**冻结/解冻 = 这两个函数互换**,不需要手写 `requires_grad=False` 遍历,也不影响 state_dict 的保存/加载。

下图把这条调用链和「张量注册状态」并排看——上排是代码里发生的事,下排是同一时刻模型内部 Parameter 与 buffer 的分布:

<div class="archify-figure">
<iframe src="../../../diagrams/dm-esft-param-buffer.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="冻结的实现:parameter 与 buffer 互换"></iframe>
</div>

```python
# esft.py 核心调用示例(核自本地源码,ESFT 仓库 2024-08 训练代码版)
from esft import to_esft, load_esft_model, add_adapter

# 训练侧:按配置决定谁可训(train.py L94 同款调用)
import json
expert_config = json.load(open("results/expert_configs/intent.json"))
model, tokenizer = load_base_model("deepseek-ai/ESFT-vanilla-lite")
to_esft(model, expert_config)   # ①默认全模型 parameter→buffer(全冻)
                                # ②逐 MoE 层:选中专家 buffer→parameter(解冻)
                                #   shared_experts/non_expert_modules 按配置开关
# 之后正常接 Trainer 训练;保存的 checkpoint 只含可训部分 = 天然 adapter

# 推理侧:基座 + adapter 合体(eval_multigpu.py L35 同款)
model, tokenizer = load_base_model("deepseek-ai/ESFT-vanilla-lite")
model = add_adapter(model, "all_models/adapters/token/translation")
# add_adapter 内部:读 expert_cfg.json + adapter 权重,校验「配置里的专家集合
# 与 adapter 文件里实际出现的专家 ID」一致(不一致直接 raise),再 to_esft + load

# 一步到位的加载器(基座 + adapter 目录):
model, tokenizer = load_esft_model("deepseek-ai/ESFT-vanilla-lite", adapter_dir)
```

`to_esft` 逐层扫描 `model.model.layers`,只处理类型为 `DeepseekV2MoE` 的 mlp(注意:**第一层不是 MoE 就跳过**,与打分脚本里 `N_LAYERS=26  # 27 layers totally, the first layer is not MoE` 一致),对每层 64 个专家查 `expert_config['experts'][str(idx)]` 决定升 parameter 还是压 buffer。

### 2.3 存储红利:adapter 即「被训专家的权重包」

因为 state_dict 里只有 parameter,**训练产物自动就是 adapter**(每任务一份,只含被训专家 + 可选共享专家/非专家模块的权重)。N 个任务共享一个基座、各挂各的 adapter——这就是「用更少资源和存储高效定制」的落点。`add_adapter` 的一致性校验(配置 vs 权重文件里的专家 ID 集合,`_dict_equal` 不符就 raise)防止「拿 A 任务的配置去挂 B 任务的 adapter」这种静默错误。

下图把「训练产物 → adapter 目录 → 与基座组装 → 一致性校验」串成一条数据流,注意校验发生在注入之前:

<div class="archify-figure">
<iframe src="../../../diagrams/dm-esft-adapter-storage.dataflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="adapter 的存储红利与加载一致性校验"></iframe>
</div>

---

## 3. 训练配置(configs/base.yaml,核自本地)

单文件配置,要点(默认值即论文实验设置,基座 ESFT-vanilla-lite):

```yaml
seq_length: 4096
per_device_batch_size: 1        # n_device: 8 卡
steps: 500
learning_rate: 0.00001          # 1e-5
optim: adamw_torch_fused        # beta (0.9, 0.95),weight_decay 0.1
gradient_checkpointing: true
gradient_accumulation_steps: 4
ep_size: 2                      # train_ep.py 的专家并行度
```

注意 `ep_size` 只对 `train_ep.py`(torchrun 专家并行版)生效;`train.py` 单进程版不用。ep_size>1 且只训专家时,train_ep.py 会按被训专家所在的最小层号切分布局(源码 L115–116)。

---

## 4. 与其它路线的关系(定位一张表)

下图把五条路线排成五行,每行读作「动的是什么 → 什么时候用 → 与 ESFT 的关系」。**ESFT 与 LoRA 正交可组合**;最容易出错的是把它和 EPLB 混为一谈(一个训练时挑专家,一个推理时摊负载):

<div class="archify-figure">
<iframe src="../../../diagrams/dm-esft-route-position.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="五条微调路线各自的定位,ESFT 站在哪一格"></iframe>
</div>

| 路线 | 训什么 | 适合 | 与 ESFT 关系 |
|---|---|---|---|
| 全量微调 | 所有参数 | 数据多、不差算力/存储 | ESFT 的省参数对照 |
| LoRA/adapter(通用 PEFT) | 低秩旁路矩阵 | dense 模型为主 | **正交可组合**:ESFT 选「哪些专家」,adapter 机制存「训了什么」;ESFT 是 PEFT 的 MoE 特化版 |
| ESFT | 任务相关专家本体 | **MoE 模型**领域定制 | — |
| R1 式 RL(见 [deepseek-r1-reasoning.md](deepseek-r1-reasoning.md)) | 全模型策略(GRPO) | 推理能力激励 | 不同目标:ESFT 管「领域知识注入省参数」,R1 管「推理行为涌现」 |
| EPLB/LPLB 负载均衡(infra 域) | 运行时专家**复制/重排** | 部署摊负载 | **别混淆**:ESFT 是训练时挑专家训,EPLB/LPLB 是推理时摊负载;见 infra 篇 §4 |

---

## 5. 局限

- **打分依赖评估集代表性**:专家分数在 `--eval_dataset` 上统计,若线上任务分布和评估集差得远,选出的专家会偏;`n_sample_tokens`(默认 131072)不够大时统计噪声也大;
- **top_p 是激进程度旋钮**:调太小 → 该训的专家没解冻、学不动;调太大 → 逼近全量微调、失去省的意义;需按任务扫;
- **基座耦合**:打分/训练脚本围绕 deepseek-ai/ESFT-vanilla-lite(26 层 MoE、64 专家、top-6、首层非 MoE,`DeepseekV2MoE` 类型硬编码在 to_esft 里)设计,换基座要核对这些结构假设;
- **只管微调,不管推理加速**:被训专家没有结构化稀疏可利用,推理仍是标准 MoE 前向。

---

## 6. 互链

- **操作流程/脚本/实践坑**(三阶段命令、download_adapters.sh、eval_multigpu.py 用法):[deepseek-infra/concepts/esft.md](../../deepseek-infra/concepts/esft.md)——本篇 §2 讲机制,那篇讲上手,分工明确。
- **MoE 基座谱系**(细粒度专家 + 共享专家,ESFT 思想的前提):[deepseek-moe.md](deepseek-moe.md);结构源码级机制见 [model-architecture/concepts/deepseek-arch-v1-moe-v2.md](../../model-architecture/concepts/deepseek-arch-v1-moe-v2.md)。
- **R1 的训练策略**(DeepSeek 模型定制的另一条路:RL 而非专家微调):[deepseek-r1-reasoning.md](deepseek-r1-reasoning.md)。
- **负载均衡对照**(部署侧「动专家」的另一件事):[deepseek-infra/concepts/lplb-epb-load-balancing.md](../../deepseek-infra/concepts/lplb-epb-load-balancing.md)。
- **推理落地**(微调产物的服务):[../../vllm/index.md](../../vllm/index.md)。

## 溯源

- 仓库(本地全文核对:esft.py 164 行、scripts/expert/generate_expert_config.py、scripts/expert/get_expert_scores.py、configs/base.yaml、README.md):https://github.com/deepseek-ai/ESFT
- 论文(arXiv:2407.01906,EMNLP 2024 Main,2024-09-20 接收):https://arxiv.org/abs/2407.01906
- 基座模型:https://huggingface.co/deepseek-ai/ESFT-vanilla-lite
