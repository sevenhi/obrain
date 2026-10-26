---
title: 结构化剪枝 + 知识蒸馏:Minitron / Puzzletron(一个大模型剪出一族小模型)
type: concept
tags: [pruning, structured-pruning, knowledge-distillation, minitron, puzzletron, nas, mip, megatron, modelopt, compression, mamba, moe]
source:
  - "pdf: Minitron(Nemotron-4 15B→8B/4B,激活幅度打分 + 蒸馏恢复)| arXiv:2407.14679 (Muralidharan/Sreenivas et al., NVIDIA;NeurIPS 2024, v2 2024-11;↑primary)"
  - "pdf: LLM Pruning and Distillation in Practice: The Minitron Approach(Llama-3.1-8B→4B、Mistral-NeMo-12B→8B,teacher correction)| arXiv:2408.11796 (Sreenivas et al., NVIDIA;v4 2024-12-09)"
  - "pdf: Puzzle: Distillation-Based NAS for Inference-Optimized LLMs(异构逐层架构 + 块级局部蒸馏 + MIP)| arXiv:2411.19146 (Bercovich et al., NVIDIA;v5 2025-06-03;Puzzletron 是其工程化实现)"
  - "pdf: Minitron-SSM(Group-aware SSM 剪枝,Nemotron-H 8B→4B)| arXiv:2504.11409"
  - "pdf: NVIDIA Nemotron Nano 2(12B→9B,mamba hybrid 剪枝 + 两阶段蒸馏)| arXiv:2508.14444"
  - "code: NVIDIA/Model-Optimizer examples/pruning(README:Minitron/Puzzletron/FastNAS 三种 mode、支持矩阵、剪枝与蒸馏超参建议)| https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/pruning/README.md"
  - "code: NVIDIA/Model-Optimizer examples/puzzletron(AnyModel、MIP sweep、runtime 延迟约束)| https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/puzzletron/README.md"
  - "code: NVIDIA/Model-Optimizer PR #2922(2026-10-23:Minitron/Puzzletron 兼容 Megatron-Core 26.10,Mamba hybrid 状态一致性三修)| https://github.com/NVIDIA/Model-Optimizer/pull/2922"
  - "web: Megatron-Bridge 剪枝文档(Minitron 在 M-Bridge 上的调用方式、默认约束)| https://docs.nvidia.com/nemo/megatron-bridge/latest/modelopt/pruning.html"
  - "web: ModelOpt Pruning 指南(三种 mode 与统一 API mtp.prune)| https://nvidia.github.io/Model-Optimizer/guides/3_pruning.html"
date: 2026-10
---

# 结构化剪枝 + 知识蒸馏:Minitron / Puzzletron

## TL;DR

「一个模型族」的标准做法是每个尺寸各训一遍,代价是 N 倍预训练算力。Minitron 给出的替代路径是:**拿最大的那个已经训好的模型,用 1024 条样本的纯前向激活幅度打分 → 按轴切掉最不重要的参数(结构化剪枝,权重矩阵直接变窄/变浅)→ 用 KL 蒸馏在 <3% 原始 token 量上恢复精度**。Nemotron-4 15B 剪出 8B/4B,8B 只花 **94B token**(Nemotron-3 8B 用了 3.8T,即 **40× 更少**),MMLU 反而比从头训高最多 16%;四个尺寸族的总算力省 **1.8×**。

Puzzletron(Puzzle 论文的工程化)把「同构剪枝(每层一样宽)」升级成**异构剪枝**:每层 FFN 可以不同宽、attention 层可以被整层删掉,用**块级局部蒸馏(blockwise local distillation)**并行探索候选 + **混合整数规划(MIP)** 在参数量/显存/延迟约束下选最优组合。成果是 Llama-3.1-8B 显存 113→96 GiB 只掉 <1% top-10 token 准确率;Llama-70B → Nemotron-51B 单卡 H100 跑出 **2.17× 吞吐**、保留 **98.4%** 基准精度,蒸馏只用了 ≤45B token。

**一句话心法**:压缩预算不要均匀砍,要**按激活/层级的敏感度花在杠杆点上**;而「剪」本身不值钱,值钱的是**剪完立刻用 teacher 的 logits 补回来**——这套路线里蒸馏不是可选收尾,是精度恢复的主体。

---

## 1. 为什么值得做:一组能自己验算的数字

> 下图把 Minitron 主线画成流程(打分发分 → 结构化剪 → 蒸馏恢复),回答「一个大模型怎么剪出一族小模型、为什么蒸馏是精度恢复的主体」。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-minitron-pipeline.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="Minitron 主线流程"></iframe>
</div>



同一个 4B 尺寸,三种造法在 iso-compute(等算力)下的对比(Minitron 论文 Table 11):

| 造法 | 训练 token | HellaSwag | MMLU |
|---|---|---|---|
| 随机初始化从头训 | 150B* | 46.22 | 24.36 |
| 随机初始化从头训 | 400B | 48.23 | 26.24 |
| 15B 剪到 4B,**常规训练**(真标签) | 150B* | 50.85 | 24.57 |
| 15B 剪到 4B,**15B 当 teacher 蒸馏** | 100B* | 51.04 | **37.81** |
| Minitron 8B 剪到 4B(迭代),**15B 蒸馏** | 100B* | 52.04 | **42.45** |

(*标 iso-compute;100B 蒸馏 token 与 150B 常规训练算力相当,因为蒸馏多了一次 teacher 前向。)

三条结论都能从这张表直接读出来:

1. **只剪不蒸馏几乎没用**:同样是 15B→4B,常规训练 24.57 与随机初始化 24.36 持平——剪枝毁掉的能力不会自己长回来。
2. **蒸馏是主体**:同算力下 MMLU 24.36 → 37.81(+13.5 个点),这是整条路线成立的全部理由。
3. **迭代剪优于一次性剪**:先 15B→8B 再 8B→4B,MMLU 再加 4.6 点(37.81→42.45)。

成本账(论文 §4.1 自述口径,可复算):三模型族的每步 FLOP 正比于参数数,从头训 `(4.4+2.5+1.2)e17 × steps`;用剪枝则是 `(4.4 + 2.5/40 + 1.2/40)e17 × steps`,比值 ≈ (4.4+2.5+1.2)/(4.4+0.0625+0.03) = 8.1/4.49 ≈ **1.8×**——与论文宣称的 1.8× 一致。

## 2. 机制一:重要性怎么算(纯前向,不用梯度)

> 下图把三宽度轴打分式与深度轴候选指标并排,回答「不用梯度怎么给参数定重要性、宽度轴与深度轴分别用什么指标」。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-minitron-importance-scoring.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="重要性打分(宽度 vs 深度)"></iframe>
</div>



### 2.1 为什么不用梯度/Taylor

LLM 规模下梯度信息(mask 学习 + 拉格朗日)需要额外反向、额外显存、还要大量数据把 mask 训出来。Minitron 的选择是**只用前向的激活幅度**,一次校准跑完四个轴(深度/神经元/头/embedding channel)的打分。校准集 1024 条样本,8B 模型打分约 5 分钟——这个量级决定了「先搜架构再逐个蒸馏」在工程上可行。

### 2.2 三个宽度轴的打分式

对 MHA / MLP / LayerNorm 的激活取范数,再沿 batch 与序列维聚合:

```
F_head(i)   = Σ_{B,S} ‖ Attn(X·W_Q^i, X·W_K^i, X·W_V^i) ‖₂
F_neuron(i) = Σ_{B,S}  X · (W₁^i)ᵀ          # W₁ 的第 i 行 = 第 i 个 FFN 神经元
F_emb(i)    = Σ_{B,S}  LN(X)_i              # 第 i 个 embedding channel
```

聚合函数**必须选对**,这是论文里最容易被忽视的细节。它比较了三种聚合(mean-abs / L2 / variance)在 batch 维与 seq 维上的 9 种组合,结论:**(batch = L2, seq = mean)** 最优;而 (batch = L2, seq = L2) 明显更差。更关键的是,**重训 400 步(~1.8B token)后这个相对排序不变**,所以可以放心用「重训前」的指标做架构比较——这是个很实用的工程性质:意味着你不必为每个候选都跑完整蒸馏才能排序。

### 2.3 深度轴:三个候选指标,选哪个有讲究

- **PPL**:删掉单层后测困惑度增量(最直觉,但要逐层各跑一次前向)。
- **BI(Block Importance)**:1 − 层输入/输出向量的余弦相似度,`BI_i = 1 − E[X_i,tᵀX_{i+1,t} / (‖X_i,t‖‖X_{i+1,t}‖)]`,一次前向可算全部层,快得多,且可扩展到连续多层的块。
- **下游任务准确率**(2408.11796 的经验):PPL 选出的层在**下游任务**上不是最准的;实测用 Winogrande 选连续层块效果更好。且**删连续层块优于删散层**(与 ShortGPT/LaCo 一路的观察一致)。

工程含义:深度剪枝的目标函数最好直接挂你要保的那个指标,别默认用 PPL 代理。

### 2.4 交叉验证得到的排序结论(可以拿来当先验)

固定到同一个参数量(≈8B)对比四种剪法(Table 10 / Table 1),重训 1.8B token 后:

| 策略 | 参数 | LM val loss |
|---|---|---|
| 深度 PPL | 9.39B | 2.155 |
| 深度 BI | 9.39B | 2.177 |
| **宽度(neuron+head+emb)** | **7.74B** | **2.049** |
| 深度 + 宽度 | 7.91B | 2.062 |

注意**重训前的排序是反的**:Table 1 里「四轴全剪」重训前 KL 5.35 优于「不剪深度」6.33,但重训 400 步后 val loss 变成 2.062 vs 2.049(宽度-only 反超)。论文明确指出翻转发生在**约 200 步(0.8B token)** 左右。这是个非常实操的坑:**用「剪完立刻的 loss」挑架构会系统性地偏向深度剪枝**;要等少量重训之后再比。

## 3. 机制二:剪的时候两个非直觉动作

### 3.1 被剪 heads 的残差要回灌

剪 attention 头时,把被删头的输出**加回**保留下来的头(MHA 版的 Layer Collapse):

```
head_i ← head_i + (head_i − head_{2K−i+1}),  i ∈ [K−(L−K), K]
```

L 个头剪到 K 个时,后半段保留的头各自「吸收」一个被删头的差分信息。GQA 模型上**只对 query heads 做**,不动 KV 头(否则破坏分组语义)。这个技巧是白送的精度,不实现就是纯亏。

### 3.2 迭代式重要性估计(iterative importance)其实没有收益

直觉上「剪一点 → 重算重要性 → 再剪」更准。论文做了 T=1/2/4 的对照(15B 的 embedding 维一次压到 4096):**重训前迭代确实更好,重训 1.8B token 后三者收敛到同一 loss**。所以别为它付额外复杂度——单次打分即可。

> 这与「迭代**剪枝+蒸馏**」不是一回事:后者(先 30% 再 30%)在 >50% 压缩率时是必须的(见 §6),差别在于**中间插入的是重训而不是重打分**。

## 4. 机制三:蒸馏损失的构成与取舍

> 下图把 L = L_CLM + L_logits + α·L_is 的三项与实测取舍画出来,回答「哪一项该留、哪一项该省、α 该取常数还是动态」。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-minitron-distill-loss.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="蒸馏损失的构成与取舍"></iframe>
</div>



总损失:

```
L = L_CLM + L_logits + α · L_is          (α 动态取 L_logits / L_is)
L_logits = (1/l) Σ_k Loss(p_teacher^k(x,τ), p_student^k(x,τ))
L_is     = (1/l) Σ_{k∈H} Σ_i Loss_k(h_teacher^{k,i}, h_student^{k,i})
```

其中 `L_is` 是选定中间隐状态(post-LayerNorm)的匹配损失,学生/教师维度不一致时学一个共享线性变换升维。

论文的实测取舍(每条都能省一半实现工作量):

| 决策点 | 结论 |
|---|---|
| logits 损失函数 | **KLD 最好**;MSE / cosine / reverse-KLD 都更差(注意 R-KLD 在 SFT 场景常被推荐,但在**剪枝后 base 模型**训练里不是最优) |
| 要不要 `L_CLM` | **只用 `L_logits` 最好**;再加真标签交叉熵反而降分 |
| 要不要 `L_is` | **深度没大幅剪时无收益**(Minitron 8B/4B 都属此列);省掉 |
| α 取常数还是动态 | 动态(`L_logits/L_is`)明显好于常数,因为两项量级差很大 |

### 4.1 teacher correction:拿不到原始训练数据时的关键一步

2408.11796 的场景才是真实工程常态(Llama-3.1 8B 的 15T 数据不可得)。若直接用原模型当 teacher、换自己的数据集蒸馏,学生的上限被压住——论文的假设是**子词分布漂移**导致 teacher 在蒸馏集上给的是次优指导。

解法:**先用约 100B token 对自己的蒸馏集轻量微调 teacher**(LR 取原训练峰值 LR 的 **1/5**、120 步 warmup、batch 与衰减 schedule 照旧),再以它为师。实测这一步显著抬高学生精度;teacher 自身下游指标有升有降(数据集 ⇒ 分布迁移的正常代价)。也试过「teacher correction 与学生蒸馏并行」,与「先修正完再蒸馏」打平。

## 5. Puzzletron:从「同构剪枝」到「逐层异构 + MIP」

> 下图把 Puzzletron 的三板斧(逐层异构搜索 + 块级局部蒸馏 + MIP 全局选优)串成闭环,回答「组合爆炸的异构搜索为什么可行」。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-puzzletron-hetero-mip.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="Puzzletron:异构 + BLD + MIP"></iframe>
</div>



### 5.1 异构为什么值得

同构剪枝(所有层一样宽)是 Minitron 的默认形态,实现简单(权重直接切片,存 HF 格式即可)。但各层的重要性本来不同:有的层 FFN 可以砍到 1/4,有的 attention 层整个删掉几乎无损。Puzzle/Puzzletron 的搜索维度:

- `ffn_intermediate_size`:**逐层**不同;
- `attention op / no_op`:**整层删除** attention(只留 FFN,退化成逐 token MLP 混合);
- 另有 KV heads(GQA 分组)、`hidden_size`、MoE `num_experts` 的逐层选择。

约束用**混合整数规划**表达:目标(参数量 / 显存 / 延迟)作为硬约束,最大化「保留精度」的加和,一次求解得到全局最优组合。这是它与「逐轴贪心」的本质区别:逐层维度之间是耦合的(删了 attention 层,FFN 的重要性会变),MIP 才处理得了这种耦合。

### 5.2 blockwise local distillation(BLD)是让它可行的关键

异构搜索空间是组合爆炸,不可能每个候选都全模型蒸馏。Puzzle 的做法是**块级局部蒸馏**:把模型切成块,只对被替换的块单独做局部蒸馏与打分,块与块之间可并行。这样「候选评估」的代价从「训一个模型」降到「训一个块」。

### 5.3 可复算的实测数字

Llama-3.1-8B → 两个约束点(ModelOpt examples/puzzletron README 的 2025-11 运行日志):

| 约束 | 解出的架构 | `stats.memory_mib` | top-10 token acc | 相对 teacher |
|---|---|---|---|---|
| teacher | 全 32 层 attention | — | 0.9424 | — |
| `target_memory=96_000` | 32 层 attention 全留,FFN 不动 | 94 708 | 0.9337 | **−0.9%**(0.9337/0.9424) |
| `target_memory=78_000` | 删 12 层 attention(block_17-28,留 20/32) | 75 796 | 0.8988 | **−4.6%**(0.8988/0.9424) |

两个不变量可以自检:① `num_kv_heads = has_attention × 8`——20×8=160 ✓、26×8=208 ✓(日志里 `gqa_4` = 32 个 Q 头、4 头一组 ⇒ 8 个 KV 头),说明 KV 头是随 attention 层一起删的;② 解出的 `stats.memory_mib`(94 708 / 75 796)都低于对应约束(96 000 / 78 000),即规划值比实测略保守。
README 正文另有一句口径不同的描述:「显存 113 GiB → 96 GiB,15% 下降,top-10 准确率退化 <1%」——它与日志里的 `stats.memory_mib`(92.5 GiB)**不是同一口径**(前者大概是端到端占用,后者是建模的权重+cache+激活),引用时别混。

Puzzle 论文的大模型成果:**Llama-3.1-Nemotron-51B** 与 **Llama-3.3-Nemotron-49B**,从 Llama-70B-Instruct 导出,单张 H100 可跑、大 batch 下 **2.17× 吞吐**,保留 **98.4%** 基准精度,蒸馏用量 **≤45B token**(对比原模型 15T)。这条数字的意义在于:**「单卡能部署」这件事的瓶颈可以靠结构搜索解决,而不只是靠量化**。

## 6. 工程落地:API、支持矩阵、超参

### 6.1 统一入口

三种 mode 都走 `mtp.prune`,区别只在 `mode` 与约束形式:

```python
import modelopt.torch.prune as mtp
from modelopt.torch.utils.plugins.mbridge import load_mbridge_model_from_hf
from modelopt.torch.utils.plugins.megatron_calibration import get_megatron_calibration_forward_loop

bridge, provider, model, unwrapped, tok = load_mbridge_model_from_hf(
    hf_model_name_or_path="Qwen/Qwen3-8B",
    provider_overrides={"pipeline_model_parallel_size": 1,
                        "pipeline_dtype": torch.bfloat16, "seq_length": 4096},
)
forward_loop = get_megatron_calibration_forward_loop(
    tok, dataset_name="nemotron-post-training-dataset-v2", num_samples=1024, seq_length=4096)

# ① 手动定架构(知道自己要什么尺寸)
constraints = {"export_config": {"num_layers": 32, "hidden_size": 3584, "ffn_hidden_size": 10240}}
config = {"forward_loop": forward_loop, "checkpoint": "/path/scores/"}   # 打分缓存,换约束不必重跑前向

# ② NAS 自动搜(只说目标参数量 + 打分函数)
def score_func(m): return megatron_mmlu(m, tok, fraction=0.1, batch_size=4)
constraints = {"params": 6e9}
config = {"forward_loop": forward_loop, "checkpoint": "/path/scores/",
          "score_func": score_func, "max_width_pruning": 0.4, "max_depth_pruning": 0.2,
          "hparams_to_skip": [], "top_k": 10}

mtp.prune(unwrapped, mode="mcore_minitron", constraints=constraints,
          dummy_input=None, config=config)     # 原地剪枝;Minitron 是同构的,存盘用常规 save
```

搜索空间的粒度是**硬约束**,不是审美:各轴取值必须能被 `hidden_size_divisor=256`、`ffn_hidden_size_divisor=512`、`mamba_head_dim_divisor=8`、`num_moe_experts_divisor=8`、`num_layers_divisor=2` 整除,否则下游 tensor 操作会因形状不友好而报错(或掉性能)。Puzzletron 的 `intermediate_size_list` 同样要求**能被 256 整除**。

### 6.2 支持矩阵里那些「剪不了」的坑

| 轴 | 不能剪的情形 |
|---|---|
| `num_attention_heads` | **Gated DeltaNet(线性注意力)、gated attention(`attention_output_gate`,如 Qwen3.5)、MLA(如 DeepSeek)** 都不支持,这些模型只能剪 `hidden_size` |
| `hidden_size` | **VLM 不剪**——它与 vision projector 共享;只剪 language model 部分,视觉塔原封不动 |
| MTP 头 | Qwen3.5 等的多 token 预测头**当前直接丢弃**(剪枝 run 里不参与),自回归推理不受影响;要做投机解码需在剪后单独做 MTP SFT |
| grouped hybrid 层 | 分组 shortcut-MoE 块拒绝整层删除(见 §7) |

这条和库内结构知识直接挂钩:MLA/GDN 这些「新注意力」在压缩工具链里是**二等公民**,不是所有轴都能动。

### 6.3 超参默认值(照抄官方指南即可,别自己发明)

| 项 | 建议 |
|---|---|
| 压缩率上限 | ≤1/3 参数删减一般能站在 Pareto 前沿上;>50% 很难恢复,必须**迭代**(先 30% + 蒸馏,再压第二轮) |
| 蒸馏 token | **80-100B** 最佳(质量好的数据集) |
| seq length | 8192(数据更短则 4096) |
| 全局 batch | 与原训练一致;不确定就 768 |
| LR | 30-50% 压缩:**1e-4 → 1e-5 线性衰减**;压得越狠 LR 越高;模型越大 LR 越低。经验法则:取原训练峰值 LR 的 **1/5**(约 50% 压缩时) |
| warmup | 100 步 |
| 数据配比 | 标准模型 100% 预训练数据;推理模型 70% 推理 + 30% 预训练 |
| 蒸馏后 | 仍要 post-training(SFT/偏好对齐),否则 reasoning 能力不回来 |
| 轴敏感度 | `ffn_hidden_size` 通常能比 `hidden_size` 与 attention/Mamba 维更激进 |

一个有说服力的官方实例:Qwen3-8B(36 层,ffn 12288,hidden 4096)→ 6B(24 层,ffn 9216,hidden 3584);Llama-3.1-8B → 4.5B(16 层);Nemotron-H-8B(ffn 21504,hidden 4096,mamba 128 头)→ 4B(ffn 12288,hidden 3072,mamba 112 头)。

## 7. 2026-10-23 上游变更:Mamba hybrid 剪枝的三处「状态一致性」坑

> 下图把三处状态一致性坑(派生缓存 / 层列表索引 / 逐层 config 副本)与附带 dataclass 小坑归成树,回答「剪完结构之后还要同步哪些对齐状态」。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-minitron-mamba-state-consistency.tree.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="Mamba hybrid 剪枝状态一致性"></iframe>
</div>



ModelOpt PR #2922 把 Minitron(Mamba hybrid)与 Puzzletron 修到兼容 Megatron-Core / Megatron-Bridge 26.10(NeMo 26.10 容器带这些改动;在 26.08 上每个修复都是 no-op)。它改的三个文件恰好是**「剪枝 = 改结构」这件事最典型的三种翻车方式**,对任何自己写结构裁剪的人都适用:

**(i) 派生缓存不跟着被剪轴走 → 形状不匹配。** 新版 `MambaMixer` 维护一份 per-head 的 `-exp(A_log)` 解码缓存(`_A_neg_exp_cache`),并在**每次 `eval()`** 时刷新。剪枝把 active heads 缩到 6 个头之后,刷新时旧缓存仍是 8 个头:`The size of tensor a (8) must match the size of tensor b (6)`。修法是在 `train()` 与 `export()` 之后按 `A_log.shape` 重新分配缓存并置 `_A_neg_exp_cache_stale = True`,让 MCore 自己重算。

> 通用规则:**剪枝后所有「由被剪维度派生的缓存 / 统计量 / scale」都失效了**,必须显式重建或标脏,别指望下游发现。这与量化里「换 active heads 必须重算 scale」是同一类错误。

**(ii) 与层列表索引对齐的元数据没跟着切 → 一路报错或静默错。** 新版 `HybridStack` 保留 `layer_config_list`、`num_layers_per_pipeline_rank`、`_execution_layer_indices`、`_execution_layer_config_list`,且 forward 用 **`zip(..., strict=True)`** 与 `layers` 对齐。旧代码只重建了 `decoder.layers`,元数据仍是旧长度 → 直接抛错。修法是记录保留层的**原始索引**,四份元数据一起按同一索引切。
反向的坑也补了:**分组(shortcut-MoE)块**在执行列表与层列表不是 1:1 时无法安全删层,现在**在改动任何东西之前**就用 assert 拒绝(先校验再变更,避免半改状态)。

**(iii) 每层各持一份 config 副本 → 只改全局 config 等于没改。** 新版 hybrid 每层持有自己的 config 副本;Megatron-Bridge 的 Mamba `in_proj` 导出读的是**层自己的** config。旧代码只把剪后架构写进 `model.config`,于是导出剪过的 Nemotron-H 时打到 device-side index out of bounds。修法是遍历 `[model.config, *decoder.layer_config_list]` 全部写入。
配套的还有一处:`_temporarily_pruned`(打分候选时临时剪、事后再恢复)现在会**额外保存并还原**上述 stack 元数据——临时剪枝如果只还原 `layers` 不还原元数据,后续 `sample()` 会读到错位的配置。

**(iv) 附带一个 dataclass 层面的小坑**:`provider_bridge` 用 `fields()` 过滤 kwargs 时,新版 Megatron-Bridge 的某些字段是 `init=False`(在 `__post_init__` 里算出来的,如 `gtp_weight_remat_size`),但 `asdict()` 仍会把它们吐出来 → 构造时 `TypeError`。修法是 `{f.name for f in fields(...) if f.init}`。

**这三条合起来是一条**:结构化剪枝的难点不在「切矩阵」,而在**「切完之后,模型里所有跟被切维度对齐的东西是否还对齐」**——缓存、执行列表、逐层 config 副本、流水线切分计数,一个漏了就是形状错或静默错。写这类工具的第一条 checklist 应该是:列出所有 `len()` 依赖 `num_layers` / `num_heads` 的状态,逐个确认跟着变了。

## 8. 实践要点(checklist)

1. **先问「原数据在不在」**:在 → 直接用原训练集蒸馏;不在 → 先做 teacher correction(100B token、LR=原峰值 1/5),否则学生上限被分布漂移压住。
2. **打分只用前向**,1024 条校准样本、聚合取 (batch=L2, seq=mean);打分结果存 checkpoint,换约束不必重跑前向。
3. **比架构要等重训**:剪完立刻的 loss 排序会系统性偏向深度剪枝,约 200-800 步(0.8-1.8B token)后宽度反超。
4. **头剪完做残差回灌**(GQA 只对 query heads),白拿精度。
5. **蒸馏只留 `L_logits`(KLD)+ 动态权重**,除非大幅剪深度才加中间隐状态损失;别再加 `L_CLM`。
6. **压缩 >50% 必须迭代**(30% → 蒸馏 → 再 30%),一次性砍超过一半很难恢复;两个数据点:15B→8B 是 47%、8B→4B 是 49%(按论文 Table 5 的 8.27B/4.19B 重算),而 15.6B 一次砍到 4.19B 是 **73.1%**(论文正文写 73.3%,口径差在 15.6 vs 15.0 的基数),论文明确说这一步一次性做会明显吃亏——实测迭代比一次剪 MMLU 高 12%。
7. **维度取整先定好**(256/512/8/2 的除数),否则搜索出来的架构下游不一定跑得动;Puzzletron 的 FFN 候选同样要 256 对齐。
8. **部署侧先确认异构架构能加载**:Puzzletron 的解是 AnyModel 格式,vLLM 需要带 AnyModel 支持的分支(vLLM#36512);异构模型不是「存成 HF 就能跑」。
9. **剪枝工具链对 MLA / GatedDeltaNet / gated attention 只能剪 `hidden_size`**,`num_attention_heads` 那条路走不通;VLM 不剪 `hidden_size`(与 projector 共享)。
10. **改结构就盘点派生状态**:`num_layers` 一变,执行列表、逐层 config、缓存形状、pipeline 分层计数全要同步(见 §7)。
11. **量化可以叠加在剪枝之后**:官方端到端教程是「剪枝 → 蒸馏 → 量化 → vLLM 部署」,Nemotron-3-Nano-30B→22B 配合 FP8 拿到 2.6× 吞吐 + 2.6× 显存下降;剪枝降参数、量化降位宽,是两个正交杠杆。

## 9. 与库内其他篇的关系

- **量化 vs 剪枝**:本库 [ptq-gptq-awq-smoothquant](ptq-gptq-awq-smoothquant.md) / [two-bit-weight-frontier](two-bit-weight-frontier.md) 解决「同一个结构、更少比特」;本篇解决「更小的结构」。前者不动 shape、后者动 shape,所以剪枝后的模型需要重新做量化校准(scale 全变)。
- **低秩路线**:[lstm-lowrank-compression](lstm-lowrank-compression.md) 的「核范数是可压缩性先验」与本篇「激活幅度是重要性先验」是同一类思想的两种度量;都是「先估计再动手」。
- **结构侧依赖**:Mamba/MoE 的剪枝轴对应 [hybrid-linear-attention](../../model-architecture/concepts/hybrid-linear-attention.md)(mamba heads / head_dim、hybrid 层配比)与 [open-model-landscape](../../model-architecture/concepts/open-model-landscape.md)(MoE experts / shared expert);为什么 MLA 与 GDN 剪不了头,机制原因在 [attention-heads-mha-mqa-gqa](../../model-architecture/concepts/attention-heads-mha-mqa-gqa.md)(KV 头分组)与 [deepseek-arch-v1-moe-v2](../../model-architecture/concepts/deepseek-arch-v1-moe-v2.md)(MLA 低秩压缩)。
- **部署侧**:剪枝产物的 engine 落地与量化导出见 [onnx-tensorrt-deployment](../../computer-vision/frameworks/onnx-tensorrt-deployment.md)(ModelOpt 是同一套工具链)与 [vllm/](../../vllm/index.md)。

## 溯源

- Minitron 主论文(15B→8B/4B、重要性打分、蒸馏配方、10 条 best practice):arXiv:2407.14679,NeurIPS 2024;本文 §1 数字取自其 Table 10/11,§2/§4 的公式与结论取自 §2.2-2.3、§4.2-4.3。
- 实战篇(Llama-3.1-8B→4B、Mistral-NeMo-12B→8B、teacher correction):arXiv:2408.11796 v4。
- 异构 + MIP + BLD:Puzzle,arXiv:2411.19146 v5;工程实现与实测日志见 ModelOpt `examples/puzzletron/README.md`(§5.3 表格数据出处,2025-11 日志快照)。
- 支持矩阵 / 超参默认值 / 搜索空间除数:ModelOpt `examples/pruning/README.md` 与 Megatron-Bridge 剪枝文档(2026-10-23 抓取)。
- Mamba hybrid 剪枝的版本坑:PR #2922(2026-10-23 合并)的 diff 与 PR 描述;PR 自带测试数字(nemo:26.08 + 新版 MCore:修复前 `tests/examples/megatron_bridge` 98 通过 2 失败、`tests/gpu_megatron` 448 通过 4 失败;修复后 pruned/NAS 62 全通过)——**数字照抄 PR,未独立复现**。
- 外部链接索引见 [../references.md](../references.md)。
