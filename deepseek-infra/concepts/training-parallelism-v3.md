---
title: DeepSeek-V3 训练并行编排 — PP16 × EP64 × ZeRO-1(DualPipe + DeepEP 把 671B 训出来)
type: concept
tags: [training, parallelism, pipeline-parallelism, expert-parallelism, data-parallelism, zero-1, dualpipe, deepep, moe, deepseek, v3, fp8, memory, hai-llm]
source:
  - "pdf: DeepSeek-V3 Technical Report(arXiv:2412.19437)| https://arxiv.org/abs/2412.19437 ↑primary"
  - "code: deepseek-ai/DeepSeek-V3 | https://github.com/deepseek-ai/DeepSeek-V3"
  - "code: deepseek-ai/DualPipe | https://github.com/deepseek-ai/DualPipe"
  - "code: deepseek-ai/DeepEP | https://github.com/deepseek-ai/DeepEP"
date: 2026-10
---

# DeepSeek-V3 训练并行编排 — PP16 × EP64 × ZeRO-1(DualPipe + DeepEP 把 671B 训出来)

> **定位**:算法域([model-architecture](../../model-architecture/concepts/deepseek-arch-v3.md))讲「网络长什么样」;**本域讲「怎么把它训出来」**——671B MoE 在 2048×H800 上的并行切分、通信重叠与显存账。DualPipe 的调度细节见 [dualpipe.md](dualpipe.md),DeepEP 的 dispatch/combine 内核见 [deepep-nvidia-vs-ascend.md](deepep-nvidia-vs-ascend.md);**本篇只做编排层总账,不重复单点机制**。

## TL;DR

**DeepSeek-V3 的训练并行编排(论文口径,arXiv:2412.19437 §3.2)是**:**16-way Pipeline Parallelism(PP)× 64-way Expert Parallelism(EP,跨 8 节点)× ZeRO-1 Data Parallelism(DP)**,跑在自研 **HAI-LLM** 框架上,**训练全程不用 Tensor Parallelism(TP)**。

一句话记法:**层切给 PP、专家切给 EP、batch 与优化器切给 ZeRO-1 DP;气泡与 all-to-all 用 DualPipe + DeepEP 藏进计算;显存用「重计算 + FP8 存储 + 大 EP 分摊」硬省,省到不需要 TP**。

| 维度 | V3 训练选择 | 为什么 |
|---|---|---|
| 流水线 | **PP16** + **DualPipe** 双向调度 | 跨节点 EP 让计算:通信 ≈ **1:1**,必须重叠;DualPipe 把 a2a 与 PP 通信全藏进计算,气泡约为 1F1B 的一半 |
| 专家 | **EP64,跨 8 节点**(NVLink 节点内 + IB 跨节点) | 256 路由专家细粒度切开;用 EP 而非 TP 切专家,避免 TP 的权重 all-reduce |
| 数据/优化器 | **ZeRO-1 DP** | 只切优化器状态(Adam moments),梯度/参数仍各 rank 全量,通信最轻 |
| 张量并行 | **不用 TP** | 显存已靠重计算 + FP8 + 大 EP 压住;TP 的通信代价「costly」 |
| 通信重叠 | **DeepEP** dispatch/combine + 20 SM 通信通道 | NVLink 160 GB/s vs IB 50 GB/s(≈3.2×),token 最多进 4 节点,IB/NVLink 两级流水重叠 |
| 显存硬省 | 重算 RMSNorm + MLA up-proj、EMA 放 CPU、MTP 共享 emb/out-head | 在不引入 TP 的前提下装下 DualPipe 的 2× 参数 |

**关键结论**:这套编排的本质是「**用 EP 换掉 TP、用 DualPipe 换掉 1F1B、用 ZeRO-1 兜住优化器**」——三者合起来才让 671B(激活 37B)在 14.8T token / 2.788M H800 GPU 小时内稳定训完(全程无不可恢复 loss spike、无回滚)。

---

## 1. 并行原语速览(为 V3 编排做铺垫)

大模型训练的并行原语,按「切什么、通信什么、显存归谁」拆开:

| 原语 | 切什么 | 主要通信 | 显存效果 | V3 是否用 |
|---|---|---|---|---|
| **TP**(Tensor Parallel) | 单层内的权重矩阵(按列/行) | 每层 all-reduce / all-gather(激活) | 参数、激活都按 TP 度数摊薄 | **训练不用**(推理 TP4) |
| **PP**(Pipeline Parallel) | **层维度**(整层归属某 stage) | 相邻 stage 传激活/梯度(send/recv) | 每设备只存 1/PP 的层;激活随 micro-batch 数涨 | **PP16** |
| **DP**(Data Parallel) | **batch 维**(每 rank 一份完整副本) | 梯度 all-reduce 或 reduce-scatter | 参数全量冗余;ZeRO 可切优化器 | **ZeRO-1 DP** |
| **EP**(Expert Parallel) | **MoE 专家维**(expert 分片到 rank) | 每 MoE 层两次 all-to-all(dispatch/combine) | 每设备只存 1/EP 的专家 | **EP64,跨 8 节点** |
| **SP**(Sequence Parallel) | 序列维(配合 TP 切激活/非 GEMM 区) | 与 TP 同步 all-gather/reduce-scatter | 降 TP 组内激活冗余 | 训练未用;推理 TP+SP |
| **CP**(Context Parallel) | 超长序列按 token 切 | 注意力相关 all-to-all/reduce | 支持超长上下文 | **未用**(预训练 seq=4K) |

**几条容易混的账**:

1. **TP 切的是「一层怎么算」,EP 切的是「MoE 里哪几个专家放哪」**。V3 有 256 个细粒度路由专家(inter=2048),天然按 expert 整体切走(EP)更干净;硬上 TP 切 expert 权重会带来每层 all-reduce,与 MoE 的 all-to-all 叠成双重通信。
2. **DP 的通信取决于 ZeRO 阶段**:ZeRO-1 只切优化器状态(moments),反传后仍需全局梯度归约;ZeRO-2 再切梯度、ZeRO-3 连参数也切。V3 选 **ZeRO-1 = 最轻的那档**——因为参数/专家已经被 PP+EP 切得够碎,优化器再切一刀就装下了。
3. **PP 的激活随 micro-batch 驻留数增长**(1F1B 约 1×PP 份),DualPipe 是 PP+1(见 [dualpipe.md](dualpipe.md));参数则随 PP 反比下降。V3 还叠了 DualPipe 的 **2× 参数**代价,靠 **EP64 把每卡专家切得极碎**来消化(论文明确:「two copies of the model parameters … does not significantly increase the memory consumption since we use a large EP size」)。
4. **通信形态**:TP 是每层高频 all-reduce;PP 是相邻 rank 点对点;EP 是每 MoE 层两次不规则 all-to-all;DP/ZeRO 是每 step 一次梯度归约。**V3 的通信大头是 EP 的 all-to-all**,所以整个 infra 篇幅都在解这一件事(DeepEP + DualPipe + node-limited routing)。

---

## 2. V3 的实际编排:PP16 × EP64 × ZeRO-1

### 2.1 硬件与框架底座(论文 §3.1–3.2)

| 项 | 论文口径 |
|---|---|
| 集群 | **2048 × NVIDIA H800** |
| 节点内 | 8 GPU,NVLink + NVSwitch 互联 |
| 节点间 | InfiniBand(IB) |
| 训练框架 | **HAI-LLM**(DeepSeek 自研轻量训练框架) |
| 模型 | **61 层** Transformer,hidden=7168;671B 总参 / 37B 激活 |
| 预训练 | seq=**4K**,batch 从 3072 渐增至 **15360**;共 **14.8T** token |
| 精度 | **FP8 混合精度**(超大规模首次验证) |

### 2.2 三维切分的具体含义

```
                 全局 2048 GPU(H800)
                        │
        ┌───────────────┼───────────────┐
        │ PP = 16 stage(层维)            │
        │  ├─ stage 0:  最浅层 + Embedding │
        │  ├─ stage 1..14: 中间层          │
        │  └─ stage 15: 最深层 + LM Head   │
        │       (DualPipe 把 emb/out-head 放同一 rank,便于 MTP 物理共享)
        │
        ├─ EP = 64(专家维,跨 8 节点 = 64 GPU)
        │   256 routed expert 切到 64 rank
        │   每 rank 约 4 个专家 + 1 共享专家(共享专家通常每 rank 复制/共置)
        │   → DualPipe 的 2× 参数由「大 EP 切碎」吸收
        │
        └─ ZeRO-1 DP(batch 维 + 优化器分片)
            每 step:梯度全局归约;Adam moments 按 DP rank 切片
            参数/梯度本体不在 DP 内切(ZeRO-1 只切优化器状态)
```

- **PP16**:61 层按 stage 切开。论文**未给出每 stage 层数的精确表**;61/16≈3.8,实践上是 ~3–4 层/stage 的**不均分**(此处不作断言,以论文口径为准)。DualPipe 额外约束:**最浅层(含 embedding)与最深层(含 output head)部署在同一 PP rank**,为 MTP 共享 embedding/output head 提供物理同卡条件。
- **EP64 跨 8 节点**:专家并行组 = 8 节点 × 8 GPU。跨节点走 IB、节点内走 NVLink 的**两级 all-to-all**(由 DeepEP 实现,见 §3)。与之配套的是 **node-limited routing(M=4)**:每个 token 最多发往 4 个节点,把 IB 流量压到可重叠范围——这是「计算-通信近完全 overlap」的前提,不是可选优化。
- **ZeRO-1 DP**:batch 维扩开;优化器(AdamW 的一阶/二阶矩)按 DP 分片。论文另有低精度优化器状态:**moments 用 BF16,master weights 与梯度累加仍 FP32**——分片归分片,数值精度归精度,两件事。

### 2.3 为什么不用 TP(训练)

论文原话:「meticulously optimize the memory footprint during training, thereby enabling us to train DeepSeek-V3 **without using costly Tensor Parallelism (TP)**」。拆开看三条理由:

1. **通信形态不匹配**:TP 的每层 all-reduce 会与 EP 的每层 all-to-all 叠在同一关键路径上;而 V3 的跨节点 EP 本身已把计算:通信推到 ≈1:1,再叠 TP 通信只会更糟。
2. **显存已被别处省出来了**:RMSNorm + MLA up-projection 全量重计算、激活 FP8 缓存、EMA 甩到 CPU、MTP 共享 embedding/output-head、DualPipe 的 2× 参数被 EP64 摊薄——不需要再靠 TP 切权重来装下模型。
3. **MoE 专家用 EP 更自然**:256 个独立专家按 expert 整体切走(EP),比把每个 expert 的矩阵再竖着切(TP)更少同步点、更贴合 grouped expert GEMM 的计算形态。

> 对照:V3 **推理**侧拓扑分阶段——prefill 用 EP32(MoE 大批量吞吐优先);decode 用 TP4+SP(注意力)+ EP40(MoE)+ DP 组合(单卡延迟与显存优先)。训练要吞吐与稳定性故砍掉 TP;推理 prefill/decode 两阶段取舍不同,分别选 EP 与 TP+SP。别把两套拓扑混着抄。

### 2.4 MTP 对训练编排的影响

MTP(多 token 预测,D=1,预测额外 1 个 token)不只是损失函数上的改动,它**落在并行拓扑里**:

- 每个 MTP 模块含 **1 个 Transformer block + 投影矩阵**,与主模型同 PP 维度一起切;
- embedding 与 output head **与主模型物理共享**(同参数同梯度)——为此 DualPipe 把最浅层与最深层绑在同一 PP rank(§2.2);
- 训练总损失 = 主损失 + `λ · L_MTP`,λ 前 10T token 取 **0.3**、后 4.8T 取 **0.1**(论文口径);
- **推理时 MTP 模块整体丢弃**,或反手当投机解码头用——训练拓扑为它多背了一个 block 的计算/显存,部署拓扑不必带走。

---

## 3. 通信-计算重叠:DualPipe + DeepEP + SM 划分

> 本节只做编排视角的连接;调度图与气泡公式见 [dualpipe.md](dualpipe.md),dispatch/combine 内核与 EPBuffer 见 [deepep-nvidia-vs-ascend.md](deepep-nvidia-vs-ascend.md)。

### 3.1 问题:跨节点 EP 把计算:通信推到 1:1

论文口径:跨节点专家并行引入的通信开销,使**计算:通信比约为 1:1**——也就是说,不做重叠的话,一半时间在等网络。这是 V3 整个 infra 设计的出发点。

### 3.2 DualPipe 在编排层做了什么

每个 chunk 被拆成四段:**attention → all-to-all dispatch → MLP → all-to-all combine**;反向 chunk 再把 attention/MLP 各拆成 **backward for input / backward for weights**(ZeroBubble 风格)。然后:

- 在**一对前向+反向 chunk** 内重排这四段,并**手工调配通信 SM 与计算 SM 的比例**;
- 全局调度是**双向流水**:micro-batch 从 pipeline 两端同时灌入;
- 结果:**all-to-all 与 PP 通信都可完全隐藏**;且保持计算:通信比不变时,模型再变大也能近乎零 all-to-all 开销(细粒度专家跨节点仍可用)。

对编排账的两个直接影响(详见 [dualpipe.md](dualpipe.md)):

| 项 | 1F1B | DualPipe | V3 如何消化 |
|---|---|---|---|
| 气泡 | (PP−1)(F+B) | (PP/2−1)(F&B+B−3W) ≈ 减半 | 双向调度本身 |
| 参数/设备 | 1× | **2×** | **EP64 把专家切碎**,2× 的绝对量不大 |
| 激活/设备 | PP | PP+1(峰值 +1/PP) | 重计算 + FP8 激活 |

约束:PP 与 micro-batch 数需为**偶数**(比 Chimera 宽松——不要求 micro-batch 整除 PP);且 DualPipe 的气泡/激活**不随 micro-batch 数增长**。

### 3.3 DeepEP:all-to-all 本身要快、还要省 SM

即便重叠,通信内核也得吃满带宽、且不能占掉太多 SM(H800 共 **132 SM**,论文口径分配 **20 SM** 给通信):

| 机制 | 论文口径 |
|---|---|
| 带宽层级 | NVLink **160 GB/s** vs IB **50 GB/s**(≈**3.2×**) |
| 路由限制 | 每 token **最多进 4 个节点**(node-limited routing,M=4),压 IB 流量 |
| 两级转发 | token 先 IB 到目标节点,再**即时** NVLink 转发到目标 expert 所在 GPU,后续 token 不阻塞 |
| 同节点 expert 密度 | 平均每节点 **3.2 个 expert**(M=4 × top-8 的自然结果),NVLink 转发几乎零额外开销 |
| SM 预算 | **20 SM / 132**,warp specialization 切 **10 个通信 channel** |
| dispatch 三段 | IB 发送 / IB→NVLink 转发 / NVLink 接收(各由独立 warp 组负责,warp 数按负载动态调) |
| combine 三段 | NVLink 发送 / NVLink→IB 转发与累加 / IB 接收与累加 |
| 实现细节 | 定制 PTX + 自动调优通信 chunk size,压 L2 干扰 |

这些内核在开源侧对应 **DeepEP**(见 [deepep-nvidia-vs-ascend.md](deepep-nvidia-vs-ascend.md)):dispatch(token→expert rank)/ combine(expert 结果→token 原 rank)、FP8 dispatch、`EventOverlap` 式计算-通信重叠。V2 起 SM 占用从 24 降到 4–6,是同一目标的延续——**通信不能抢计算的 SM**。

### 3.4 时间线视图(一层内)

```
时间 ──────────────────────────────────────────────────────────►
计算流: [attention][  MLP/expert GEMM  ][attention][  MLP ... ]
通信流:      └─a2a dispatch─┘└─a2a combine─┘
                  ↑藏进 attention 尾/MLM 头间隙 ↑
PP 流:   ──send act──►                  ◄──recv grad──  (与另一方向微批的计算重叠)
SM 划分: 计算 SM 跑 GEMM | 通信 SM(20)跑 IB/NVLink 转发| 同卡并发
```

要点:**重叠不是「异步一下就行」**,而是 (a) chunk 内四段重排、(b) 通信/计算 SM 显式配比、(c) node-limited routing 把 IB 流量限到可藏、(d) DeepEP 内核把两级网络吃满——四件事缺一不可。

---

## 4. 显存预算账:参数/梯度/优化器/激活怎么被摊掉

### 4.1 四类占用的分摊路径

| 显存项 | 谁来摊 | V3 具体做法 |
|---|---|---|
| **参数** | PP(层)+ EP(专家) | 每卡只有 1/PP 层 × 其中 1/EP 专家;DualPipe 再 ×2,被大 EP 吸收 |
| **梯度** | 随参数分片(PP/EP 同切) | 反传后按 ZeRO-1 做全局归约;不切梯度本身(ZeRO-1 边界) |
| **优化器状态** | **ZeRO-1 DP 分片** + 低精度 | Adam moments 改 **BF16**(FP32 master weights + FP32 梯度累加保留) |
| **激活** | PP 决定驻留 micro-batch 数;重计算/低精度压单份大小 | 见 4.2;DualPipe 峰值 PP+1 份 |

### 4.2 激活侧的三板斧(论文 §3.2.3 + §3.3)

1. **重计算(Selective Recomputation)**:**所有 RMSNorm + MLA up-projection** 在反传时重算,不持久保存其输出激活——「minor overhead, significantly reduces memory」。注意这不是全层重计算,是**点名这两类算子**(MLA 的 up-projection 激活大、RMSNorm 便宜好算)。
2. **SwiGLU 输入缓存 + 输出重算**:MoE 内 SwiGLU **只缓存输入**(FP8 细粒度量化),反传时重算输出——又省一段大激活。
3. **FP8 激活缓存 + 特例**:Linear 的激活按 FP8 缓存;但 attention 之后那个 Linear 的输入敏感(反传给 attention 用),改用定制 **E5M6** 格式,且 scale 取 2 的整数幂(round scale),1×128 → 128×1 转置时不引入额外量化误差。

### 4.3 训练侧 FP8 的「存什么、算什么、传什么」

| 对象 | 精度 | 备注 |
|---|---|---|
| Fprop / Wgrad GEMM | **FP8** | 细粒度 tile/block 量化 + 高精度累加 |
| 缓存激活 | **FP8**(特例 E5M6) | 见 4.2 |
| MoE up-proj 前激活(dispatch 载荷) | **FP8** | 通信与 Fprop 兼容,scale 为 2 的幂 |
| MoE down-proj 前的激活梯度 | **FP8**(类似策略) | combine 相关 |
| **combine 输出(前向/反向)** | **BF16** | 关键路径保精度 |
| Adam moments | **BF16** | 无明显精度损失 |
| master weights / 梯度累加 | **FP32** | 数值稳定性底线 |

### 4.4 杂项省显存

- **EMA(指数移动平均)放 CPU**:用于学习率衰减后性能预估;CPU 内存异步更新,「不占 GPU 显存、不占额外时间」。
- **No token-dropping**:负载均衡(aux-loss-free + 补充 sequence-wise loss + EPLB)足够好,训练/推理都不丢 token——省掉了 token drop 带来的缓冲与重排显存。

---

## 5. 与 Megatron-LM/3D 并行惯例对照 + 实践 checklist

### 5.1 对照表

| 维度 | Megatron-LM 惯例(3D 并行) | DeepSeek-V3 训练 | 差异根源 |
|---|---|---|---|
| 典型拓扑 | **TP × PP × DP**(TP 常 2/4/8,放节点内) | **PP16 × EP64 × ZeRO-1 DP,无 TP** | MoE 用 EP 切专家,不必 TP 切矩阵 |
| 流水线调度 | 1F1B / Interleaved 1F1B / ZB | **DualPipe**(双向 + 通信重叠) | 跨节点 EP 通信太重,1F1B 藏不住 |
| MoE 并行 | EP(常与 TP 组合)+ token drop / aux loss | **纯 EP64** + aux-loss-free + node-limited routing | 通信与质量都不想付 token-drop 代价 |
| all-to-all | NCCL 集合通信语义 | **DeepEP** 定制 kernel(GPU 发起、IB/NVLink 两级) | 细粒度不规则 a2a,NCCL bulk 语义不匹配 |
| 优化器分片 | 常 ZeRO-1 或无分片 | **ZeRO-1** + moments BF16 | 参数已被 PP/EP 切碎,只需再切优化器 |
| 激活策略 | 全量/选择性重计算 + SP | **点名重算** RMSNorm/MLA-up + SwiGLU 输出 + FP8 激活 | 更激进的「算子级」重计算清单 |
| 序列并行 | SP 常与 TP 捆绑 | 训练未用(seq=4K) | 4K 不需要 CP/SP |
| 精度 | BF16 为主,FP8 渐进 | **FP8 混合精度**全栈(GEMM/激活/通信) | 671B + 14.8T token 的成本压力 |

### 5.2 实践 checklist(想复现 / 调参时先看什么)

**拓扑层**

1. **先定 EP 度数与是否跨节点**:expert 数能否被 EP 整除?EP 组是否跨节点?跨节点就必须有 IB→NVLink 两级 a2a(或等价物),并按 M=4 这类 node-limited routing 控流量。
2. **PP 必须偶数**(DualPipe 公式含 PP/2);micro-batch 数也要偶数;不必整除 PP。
3. **DualPipe 的 emb/out-head 同 rank**:若你也要挂 MTP 并共享 embedding/output head,PP 划分时把最浅+最深绑在同一 stage。
4. **DP/ZeRO 阶段按剩余显存选**:优先 ZeRO-1;只有优化器仍装不下才上 ZeRO-2/3(通信会变重)。
5. **先算「不用 TP」能不能装下**:参数/激活/优化器三笔账都过不了,再考虑 TP;能过就别加 TP(每层 all-reduce 对 MoE 是纯负担)。

**通信层**

6. **测计算:通信比**:若接近 1:1(跨节点细粒度 EP 的典型值),重叠是必需项,不是优化项——排进 DualPipe 式 chunk 重排 + SM 配比。
7. **通信 SM 预算要显式**:留 15–20% SM 给通信(V3:20/132),并用 warp specialization 分 channel;别让 NCCL 默认路径占满 SM。
8. **combine 精度别乱降**:V3 前向/反向 combine 保持 BF16;只有 dispatch 载荷与部分激活梯度敢上 FP8。

**显存层**

9. **重计算清单点名写**:RMSNorm、MLA up-projection、SwiGLU 输出——便宜的算子重算,贵的激活缓存,比「一刀切全重算」划算。
10. **优化器状态先降精度再分片**:moments 用 BF16 能直接省一半,master weights/梯度累加保持 FP32。
11. **EMA / 评估副本放 CPU**,异步更新。

**质量与负载**

12. **负载均衡三件套**:aux-loss-free bias(主)+ 极小 α 的 sequence-wise loss(兜底)+ EPLB 冗余专家(热点);目标是 **no token-dropping**。
13. **MTP 的 λ 调度**:前期重、后期轻(0.3 → 0.1),别全程一个值。
14. **稳定性优先于峰值**:V3 全程无不可恢复 loss spike、无回滚——FP8 累加精度、master FP32、combine BF16 都是为此服务,调参时别为吞吐砍这几处。

---

## 溯源

- **DeepSeek-V3 Technical Report**(主源:§3.1 计算集群、§3.2 训练框架/DualPipe/all-to-all/省显存、§3.3 FP8、§4.2 超参,arXiv:2412.19437):https://arxiv.org/abs/2412.19437
- **DeepSeek-V3 仓库**(配置与推理实现对照):https://github.com/deepseek-ai/DeepSeek-V3
- **DualPipe 仓库**(双向调度、气泡/显存对比表):https://github.com/deepseek-ai/DualPipe
- **DeepEP 仓库**(EP all-to-all dispatch/combine 内核):https://github.com/deepseek-ai/DeepEP
- **profile-data**(V3 训练各算子通信-计算重叠测量):https://github.com/deepseek-ai/profile-data
- **库内交叉**:[dualpipe.md](dualpipe.md)(PP 调度与气泡公式的专篇,本篇不重复)、[deepep-nvidia-vs-ascend.md](deepep-nvidia-vs-ascend.md)(EP 通信内核)、[deepep-ascend.md](deepep-ascend.md)(昇腾侧 EP 后端)、[lplb-epb-load-balancing.md](lplb-epb-load-balancing.md)(EP 负载均衡,与并行拓扑正交但决定 EP 能否跑满)、[../../model-architecture/concepts/deepseek-arch-v3.md](../../model-architecture/concepts/deepseek-arch-v3.md)(网络结构:MLA/DeepSeekMoE/MTP,「算法域讲结构」篇)、[open-infra-index.md](open-infra-index.md)(开源 infra 总账)。

## 相关条目(库内互链)

- **[dualpipe.md](dualpipe.md)** — 本篇 §3 的调度机制专篇:双向流水、气泡公式 `(PP/2−1)(F&B+B−3W)`、参数 2×/激活 PP+1、DualPipeV。编排账在本篇,调度图在彼篇。
- **[deepep-nvidia-vs-ascend.md](deepep-nvidia-vs-ascend.md)** — 本篇 §3.3 的通信内核专篇:dispatch/combine、EPBuffer、SM 占用 24→4–6、NCCL Gin 后端。
- **[deepep-ascend.md](deepep-ascend.md)** — 同一 API 的昇腾后端(HCCL/HCOMM/UBMEM),移植 EP 通信时对照。
- **[lplb-epb-load-balancing.md](lplb-epb-load-balancing.md)** — EPLB 静态冗余专家 / LPLB 动态 LP 分流:决定 EP64 上专家是否跑满,是「拓扑定了之后」的下一层问题。
- **[../../model-architecture/concepts/deepseek-arch-v3.md](../../model-architecture/concepts/deepseek-arch-v3.md)** — 算法域:MLA + DeepSeekMoE + MTP + aux-loss-free 的结构细节(61 层、7168 维、256 专家 top-8)。互链不重复:结构在彼篇,训练编排在本篇。
- **[open-infra-index.md](open-infra-index.md)** — 开源 infra 官方总账;V3 训练口径的一手出处与本域各专篇索引。
