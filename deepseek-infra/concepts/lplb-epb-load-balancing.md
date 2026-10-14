---
title: LPLB + EPLB — 专家并行负载均衡(动态线性规划 vs 静态复制)
type: concept
tags: [eplb, lplb, expert-parallel, load-balancing, moe, linear-programming, redundant-expert, deepseek, v3]
source:
  - "code: deepseek-ai/EPLB | https://github.com/deepseek-ai/EPLB ↑primary"
  - "code: deepseek-ai/LPLB | https://github.com/deepseek-ai/LPLB ↑primary"
  - "code: deepseek-ai/DeepEP(负载均衡原语消费方)| https://github.com/deepseek-ai/DeepEP"
date: 2026-10
---

# LPLB + EPLB — 专家并行负载均衡(动态线性规划 vs 静态复制)

## TL;DR

DeepSeek 有两个专家并行(EP)负载均衡器,解决**不同时间尺度**的负载不均:

- **EPLB(Expert Parallelism Load Balancer,静态/离线)**:处理**稳定的**负载不均(某 expert 持续过热,源于数据分布)。用「冗余专家(redundant experts)」策略复制重载 expert,再启发式打包到 GPU 上。这是 DeepSeek-V3 论文里说的、**已部署上线**的算法。
- **LPLB(Linear-Programming-Based Load Balancer,动态/在线)**:处理**逐 batch 波动**的负载不均(小 batch 随机性导致的瞬时不均)。在 EPLB 选出的冗余专家拓扑上,**用线性规划(LP)求解每个 batch 的最优 token 分配**,最小化 EP 组内负载失衡。**早期研究阶段,性能仍在评估**。

**关键区分**:EPLB 解决「谁是热点」(静态,离线算一次),LPLB 解决「这一批 token 怎么分到冗余专家」(动态,每 batch 解一次 LP)。两者是**接力关系**:EPLB 定复制拓扑(重排/replication),LPLB 在拓扑上做逐 batch 的最优分流。

---

## 1. EPLB:静态冗余专家 + 启发式打包

### 1.1 为什么需要冗余专家

EP 下不同 expert 分到不同 GPU,而 expert 负载随 workload 变化(有的 expert 老被路由选中)。V3 的做法是**复制重载 expert**(冗余专家),让多个 GPU 各持一份热 expert 的副本,把负载摊开。EPLB 根据**估计的 expert 负载**算一个均衡的复制 + 放置方案。

> ⚠ 负载**预测**方法不在 EPLB 仓库范围内——常用做法是对历史统计做滑动平均,调用方自己提供权重。

### 1.2 两种策略

- **Hierarchical Load Balancing(分层)**:当「节点数」整除「expert 组数」时用。利用 V3 的 **group-limited expert routing**(同组 expert 尽量放同节点以减少跨节点流量):先把 expert 组均匀打包到节点 → 节点内复制 → 再打包到单 GPU。用于 **prefill 阶段(EP 较小)**。
- **Global Load Balancing(全局)**:其他情况用。不管 expert 组,全局复制 + 打包到单 GPU。用于 **decoding 阶段(EP 较大)**。

### 1.3 接口

```python
import eplb
weight = torch.tensor([[90,132,40,61,104,165,39,4,73,56,183,86],
                       [20,107,104,64,19,197,187,157,172,86,16,27]])
phy2log, log2phy, logcnt = eplb.rebalance_experts(
    weight, num_replicas=16, num_groups=4, num_nodes=2, num_gpus=8)
# phy2log: 每个物理专家槽位对应的逻辑 expert id
```

输出是复制 + 放置计划(phy2log/log2phy/logcnt),纯 Python + PyTorch,无 CUDA 依赖。

---

## 2. LPLB:动态线性规划分流

### 2.1 机制(三步)

1. **冗余专家边**:每个冗余专家连到它的原 expert,在 GPU 之间形成边;
2. **边容量**:边容量 = 当前 batch 里分配给该冗余专家的 token 数,即平衡时的最大 token 流量;
3. **LP 优化**:解一个**线性规划**问题,沿这些边重新分配 token,在尊重边容量的前提下最小化 EP 组内负载失衡。

被复制的 expert 由 **EPLB(只重排、不复制)** 选出,最热的 expert 再按 LPLB 拓扑复制。实时负载同步用 **NVLink + NVSHMEM** 而非 `torch.distributed.allreduce`,降低通信开销(因此**强依赖 DeepEP**)。

### 2.2 嵌入式 LP 求解器

LPLB 内嵌**单 SM Interior Point Method(IPM)** 求解器,线性代数用 NVIDIA 的 **cuSolverDx + cuBLASDx**(Device API 扩展)高效实现,求解在 GPU 上跑。

### 2.3 接口

```python
planner = Planner(r2o, n_logical_experts + n_redundants_per_rank*ep_size,
                  n_logical_experts, group=ep_group)
# 可选:从 DeepEP buffer 初始化实时统计
# planner.init_from_deep_ep(buffer)
redirected_indices = planner.run(indices, avail_counter, N_SMS)  # 逻辑 → 物理 expert
```

`r2o` 矩阵定义冗余专家拓扑(哪几个物理槽复制哪个逻辑 expert)。

### 2.4 典型拓扑

| 拓扑 | 复制方式 | 需求 | 适用 |
|---|---|---|---|
| **Cube** | 在部分 GPU 上复制 expert,带对角边 | ≥2 experts/GPU | 8-GPU EP 子组内均衡,不牺牲跨节点通信 |
| **Hypercube** | 无对角边 | 16 GPU | 16-GPU 专家并行 |
| **Torus** | 同节点邻居复制一份 + 邻节点复制一份 | ≥2 experts/GPU | 全局均衡,但节点内通信效率低于 Cube |

自定义拓扑可改 `r2o` 矩阵探索。

---

## 3. 数据 / 局限(LPLB 明确列出的三条)

- **只平衡 token 总数**:不考虑 grouped 矩阵乘时间的非线性(不同 expert 的形状/计算量可能不同),可能次优;
- **求解耗时 ~100 µs(节点内)**:节点间更长,小 batch 下不可忽略;
- **极端全局失衡时 LPLB 可能不如 EPLB**:因为 LPLB 避免给同一原 expert 分配多个副本,复制策略与 EPLB 不同。

EPLB 无公开 benchmark,其价值在「V3/V4 已部署」的生产验证。

---

## 4. 实践要点 / 坑

- **EPLB 是离线静态、LPLB 是在线动态**,别混:EPLB 算一次复制方案,LPLB 每 batch 解一次 LP。生产里 EPLB 定拓扑 → LPLB(可选)逐 batch 分流。
- **EPLB 依赖调用方提供负载估计**:滑动平均是常见做法,但预测方法仓库不管。EPLB 纯 Python,无 GPU 依赖,可直接离线算。
- **LPLB 前置条件**:CUDA ≥ 12.6.3(带 cuSolverDx/cuBLASDx 依赖,`./download-mathdx.sh` 下载),DeepEP **强推荐**(实际必选,实时统计走 NVLink+NVSHMEM),EPLB 内嵌。
- **LPLB 拓扑选择看规模**:≤8 GPU 用 Cube、16 GPU 用 Hypercube、跨节点全局用 Torus;拓扑与 EP 规模/网络层级要匹配。
- **「LPLB 一定比 EPLB 好」是错的**:极端全局失衡下 LPLB 因复制策略差异可能更差(README 明说),小 batch 下 ~100µs 求解开销也可能得不偿失。
- **与 DeepEP 的接力**:DeepEP `EPBuffer.lb_prefetch_weights`/`lb_reduce_grads` 提供复制后专家的权重预取/梯度归约原语,复制计划(即 LPLB/EPLB 的输出)由调用方喂给 DeepEP(见 [deepep-nvidia-vs-ascend.md](deepep-nvidia-vs-ascend.md))。

---

## 溯源

- **EPLB 仓库**(主源:冗余专家策略/分层+全局两策略/接口/license MIT):https://github.com/deepseek-ai/EPLB
- **LPLB 仓库**(主源:LP 机制/IPM+cuSolverDx/cuBLASDx/三局限/拓扑):https://github.com/deepseek-ai/LPLB
- **DeepEP**(负载均衡通信原语 + LPLB 实时统计来源):https://github.com/deepseek-ai/DeepEP
- **DeepSeek-V3 技术报告**(冗余专家策略出处):https://arxiv.org/abs/2412.19437
- **库内交叉**:[deepep-nvidia-vs-ascend.md](deepep-nvidia-vs-ascend.md)(负载均衡通信原语)、[dualpipe.md](dualpipe.md)(PP 侧,与 EP 负载均衡正交)、[../../model-architecture/concepts/open-model-landscape.md](../../model-architecture/concepts/open-model-landscape.md)(MoE/EP 背景)。
