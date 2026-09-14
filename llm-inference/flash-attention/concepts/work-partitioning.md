---
title: FlashAttention-2 的并行与 Work Partitioning:为什么快了 2×
type: concept
tags: [flashattention, fa2, work-partitioning, warp, split-k, threadblock, parallelism]
source:
  - "pdf: FlashAttention-2 | Dao arXiv:2307.08691 ICLR 2024"
  - "web: Tri Dao FlashAttention-2 博客 (work partitioning) | https://princeton-nlp.github.io/flash-atttention-2/"
  - "doc: tracker/flash-attention-tracker/flash-attention-annual-report.md §2.3 (canonical)"
date: 2026-09
---

# FlashAttention-2 的并行与 Work Partitioning:为什么快了 2×

## TL;DR

FA1 已达 **IO 下界**(HBM 访存最优),但在 A100 上只有 **25-40%** 的 FLOP 利用率。**瓶颈不再是"搬多少数据",而是"算得不够满"**——并行拓扑把 GPU 的资源没用满。

**FA2 主要做两件事**(配合 CUTLASS 3.x/CuTe 重写),把利用率提到 **50-73%**、A100 上到 ~230 TFLOP/s:
1. **减少 non-matmul FLOPs**:把"每次迭代都对未归一化输出除 `ℓ`"**推迟到最终一次**(rescale 本身仍按 max 做,但省掉每步矩阵除 ℓ;详见 §2)——标量运算每步随块数线性累积,A100 上 non-matmul 比 matmul 贵约 16×;
2. **更好的 work partitioning**:threadblock/warp 的并行划分更贴合 GPU——**warp 不再各算不同 K/V 块(要通信合结果),而是各算同一 K/V 块的连续行(几乎无通信)**;并新增**序列维并行(沿 Q 切,免归并,见 §4)**。

> **心法**:IO-optimal 之后,下一个瓶颈是利用率。FA2 证明:attention kernel 的并行拓扑(threadblock/warp 怎么分活)对性能的影响,不亚于访存优化。

---

## 1. 为什么 FA1 到 25-40% 就上不去了

FA1 的 IO 已最优,但计算没跑满,三个原因:

| 瓶颈 | 说明 |
|---|---|
| **大量 non-matmul FLOPs** | FA1 每次加载新 K/V 块都要对未归一化输出按 max 做矩阵 rescale 并**除一次 `ℓ`**,标量运算每行随块数线性累积;而 Tensor Core 的 matmul 吞吐远超标量(**A100:matmul 312 vs non-matmul FP32 仅 ~19.5 TFLOPs/s,贵约 16×**),这些 non-matmul 成了拖累 |
| **warp 间通信开销** | FA1 让 4 个 warp 各算不同的 K/V 块,最后要 **shared memory sync + reduce** 合并各行分数——warp 间大量通信 |
| **并行维度不足** | FA1 只在 **batch × head** 维度并行;**batch×heads 少**(如 batch 小且序列长)时 grid 喂不满 SM、occupancy 不足(batch 足够大时长序列并不缺线程块) |

FA2 逐个解决。

---

## 2. 减少 non-matmul FLOPs:把 rescale 推迟到最终

**FA1 的写法**(每次加载新 K/V 块,rescale + 除行和):

```
每次更新:
  ℓ_new = ℓ_old·e^{m_old−m_new}   + rowsum(P̃)          # P̃ = e^{S−m_new}
  Õ_new = diag(e^{m_old−m_new})·Õ_old + e^{S−m_new}@V    # 未归一化分子,按 max rescale
  Õ = Õ_new / ℓ_new                                          # 每步都除行和 ℓ
```

每步对已累加输出做一次矩阵 rescale 且**每步都除一次 ℓ_old**(FA1 的 rescale 系数是 `diag(ℓ_old/ℓ_new)`,含 ℓ 之比),这些是量化 non-matmul 运算。

**FA2 的写法**(推迟除 ℓ,但 Õ 仍按 max rescale):

```
每次更新:
  ℓ_new = ℓ_old·e^{m_old−m_new} + rowsum(P̃)     # rescale 运行行和
  Õ_new = diag(e^{m_old−m_new})·Õ_old + P̃@V     # 未归一化分子仍按 max rescale
  ...直到最后:
  Oi = Õ_final / ℓ_final                          # 只除一次 ℓ
```

- **FA2 真正省的不是"rescale"(Õ 仍每块按 max rescale),而是把"每次迭代对 Õ 除 ℓ"推迟到最终一次**——FA1 每步除 ℓ,FA2 只在最后除;
- 省掉的 non-matmul 量级**不是 Θ(N²) 而是每行随块数线性累积**(每 KV 块 O(Br·d),总计 Θ(N²d/Bc));
- **为什么有效(论文核心论据)**:A100 上 matmul 达 312 TFLOPs/s,而 non-matmul(FP32 标量)仅 ~19.5 TFLOPs/s——**每个 non-matmul FLOP 比 matmul 贵约 16×**。省 rescale 的标量运算,把算力让给 Tensor Core 的 matmul,收益被放大。

---

## 3. 更好的 Warp 划分:通信大减

**FA1 的 warp 划分**(4 warp 各算不同 K/V block,需合并):

```
Warp 0: S(Q_block, K[b0]) @ V[b0]
Warp 1: S(Q_block, K[b1]) @ V[b1]   → 结果要 shared memory sync + reduce 合并
Warp 2: S(Q_block, K[b2]) @ V[b2]
Warp 3: S(Q_block, K[b3]) @ V[b3]
```

四个 warp 对**同一个 Q block** 各算一部分 K/V,产生的分数行是"按列切"的,最后要**跨 warp 归约**,通信 + 同步多。

**FA2 的 warp 划分**(4 warp 各算同一 K/V block 的连续行,独立写):

```
              K_block, V_block (载入 shared memory)
                    ↓
Warp 0:  Q rows 0-31   → S[0:32,:] → P[0:32,:] → O[0:32]
Warp 1:  Q rows 32-63  → S[32:64,:]→ P[32:64,:]→ O[32:64]
Warp 2:  Q rows 64-95  → ...
Warp 3:  Q rows 96-127 → ...
→ 每个 warp 独立写自己的输出行,几乎不需要 warp 间通信
```

- **Q 的各行在 K/V 维度上的 softmax 是独立的**(行间不共享 max/sum——除 split-K 场景),所以**按 Q 行切分 = 各 warp 完全独立**;
- 每个 warp 自己算、自己写,免除合并同步;
- 这正是"work partitioning 贴合注意力结构"的体现:attention 的行间独立性天然适合按行分活。

---

## 4. 序列维并行:切 Q(免合并)与切 KV(需归并)要分清

**容易混淆的关键结构**:FA2 论文的 threadblock 级序列并行是**沿 Q 行切**——`grid ≈ (num_q_blocks, batch, nheads)`,每个 block 算一段 **Q 行**,各行 softmax 独立,**block 之间"no communication at all"**、**根本不需要 reduction kernel**。注意:**网格结构与 FA1 本质相同**(FA1 的 grid = (batch, nheads, ceil(N/Br)) 也已把 Q 切块、每 block 算 Br 行);FA2 的关键改进**不在 grid 层级新增并行维,而在 block 内部**:把 warp 划分从"各 warp 按 K/V 段切"(要跨 warp 合并)改成"各 warp 按 Q 行切"(行间独立、免通信),并利用** Q 行天然可独立计算**这一性质充分并行:

```
grid = (num_q_blocks, batch, nheads)   # 沿 Q 切
每个 block 算自己的 Q 行块 → 各行独立 → 无需合并,直接写各自的 O 行
```

**只有沿 K/V 切**(FlashDecoding 一族,如长序列 decode 时)才需要 split-K 归并:

```
grid = (batch, nheads, n_splits)       # 沿 K/V 切(分 split)
每个 block 处理同样的 Q 行、不同的 K/V 段 → 产生局部 O_split + 局部 LSE(logsumexp)
最后 reduction kernel 合并(按 LSE 加权):
  for s in splits:
      m_final  = max(m_final, m_s)
      scale    = exp(m_s - m_final)
      O_final += O_split_s * scale       # 按新 max 加权
      l_final += l_split_s * scale
  O_final /= l_final                     # 最终归一化
```

- **切 Q = 各行独立,免合并**——因为 attention 的 softmax 在 Q 行间完全独立;
- **切 KV = 各行分成多段,须按 LSE 加权合并**(就是 split-K 归并,类比 CUTLASS split-K/Stream-K,见 cutlass tile-scheduler 篇);
- **二者动机相同**:长序列/少 batch 时把更多的 threadblock 塞进 GPU(补 occupancy);

> 一句话:FA2 序列并行加的是"切 Q"(免归并);"切 KV 要归并"是 FlashDecoding/split-K 的事,别混。

---

## 5. 实践要点(checklist)

1. **FA2 on 优先于 FA1(但 FA1 不算"过时")**:FA1 仍是 IO 最优算法框架,FA2 在其上优化利用率(50-73% MFU);计算密集且长序列的 workload,FA2 收益(2×)更值得。
2. **序列并行分清切 Q / 切 KV**:沿 Q 切 → 各行独立、免归并;沿 KV 切 → 需按 LSE 加权合并(split-K/FlashDecoding)。batch×head 够大时可不拆,免得无谓归并。
3. **non-matmul FLOPs 是隐藏的坑**:Tensor Core 时代 non-matmul 比 matmul 贵 ~16×(A100);任何"每块更新都除 ℓ"的实现都在浪费算力——FA2 把除 ℓ 推迟到最后。
4. **warp 按 Q 行切分是 FA2 的关键**:attention 行间独立 → 按行分 warp 免通信;写自定义 attention kernel 时优先模仿这个划分。
5. **causal mask 也是 non-matmul 削减**:对每行只要算 1 个被 mask 的块/可整块跳过,SAVE 大量无用计算——别忽略这一半优化。
6. **反向也讲 work partitioning**:FA2 backward 沿 KV 块切、固定外层循环以避免对 dQ 的 atomic add;且反向只存 logsumexp L = m + log ℓ(不再同时存 m、ℓ),省 HBM 写 + 标量运算——这是全库到处传 `lse` 的由来。
7. **配合工具链**:FA2 基于 CUTLASS 3.x/CuTe(见 cutlass 篇);要到 H100 看 FA3 的 warp specialization 如何进一步改写划分(见 fa3-hopper-async 篇)。

---

## 6. 溯源

- FlashAttention-2 论文(§3.2 减 non-matmul FLOPs、§3.3 序列维并行): Dao, https://arxiv.org/abs/2307.08691 (ICLR 2024)
- Tri Dao FA2 博客(25-40%→72% MFU、A100 230 TFLOP/s、warp 划分示意图;`flash-atttention-2` 为站点真实拼写): https://princeton-nlp.github.io/flash-atttention-2/
- canonical 库内文档 §2.3(FA2 伪代码与细节): `~/zhanbin/tracker/flash-attention-tracker/flash-attention-annual-report.md`

---

## 7. 可复现参数速查(FA2 常见配置)

| 参数 | 取值(head_dim=128 时) | 说明 |
|---|---|---|
| Br(Q block 行) | 128 | SRAM 每块 Q 行数 |
| Bc(K/V block 列) | 128 | 每个 K/V 块列数(受 SRAM 预算约 4d 约束,见 io-aware 篇) |
| Warp tile(每个 warp 的 Q 行) | ~32-64 | FA2 warp 沿 Q 行切分后的粒度 |
| MFU | ~50-73%(A100,**up to 72%** 为博客口径,随 head_dim/batch/config 变化) | 相对 A100 理论峰值 |
