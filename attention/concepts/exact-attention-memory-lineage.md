---
title: 精确注意力的省显存谱系 — Rabe & Staats 的 O(1) 存注意力(lazy softmax + max rescale)
type: concept
tags: [attention, memory-efficient-attention, online-softmax, rabe-staats, jax, tpu, gradient-checkpointing, flashattention-lineage]
source:
  - "pdf: Self-Attention does not Need O(n²) Memory(Rabe & Staats, Google Research, ↑primary) | arXiv:2112.05682v3 (2022-10-10) | ~/zhanbin/library/papers/Self-Attention does not Need O(n2) Memory.pdf"
  - "pdf: FlashAttention FA1 论文 §3.1 / 附录 B.1、B.5(引用并设专节比较本文) | arXiv:2205.14135 | ~/zhanbin/library/papers/FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.pdf"
  - "web: 官方 JAX 实现 + colab 复现 | https://github.com/google-research/google-research/tree/master/memory_efficient_attention (2026-09 实测 200)"
  - "web: PyTorch 复现(lucidrains) | https://github.com/lucidrains/memory-efficient-attention-pytorch (2026-09 实测 200)"
  - "web: 上游方法脉络 — Online normalizer calculation for softmax(Milakov & Gimelshein, FA1 附录标注为[60]) | https://arxiv.org/abs/1805.02867 (2026-09 实测 200);lazy softmax 首见 MNNFast, Jang et al. ISCA 2019"
date: 2026-09
---

# 精确注意力的省显存谱系:Rabe & Staats 的 O(1) 存注意力

> 定位:本篇是 attention 域**内核/实现层的上游前驱**。FA1 论文两处引它(附录 B.1 说"这一技巧已被用于证明 attention 不需要二次额外显存",B.5 专节做三点比较)。理解它 = 理解 FlashAttention 究竟"新"在哪里:**R&S 解决"显存占用",FA 解决"访存次数"**——两者都是精确(非近似)attention,但优化目标不同,后果是 FA 快 2-4×、R&S 与标准实现同速或略慢。

## TL;DR

Rabe & Staats(Google Research, arXiv:2112.05682)指出社区长期误传"self-attention 需要 O(n²) 显存"其实不成立。做法极简:**把 softmax 的除法推到最后一刻**(lazy softmax,`attention = Σ_i v_i e^{s_i} / Σ_j e^{s_j}`),于是单个 query 只需维护两个累加量(向量 `v*`、标量 `s*`)→ **单 query attention O(1) 额外显存,self-attention O(log n)**(多一个 query 索引)。数值稳定靠**在线 max 重归一**(和 online softmax 同一招:遇新 max 就把已累加量乘 `e^{m*-m_i}` 收缩)。工程实现上为利用 TPU/GPU 并行,分块处理 → 实际 **O(√n) 显存**;反向传播用 **gradient checkpointing** 重算而非存中间量。效果:n=16384 时 self-attention 显存开销**推理降 59×、反传降 32×**,运行时与标准实现差几个百分点(WMT en-de 训到 100K step:62.69 vs 62.59 BLEU,几乎重合)。

一句话心法:**显存爆炸来自"先物化 N×N 矩阵再归一化"这个实现习惯,不是 attention 的数学本质——把归一化延迟到末尾,N×N 就永远不必存在**。

---

## 1. 机制原理(为什么 O(1) 成立)

### 1.1 除法延迟(lazy softmax)

标准写法先算全部 `s_i`,做 softmax(需要全行的分母),再加权求和——这一步强制物化整行分数。R&S 用分配律改写:

```
s_i = dot(q, k_i);  s'_i = e^{s_i}
attention(q,k,v) = (Σ_i v_i s'_i) / (Σ_j s'_j)        # 式(1)
```

分母是**标量**,可以边遍历边累加。于是状态只有 `v* ∈ R^d`(初始 0)和 `s* ∈ R`(初始 0):读入 `(k_i, v_i)` → `s_i = dot(q,k_i)` → `v* += v_i e^{s_i}`、`s* += e^{s_i}`;遍历完 `v*/s*` 即结果。

**空间复杂度成立的前提是输入顺序**:先读 query、再顺序读 (k,v) 对。若输入乱序,需额外存序列索引 → O(log n)。self-attention 对每个 query 顺序跑一遍,多一个 query 索引 → **O(log n)**;输出本身 O(n) 不计入额外显存。

> **出处诚实标注**:论文自己在 §2 和 §6 承认,式(1)是 **Jang et al. (MNNFast, ISCA 2019) "lazy softmax" 的再发现**——但 Jang 用它做多芯片 KV 分片的带宽优化,**完全没讨论显存复杂度、数值稳定与反向传播**,也无公开实现。R&S 的增量贡献在后三者 + 分块实现 + 实测。

### 1.2 数值稳定:在线 max 重归一

朴素累加 `e^{s_i}` 会溢出(bfloat16/float32 下 `s_i ≥ 89` 即 inf)。标准 softmax 靠"全行减 max"解决,但增量式看不到未来的 max。R&S 的解法与 online softmax 同源——**加第三个状态 `m*`(至今见过的最大分数),每次更新前重归一已累加量**:

```
m_i = max(m*, s_i)
v* ← v* · e^{m*-m_i} + v_i · e^{s_i-m_i}
s* ← s* · e^{m*-m_i} +       e^{s_i-m_i}
m* ← m_i
```

因果链:**max 未知 → 已累加量按旧 max 记的账,新 max 出现时全部要收缩同一因子 `e^{m_old-m_new}` → 乘法收缩可后置、不必重算**。这正是 io-aware-tiling 篇 §3 的 running max/running sum rescale——**同一数学,不同工程目标**(R&S 为省显存,FA 为省 HBM 访存)。

### 1.3 为什么"省显存"不等于"变快"

R&S 的时间复杂度仍是 **self-attention O(n²)、单 query O(n)**。作者点明现代加速器上"显存常是瓶颈、算力相对便宜",所以省显存让长序列**可行**(n=1M 时算法在做 >1 万亿次 query-key 组合);但 TPU 上标准 attention 已经把 FLOPs 和带宽平衡得不错,**R&S 换不来显著加速**(推理 -8~-13%,反传 -30~-35%,后者是 checkpointing 重算的必然代价)。FA1 论文 B.5 正是抓住这一点做差异化:**"R&S 优化 total memory footprint,FlashAttention 优化 memory accesses;访存次数才是 runtime 的主导因素"**。

## 2. 可复现细节

### 2.1 分块实现(O(√n) 的实际形态)

朴素逐 key 累加对编译器不友好(全序列依赖链)。论文 Figure 1 的 JAX 实现分两级块:

```
外层:jax.lax.scan 按 query_chunk_size(默认 1024)切 query
      → 每次迭代结果直接写进输出张量 res,不跨迭代累积显存
内层:_query_chunk_attention 按 key_chunk_size(默认 4096)切 K/V
      summarize_chunk(query, key_chunk, value_chunk) 返回三元组:
        exp_values = einsum('vhf,qhv->qhf', value, exp_weights)
        exp_weights.sum(-1)      # 分母部分和
        max_score                # 块内 max(带 stop_gradient)
      → √n 个块的摘要 → 全局 max 重缩放 → 求和 → all_values/all_weights
```

- **`jax.checkpoint(prevent_cse=False)` 包在 `summarize_chunk` 上**是反向省显存的关键:前向丢弃块内中间量,反传时重算;
- **`stop_gradient(max_score)`**:max 只当归一化常数,不参与梯度(否则梯度路径会走 max 的分支);
- **块大小是内存/速度的调节旋钮**:严格省内存该取 query_chunk=常数、key_chunk=√n,但实际运行时受硬件影响,论文把两个参数暴露给程序员;
- 作者注:**多阶段摘要可做到 O(log n)**,但会让实现复杂化,故未采用;
- 官方实现:github.com/google-research/google-research/tree/master/memory_efficient_attention(含 colab)。

### 2.2 为什么"给标准 attention 加 checkpointing"达不到同样效果

论文 §5.2 的辨析(容易忽略但重要):对标准实现做 checkpointing,它仍会**先形成完整 N×N attention matrix**、只是事后忘掉;R&S **从不形成完整矩阵**。所以两者的显存峰值不是一个量级——省显存来自算法改写,checkpointing 只是让反向不把它加回来。

## 3. 数据对比(TPUv3 单芯片,1 head)

**推理(Table 2)**:

| n | 输入+输出 | 标准 attn 额外显存 | R&S 额外显存 | 计算时间 | 相对速度 |
|---|---|---|---|---|---|
| 2¹² | 2.5MB | 64MB | 16MB | 0.7ms | -8±2% |
| 2¹⁴ | 10MB | 1GB | 17MB | 11.3ms | -13±2% |
| 2¹⁶ | 40MB | OOM | 21MB | 177ms | — |
| 2²⁰ | 640MB | OOM | 256MB | 2.82s | — |

**反向(Table 3)**:n=2¹² 时 128MB → 41MB,n=2¹⁴ 时 2.0GB → 64MB,n=2¹⁶ 时 OOM → 257MB,n=2²⁰ 时 OOM → 4.0GB;相对速度 -30±5%(n=2¹²)/ -35±5%(n=2¹⁴)。

其他关键数字:
- **n=16384 时显存开销推理降 59×、反传降 32×**(摘要口径);
- **正确性**:与 Flax 标准实现在不 OOM 的档位上,最大绝对差 < 1.8×10⁻⁷(σ=1 正态输入,n=2¹⁴)——**精确同函数,不是近似**;
- **训练等价**:WMT en-de,Flax Transformer 换成本实现,100K step 后 62.69 vs 62.59 BLEU,曲线几乎重合;训练中还观察到 steps/sec **+4%** 的小幅提升(算子孤立测是变慢的,嵌进大架构后反而略快——作者提醒孤立基准不等于端到端);
- **vs 只切 query(Reformer 式 query chunking)**:query chunk ≤64 时显著变慢;固定同样显存预算下,序列越长 query-chunking 越吃亏(被迫把 chunk 压到 ≤64),而 R&S 因为**同时切 K/V**不受此限——这是"必须两个维度都分块"的实证理由;
- 与 FA1 的三点差异(FA1 附录 B.5):① 优化目标(footprint vs 访存次数)→ FA 2-4× 加速、R&S 同速;② 块间传递(R&S 存**每块临时输出**+统计量、末尾合并 K 份;FA **增量更新单份输出**)→ FA 总显存更小;③ 反向(R&S 用 gradient checkpointing 重算矩阵与临时输出;FA **解析推导反向**,只重算 attention matrix)→ FA 反向更省更快。

## 4. 实践要点

1. **想省显存又要快 → 用 FlashAttention;只想要一个"数学上等价、可 drop-in 替换"的省显存实现(JAX/TPU 栈)→ memory_efficient_attention**。R&S 的价值一半是工程、一半是**认知纠偏**:"attention 天生吃 O(n²) 显存"是实现习惯,不是数学约束——做架构/硬件设计时别把它当前提。
2. **chunk size 要自己调**:默认值(query 1024 / key 4096)是 TPU 上"运行时影响最小"的折中,不是内存最优;GPU 上要按 SRAM/HBM 重新扫(参照 io-aware-tiling 篇 §7 的 `Bc≈M/(4d)` 思路)。
3. **反向必须 checkpointing**:否则前向省的显存被反传的中间量全吃回去;注意 checkpoint 要包在 `summarize_chunk` 这一层,`prevent_cse=False` 避免公共子表达式消除干扰重算语义。
4. **max 的 `stop_gradient` 不能省**:否则梯度会流进 max 选择分支,与标准 softmax 梯度不一致。
5. **rescale 是正确性要求,不是精度优化**:去掉 `e^{m*-m_i}` 收缩,各块分母基准不同 → 结果数学上错(与 io-aware-tiling 篇 §7.6 同一条坑)。
6. **孤立算子基准会误导**:本实现单测算子慢 8-13%,放进 Transformer 训练反而快 4%——评估省显存 kernel 要看端到端 steps/sec。
7. **谱系记法**(便于串库):Milakov & Gimelshein 2018(online softmax 归一化,arXiv:1805.02867)→ Jang et al. 2019(lazy softmax,带宽视角)→ **Rabe & Staats 2021(显存视角 + 稳定化 + 反传 + 实现)** → FlashAttention 2022(访存次数视角 + CUDA + 解析反向)→ FA2/FA3/FA4(并行度与硬件特性)。

## 5. 溯源与多源综合说明

- 主源 ↑primary:本地 PDF(arXiv:2112.05682v3, 2022-10-10),含 Figure 1 完整 JAX 代码、Table 2/3 实测、§5.4 query-chunking 对比、§6 lazy softmax 出处自陈;
- 第二本地源:FA1 PDF(附录 B.1/B.5)——**三点差异对比全部出自 FA1 原文 B.5,不是本篇推断**;B.1 把 online normalizer 技巧归给 [60](Milakov & Gimelshein)与 [51,66];
- 网络核对:arXiv 摘要页(2026-09 实测 200,标题 *Self-attention Does Not Need O(n²) Memory*)、官方实现仓库(google-research/memory_efficient_attention,实测 200)、lucidrains PyTorch 复现(实测 200)、Milakov & Gimelshein arXiv:1805.02867(实测 200);论文中 Rezaei 2021 JAX 复现与 Wang 2022 PyTorch 复现亦在致谢列出;
- 发表状态:arXiv 预印本(v1 2021-12,v3 2022-10),**未见正式会议收录标注**——引用时按 preprint 处理,勿写成某会议论文;
- 与库内其他篇的关系:online softmax 推导与 FA1 算法细节见 [io-aware-tiling.md](io-aware-tiling.md);反向重算 vs 解析反向的取舍见 [io-aware-tiling.md §6.5](io-aware-tiling.md);并行维度切分见 [work-partitioning.md](work-partitioning.md)。
