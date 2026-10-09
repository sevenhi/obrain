---
title: FlashAttention 实现全解:FA1 IO-aware tiling → FA2 Work Partitioning → FA3 Hopper 异步 → FA4 Blackwell
type: concept
tags: [flashattention, attention, tiling, online-softmax, io-awareness, gpu, sram, fa2, work-partitioning, warp, split-k, threadblock, parallelism, fa3, hopper, wgmma, tma, asynchrony, fp8, pipeline]
source:
  - "pdf: FlashAttention FA1 (primary) | Dao et al. arXiv:2205.14135 NeurIPS 2022"
  - "pdf: FlashAttention-2 (primary) | Dao arXiv:2307.08691 ICLR 2024"
  - "pdf: FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (primary) | arXiv:2407.08608 NeurIPS 2024"
  - "pdf: FlashAttention-4 (Blackwell, CuTe DSL) | arXiv:2603.05451 MLSys 2026"
  - "web: Tri Dao FlashAttention-2 博客 (work partitioning) | https://princeton-nlp.github.io/flash-atttention-2/"
  - "web: Tri Dao FA3 Blog (overlap、WGMMA/TMA、FP8、pingpong) | https://tridao.me/blog/2024/flash3/"
  - "pdf: UW CSE 599M 讲义 From Online Softmax to FlashAttention (Zihao Ye) | https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf"
  - "pdf: Self-Attention does not Need O(n²) Memory (Rabe & Staats, FA1 引用[66]并设附录 B.5 专节比较;online softmax/lazy softmax 前驱) | arXiv:2112.05682"
  - "pdf: Online normalizer calculation for softmax (Milakov & Gimelshein, FA1 引用[60], online softmax 最早出处) | arXiv:1805.02867 | https://arxiv.org/abs/1805.02867 (2026-09 实测 200)"
  - "code: github.com/Dao-AILab/flash-attention | Star 24.7k"
date: 2026-09
---

# FlashAttention 实现全解:四代内核主线(FA1→FA2→FA3→FA4)

> 本篇由原 `io-aware-tiling.md`(FA1)、`work-partitioning.md`(FA2)、`fa3-hopper-async.md`(FA3)三篇于 2026-09 合成,按代次串成完整主线,全部内容(公式/伪代码/数据表/实践/溯源)保留。FA 前驱谱系(Rabe & Staats O(1) 显存)单独见 [../algorithm/exact-attention-memory-lineage.md](../algorithm/exact-attention-memory-lineage.md)。

## 0. TL;DR:四代怎么解决"attention 访存受限"

attention 的瓶颈不是计算,是**内存搬运**:标准实现要把 N×N 的 `S`/`P` 中间矩阵写进 HBM 再读回,访存 Θ(Nd + N²),长序列下时间全花在搬数据上。FlashAttention 家族四代,每代解决一层瓶颈:

| 代次 | 时间 | 论文 | 目标 GPU | 解决什么 | 核心实现技术 | MFU(前向) |
|---|---|---|---|---|---|---|
| FA1 | 2022.05 | NeurIPS 2022 | A100 | **搬得少**(IO 最优) | IO-aware tiling + online softmax | 25-40% |
| FA2 | 2023.07 | ICLR 2024 | A100+H100 | **算得满**(利用率) | 更好 work partitioning, 减 non-matmul FLOPs | 50-73% |
| FA3 | 2024.07 | NeurIPS 2024 | H100 | **算搬同时**(异步隐藏延迟) | WGMMA async, TMA, overlap, FP8 | ~75% FP16 / ~61-66% FP8 |
| FA4 | 2026.03 | arXiv:2603.05451 (MLSys 2026) | B200 | **非对称硬件缩放**(TC 翻倍但 smem 带宽/exp 单元没跟上) | CuTe DSL, 异步 UMMA/tcgen05, 软 exp+软 softmax rescale, TMEM+2-CTA MMA | ~71% (BF16, up to 1613 TFLOPS) |

**一句话收益**:标准 attention 的 HBM 访存 Θ(Nd + N²)(元素访问数),FlashAttention 降到 Θ(N²d²/M)(M = SRAM 大小)——对大 N 是**数倍到接近一个量级的访存减少、端到端实测 2-4× 加速**,省显存(显存 O(N²)→O(N))。**不损失精度(数学精确,非近似,但浮点结果与朴素实现有 ulp 级差异)**。

**心法**:FA1 解决"少搬数据"(IO-optimal);FA2+ 解决"算得更满"(并行/利用率/软硬件特性)。把"算注意力分数"和"写回中间矩阵"解耦,让每次 HBM 读写都服务于"计算整行的最终输出",这就是 IO-aware 的意义;之后每一代都在此框架上叠加硬件特性的利用。

---

## 1. FA1:IO-aware tiling 与 Online Softmax

### 1.1 先看问题:标准 Attention 为什么慢

标准 attention 分三步,**每一步都把中间矩阵写进 HBM 再读回**:

```python
S = Q @ K^T * scale    # [N,N] 注意力分数    → 写入 HBM
P = softmax(S)          # [N,N] softmax 概率 → 写入 HBM
O = P @ V              # [N,d] 输出
```

**问题焦点:N×N 中间矩阵**。

- 长序列 N=8192、head dim d=128 时,`S` 占 `8192×8192×2B ≈ 128MB`——而 GPU SRAM(~192KB,下同)完全装不下;
- 结果:必须把 `S`、`P` 写到片外 HBM,再读回来做下一步,中间矩阵**来回复制两次**;
- **IO 复杂度**:标准 attention 需要 **Θ(Nd + N²)**(元素访问)次 HBM 读写(实例化完整的 N×N 矩阵;N=序列长,d=头维)。瓶颈是那个 N²(N×N 中间矩阵的读写随序列长度二次方增长)。

**核心洞察:attention 是 memory-bound 的**。它的计算量(FLOPs)是 Θ(N²d),但内存搬移才是瓶颈——尤其长序列下,时间都被"在 HBM 里搬 N×N 矩阵"吃掉,而不是真正在算。

### 1.2 为什么不能直接 tiling:softmax 不结合

矩阵乘法可以 tiling 是因为**加法按块可结合**(C 的每个输出 tile = 相关 A/B tile 的部分和累加)。但 **self-attention 里有 softmax,它不结合**:

```
softmax(合并整个 S 行) ≠ 合并各块单独算的 softmax
```

如果你硬把一整行 S 切成几块、各自单独做 softmax,合的 `∑exp` 和 max 对不上,结果就是**错的(近似)**。这让"像 GEMM 一样按块算 attention"变得非平凡。

**解法就是 online softmax**:用**增量式**维护 running max `m` 和 running sum `l`,让每块可以"边算边把信息并入全局统计",最后得到**精确**的 softmax(见下节推导)。

> **前驱交叉引用**:这个"softmax 不必先看完整行"的观察早有谱系——Milakov & Gimelshein 的 online normalizer(arXiv:1805.02867,FA1 引用[60])提出增量归一化;Rabe & Staats《Self-Attention does not Need O(n²) Memory》(arXiv:2112.05682,FA1 引用[66])据此证明单 query attention 只需 **O(1) 额外显存**(lazy softmax + 在线 max 重归一),并给出 JAX 分块实现(O(√n) 显存,n=16384 时推理省 59×/反传省 32×)。**FA 的差异化**:R&S 优化"显存占用峰值"(速度与标准实现持平或略慢),FA 优化"HBM 访存次数"(2-4× 加速)——同一数学、不同工程目标。详见本域 [exact-attention-memory-lineage.md](../algorithm/exact-attention-memory-lineage.md)(含 FA1 附录 B.5 三点比较的提炼)。

### 1.3 核心机制:Online Softmax(让 softmax 可增量合并)

online softmax 的关键:维护**每行各两个标量(即 m、l 是长度为 Br 的行向量)**(running max `m`、running sum `l`),遍历 K/V 块时增量更新,并对已累加的旧输出做 rescale,最终得到**精确**的行 softmax。**注意不是单个全局标量——每行独立维护**。

**为什么式子长这样**(一行推演):
- 标准 softmax 对整个行:`p_i = exp(s_i - max_s) / Σ_j exp(s_j - max_s)`;
- 逐块遍历时,每块会遇到**新的更大的 max**。一旦 max 变大(`m_new > m_old`),**之前所有块已经按旧 max 算的 exp 值都要整体收缩** `exp(m_old - m_new)` 倍;
- 所以 running sum 要 rescale:`l_new = l_old · e^{m_old - m_new} + Σ_block e^{s_k - m_new}`;
- 已累加的输出版本同理:`O ← O · e^{m_old - m_new} + ...`(输出的每行按旧 max 的权重整体收缩)。

这正是论文里那段伪代码做的事(`li = li * exp(mi - mi_new) + sum(exp(Sij - mi_new))`),用**标量 rescale** 换来**块间精确合并**——softmax 在该增量形式下"变可结合"了。

### 1.4 完整算法:FA1 Forward 伪代码

> ⚠ 循环顺序说明:论文 Algorithm 1 的**外层是 K/V 块、内层遍历 Q 块**(FA2 才翻成内/外层)。下面为了直观按"外层 Q、内层 K/V"展示(**简化变体**,逻辑等价、更贴合"每行输出遇到的 KV 序列");要对读论文,注意其外层是 K/V 块。

> 为什么是 4d:Bc≈M/(4d) 来自 SRAM 预算推导——SRAM 需同时装 Q 块(Br·d)、K 块(Bc·d)、V 块(Bc·d)、S 分数块(Br·Bc)与输出 O 块(Br·d);取 Br=Bc 后,**Q/K/V/O 四块各占 Bc·d**(加 S 块 `Bc²`)约 **4·Bc·d + Bc² ≈ M**,解出 Bc ≈ M/(4d)(FA1 论文 Algorithm 1 的原始约束;4d 项主导)。

```
Input: Q,K,V ∈ R^{N×d}; SRAM 容量 M
Bc = ceil(M / (4d))           # K/V block 列数(论文 Algorithm 1 为 ceil;实现常简写为 floor)
Br = min(ceil(M/(4d)), d)     # Q block 行数(FA1 论文)
for i = 1..ceil(N/Br):        # (简化变体)外层:遍历 Q blocks
  Qi = Q[i...];  Oi = 0;  mi = -inf;  li = 0
  for j = 1..ceil(N/Bc):      # 内层:遍历 K/V blocks
    Kj, Vj = K[j...], V[j...] # 载入 SRAM
    Sij = Qi @ Kjᵀ * scale     # [Br,Bc] 分数(仅存 SRAM)
    mij = rowmax(Sij)
    mi_new = max(mi, mij)
    li = li * e^{mi - mi_new} + Σ e^{Sij - mi_new} # ① rescale running sum
    Oi = Oi * e^{mi - mi_new} + e^{Sij - mi_new} @ Vj # ② rescale+累加
    mi = mi_new
  Oi = Oi / li                     # 最终归一化(每行只除一次)
  write Oi                         # 只写一次,O(N×d)
```

>`Sij` / `e^{Sij}` 只存在于 SRAM,完整行 softmax 通过逐块 rescale(`l`、`Oi` 的标量/向量修正)在最外层结束一次性除 `l` 完成——不写完整的 N×N 矩阵到 HBM。

### 1.5 IO 复杂度:为什么快得有理有据

元素访问数口径(M = SRAM 容量,单位元素):

| 版本 | HBM 访存(元素访问) | 说明 |
|---|---|---|
| 标准 attention | Θ(Nd + N²) | 读 Q/K/V + 写读 S、P 两个 N×N 矩阵 |
| FlashAttention FA1 | Θ(N²d²/M) | 分块后,每个 Q/V 块在 SRAM 内复用,中间矩阵不落 HBM |

- **标准**:Θ(N²) 项是瓶颈(随 N 二次方);
- **FA1**:固定 d、M 常数下是 Θ(N²),与序列长度二次方同阶,但隐藏常数被 SRAM 复用压到最小;
- 对比口径**统一为元素访问数**,二者才有意义(quick-assistant 2026-09 评:不要用 Naive 的 Θ(N²d) 去对 FA 的 Θ(N²d²/M),单位不一致会导致自相矛盾)。

**关于"下界"的准确表述**(FA1 论文 Proposition 3):不存在对**所有** SRAM 容量 M ∈ [d, Nd] 都达到 o(N²d²/M) 访问的算法;当 **M ≥ Nd(SRAM 装得下全部 K/V)** 时,标准读取可达 Θ(Nd) 访问(此时不用分块、K/V 驻留 SRAM 复用),FA 的"每块复用"把 M < Nd 的场景也压到接近这个下界。注意:**不是"每行必须碰 N 个 K/V 列就是 N² 次 HBM 读"**——K/V 块在 SRAM 里可复用,分数依赖关系不等于 HBM 访问次数。

### 1.6 FA1 的遗留问题(为什么还要 FA2)

FA1 已达 IO 下界,但在 **A100 上只有 25-40% 的理论 FLOP 利用率**(最高 124 TFLOP/s)——IO 最优 ≠ 计算利用率最高。问题在工作划分:**FA1 的并行维度不足**——线程块**只沿 batch × head 维度划分、不沿序列维**,batch×heads 少(batch 小、序列长)时 grid 太小、喂不满 SM,occupancy 低;同时 FA1 的 warp 把 **K/V 沿序列维切给各 warp(split-K)**,每个 warp 产出一份全宽的部分输出,聚合要经 shared memory 归约,通信 + 同步多。FA2 逐个解决(§2)。

---

## 2. FA2:并行与 Work Partitioning——为什么快了 2×

FA1 已达 **IO 下界**(HBM 访存最优),但瓶颈不再是"搬多少数据",而是"算得不够满"——并行拓扑把 GPU 的资源没用满。**FA2 主要做两件事**(配合 CUTLASS 3.x/CuTe 重写,约 2× 加速),把利用率提到 **50-73%**、A100 上到 ~230 TFLOP/s:
1. **减少 non-matmul FLOPs**:把"每次迭代都对未归一化输出除 `ℓ`"**推迟到最终一次**(rescale 本身仍按 max 做,但省掉每步矩阵除 ℓ;详见 §2.2)——标量运算每步随块数线性累积,A100 上 non-matmul 比 matmul 贵约 16×;
2. **更好的 work partitioning**:threadblock/warp 的并行划分更贴合 GPU——**warp 不再各算不同 K/V 块(要通信合结果),而是各算同一 K/V 块的连续行(几乎无通信)**;并新增**序列维并行(沿 Q 切,免归并,见 §2.4)**。

> **心法**:IO-optimal 之后,下一个瓶颈是利用率。FA2 证明:attention kernel 的并行拓扑(threadblock/warp 怎么分活)对性能的影响,不亚于访存优化。

### 2.1 为什么 FA1 到 25-40% 就上不去了

FA1 的 IO 已最优,但计算没跑满,三个原因:

| 瓶颈 | 说明 |
|---|---|
| **大量 non-matmul FLOPs** | FA1 每次加载新 K/V 块都要对未归一化输出按 max 做矩阵 rescale 并**除一次 `ℓ`**,标量运算每行随块数线性累积;而 Tensor Core 的 matmul 吞吐远超标量(**A100:matmul 312 vs non-matmul FP32 仅 ~19.5 TFLOPs/s,贵约 16×**),这些 non-matmul 成了拖累 |
| **warp 间通信开销** | FA1 让 4 个 warp 各算不同的 K/V 块,最后要 **shared memory sync + reduce** 合并各行分数——warp 间大量通信 |
| **并行维度不足** | FA1 只在 **batch × head** 维度并行;**batch×heads 少**(如 batch 小且序列长)时 grid 喂不满 SM、occupancy 不足(batch 足够大时长序列并不缺线程块) |

FA2 逐个解决。

### 2.2 减少 non-matmul FLOPs:把除 ℓ 推迟到最终

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

### 2.3 更好的 Warp 划分:通信大减

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

### 2.4 序列维并行:切 Q(免合并)与切 KV(需归并)要分清

**容易混淆的关键结构**:FA2 论文的 threadblock 级序列并行是**沿 Q 行切**——`grid ≈ (num_q_blocks, batch, nheads)`,每个 block 算一段 **Q 行**,各行 softmax 独立,**block 之间"no communication at all"、根本不需要 reduction kernel**。这正是 FA2 在论文 §3.2 明确宣称的新增并行维:"The first version of FlashAttention parallelizes over batch size and number of heads... we now additionally parallelize over the sequence length dimension."——即 **FA1 只沿 batch×heads 并行,序列长而 batch×heads 少时喂不满 GPU;FA2 才新增沿序列维(把 Q 切块)的并行**。在此基础上,FA2 的另一关键改进在 **block 内部**:把 warp 划分从"各 warp 按 K/V 段切"(要跨 warp 合并)改成"各 warp 按 Q 行切"(行间独立、免通信),并利用 **Q 行天然可独立计算**这一性质充分并行:

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
- **切 KV = 各行分成多段,须按 LSE 加权合并**(就是 split-K 归并,类比 CUTLASS split-K/Stream-K,见 [cutlass tile-scheduler 篇](../../cutlass/concepts/tile-scheduler.md));
- **二者动机相同**:长序列/少 batch 时把更多的 threadblock 塞进 GPU(补 occupancy);

> 一句话:FA2 序列并行加的是"切 Q"(免归并);"切 KV 要归并"是 FlashDecoding/split-K 的事,别混。

> **防御性脚注(防再翻烧饼)**:FA1 开源实现(v1.0.x)源码里 `dim3 grid(params.b, params.h, params.num_splits)` 的**第三维是 num_splits 占用率启发式**(默认 1、上限 ~30),**不是** ceil(N/Br);Q 块在 kernel 内由 `for (l = blockIdx.z; l < steps; l += num_splits)` 串行遍历。"FA1 的 grid 第三维就是 ceil(N/Br)、早已沿 Q 并行"是误读开源实现的产物——论文口径(FA1 仅 batch×heads 并行)才是对的。

### 2.5 可复现参数速查(FA2 常见配置)

| 参数 | 取值(head_dim=128 时) | 说明 |
|---|---|---|
| Br(Q block 行) | 128 | SRAM 每块 Q 行数 |
| Bc(K/V block 列) | 128 | 每个 K/V 块列数(受 SRAM 预算约 4d 约束,见 §1.4) |
| Warp tile(每个 warp 的 Q 行) | ~32-64 | FA2 warp 沿 Q 行切分后的粒度 |
| MFU | ~50-73%(A100,**up to 72%** 为博客口径,随 head_dim/batch/config 变化) | 相对 A100 理论峰值 |

端到端:FA2 论文 Table 1 训练 196~225 TFLOPs/s(最高 72% MFU,GPT3-2.7B 8k);attention kernel 单独基准最高 ~230 TFLOPs/s(73% 理论峰值);**不同配置数字不同,以原文献口径为准**。

---

## 3. FA3:Hopper 异步内核——WGMMA + TMA + 计算通信重叠

FA3(2024.07,Hopper H100/H800)的目标是在 Hopper 上达到**接近 GEMM 的效率**。它用 Hopper 的**三件硬件特性**把 attention kernel 推满:

1. **异步 WGMMA**(warp-group MMA):一个 warp group(4 warps/128 线程)可**异步发起**矩阵乘,不用等它做完就能干别的;
2. **TMA**(Tensor Memory Accelerator):硬件单元**异步搬数据**(HBM→shared memory),不占 warp;
3. **计算-通信重叠 pipeline**:让 **softmax、matmul、TMA load 三者重叠**执行,配合 **warp specialization**(producer warp 搬数据、consumer warpgroup 算)。

外加 **FP8 低精度**(block quantization + incoherent processing 保精度):FP8 前向达 ~1.2-1.3 PFLOPS;BF16 相对 FA2 加速 1.5-2×。

> **心法**:FA1/FA2 解决"搬得少"(IO)和"算得满"(并行),FA3 再进一步——**在 Hopper 上让运算与搬运同时发生**,用异步**隐藏内存延迟**,让 GPU 计算单元几乎不空闲。

### 3.1 为什么需要异步:Ampere 的同步局限

FA2 在 Ampere(A100)用 `mma.sync`(同步 MMA),是**串行**的:

```
FA2 (Ampere, 同步 mma.sync):
  warp: mma.sync → S[0:16]     (等待完成)
  warp: softmax(S[0:16])        (只能串行)
  warp: mma.sync P@V → O[0:16]  (等待完成)
  # softmax 和 matmul 不能重叠
```

同步 `mma.sync` 指令**阻塞调用者**直到算完——并注意 **Ampere 的 `cp.async` 本来也是异步拷贝**(FA2 已用双缓冲重叠 HBM→SMEM 搬运与计算)。FA2 在 Hopper 上利用率上不去的**真正瓶颈是 GEMM 与 softmax 串行**、以及 cp.async 需要**每线程参与寻址和发射**(占 warp 发射槽与寄存器),无法做干净的 producer/consumer 分工——典型配置下 FA2 只有 **~35% 利用率**(Tri Dao 博客口径),TMA 单线程发射 + mbarrier 完成通知 + 硬件生成 tile 地址,才让**对称的 warp specialization(producer 搬 / consumer 算)** 成为可能。

FA3 换成 Hopper 的**异步指令**(异步 WGMMA + TMA),让"发起"和"等待结果"分离,中间就能插别的活——这是与 FA2 的关键分野。

### 3.2 三层异步:WGMMA、TMA、overlap

**异步 WGMMA(warp-group MMA)**:Hopper 的 `wgmma` 由**一个 warp group(4 warps = 128 线程)异步发起**:

```
FA3 (Hopper, 异步 WGMMA):
  wgmma.async(Q_block @ Kᵀ)   # 发起,立即返回(不阻塞)
  wait() → S                   # 等结果就绪
  softmax(S)                   # 同时已发起下一个 K/V 的 TMA load
  wgmma.async(P @ V)          # 发起,立即返回
```

`wgmma.async` **发出即返回**,计算单元可以继续发起下一次 matmul 或做 softmax,结果到了再 `wait()`——重叠由此产生。

> **典型 tile 配置**:FA3 常用 Q/K/V tile 128×128(/64×64 视 head_dim),CTA 内 1 producer + 2 consumer warpgroup(pingpong);`wgmma` 显式指令流是 `fence`(刷寄存器依赖)→ `commit_group`(提交批)→ `wait_group`(等整批完成),配合 `setmaxnreg` 给 producer/consumer 分配不同寄存器数。

**TMA(Tensor Memory Accelerator,Hopper 硬件搬运)**:

```
FA2 (cp.async, 需 warp 指令参与):
  issue cp.async(K→smem); cp.async_wait()   # 等;warp 被占用不能干别的

FA3 (TMA, 硬件单元,不占 warp):
  tma.load(K→smem, mbarrier=bar)   # 硬件搬,不占 warp
  mbarrier.wait(bar)                # K 就绪时通知
```

TMA 优势:
- **支持 2D/3D tensor tile 拷贝**:不用手动地址计算;
- **swizzle 布局**:配合 shared memory bank conflict 优化;
- **不占用 warp 资源**:搬运由硬件单元完成,腾出的 warp 去算。

**计算-通信重叠 pipeline**(FA3 的关键创新):让 **TMA load、WGMMA、softmax 三者重叠**:

```
时间轴 →
Stage1 TMA load:   |load K0V0| |load K1V1| |load K2V2|
Stage2 WGMMA:        |QK0ᵀ|      |QK1ᵀ|
Stage3 Softmax:                   |softmax|
Stage4 WGMMA:                           |P0V0|
```

- TMA 搬 K1 的同时,WGMMA 算 QK0;
- softmax 与下一个 matmul 重叠;
- **用 warp specialization**:producer warp 只做 TMA load,consumer warpgroup 做 WGMMA + softmax——两拨分工并发。

**结果**:GPU 计算单元几乎不空闲,attention 逼近 GEMM 效率(BF16 ~75% MFU;FP8 ~61-66% MFU,见 §3.3)。

### 3.3 FP8 低精度(低精度路径)

利用 Hopper FP8 Tensor Core。**FA3 的 FP8 不是简单把 QK 换 E4M3 就完事**(部署时最常见的坑),核心是两项保精度技术:

```
Q, K, V 各自 block quantization 到 FP8(按 128×128 或 1×128 块做 scale,scale 可与 RoPE 融合零开销)
S      = Q_fp8 @ K_fp8ᵀ        # FP8 matmul,FP32 accumulate
P      = softmax(S) → downcast 成 FP8 操作数   # 用 shuffle 指令做 accumulator→operand 布局转换
O      = P_fp8 @ V_fp8          # 第二个 WGMMA 也是 FP8 GEMM(不是 V 保持 BF16!)
```

**保精度两项官方技术(FA3 论文标题级贡献)**:
- **Block quantization**:Q/K/V 都按 128×128 或 1×128 块独立量化、每块一个 scale;避免全局量化被 outlier 拖垮;scale 可与 RoPE 融合、零额外开销;
- **Incoherent processing**:对 Q、K 各乘一个随机正交/Hadamard 矩阵(复杂度 O(d log d),**数学上不改变 attention 输出**——因为正交矩阵满足 H·Hᵀ=I,故 (QH)·(HᵀKᵀ)=Q·(H·Hᵀ)·Kᵀ=QKᵀ,分数内积不变),消除注意力分数里的 outlier,让 E4M3 能表示;
- 二者合计带来 **2.6× 的数值误差降低**。

**性能口径(注意别混)**:
- **FP16**:最高 ~740 TFLOPS(论文摘要 "up to 740 TFLOPs/s";博客 620-640 常规流水),约 ~75% 利用率(≈H100 FP16 dense 峰值 989 的 75%,摘要口径);
- FP8:~**1.2-1.3 PFLOPS** 前向,相对 H100 FP8 理论峰值约 **61-66% MFU**(不是 80%);
- **加速基线**:BF16 相对 FA2 实测 **1.5-2×(指 BF16 加速)**,不是 FP8 的收益。

### 3.4 实现细节与特性

- 基于 **CUTLASS 3.x / CuTe 模板**(C++ 模板,非 DSL;独立包 `flash_attn_3`);
- **Pingpong scheduling(FA3 论文招牌技术)**:CTA 内 **1 个 producer warpgroup + 2 个 consumer warpgroup**,用 `bar.sync` 强制交错,使 WG1 的 softmax 落在 WG2 的 GEMM 影子里(FP16 前向 570→620-640 TFLOPS;论文 §4.2 ablation 同源 570→661 TFLOPS 是 pingpong + warp specialization 合计口径,勿与单独 pingpong 混淆);
- **前向 2-stage / 反向 3-stage pipeline**:论文的 2-stage 特指 **GEMM-softmax 交错流水**(Algorithm 2,跨迭代用寄存器缓冲打破 S→softmax→P 的串行依赖),不是简单的 load-then-compute;反向 3-stage 更复杂 overlap;
- **反向:producer + consumer + dQ writer 三角色**,仍 per-tile 重算 attention matrix + 3-stage overlap 训练读者最易在这踩坑——反向不只是"重叠",多一个写 dQ 的角色参与流水;
- **动机数字(为什么必须隐藏 softmax)**:H100 FP16 Tensor Core matmul 989 TFLOPS vs 特殊函数单元(exp)吞吐极低(论文/博客估算 ≈3.9 TFLOPS 量级,比 matmul **低 2-3 个数量级**),seqlen 8K/hd128 下 exp 可占 ~50% 周期,FP8 下更糟——所以要把 softmax 藏进 GEMM 的影子;
- 配合 `setmaxnreg` 寄存器重分配(producer 少寄存器、consumer 多);WGMMA 用 `fence/commit_group/wait_group` 显式指令流;
- 特性:FP16/BF16 前后向、**FP8(E4M3)前向**、varlen、**paged KV cache**、**SplitKV**(长序列把 KV 维 split,最后 reduce)。

> **当前状态(2026 时效)**:FA3 的 `hopper/` 目录长期停留在 **beta**(至 beta25);**Blackwell SM100 不支持 WGMMA,改用第五代 Tensor Core 指令 tcgen05(其异步矩阵乘指令 UMMA,Unordered Multi-precision Matrix Multiply),故 FA3 无法上 B200**——由 FA4(CuTe DSL)接管(B200 用户走 FA4)。

---

## 4. FA4:Blackwell(CuTe DSL + 异步 UMMA)

FA4(2026,MLSys,arXiv:2603.05451)聚焦 **Blackwell(B200/GB200)**:瞄准"张量核翻倍但 smem 带宽/exp 单元没跟上"的**非对称硬件缩放**:

- **完全异步 MMA + 更大 tile**:SM100 不支持 WGMMA,改用第五代 Tensor Core 指令 **tcgen05**(异步矩阵乘 UMMA);
- **软件模拟 exp 与条件 softmax rescale**(砍 non-matmul——延续 FA2"non-matmul 是隐藏成本"的主线);
- **TMEM + 2-CTA MMA**:减 smem 流量与反传 atomic add;
- 整条用 **CuTe DSL(Python)** 实现:BF16 B200 上 up to **1613 TFLOPS / 71% util**,比 cuDNN 9.13 快至 1.3×、比 Triton 至 2.7×,编译比 C++ 模板快 20-30×。

**实现逻辑(论文 §3-4 + GitHub 实现,至少对齐 FA1 的信息密度)**:

1. **2-CTA MMA(两个 CTA 协作算一个 tile)**:Blackwell 的 tcgen05/UMMA 允许**两个相邻 CTA 组成一对(cluster)** 协同发射同一条 MMA,各自把 M 维切成上下两半。收益:**TMEM 容量加倍**(accumulator 放在 TMEM 而非寄存器,一个 CTA 的 TMEM 有限,2-CTA 拼接把可驻留的 tile 尺寸翻倍)、smem 带宽减半(两 CTA 共享同一份 K/V tile,读一次各用一半)。这是 FA4 应对"smem 带宽没跟上张量核"的**首要结构手段**。
2. **TMEM 布局**:accumulator(C 的部分和)不落在寄存器,而是驻留在 **TMEM**(Blackwell 新增的 Tensor Memory),tcgen05 的操作数/累加器直接读写 TMEM。好处:寄存器压力大减 → 单 CTA 能塞更大的 Q/K/V tile;代价:TMEM 容量有限、且 TMEM↔寄存器/共享的搬运要显式管理(类似 Hopper 的 smem,但更贴近 MMA 单元)。
3. **软 exp + 软 softmax rescale(砍 non-matmul 的核心)**:Hopper 上 softmax 的 exp 由特殊函数单元算(慢,见 §3.4 的 exp≈3.9 TFLOPS),FA4 用 **软件多项式近似 exp(软 exp)** 放进普通 FP 流水线;同时把 online softmax 的 rescale 做成**条件触发**(soft rescale)——不是每块都 rescale,而是用 running max 的差值判断是否越过阈值才 rescale(类比 FlashMLA sparse 内核的 RESCALE_THRES,见 [deepseek-infra FlashMLA](../../deepseek-infra/concepts/flashmla.md))。二者把原本昂贵的 non-matmul 降到最低,让"算满"进一步逼近 GEMM。
4. **与前代的对应关系**:FA1 的"搬得少"(tiling/online softmax)、FA2 的"算得满"(work partitioning)、FA3 的"算搬同时"(WGMMA/TMA 异步)在 FA4 上全部延续,只是载体换成 Blackwell 的 tcgen05/TMEM/2-CTA——**主线不变,硬件载体换代**。dtype 走 BF16(1613 TFLOPS)而非 FP16,因为 Blackwell 数据中心路径以 BF16 为主。

> 一句话:FA1 解决"少搬数据"(IO-optimal);FA2+ 解决"算得更满"(并行/利用率/软硬件特性)。两者都在 FlashAttention 家族里层层叠加,FA4 把这条主线延伸到 Blackwell。

---

## 5. 反向重算(Recomputation):训练省显存的关键

TL;DR 里说"显存 O(N²)→O(N)",这个 O(N) 的胜利**来自前向只存统计量、反向重算**:

- **前向不存中间矩阵**:FA 前向只保存每行的 **m、l(FA1)或合并的 logsumexp LSE = m + log l(FA2 起,少一次 HBM 读)** 等 O(N) 统计量,不存完整的 N×N attention matrix;
- **反向重算 P**:反向时重新加载 Q/K/V 块,用存的 m、l **重算 attention scores/P**(而不是从显存读回),代价是约 **2× 的重复 FLOPs**;
- **为什么值得**:attention 是 memory-bound——重算的 FLOPs 便宜(HBM 带宽贵得多),显存从 O(N²) 省到 O(N),换来长序列训练/推理可行;
- 反向本身也 pass 化:FW 的 tiling 结构在 BW 同样能复用,配合(§3.4)FA3 的多级 pipeline 重叠;FA2 backward 沿 KV 块切、固定外层循环以避免对 dQ 的 atomic add,且反向只存 logsumexp L = m + log ℓ(不再同时存 m、ℓ),省 HBM 写 + 标量运算——这是全库到处传 `lse` 的由来。

> **一句话**:FA 用"前向存统计量 + 反向重算"换取显存 O(N);算力换显存,在 memory-bound 的 attention 里是合算的买卖。

---

## 6. 实践要点(checklist,三代合并去重)

**选型与适配**

1. **判断是否该上 FA**:attention 是 memory-bound;先用 roofline 估算算术强度,若低于脊点(内存受限)→ FA 收益最大。长序列(N 大)尤其明显。
2. **先看 mask 是否适配(选型第一问)**:FA 的 fused kernel 只支持 **有结构、可跳块** 的 mask(因果 causal / sliding window / ALiBi / padding);传入**任意 dense soft-mask** 需逐块读 mask、甚至回退到物化 S,FA 优势尽失——先确认你的 mask 类型再选 FA。
3. **按 GPU 选代次**:A100(Ampere)用 FA2;**FA3 只在 Hopper(H100)有效**——依赖 WGMMA/TMA(SM90);**Blackwell SM100 不支持 WGMMA(改 tcgen05),FA3 无法上 B200,由 FA4(CuTe DSL)接管**。
4. **FA2 优先于 FA1(但 FA1 不算"过时")**:FA1 仍是 IO 最优算法框架,FA2 在其上优化利用率(50-73% MFU);计算密集且长序列的 workload,FA2 收益(2×)更值得。**规避 FA1 低 occupancy**:FA1 只在 batch×head 维度并行,长序列小 batch(batch×heads 少)时 grid 喂不满 SM → 换 FA2 或开 FA2 的序列维并行。

**正确性与数值**

5. **记得它是 exact(数学精确,非 bitwise)**:FA 输出与标准 softmax **数学上相同、无近似**(不是稀疏近似),但浮点累加顺序不同,结果与朴素实现有 **ulp 级数值差异**(fp16 下更明显)——换了 kernel 后 loss 曲线不完全 bitwise-repro 是正常,不是 bug。
6. **数值稳定性**:online softmax 的 running max 保证 `exp(s-m)` 不溢出;rescale 必须保留——去掉它**数学上错**(各块分母不一致),不是数值精度问题。
7. **FP8 必须做 block quantization + incoherent processing**:FA3 的 FP8 是 Q/K/V 都按块量化 + 对 Q/K 乘 Hadamard 消 outlier(2.6× 误差降低)——不是简单把 Q 换 E4M3;跳过这两项,outlier 会让精度崩。
8. **性能口径别混**:FA3 BF16(~75% 利用率)相对 FA2 加速 1.5-2×;FP8 是 ~1.2-1.3 PFLOPS(61-66% MFU)——两条曲线是不同维度。FA2 的 MFU 随 head_dim/batch/config 变化(up to 72% 为博客口径)。

**性能调优**

9. **显存收益来自反向重算**:无需存 N×N 中间矩阵,前向只存 LSE/m,l 统计量;训练长序列显存峰值 O(N²)→O(N),这是它让长 context 训练可行的关键(见 §5)。
10. **Block 大小 Br/Bc 的取舍**:受 SRAM 容量 M 和 head dim d 约束(`Bc≈M/(4d)`、`Br≈min(⌈M/(4d)⌉,d)`);**实数参考:A100 上 d=64 时 Br/Bc≈128、d=128 时≈64**。Br/Bc 越大,每次 HBM 加载复用越多,但 SRAM 占满 → 调优平衡点。**口径注意**:`Bc≈M/(4d)`(d=128 时≈64)是 **FA1 口径**(4d 预算下 K/V+O 都进 SRAM);FA2 把 Q 常驻寄存器、不再占 SRAM,block 尺寸才放宽到 128×128(见 §2.5 FA2 常见配置)——两处不矛盾。
11. **non-matmul FLOPs 是隐藏的坑**:Tensor Core 时代 non-matmul 比 matmul 贵 ~16×(A100);任何"每块更新都除 ℓ"的实现都在浪费算力——FA2 把除 ℓ 推迟到最后。
12. **warp 按 Q 行切分是 FA2 的关键**:attention 行间独立 → 按行分 warp 免通信;写自定义 attention kernel 时优先模仿这个划分。
13. **序列并行分清切 Q / 切 KV**:沿 Q 切 → 各行独立、免归并;沿 KV 切 → 需按 LSE 加权合并(split-K/FlashDecoding)。batch×head 够大时可不拆,免得无谓归并。**长序列 decode 用 SplitKV**:KV 维 split 到多 block 增高并行,注意有最终 reduce(同 cutlass split-K)。
14. **causal mask 也是 non-matmul 削减**:对每行只要算 1 个被 mask 的块/可整块跳过,SAVE 大量无用计算——别忽略这一半优化。
15. **pipeline 级数要调**:FA3 前后向 2/3-stage 是 GEMM-softmax 交错流水的骨架;**改动 barrier/TMA mbarrier、WGMMA 的 commit_group/wait_group 的任何同步次序都可能引入 race**,务必参照 cutlass 流水篇的同步模型。
16. **warp specialization 是 FA3 核心且是"1 producer + 2 consumer"的 pingpong**:producer/consumer 分工交错让 softmax 藏进 GEMM 影子——读 FA3 kernel 先看这层。
17. **反向也讲 work partitioning / 也重叠**:FA2 backward 沿 KV 块切、只存 LSE(见 §5);FA3 反向靠 3-stage pipeline 重算 attention matrix 来 overlap——训练场景它同样重要。
18. **精细化路线**:要更高利用率先看 **work partitioning**(FA1→FA2 的主改进:threadblock/warp 划分、层级并行),再看硬件特性(TMA/WGMMA/FP8)。
19. **配合工具链**:FA2/FA3 基于 CUTLASS 3.x/CuTe(见 [cutlass 域](../../cutlass/index.md));FA4 走 CuTe DSL(Python)。

---

## 7. 溯源(三篇合并)

**FA1(IO-aware、tiling、online softmax、Θ(N²d²/M))**
- FA1 论文: Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022, https://arxiv.org/abs/2205.14135
- UW CSE 599M 讲义 *From Online Softmax to FlashAttention*(softmax 不结合→online softmax 的推导): https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
- Online softmax 前驱:Milakov & Gimelshein, *Online normalizer calculation for softmax*, https://arxiv.org/abs/1805.02867 (2018)
- O(1) 显存前驱(FA1 [66],附录 B.5 专节比较):Rabe & Staats, *Self-Attention does not Need O(n²) Memory*, https://arxiv.org/abs/2112.05682 → 本域详注 [exact-attention-memory-lineage.md](../algorithm/exact-attention-memory-lineage.md)

**FA2(work partitioning、基于 CUTLASS 3.x;§3.2 减 non-matmul FLOPs、§3.3 序列维并行)**
- FA2 论文: Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, ICLR 2024, https://arxiv.org/abs/2307.08691
- Tri Dao 的 FA2 博客(25-40%→72% MFU、A100 230 TFLOP/s、warp 划分示意图;`flash-atttention-2` 为站点真实拼写): https://princeton-nlp.github.io/flash-atttention-2/

**FA3(§3.2 pingpong scheduling、§3.3 block quantization + incoherent processing 的 128×128/1×128 粒度、§5 性能与 2.6× 误差)**
- FA3 论文: Dao et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, NeurIPS 2024, https://arxiv.org/abs/2407.08608
- Tri Dao FA3 博客(overlap、WGMMA/TMA、FP8、pingpong 拓扑;570→620-640 TFLOPS、exp≈3.9 TFLOPS 等博客口径数字): https://tridao.me/blog/2024/flash3/

**FA4(Blackwell)**
- FA4 论文(MLSys 2026): arXiv:2603.05451

**实现仓库 / 跨域**
- 官方实现仓库(含 FA1/FA2 kernel 与 `benchmarks/` 复测): https://github.com/Dao-AILab/flash-attention
- 相关:CUTLASS 领域的 WGMMA/TMA/epilogue 篇(同硬件特性,跨参考,见 [cutlass/concepts/hopper-wgmma.md](../../cutlass/concepts/hopper-wgmma.md))
