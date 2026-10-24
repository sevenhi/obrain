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
  - "code: github.com/Dao-AILab/flash-attention CuTe 后端源码一手实读 (HEAD 98c1cee, 2026-10-24) | flash_attn/cute/config.py、interface.py、flash_fwd_sm90.py、flash_fwd_sm100.py(NOTE [SM100 forward coordinate spaces])、tile_scheduler.py(NOTE [SM100 forward work coordinates] / CLC 与 static persistent 的坐标契约)、flash_bwd_sm100.py、flash_bwd_preprocess.py、flash_bwd_postprocess.py、pack_gqa.py、seqlen_info.py、buffer 相关的 tensormap workspace 逻辑"
  - "code: github.com/NVIDIA/cutlass | cutlass/tensor_utils/tensormap.py 的 TensorMapManager / TensorMapUpdateMode(GMEM 模式)"
date: 2026-10
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

下面这张交互图概括了四代内核的演进架构与各代关键实现技术:

<iframe src="../../../diagrams/flash-attention-implementation.architecture.html"
        style="width:100%;height:640px;border:1px solid #ddd;border-radius:8px"
        loading="lazy" title="FlashAttention 四代内核架构交互图"></iframe>

---

## 1. FA1:IO-aware tiling 与 Online Softmax

### 1.1 先看问题:标准 Attention 为什么慢

下图展示标准 attention 的三步流水,以及 S/P 两个 N×N 中间矩阵如何被迫在 HBM 里来回复制——这是 FlashAttention 要解决的根本问题:

<div class="archify-figure">
<iframe src="../../../diagrams/fa-standard-attention-hbm.dataflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="标准 Attention 的 HBM 访存瓶颈"></iframe>
</div>

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

下图用一条因果链说明"为什么不能像 GEMM 那样直接切块"——错在 softmax 的全局依赖,解在 online softmax:

<div class="archify-figure">
<iframe src="../../../diagrams/fa-why-not-tiling.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="softmax 不可分块的因果链"></iframe>
</div>

矩阵乘法可以 tiling 是因为**加法按块可结合**(C 的每个输出 tile = 相关 A/B tile 的部分和累加)。但 **self-attention 里有 softmax,它不结合**:

```
softmax(合并整个 S 行) ≠ 合并各块单独算的 softmax
```

如果你硬把一整行 S 切成几块、各自单独做 softmax,合的 `∑exp` 和 max 对不上,结果就是**错的(近似)**。这让"像 GEMM 一样按块算 attention"变得非平凡。

**解法就是 online softmax**:用**增量式**维护 running max `m` 和 running sum `l`,让每块可以"边算边把信息并入全局统计",最后得到**精确**的 softmax(见下节推导)。

> **前驱交叉引用**:这个"softmax 不必先看完整行"的观察早有谱系——Milakov & Gimelshein 的 online normalizer(arXiv:1805.02867,FA1 引用[60])提出增量归一化;Rabe & Staats《Self-Attention does not Need O(n²) Memory》(arXiv:2112.05682,FA1 引用[66])据此证明单 query attention 只需 **O(1) 额外显存**(lazy softmax + 在线 max 重归一),并给出 JAX 分块实现(O(√n) 显存,n=16384 时推理省 59×/反传省 32×)。**FA 的差异化**:R&S 优化"显存占用峰值"(速度与标准实现持平或略慢),FA 优化"HBM 访存次数"(2-4× 加速)——同一数学、不同工程目标。详见本域 [exact-attention-memory-lineage.md](../algorithm/exact-attention-memory-lineage.md)(含 FA1 附录 B.5 三点比较的提炼)。

### 1.3 核心机制:Online Softmax(让 softmax 可增量合并)

下图展示 m/l/O 三元组如何在每个 KV 块上增量更新——这是 FA1 能分块计算的根本原因:

<div class="archify-figure">
<iframe src="../../../diagrams/fa-online-softmax.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="Online Softmax 增量更新流程"></iframe>
</div>

online softmax 的关键:维护**每行各两个标量(即 m、l 是长度为 Br 的行向量)**(running max `m`、running sum `l`),遍历 K/V 块时增量更新,并对已累加的旧输出做 rescale,最终得到**精确**的行 softmax。**注意不是单个全局标量——每行独立维护**。

**为什么式子长这样**(一行推演):
- 标准 softmax 对整个行:`p_i = exp(s_i - max_s) / Σ_j exp(s_j - max_s)`;
- 逐块遍历时,每块会遇到**新的更大的 max**。一旦 max 变大(`m_new > m_old`),**之前所有块已经按旧 max 算的 exp 值都要整体收缩** `exp(m_old - m_new)` 倍;
- 所以 running sum 要 rescale:`l_new = l_old · e^{m_old - m_new} + Σ_block e^{s_k - m_new}`;
- 已累加的输出版本同理:`O ← O · e^{m_old - m_new} + ...`(输出的每行按旧 max 的权重整体收缩)。

这正是论文里那段伪代码做的事(`li = li * exp(mi - mi_new) + sum(exp(Sij - mi_new))`),用**标量 rescale** 换来**块间精确合并**——softmax 在该增量形式下"变可结合"了。

### 1.4 完整算法:FA1 Forward 伪代码

下图展示 FA1 前向的双层 tiling 结构——外层 Q 块只读一次,内层 K/V 块逐块载入 SRAM,分数与 softmax 全程不出片上:

<div class="archify-figure">
<iframe src="../../../diagrams/fa-fa1-tiling.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="FA1 双层 tiling 循环结构"></iframe>
</div>

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

下图对比 FA1 与 FA2 在每个 KV 块上的更新写法——FA2 省掉的是"每步除 ℓ"这次矩阵级标量运算:

<div class="archify-figure">
<iframe src="../../../diagrams/fa-fa2-defer-divide.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="FA1 每步除 ℓ vs FA2 推迟除 ℓ"></iframe>
</div>

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

下图对比两种 warp 分工的通信路径:FA1 按 K/V 切要跨 warp 归约,FA2 按 Q 行切几乎零通信:

<div class="archify-figure">
<iframe src="../../../diagrams/fa-warp-partition.architecture.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="FA1 split-KV vs FA2 按 Q 行切"></iframe>
</div>

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

下图并排展示两条序列并行路径——切 Q 各行独立直接写,切 KV 必须按 LSE 加权归并:

<div class="archify-figure">
<iframe src="../../../diagrams/fa-splitq-vs-splitkv.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="切 Q 免归并 vs 切 KV 需归并"></iframe>
</div>

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

下图用时间轴展示 FA3 的三层异步流水——TMA 搬数据、WGMMA 算矩阵乘、softmax 三者重叠,计算单元几乎不空闲:

<div class="archify-figure">
<iframe src="../../../diagrams/fa-fa3-async-pipeline.waterfall.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="FA3 三层异步重叠时序"></iframe>
</div>

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

下图展示 FA3 FP8 路径的两项保精度技术——block quantization 和 incoherent processing 分别在量化和预处理阶段起作用:

<div class="archify-figure">
<iframe src="../../../diagrams/fa-fp8-precision.dataflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="FP8 block quantization + incoherent processing"></iframe>
</div>

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

> **当前状态(2026 时效)**:FA3 的 `hopper/` 目录长期停留在 **beta**(至 beta25);**Blackwell SM100 不支持 WGMMA,改用第五代 Tensor Core 指令 tcgen05(其异步矩阵乘指令 UMMA——CUTLASS 对 tcgen05.mma 的称呼,官方未展开全称),故 FA3 无法上 B200**——由 FA4(CuTe DSL)接管(B200 用户走 FA4)。

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

## 4.5 2026-10 的 CuTe 后端工程增量(上游 10-12 → 10-24,HEAD `98c1cee`)

FA4 的 CuTe DSL 后端(同时服务 Hopper/SM90 与 Blackwell/SM100)在 2026-10 有一批**不是新算法、但决定"能不能上线"**的工程改动。它们比论文更能说明真实部署里会踩什么,故单列一节。

### 4.5.1 一类反复出现的 bug:未用量 KV 行的 NaN 污染(`0 × NaN = NaN`)

下图按因果链展示 NaN 如何绕过 mask 污染输出,以及前向/反向两个独立的感染路径:

<div class="archify-figure">
<iframe src="../../../diagrams/fa-nan-pollution.sequence.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="NaN 污染因果链与前向/反向路径"></iframe>
</div>

现网 KV cache 是**复用**的:paged/ragged 布局里 `[seqlen_k, tile_n)` 这段"本序列用不到的行"保留的是上一轮或 padding 的脏数据,完全可能是 NaN。直觉认为"mask 掉就没事",但**mask 只能让系数为 0,消不掉乘法本身**:

| 位置 | 计算 | 后果 |
|---|---|---|
| 前向 | `O = P @ V`,P 在尾行被 mask 成 0 | 仍逐元素读 V 尾部 → `0×NaN=NaN` → **输出 NaN**(K 尾 NaN 无害,它在 exp 前被 mask 成 −inf) |
| 反向 | `dS` 被 mask 成 0,但 `dP = dO @ Vᵀ` 读 V 尾、`dQ = dS @ K` 对整个 K tile 求和 | K 尾**或** V 尾有 NaN 就让 `dQ` 全 NaN,而 `dK/dV` 仍有限 → **与前向独立**,前向干净不代表反向干净 |

所以正确做法不是改 mask,而是**在加载层面把 tile 上界收缩到 `seqlen_k`**,让越界行由 TMA 自己补 0(zero-fill),根本不进 SMEM。上游为此换了三种写法(#2983 → #3012 → #3013 → #3071 逐步收敛):

| 手段 | 做法 | 代价/失败模式 |
|---|---|---|
| 5-D wraparound descriptor | 把 `(rows, features)` 伪装成高维张量,用维度上界让 TMA 判越界 | **足迹无界**:TMA 会预取 box 之后最多 ~128 KB,遇未映射页即 `cudaErrorIllegalAddress`;PyTorch caching allocator 通常把这块藏起来,于是表现为"偶发、与执行顺序相关"(issue #3068) |
| ptr_shift | 把 descriptor 的 base 前移 | base 落到已映射区之外,load 直接 fault——**只适合 store 侧**(dK/dV/O 的写出仍用它) |
| **per-CTA GMEM descriptor** | 每个 CTA 一份 descriptor,进 work tile 时按该 batch 的 `seqlen_k` 改 rows extent | 需要 workspace、fence/prefetch 协议与上界校验(见下) |

最终实现的骨架(`flash_attn/cute/flash_fwd_sm90.py`,SM100 同构):

```python
# 每个 CTA 一份 descriptor,放在 gmem workspace 里(128 B/槽)
tensormap_manager = TensorMapManager(TensorMapUpdateMode.GMEM, 128)
cta_linear_idx = bidx + bidy * gdx + bidz * gdx * gdy
if cta_linear_idx >= cute.size(tensormap_workspace, mode=[0]):
    cute.arch.inline_ptx("trap;")           # workspace 不够就硬失败,不静默越界
tensormap_V_gmem = tensormap_manager.get_tensormap_ptr(workspace[cta_linear_idx, None].iterator)
tensormap_manager.init_tensormap_from_atom(tma_atom_V, tensormap_V_gmem, 0)   # warp 0 单线程 copy
tensormap_manager.fence_tensormap_initialization()                            # fence.acq_rel.cta

v_desc_batch_idx = Int32(-1)
while work_tile.is_valid_tile:
    batch_idx = work_tile.tile_idx[2]
    if batch_idx != v_desc_batch_idx:            # 只在 batch 变化时改一次,不是每 tile
        v_rows_end = max(offset_k + seqlen_k if packed else seqlen_k, 1)
        tensormap_manager.update_tensormap((mV_bounded,), (tma_atom_V,), (tensormap_V_gmem,), 0, (None,))
        tensormap_manager.fence_tensormap_update(tensormap_V_gmem)   # fence.proxy.tensormap::generic.acquire
        cute.arch.inline_ptx("prefetch.tensormap [{$r0}];", read_only_args=[tensormap_V_gmem.toint()])
```

反向要同时管 K、V、Kt **三份** descriptor(`workspace` 形状 `(n_ctas, 3, 16) int64`;前向是 `(n_ctas, 16)`),因为 `dP = dO@Vᵀ` 与 `dQ = dS@K` 都要读尾行。

**工程教训(比这个 bug 本身可复用)**:
- **workspace 上界要自己推一遍**:前向 grid 上界 `rows // tile_m + 2 * batch + 1`(`2` 来自变长调度器每 batch 的 padding tile,cluster 尺寸与 Q stage 数都被这条覆盖);反向 `total_k // n_block_size + 2 * batch + 1`。上游最初漏算 padding,H100 上 `B=49, seqlen_q=128, H=6` 排出 **582 个 CTA 却只有 302 个槽**,多出的 CTA 改写别的 descriptor 后照旧加载——**这类"算错容量"的 bug 不会报错,只会偶发错数**。
- **block-sparse 路径要单独补 fence**:SM100 前向把 descriptor 的 acquire fence 推迟到首个 V load 前,而 block-sparse 走的是另一条 producer 函数,永远到不了那行 → 持久 CTA 会拿上一序列的 extent 加载。
- **验证方法**:"偶发非法地址"先把 caching allocator 关掉(`PYTORCH_NO_CUDA_MEMORY_CACHING=1` + 子进程单序列)复现,再把 V 放到 VMM 映射尾部、后面留一段**保留未映射的 guard range**——这样"改前每次必 fault、改后必过",测试才立得住。
- **性能代价可接受**:B200 上 194 个正确性用例通过,16 组对照 × 3 次运行,最大延迟增幅 **2.23%**。

### 4.5.2 typed forward config(#2732):把"调度策略"变成可校验的数据

`flash_attn/cute/config.py` 新增不可变、全解析的 `FwdConfig`(`device_capacity / tile_m / tile_n / mma_pv_is_rs / intra_wg_overlap / q_stage / use_clc_scheduler / is_static_persistent / use_tma_o / use_s_ping_pong / num_splits / use_2cta_instrs / registers`),把主机侧选择器输入隔离成 `FwdHeuristicInputs`;`registers` 按架构给 `Sm90Registers(mma, producer)` 或 `Sm100Registers(softmax, correction, other)`。

三条设计约束值得抄到任何 kernel 自动调优代码里:
1. **显式配置被"校验"而不是"归一化"** —— 传了非法组合就报错,不静默改成"能跑的组合"(否则调优结果与配置永远对不上);
2. **选择器不许读 tensor、不许同步 CUDA、不许分配显存**(纯 host 计算,可缓存、可复现);
3. **每个调度特性只写一条 SM100 支持谓词,选择与校验共用** —— 防止"选得出来但跑不了"。

### 4.5.3 hd256 反向统一到通用 SM100 内核(opt-in,#2965/#2966)

head_dim=256 原本走独立的 dQ / dKdV 双内核,现在可以合并到通用 `FlashAttentionBackwardSm100`(环境变量 `FLASH_ATTENTION_HD256_GENERIC_BWD=1` 灰度,默认仍是专用内核):2CTA、`tile_m=128`、每 CTA 64 行 KV tile(`M=64` 的 TMEM 布局)、serial 2CTA 调度。

- **TMEM 预算正好用满 512 列**:`dV 128 | dK 128 | S 64(P 别名) | dP 64(dS 别名,与 dQ 尾重叠) | dQ 128`;
- **SMEM 231,424 B / 232,448 B**——只剩 16 KB 给 dQ 归约暂存(两个 16 列 stage),"把 SMEM 抠到字节"是这类内核的常态;
- 关键收益来自**删掉交换暂存**:M=64 布局下,对端需要的 dS 那一半可以直接从本 CTA 的 `sdS_dK`(对端的 dSᵀ,K-major)读——它正好落在对端 dQ A 操作数的半块布局里,省掉一份交换缓冲和 `dQaccum_empty` 等待,而那个等待曾把整条 dQ 归约链放到关键路径上、在 hd256 上吃掉约 **17%**;
- **实测(B200 @1400 MHz,vs 专用双内核)**:MHA h8 在 1k 持平、2k–32k 快 **3–5%**(causal 6–11%);varlen h8 在 2k 持平、8k/32k 快 **4–17%**;GQA(24/4)在 8k 持平、32k 快 4–5%,但 **2k 慢 22–29%**(**前提:这是 #2966 的口径,当时通用内核还没有反向 pack_gqa**——专用 dKdV 内核在 kernel 内循环 group 的 q-head,通用内核还没做这个循环;#2967 把 q-head 折进 Q tile 正是补上这个循环,之后同一"通用内核 GQA 小序列"现象翻正,见 §4.5.4);
- **默认行为零变化**:36 组(2CTA + 1CTA,280 个文件)的 CUBIN/PTX 与父 commit **逐字节相同**。

### 4.5.4 pack_gqa:把 GQA 的 q-head 折进反向的 Q tile(#2967)

上游 commit `bc52a40`(2026-10-23,`[CuTe, SM100] Unify hd256 backward (4/7)`,10 files +709/−116)。这是 hd256 反向统一系列的第 4 步,但机制上独立成篇:**SM100 反向的 GQA 从"一个 Q head 一个 tile + 跨 group fp32 归约"改为"一个 KV head 一个 tile,qh 个 Q head 折进 tile 的行维"**——正是 §4.5.3 里"通用内核还没做的那个 kernel 内 q-head 循环"的补法(循环不在 kernel 内做,而是折进行维由 MMA 天然吃下)。

#### ① 问题与代价:原来 GQA 反向是怎么做的

GQA(`num_head/num_head_kv = qh > 1`)时,SM100 反向原路径是**一个 Q head 一个 tile**:同一 KV head 的 qh 个 Q head 各自算出自己那一份 dK/dV 贡献,再跨 group 归约。代价三条:

1. **K/V 被读 qh 次**——K/V 是反向的访存大头(反向要重算 P,每 tile 都要 K/V);
2. **dK/dV 必须落 fp32 累加缓冲**(HBM),再加**每张量一趟 postprocess kernel**(fp32→bf16)+ **semaphore 握手**(`cp.reduce.async.bulk` 的完成信号);
3. 归约走 **cluster 内跨 CTA 通信**,压在 dQ 归约链旁边抢资源。

前向早已有解:`pack_gqa` 把 qh 个 Q head 折进 **Q tile 的行维**(packed 行 r = 第 `r % qh` 个 head、第 `r // qh` 个 token 的行),一个 tile 就覆盖一个 KV head 的全部 Q head。本 commit 把这套搬到反向。

| | unpacked(原路径) | packed(#2967) |
|---|---|---|
| tile 划分 | 一个 Q head 一个 tile | 一个 KV head 一个 tile,行维含全部 qh 个 Q head |
| K/V 加载 | 每 KV head 读 **qh 次** | 每 KV head 读 **1 次** |
| dK/dV 累加 | fp32 HBM 累加缓冲 + `cp.reduce.async.bulk` 跨 CTA 归约 | **留在 TMEM**,组内 direct store |
| postprocess | dK/dV 各一趟 fp32→bf16 kernel | **无**(dK/dV 不需要;接口里 `dKV_postprocess = qhead_per_kvhead > 1 and not use_dedicated_hd256_kernel and not bwd_pack_gqa`) |
| semaphore | dK/dV 归约握手 | **无 dK/dV semaphore**;deterministic 模式只剩每 `(KV head, packed m block)` 一个 dQ semaphore 槽 |
| LSE/dPsum/dQaccum | 按 Q head 分配 | **按 KV head 分配、按 packed 行序存放** |
| dQ 写出 | postprocess 读 dQaccum | 直接经 `pack_gqa_layout` 视图 `(b, (qh, s), h_kv, d)` 写 dQ,**没有 unpack pass** |

即"packed 省掉的四件事":**K/V 重复加载、fp32 dK/dV 累加缓冲、per-tensor postprocess kernel、dK/dV semaphore 握手**。

#### ② 机制推导:改动落在哪几层

- **张量视图层**:Q / dO(以及 2CTA 的转置视图)都变成 per-head 张量的 `pack_gqa_layout` 视图;`(qh, seqlen)` 这个行模式作为**额外一维 TMA box**。`make_packgqa_tiled_tma_atom`(pack_gqa.py)先把 head/dim 并成一维再 group 回 `((qh, seqlen), d, nheads_kv, b)`——**保持与原来相同的 TMA 维数**(源码注释:`for (seqlen, d, nheads, b) layout, we still have 4D TMA after packing`;直接 pack 成 `((qh, seqlen), d, nheads_kv, b)` 会变成 5D),box 是 `((qh, tile_m // qh), tile_n)`,并要求 `tile_m % qh == 0`。
- **派生缓冲层**:LSE / dPsum / dQaccum 按 KV head 分配、按 packed 行序存放(kernel 用 1-D bulk copy 加载 LSE/dPsum)。preprocess 用新枚举 `PackGQA.PACKED_ROW_OUTPUTS` 写它们并把 packed dQaccum 清零(见 ③ 代码)。
- **varlen 层**:`SeqlenInfo` **保持 token 空间**(review 后撤掉了 q_scale 参数侵入),新增 `seqlen_info.sequence_row_offset(token_offset, batch_idx, rows_per_token, tile_rows, padded)` 把 cu_seqlens 的 token 偏移换算成 per-KV-head scratch 的存储行偏移:先 `× rows_per_token`(=qh)scale,再按同样的 `cute.assume(..., divby=tile_rows)` 对齐 pad 到 tile 边界。load / dQ-reduce 的 varlen 分支、preprocess、postprocess 都调它。
- **mask / 调度层**:`block_info`、转置 mask、deterministic 锁值都吃 `qhead_per_kvhead_packgqa`(packed 行 r 的 token 位置是 `r // qh`,位置上限按 qh 放大);**反向的 KV tile scheduler 保持 1**(packing 只动 M 维语义,不动 KV 遍历)。
- **数值口径**:dQ 与 unpacked 路径**逐 bit 相同**;dK/dV **差 1 ulp**(packed 是"一次 bf16 TMEM 累加",unpacked 是 fp32 atomics + postprocess,舍入路径不同);deterministic 模式可逐 bit 重复;`FLASH_ATTENTION_BWD_PACK_GQA=0` 时 36 组配置矩阵的 **CUBIN/PTX 与父 commit 逐字节相同**(不开启就零影响)。

#### ③ 可复现代码

**启用条件**(interface.py,决定"能不能用"):

```python
bwd_pack_gqa = (
    bwd_pack_gqa_request                                   # 未设 tile_n 测试旋钮
    and not use_dedicated_hd256_kernel                     # hd256 专用内核路径不参与
    and qhead_per_kvhead > 1
    and m_block_size % qhead_per_kvhead == 0               # qh 必须整除 tile_m
    and block_sparse_tensors is None
    and score_mod is None
    and mask_mod is None
    and learnable_sink is None
)
```

通过的 qh ∈ {2,4,8,16,32,64,128}。环境变量 `FLASH_ATTENTION_BWD_PACK_GQA=0/1/auto` 可强制覆盖(0=关、1=开、auto/未设=交给启发式;`utils.py::_get_bwd_pack_gqa_override`,非法值直接 assert)。

**preprocess 的三态枚举**(flash_bwd_preprocess.py,决定 LSE/dPsum/dQaccum 写去哪):

```python
class PackGQA(enum.Enum):
    OFF = "off"
    # per query head, written through the same packed views (MLA backward)
    PER_QHEAD_OUTPUTS = "per_qhead_outputs"
    # allocated per KV head in packed row order, qhead_per_kvhead rows per token
    # (the SM100 backward's pack_gqa, NOTE [bwd pack_gqa] in pack_gqa.py)
    PACKED_ROW_OUTPUTS = "packed_row_outputs"
```

**启发式两道闸**(interface.py::_bwd_pack_gqa_heuristic,决定"该不该用",缺一不可):

```python
qh = num_head // num_head_kv
kv_tile = n_block_size * cluster_size
n_blocks_kv = (seqlen_k + kv_tile - 1) // kv_tile
slots = max(1, num_sms // cluster_size)

# 闸 1:关键路径——最长 packed CTA 的迭代数 vs 平均每个 cluster slot 的活
packed_max_iters = qh * ((seqlen_q + m_block_size - 1) // m_block_size)
mean_q = total_q / max(batch_size, 1) if varlen else seqlen_q
mean_k = total_k / max(batch_size, 1) if varlen else seqlen_k
total_iters = (batch_size * num_head * math.ceil(mean_k / kv_tile)
               * math.ceil(mean_q / m_block_size) * (0.5 if causal else 1.0))
if packed_max_iters > 1.25 * total_iters / slots:
    return False

# 闸 2:毛收益 vs wave 量化损失
def wave_loss(num_ctas):
    waves = num_ctas / num_sms
    return (math.ceil(waves) - waves) / math.ceil(waves)

ctas_unpacked = batch_size * num_head    * n_blocks_kv * cluster_size
ctas_packed   = batch_size * num_head_kv * n_blocks_kv * cluster_size
gain = 0.25 * 2048 / max(seqlen_q, 1)
if causal or local:
    gain *= 0.6; gain -= 0.02
return gain > wave_loss(ctas_packed) - wave_loss(ctas_unpacked)
```

**两道闸的因果**:packing 把 tile 数除以 qh(每 tile 的省:K/V 只读一次、dK/dV 留 TMEM、免 fp32 累加与 postprocess),但付出两种调度代价——①单个 packed tile 要循环 **qh 倍的 Q 行**,网格变小时**长尾 CTA 无法被短 CTA 填满**(闸 1);②网格小 qh 倍,**wave 量化损失**(最后一个不满的波)占比变大(闸 2)。毛收益随 seqlen_q 变小(B200 实测 2k ~25%、8k ~7%),而 3.5 波的 packed 网格损失 ~13%——`gain = 0.25 * 2048 / seqlen_q` 就是这条"2k 时 25%、8k 时 6.25%"的经验曲线,causal/local 的 `×0.6 −0.02` 是"tile 不均匀、长 packed tile 尾部更难配平"的保守余量。

**闸 1 是被一个实测反例逼出来的(工程教训)**:原本只有闸 2。MQA(16/1)hd256 在**一条 8k causal 序列**(varlen 包里还混 16 条短序列)上,B200 实测 **packed 3946 µs vs unpacked 2297 µs**——wave-loss 规则仍然选了 packing,因为 wave 损失只看"网格有多大",看不到"最长那个 CTA 有多长"。补上闸 1 后,varlen 用**平均序列长**做分母,所以"一长串短序列混一条长序列"的 ragged 包会保持 unpacked。启发式抽成独立函数 `_bwd_pack_gqa_heuristic` 也是为了可单测。

#### ④ 实测数据(B200 @1400 MHz,两轮交替,vs unpacked 路径)

| 配置 | 2k | 8k | 32k |
|---|---|---|---|
| hd128 GQA(8/2、16/4、32/8、64/8) | **+13…26%** | +0…10% | +0…5% |
| hd256 通用内核 vs 专用内核(8/4、16/4、16/8) | **+4…10%** | **+1…11%** | — |

启发式让 hd128 8/2@8k 和 causal 32k 留在 unpacked。测试:`tests/cute/test_flash_attn_bwd_pack_gqa.py`(packed vs unpacked vs FP64 eager:hd 64/128/256,GQA 8/2、16/4、32/4、MQA,dense 与含空序列的 varlen,local、softcap、deterministic、causal)**480 passed / 96 skipped**;全量 `test_flash_attn_output` + d=256 varlen 子集 + hd256 文件 **18,032 passed / 29 failed / 9,416 skipped**(27,477 ids,5 GPU 3h16m)——29 个失败中 18 个是 has_learnable_sink 的 dSink 容差问题(父 commit 同样失败、ids 相同),11 个是 `test_hd256_varlen_scheduler.py` 断言"非确定性调用可 bit 重复"(父 commit 32 个的子集),均非本 commit 引入。另有专门的边界测试:packed K/V 在 `cu_seqlens_k + seqused_k` / dense `(b,s,h,d)+seqused_k` 下,`seqused_k` 之后填 NaN(K/V/both 三种),要求 dQ/dK/dV 有限、等于干净 packed 跑,干净 packed 又等于 unpacked(hdim 64/128,以及通用内核上的 256)——即 §4.5.1 那套 NaN 污染防线在 packed 网格上重新覆盖了一遍。

#### ⑤ 实践要点

- **什么时候强制开/关**:GQA 且 seqlen_q 短(≤2k 收益最大 13–26%)、网格够大(波数多、wave 损失小)→ 默认启发式就会开;想对照实验用 `FLASH_ATTENTION_BWD_PACK_GQA=0/1`。qh 不整除 tile_m、或用了 mask_mod/score_mod/block sparsity/learnable_sink → 自动退回 unpacked,这些组合**目前就是不支持**,别指望环境变量强开(强开只覆盖启发式,不覆盖硬条件)。
- **varlen ragged 包为什么不 pack**:启发式闸 1 用平均序列长估关键路径,"一条长序列 + 一堆短序列"的包里,最长 packed CTA 的迭代数远超平均 slot 工作量,长 tile 串行化(实测 3946 vs 2297 µs)——这类包保持 unpacked 是对的,不是启发式失灵。
- **deterministic 的代价**:仍有 dQ semaphore(每 `(KV head, packed m block)` 一槽),但 dK/dV 不再有 semaphore/postprocess;deterministic 模式可逐 bit 重复。
- **数值预期管理**:开 pack_gqa 后 dK/dV 与旧路径差 1 ulp(bf16 TMEM 累加 vs fp32 atomics),dQ 逐 bit 相同——训练 loss 曲线出现 ulp 级偏移是预期行为(呼应 §6 checklist 第 5 条"exact 但非 bitwise")。
- **对照 §4.5.3 的读法**:旧的"GQA 2k 慢 22–29%"(#2966 口径,hd256 24/4,通用内核 vs 专用内核)与本节的"+13…26%"(#2967 口径,hd128 8/2·16/4·32/8·64/8 与 hd256 8/4·16/4·16/8,packed vs unpacked)**不是同一个 benchmark 从负翻正**,而是同一现象(通用内核缺 q-head 循环)的修复方向:根因补上后,通用内核在 GQA 小序列上不再结构性落后。

### 4.5.5 调度坐标契约:CLC 2CTA 张冠李戴(#3034)

上游 commit `98c1cee`(2026-10-24,`[CuTe, SM100] Fix CLC tile coordinates and make forward work-tile coordinates logical`,4 files +284/−59)。它是**这一类 bug 的教科书形态**:不是算错公式,而是**两套坐标系被混着用**,而且错的那一侧只在 CLC + 2CTA 这一种调度组合下才暴露。

#### ① 症状与根因:两个 CTA 拿到了不同的 tile

SM100 前向的行地址由这一条式子决定(commit 原文):

```
rows = (m_block * q_stage + stage) * cta_group_size + mma_tile_coord_v
```

也就是说 `m_block` 必须是 **cluster 级的逻辑 tile 编号**,`cta_group_size` 与 `mma_tile_coord_v` 已经负责把 CTA 内的行折进去。而在 CLC(Cluster Launch Control,见 [cutlass tile-scheduler](../../cutlass/concepts/tile-scheduler.md))路径上:

1. CLC 调度器在代码里**被算作 persistent**(`is_persistent` 为真),所以前向传下去的是 **`use_cluster_idx=False`** 且 **CTA 级 `num_block`**(改动前那行是 `_num_block_divisor = cta_tiler[0] * (cta_group_size if not is_persistent and cta_group_size > 1 else 1)`);
2. `clc_work_to_coords` 于是在 `use_cluster_idx=False` 分支里自己做 per-CTA 映射:`block_idx = block_idx * cluster_shape_m + block_in_cluster_idx()`;
3. 两步叠加的结果是**每个 CTA 拿到 `block_idx * 2 + rank`**——两个 CTA 认为自己在**两个不同的 work tile** 上,而 2CTA MMA 是一条指令横跨两个 CTA 的 M 维。**于是 MMA 把不同 tile 的 Q 行拼在一起做了一次 attention**,O/LSE 都写到了错误的行。

数值表现:GB300 上,CLC(`FA_CLC=1`)跑 **K-only varlen + packed GQA + hd128**,**max error 0.39**(相对 ~1 量级的输出,基本等于随机),修完**与非 CLC 内核逐 bit 相等**。

顺带把一条既有注释的**病因**说清了:`flash_attn/cute/interface.py:1128`(2026-10-06 起)写着 `CLC does not map hd256 2CTA tiles correctly (wrong output, traps, hangs)` 并把 `hd256 + 2CTA` 排除在 CLC 之外——本 commit 明确"两个 CTA 拿到不同 tile 也是这条排除的成因之一",但**排除保留不动**(见 ④)。

#### ② 修法:把"逻辑坐标"写成显式契约,而不是就地打补丁

作者没有在 CLC 分支里改那一行乘法,而是把**契约**写清楚,再做统一:

- **调度器只发"逻辑 work tile"**:无论 grid / static persistent / CLC 哪一种策略,一个 work tile 都是 `m_block_size * cta_group_size * q_stage` 行,cluster 内两个 CTA 拿到**同一个** `(work_m, head, batch, split)`;
- 统一 `num_work_m = ceil_div(eff_seqlen_q, cta_tiler[0] * cta_group_size)`(原来按 `is_persistent` 分叉),`use_cluster_idx = (cta_group_size > 1)` **不再看是否 persistent**;
- `StaticPersistentTileScheduler.create` 在 `use_cluster_idx` 时**直接把 `num_block` 当 cluster 数**(不再 `ceil_div` 一次),因为它的唯一调用者就是前向;
- `clc_work_to_coords` **删掉 per-CTA 映射**,并加断言:cluster 化的 grid 必须 `use_cluster_idx`;
- LPT(causal/local 的最长处理时间优先)反转也随之简化成 `num_block - 1 - block_idx`(旧代码要在 `cluster_shape_m > 1 and not use_cluster_idx` 时先除一次)。

核心是一个**唯一的行映射函数**(`flash_fwd_sm100.py`,导出给 host 侧测试):

```python
# NOTE [SM100 forward coordinate spaces]
#   physical CTA / CLC response
#        |  scheduler(NOTE [SM100 forward work coordinates], tile_scheduler.py)
#   (work_m, head, batch, split)        cluster 内每个 CTA 相同
#        |  work_q_block(work_m, stage, cta_rank)
#   q_block                             R 行的块号
#        |  * R + row_in_cta
#   q_row                               batch 内有效行(在 pack_gqa 下为 packed 行)
def work_q_block(work_m, stage, cta_rank, *, q_stage: int, cta_group_size: int):
    return (work_m * q_stage + stage) * cta_group_size + cta_rank
```

**行序不变量的显式写法**:`(row_in_cta, cta_rank, stage, work_m) : (1, R, R*C, R*C*S)`。三条维度相乘就是 work tile 的行数,把 `R=128, C=S=2, work_m=0` 代进去就是源码注释里的那张表:

| | CTA 0 | CTA 1 |
|---|---|---|
| stage 0 | 0..127 | 128..255 |
| stage 1 | 256..383 | 384..511 |

自检一下:work_m=1、stage=0、rank=0 → `(1*2+0)*2+0 = 4` → 行 512,正好接在 511 之后 ✓;而按 `(row_in_cta, cta_rank, stage, work_m)` 的 strides 算 `row + 128*rank + 256*stage + 512*work_m` 也对得上 → 两套写法等价,不变量成立。

**"谁用哪个坐标系"也一起写进注释**(这是本 commit 最有复用价值的部分):

| 使用者 | 用什么坐标 | 为什么 |
|---|---|---|
| TMA Q / O 的 `cute.local_tile` | `work_m` | 一个 box 覆盖 `q_stage * m_block_size` 行,再按 stage/CTA 切 |
| LSE、非 TMA 的 Q/O 下标 | 本 CTA 的 `q_block` | 直接算行偏移,不需要 cluster 概念 |
| mask / score_mod | `q_block(work_m, stage, **0**)` | `thr_mma` 的 partition **已经**含了本 CTA 的行,再传 rank 会加两次 |
| block-sparse 的 stage 终点 | `(work_q_block(work_m, stage, 0) + C) * R` | 一个 stage 跨 C 个 CTA |
| learnable sink 的 packed 行 | `_q_block(work_m, stage, mma_tile_coord_v) * R + tidx` | sink 按 packed q-head 取行 |
| `aux_tensors` 的边界判定 | `m_tile_end = (q_block + C) * m_block_size` | 只有边缘 tile 需要 wrap 越界 index |

#### ③ 测试怎么写才抓得住这类 bug

这类错误**不会被单点数值测试抓到**(每个 shard 单独看都"像对"的,错的是行的归属),因此本 commit 补了两层:

**a) host 侧的纯 Python 行映射检查**(`tests/cute/test_fwd_work_coords.py`,不需要 GPU kernel):参数化 `cta_group_size ∈ {1,2} × q_stage ∈ {1,2} × seqlen_q ∈ {1,127,128,129,511,1000}`,断言四件事:

```python
work_rows = 128 * cta_group_size * q_stage
num_work_m = ceil_div(seqlen_q, work_rows)
# 1) 行不重叠且覆盖:每个 q_row 只被一个 (work_m, stage, cta_rank) 拥有
assert q_row not in owner
assert sorted(owner) == list(range(num_work_m * work_rows))
# 2) 有效行恰好是 0..seqlen_q-1(不多不少)
assert sorted(r for r in owner if r < seqlen_q) == list(range(seqlen_q))
# 3) 尾行只出现在最后一个 work tile(不允许中间的 tile 有空洞)
assert all(owner[r][0] == num_work_m - 1 for r in owner if r >= seqlen_q)
# 4) rank 只加一次:q_block(rank) == q_block(rank=0) + rank
```

**这就是"能用 CPU 验的不变量就别上 GPU 验"**——坐标契约的正确性是纯组合问题,`4`(2×2 的 cta_group_size × q_stage) `× 6` 个序列长就穷举完了,而 GPU 测试只能抽样。

**b) 强制调度策略的覆盖测试**(`test_flash_attn_work_tiles_cover_q`):`policy ∈ {grid, static_persistent, clc} × 1CTA/2CTA × q_stage {1,2} × epilogue {corr_epi, tma_o} × causal {False,True}`,方法是在 `FlashAttentionForwardSm100.__init__` 上 monkeypatch 强制 `use_clc_scheduler` / `is_static_persistent` / `use_2cta_instrs` / `q_stage`(**并换一个新的 `JITCache`**,否则编译缓存会让"强制"失效),然后用 **NaN 预填的 O 和 LSE** 与参考实现比——NaN 预填是关键:漏写的行不会被旧值掩盖,直接暴露成 NaN。另有 `test_flash_attn_clc_2cta_matches_default` 对 FA_CLC=1 做**逐 bit** 对比(dense Q + `seqused_k`、varlen K + 半个 cluster 的尾 tile)。

#### ④ 为什么 hd256 2CTA 的 CLC 依然禁着

commit 原话:`This was also the cause of the hd256 2CTA CLC exclusion, which stays in place here.` 即:坐标混用是**已知的一类**根因,但 hd256 一侧**还有别的问题**(trap/hang,不只是错数),所以 `interface.py` 的 `and not (is_hdim256 and use_2cta_instrs)` 保留。实践含义:**不要把这次修复当成"hd256 2CTA + CLC 可以开了"**。

同时注意 CLC 的**默认状态**:`utils.py:64` 是 `_fa_clc_enabled = os.environ.get("FA_CLC", "0") == "1"` → **默认关**;`interface.py:1124` 还会在 varlen MHA、dense non-causal、hd256+2CTA 三种情形下把它关掉(注释给的理由是 L2 抖动与 work-stealing 开销)。所以要复现本 bug 也必须显式 `FA_CLC=1`。

#### ⑤ 实践要点(可迁移到任何 cluster/2CTA 内核)

- **一个 tile 只能有一个编号空间**。cluster 化之后,`num_block`(逻辑 tile 数)与 grid size(物理 CTA 数)是**两个数**,关系是 `physical_x = C * cluster_slot + cta_rank`。凡是"调度器发号"的地方都要问一句:这个号是 CTA 的还是 tile 的?本 bug 就是 `num_block` 用 CTA 口径、内核按 tile 口径解码。
- **persistent 与 cluster 是两个正交属性**。旧代码把 `use_cluster_idx` 绑在 `not is_persistent` 上(因为"非 persistent 的 grid 才是按 CTA 数的"),而 CLC 恰好"既 persistent 又 cluster 化"——两个属性一交叉,绑定的假设就破了。**把互不相关的条件做成一个布尔量,迟早会在某个交叉组合上错**。
- **行映射函数要能被 host 侧单测**。`work_q_block` 单独导出 + 纯 Python 穷举,是这次能在没有 GPU 迭代的情况下把契约钉死的直接原因;反之,把映射写在 `@cute.jit` 的 kernel 体里就只能靠 GPU 抽样。
- **`thr_mma` 已经含 rank 时,给 mask/score_mod 传 rank=0**(否则行偏移加两次)。凡"同一行号既被 tile 级代码用、又被 warp 级 partition 用"的接口,都要在注释里写清谁已经加过了。
- **一个共享 work tile 的 CTA 群必须看到同一个 `is_valid_tile`**:cluster 内一个 CTA 判定无效、另一个继续跑,等于让硬件做半条 MMA。
- **验证顺序**:先 CPU 穷举不变量 → 再强制策略的 GPU 覆盖测试(NaN 预填输出)→ 最后才是性能;本次改动**没有性能数字**(纯正确性修复),别在引用时给它编加速比。

### 4.5.6 版本与依赖快照(2026-10-24 实测,`98c1cee`)

| 组件 | 版本 | 备注 |
|---|---|---|
| FA2(C++ 主线) | `2.8.4`(`flash_attn/__init__.py`) | 论文语义那条线仍是 2.8.x |
| FA3(Hopper) | `3.0.1`(`hopper/__init__.py`) | `flash_attn_3` 包;`hopper/` 长期 beta 的状态未变 |
| FA4(CuTe DSL) | PyPI 包名 `flash-attn-4`,setuptools-scm 动态版本,当前 tag `fa4-v4.0.0.beta36` | 依赖 `nvidia-cutlass-dsl>=4.6.2`、`quack-kernels>=0.5.3`、`apache-tvm-ffi` |
| 同期修复 | #2984 paged KV cache 的 CUDA graph 捕获(csrc 侧 12 行)、#3007 给 FA4/FA2 入口补 CUDA graph 捕获-重放测试、#2758 SM120 自动 `num_splits` | 都是"能上线"级的小改动,不是算法变化 |
| 同期实质改动 | **#2967 反向 pack_gqa**(见 §4.5.4,10 files +709/−116,hd128 GQA 2k +13…26%) | 不是算法变化,但改的是反向数据流与派生缓冲布局,量级远超上面的小修 |
| 本轮新增 | **#3034 CLC 2CTA 坐标修复**(见 §4.5.5,4 files +284/−59;`git describe` = `fa4-v4.0.0.beta36-7-g98c1cee`) | 纯正确性修复(GB300 hd128 K-only varlen packed GQA:max error 0.39 → 逐 bit 相等),**无性能数字** |

> **读源码的顺序建议**:先看 `config.py`(调度决策的数据形态)→ `interface.py`(host 侧选型/校验/workspace 上界)→ `flash_fwd_sm90.py` / `flash_fwd_sm100.py`(producer:descriptor 与流水)→ 再看 `flash_bwd_*`。FA4 论文给的算法框架没变,但"合法且不崩"的边界条件全在这几层里。

---

## 5. 反向重算(Recomputation):训练省显存的关键

下图展示"前向只存统计量、反向重算"这笔算力换显存的交易:

<div class="archify-figure">
<iframe src="../../../diagrams/fa-backward-recompute.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="反向重算:算力换显存"></iframe>
</div>

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
19. **配合工具链**:FA3 基于 CUTLASS 3.x/CuTe,FA2 亦用 CuTe 原语写 tiling(外层调度自写)(见 [cutlass 域](../../cutlass/index.md));FA4 走 CuTe DSL(Python),要求 `nvidia-cutlass-dsl>=4.6.2`。

**CuTe 后端(FA4/FA3-Hopper 实现在用的这套 Python kernel)专项**

20. **KV cache 复用场景必须把 TMA 上界收到 `seqlen_k`**:mask = 0 挡不住 `0 × NaN`——前向 NaN 出在 V 尾,反向 NaN 出在 dQ(K 尾或 V 尾都会中招,且**前向干净不代表反向干净**)。凡是自己写 ragged/paged attention kernel,先确认"越界行是 TMA zero-fill 补的,还是 mask 挡的"。
21. **别用 wraparound descriptor 兜可变长度**:TMA 的预取足迹会超出 box 最多 ~128 KB,撞到未映射页就是 `cudaErrorIllegalAddress`,而 PyTorch caching allocator 会把它伪装成"偶发/与顺序相关"。改用**每 CTA 一份 GMEM descriptor + `tensormap.replace` 改 rows extent**,并补 `fence_tensormap_update` + `prefetch.tensormap`。
22. **descriptor workspace 容量自己推导并 `trap` 兜底**:`rows // tile + 2 * batch + 1`(2 = 变长调度器每 batch 的 padding tile);算少了不会报错,只会让 CTA 改写别人的 descriptor 后照旧加载。写 `if cta >= size: trap;`,宁可硬失败。
23. **block-sparse / 特殊 producer 路径要单独过一遍 fence**:主路径加的 acquire fence 往往在另一条 producer 函数里到不了(FA4 #3071 的实际坑),持久 CTA 会用上一序列的 extent。
24. **调优配置用"不可变 + 校验,不归一化"的数据结构**:FA4 的 `FwdConfig` 就是范式——非法组合直接抛错,选择器纯 host 计算(不读 tensor / 不同步 / 不分配),每个特性一条支持谓词供选择与校验共用。
25. **"偶发非法地址"的测试写法**:先关 caching allocator 单序列子进程复现;再把被测 tensor 放到 VMM 映射尾部、后面留保留未映射 guard range,做到"改前必 fault、改后必过"。否则测试在有 allocator 时 100% 假通过。
26. **hd256 这类极端 head_dim 要先看资源账**:FA4 的 hd256 反向 TMEM 正好占满 512 列、SMEM 用掉 232,448 B 中的 231,424 B;这种配置下"少一次等待"(如删掉 dQaccum 别名握手)就是 17% 的差别,而 GQA 小序列反而可能退 22–29%(**#2966 旧口径,当时无反向 pack_gqa;#2967 折 q-head 进 Q tile 后翻正为 +13…26%,见 §4.5.4**)——按 shape 分派,别一套配置通吃。
27. **cluster/2CTA 内核先画清坐标系再写代码**:逻辑 work tile 数、物理 CTA 数(`physical_x = C * cluster_slot + cta_rank`)、以及 CTA 内行偏移是三个不同的数;把"是不是 persistent"和"要不要 cluster 索引"合成一个布尔量一定会在交叉组合上翻车(CLC 恰好"既 persistent 又 cluster 化",FA4 #3034 就是这么错的——两个 CTA 拿到不同 tile,2CTA MMA 把不同 tile 的 Q 行拼在一起,max error 0.39)。行映射抽成可被 host 侧纯 Python 穷举的独立函数,别埋在 `@cute.jit` kernel 体里(见 §4.5.5)。**另:`thr_mma` 的 partition 已含 CTA rank,mask/score_mod 要传 rank=0**,否则行偏移加两次。

---

## 7. 溯源(三篇合并)

**FA1(IO-aware、tiling、online softmax、Θ(N²d²/M))**
- FA1 论文: Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022, https://arxiv.org/abs/2205.14135
- UW CSE 599M 讲义 *From Online Softmax to FlashAttention*(softmax 不结合→online softmax 的推导): https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
- Online softmax 前驱:Milakov & Gimelshein, *Online normalizer calculation for softmax*, https://arxiv.org/abs/1805.02867 (2018)
- O(1) 显存前驱(FA1 [66],附录 B.5 专节比较):Rabe & Staats, *Self-Attention does not Need O(n²) Memory*, https://arxiv.org/abs/2112.05682 → 本域详注 [exact-attention-memory-lineage.md](../algorithm/exact-attention-memory-lineage.md)

**FA2(work partitioning、基于 CUTLASS 3.x/CuTe 重写;§3.2 减 non-matmul FLOPs、§3.3 序列维并行)**
- FA2 论文: Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, ICLR 2024, https://arxiv.org/abs/2307.08691
- Tri Dao 的 FA2 博客(25-40%→72% MFU、A100 230 TFLOP/s、warp 划分示意图;`flash-atttention-2` 为站点真实拼写): https://princeton-nlp.github.io/flash-atttention-2/

**FA3(§3.2 pingpong scheduling、§3.3 block quantization + incoherent processing 的 128×128/1×128 粒度、§5 性能与 2.6× 误差)**
- FA3 论文: Dao et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, NeurIPS 2024, https://arxiv.org/abs/2407.08608
- Tri Dao FA3 博客(overlap、WGMMA/TMA、FP8、pingpong 拓扑;570→620-640 TFLOPS、exp≈3.9 TFLOPS 等博客口径数字): https://tridao.me/blog/2024/flash3/

**FA4(Blackwell)**
- FA4 论文(MLSys 2026): arXiv:2603.05451

**实现仓库 / 跨域**
- 官方实现仓库(含 FA1/FA2 kernel 与 `benchmarks/` 复测): https://github.com/Dao-AILab/flash-attention
- 2026-10 CuTe 后端增量的上游一手源(均于 2026-10-23 复核 commit message 与 diff):
  - #2983 / #3012 / #3013 / #3071:未用量 K/V 尾行的 NaN 污染防治(前向 V TMA 上界 → 反向 K/V/Kt → SM90 对齐 → 改用 per-CTA GMEM descriptor)https://github.com/Dao-AILab/flash-attention/pull/3071
  - #2732:typed forward config(`FwdConfig` / `FwdHeuristicInputs`)https://github.com/Dao-AILab/flash-attention/pull/2732
  - #2965 / #2966:hd256 反向统一到通用 SM100 内核(opt-in)https://github.com/Dao-AILab/flash-attention/pull/2966
  - #2967(commit bc52a40):反向 pack_gqa——GQA 的 q-head 折进 Q tile,启发式 `_bwd_pack_gqa_heuristic` 两道闸 https://github.com/Dao-AILab/flash-attention/pull/2967
  - #3034(commit 98c1cee,2026-10-24):CLC tile 坐标修复 + 前向 work tile 坐标逻辑化(`work_q_block` / NOTE [SM100 forward coordinate spaces] / NOTE [SM100 forward work coordinates];GB300 CLC 2CTA hd128 varlen packed GQA max error 0.39 → 逐 bit 相等)https://github.com/Dao-AILab/flash-attention/pull/3034
  - #2984 paged KV cache 的 CUDA graph 捕获修复、#3007 FA4/FA2 的 CUDA graph 捕获-重放测试、#2758 SM120 自动 `num_splits`
- CuTe DSL 侧:`TensorMapManager` / `TensorMapUpdateMode`(GMEM |=在设备端改 descriptor)与 `prims.tensormap_replace` 见 CUTLASS 仓库 `python/CuTeDSL/cutlass/tensor_utils/tensormap.py`、`python/CuTeDSL/cutlass/experimental/primitives/nvvm_wrapper.py`;CUTLASS v4.8.1 的 release note 明确修了 `tensormap_replace` 在 descriptor 被缩小后的越界风险(可用 `keep_oob_fetch_window=True` 退回旧行为)——**这是 FA 侧 per-CTA 写法的上游前提**
- 相关:CUTLASS 领域的 WGMMA/TMA/epilogue 篇(同硬件特性,跨参考,见 [cutlass/concepts/hopper-wgmma.md](../../cutlass/concepts/hopper-wgmma.md))
