---
title: FlashAttention 核心概念:IO-aware tiling 与 Online Softmax
type: concept
tags: [flashattention, attention, tiling, online-softmax, io-awareness, gpu, sram]
source:
  - "pdf: FlashAttention FA1 | Dao et al. arXiv:2205.14135 NeurIPS 2022"
  - "pdf: FlashAttention-2 | Dao et al. arXiv:2307.08691 ICLR 2024"
  - "web: Tri Dao FlashAttention-2 博客 (work partitioning) | https://princeton-nlp.github.io/flash-atttention-2/"
  - "pdf: UW CSE 599M 讲义 From Online Softmax to FlashAttention (Zihao Ye) | https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf"
  - "doc: tracker/flash-attention-tracker/flash-attention-annual-report.md §2 (canonical)"
date: 2026-09
---

# FlashAttention 核心概念:IO-aware tiling 与 Online Softmax

## TL;DR

FlashAttention(FA1, NeurIPS 2022)的核心思想是 **IO-aware**:把 attention 的计算**按块(tile)放进 SRAM** 做,用 **online softmax** 让"不可结合"的 softmax 变得可以增量合并,从而**不在 HBM 里实例化完整的 N×N attention 矩阵**。

**一句话收益**:标准 attention 的 HBM 访存 Θ(Nd + N²)(元素访问数),FlashAttention 降到 Θ(N²d²/M)(M = SRAM 大小)——对大 N 是**数倍到接近一个量级的访存减少、端到端实测 2-4× 加速**,省显存(显存 O(N²)→O(N))。**不损失精度(数学精确,非近似,但浮点结果与朴素实现有 ulp 级差异)**。

**心法**:attention 的瓶颈不是计算,是**内存搬运**。把"算注意力分数"和"写回中间矩阵"解耦,让每次 HBM 读写都服务于"计算整行的最终输出",这就是 IO-aware 的意义。

---

## 1. 先看问题:标准 Attention 为什么慢

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

---

## 2. 为什么不能直接 tiling:softmax 不结合

矩阵乘法可以 tiling 是因为**加法按块可结合**(C 的每个输出 tile = 相关 A/B tile 的部分和累加)。但 **self-attention 里有 softmax,它不结合**:

```
softmax(合并整个 S 行) ≠ 合并各块单独算的 softmax
```

如果你硬把一整行 S 切成几块、各自单独做 softmax,合的 `∑exp` 和 max 对不上,结果就是**错的(近似)**。这让"像 GEMM 一样按块算 attention"变得非平凡。

**解法就是 online softmax**:用**增量式**维护 running max `m` 和 running sum `l`,让每块可以"边算边把信息并入全局统计",最后得到**精确**的 softmax(见下节推导)。

---

## 3. 核心机制:Online Softmax(让 softmax 可增量合并)

online softmax 的关键:维护**每行各两个标量(即 m、l 是长度为 Br 的行向量)**(running max `m`、running sum `l`),遍历 K/V 块时增量更新,并对已累加的旧输出做 rescale,最终得到**精确**的行 softmax。**注意不是单个全局标量——每行独立维护**。

**为什么式子长这样**(一行推演):
- 标准 softmax 对整个行:`p_i = exp(s_i - max_s) / Σ_j exp(s_j - max_s)`;
- 逐块遍历时,每块会遇到**新的更大的 max**。一旦 max 变大(`m_new > m_old`),**之前所有块已经按旧 max 算的 exp 值都要整体收缩** `exp(m_old - m_new)` 倍;
- 所以 running sum 要 rescale:`l_new = l_old · e^{m_old - m_new} + Σ_block e^{s_k - m_new}`;
- 已累加的输出版本同理:`O ← O · e^{m_old - m_new} + ...`(输出的每行按旧 max 的权重整体收缩)。

这正是 canonical 里那段伪代码做的事(`li = li * exp(mi - mi_new) + sum(exp(Sij - mi_new))`),用**标量 rescale** 换来**块间精确合并**——softmax 在该增量形式下"变可结合"了。

---

## 4. 完整算法:FA1 Forward 伪代码

> ⚠ 循环顺序说明:论文 Algorithm 1 的**外层是 K/V 块、内层遍历 Q 块**(FA2 才翻成内/外层)。下面为了直观按"外层 Q、内层 K/V"展示(**简化变体**,逻辑等价、更贴合"每行输出遇到的 KV 序列");要对读论文,注意其外层是 K/V 块。

> 为什么是 4d:Bc≈M/(4d) 来自 SRAM 预算推导——SRAM 需同时装 Q 块(Br·d)、K 块(Bc·d)、V 块(Bc·d)、S 分数块(Br·Bc)与输出 O 块(Br·d);取 Br=Bc 后,**Q/K/V/O 四块各占 Bc·d**(加 S 块 `Bc²`)约 **4·Bc·d + Bc² ≈ M**,解出 Bc ≈ M/(4d)(FA1 论文 Algorithm 1 的原始约束;4d 项主导)。

```
Input: Q,K,V ∈ R^{N×d}; SRAM 容量 M
Bc = ceil(M / (4d))           # K/V block 列数(论文 Algorithm 1 为 ceil;canonical 里简写为 floor)
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

---

## 5. IO 复杂度:为什么快得有理有据

元素访问数口径(M = SRAM 容量,单位元素):

| 版本 | HBM 访存(元素访问) | 说明 |
|---|---|---|
| 标准 attention | Θ(Nd + N²) | 读 Q/K/V + 写读 S、P 两个 N×N 矩阵 |
| FlashAttention FA1 | Θ(N²d²/M) | 分块后,每个 Q/V 块在 SRAM 内复用,中间矩阵不落 HBM |

- **标准**:Θ(N²) 项是瓶颈(随 N 二次方);
- **FA1**:固定 d、M 常数下是 Θ(N²),与序列长度二次方同阶,但隐藏常数被 SRAM 复用压到最小;
- 对比口径**统一为元素访问数**,二者才有意义(quick-assistant 2026-09 评:不要用 Naive 的 Θ(N²d) 去对 FA 的 Θ(N²d²/M),单位不一致会导致自相矛盾)。

**关于"下界"的准确表述**(FA1 论文 Proposition 3):不存在对**所有** SRAM 容量 M ∈ [d, Nd] 都达到 o(N²d²/M) 访问的算法;当 **M ≥ Nd(SRAM 装得下全部 K/V)** 时,标准读取可达 Θ(Nd) 访问(此时不用分块、K/V 驻留 SRAM 复用),FA 的"每块复用"把 M < Nd 的场景也压到接近这个下界。注意:**不是"每行必须碰 N 个 K/V 列就是 N² 次 HBM 读"**——K/V 块在 SRAM 里可复用,分数依赖关系不等于 HBM 访问次数。

---

## 6. 局限与演进(为什么还要 FA2/FA3)

FA1 已达 IO 下界,但在 **A100 上只有 25-40% 的理论 FLOP 利用率**(最高 124 TFLOP/s)——IO 最优 ≠ 计算利用率最高。问题在工作划分:

- **FA1 的并行维度不足**:线程块**只沿 batch × head 维度划分、不沿序列维**——batch×heads 少(batch 小、序列长)时 grid 太小、喂不满 SM,occupancy 低;同时 FA1 的 warp 把 **K/V 沿序列维切给各 warp(split-K)**,每个 warp 产出一份全宽的部分输出,聚合要经 shared memory 归约,通信 + 同步多;
- **FA2**(ICLR 2024,基于 CUTLASS 3.x/CuTe 重写,约 2×)改进 **work partitioning**:把 threadblock 和 warp 的并行更好地贴合 GPU(减少共享内存读写、提高 occupancy)——FA2 论文 Table 1 端到端训练 196~225 TFLOPs/s(最高 72% MFU,GPT3-2.7B 8k);attention kernel 单独基准最高 ~230 TFLOPs/s(73% 理论峰值);**不同配置数字不同,以原文献口径为准**;
- **FA3**(H100)进一步用 **WGMMA 异步、TMA、低精度(FP8)** 挖 Hopper 特性(见 work-partitioning 与 fa3 篇);
- **FA4**(2025)用 **CuTe DSL(Python)生成 kernel**,支持 H100+B200 及新特性。

> 一句话:FA1 解决"少搬数据"(IO-optimal);FA2+ 解决"算得更满"(并行/利用率/软硬件特性)。两者都在 FlashAttention 家族里层层叠加。

---

## 6.5 反向重算(Recomputation):训练省显存的关键

TL;DR 里说"显存 O(N²)→O(N)",这个 O(N) 的胜利**来自前向只存统计量、反向重算**:

- **前向不存中间矩阵**:FA 前向只保存每行的 **m、l(FA1)或合并的 logsumexp LSE = m + log l(FA2 起,少一次 HBM 读)** 等 O(N) 统计量,不存完整的 N×N attention matrix;
- **反向重算 P**:反向时重新加载 Q/K/V 块,用存的 m、l **重算 attention scores/P**(而不是从显存读回),代价是约 **2× 的重复 FLOPs**;
- **为什么值得**:attention 是 memory-bound——重算的 FLOPs 便宜(HBM 带宽贵得多),显存从 O(N²) 省到 O(N),换来长序列训练/推理可行;
- 反向本身也 pass 化:FW 的 tiling 结构在 BW 同样能复用,配合(见 FA3 篇)多级 pipeline 重叠。

> **一句话**:FA 用"前向存统计量 + 反向重算"换取显存 O(N);算力换显存,在 memory-bound 的 attention 里是合算的买卖。

---

## 7. 实践要点(checklist)

1. **判断是否该上 FA**:attention 是 memory-bound;先用 roofline 估算算术强度,若低于脊点(内存受限)→ FA 收益最大。长序列(N 大)尤其明显。
1.5. **先看 mask 是否适配(选型第一问)**:FA 的 fused kernel 只支持 **有结构、可跳块** 的 mask(因果 causal / sliding window / ALiBi / padding);传入**任意 dense soft-mask** 需逐块读 mask、甚至回退到物化 S,FA 优势尽失——先确认你的 mask 类型再选 FA。
2. **记得它是 exact(数学精确,非 bitwise)**:FA 输出与标准 softmax **数学上相同、无近似**(不是稀疏近似),但浮点累加顺序不同,结果与朴素实现有 **ulp 级数值差异**(fp16 下更明显)——换了 kernel 后 loss 曲线不完全 bitwise-repro 是正常,不是 bug。
3. **显存收益来自反向重算**:无需存 N×N 中间矩阵,前向只存 LSE/m,l 统计量;训练长序列显存峰值 O(N²)→O(N),这是它让长 context 训练可行的关键(见 §6.5)。
4. **Block 大小 Br/Bc 的取舍**:受 SRAM 容量 M 和 head dim d 约束(`Bc≈M/(4d)`、`Br≈min(⌈M/(4d)⌉,d)`);**实数参考:A100 上 d=64 时 Br/Bc≈128、d=128 时≈64**(见 canonical §2.2.5)。Br/Bc 越大,每次 HBM 加载复用越多,但 SRAM 占满 → 调优平衡点。
4.5. **规避 FA1 低 occupancy**:FA1 只在 batch×head 维度并行,长序列小 batch(batch×heads 少)时 grid 喂不满 SM → 换 FA2 或开 FA2 的序列维 split-K。
5. **精细化路线**:要更高利用率先看 **work partitioning**(FA1→FA2 的主改进:threadblock/warp 划分、层级并行),再看硬件特性(TMA/WGMMA/FP8)。
6. **数值稳定性**:online softmax 的 running max 保证 `exp(s-m)` 不溢出;rescale 必须保留——去掉它**数学上错**(各块分母不一致),不是数值精度问题。

---

## 8. 溯源

- FA1 论文(IO-aware、tiling、online softmax、Θ(N²d²/M)): Dao et al., https://arxiv.org/abs/2205.14135 (NeurIPS 2022)
- FA2 论文(work partitioning、基于 CUTLASS 3.x): Dao, https://arxiv.org/abs/2307.08691 (ICLR 2024)
- Tri Dao 的 FA2 博客(25-40%→72% MFU、A100 230 TFLOP/s): https://princeton-nlp.github.io/flash-atttention-2/
- UW CSE 599M 讲义 *From Online Softmax to FlashAttention*(softmax 不结合→online softmax 的推导): https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
- canonical 库内文档 §2(各代原理/伪代码): `~/zhanbin/tracker/flash-attention-tracker/flash-attention-annual-report.md`
- 官方实现仓库(含 FA1/FA2 kernel 与 `benchmarks/` 复测): https://github.com/Dao-AILab/flash-attention
