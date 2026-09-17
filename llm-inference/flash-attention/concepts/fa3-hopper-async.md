---
title: FlashAttention-3 的 Hopper 异步内核:WGMMA + TMA + 计算通信重叠
type: concept
tags: [flashattention, fa3, hopper, wgmma, tma, asynchrony, fp8, pipeline]
source:
  - "web: Tri Dao FA3 Blog | https://tridao.me/blog/2024/flash3/"
  - "pdf: FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | arXiv:2407.08608 NeurIPS 2024"
date: 2026-09
---

# FlashAttention-3 的 Hopper 异步内核:WGMMA + TMA + 计算通信重叠

## TL;DR

FA3(2024.07,Hopper H100/H800)的目标是在 Hopper 上达到**接近 GEMM 的效率**。它用 Hopper 的**三件硬件特性**把 attention kernel 推满:

1. **异步 WGMMA**(warp-group MMA):一个 warp group(4 warps/128 线程)可**异步发起**矩阵乘,不用等它做完就能干别的;
2. **TMA**(Tensor Memory Accelerator):硬件单元**异步搬数据**(HBM→shared memory),不占 warp;
3. **计算-通信重叠 pipeline**:让 **softmax、matmul、TMA load 三者重叠**执行,配合 **warp specialization**(producer warp 搬数据、consumer warpgroup 算)。

外加 **FP8 低精度**(block quantization + incoherent processing 保精度):FP8 前向达 ~1.2-1.3 PFLOPS;BF16 相对 FA2 加速 1.5-2×。

> **心法**:FA1/FA2 解决"搬得少"(IO)和"算得满"(并行),FA3 再进一步——**在 Hopper 上让运算与搬运同时发生**,用异步**隐藏内存延迟**,让 GPU 计算单元几乎不空闲。

---

## 1. 为什么需要异步:Ampere 的同步局限

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

---

## 2. 三层异步:WGMMA、TMA、overlap

### 2.1 异步 WGMMA(warp-group MMA)

Hopper 的 `wgmma` 由**一个 warp group(4 warps = 128 线程)异步发起**:

```
FA3 (Hopper, 异步 WGMMA):
  wgmma.async(Q_block @ Kᵀ)   # 发起,立即返回(不阻塞)
  wait() → S                   # 等结果就绪
  softmax(S)                   # 同时已发起下一个 K/V 的 TMA load
  wgmma.async(P @ V)          # 发起,立即返回
```

`wgmma.async` **发出即返回**,计算单元可以继续发起下一次 matmul 或做 softmax,结果到了再 `wait()`——重叠由此产生。

> **典型 tile 配置**:FA3 常用 Q/K/V tile 128×128(/64×64 视 head_dim),CTA 内 1 producer + 2 consumer warpgroup(pingpong);`wgmma` 显式指令流是 `fence`(刷寄存器依赖)→ `commit_group`(提交批)→ `wait_group`(等整批完成),配合 `setmaxnreg` 给 producer/consumer 分配不同寄存器数。

### 2.2 TMA(Tensor Memory Accelerator,Hopper 硬件搬运)

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

### 2.3 计算-通信重叠 pipeline

FA3 的关键创新:让 **TMA load、WGMMA、softmax 三者重叠**:

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

**结果**:GPU 计算单元几乎不空闲,attention 逼近 GEMM 效率(BF16 ~75% MFU;FP8 ~61-66% MFU,见 §3)。

---

## 3. FP8 低精度(低精度路径)

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
- BF16:最高 **~740 TFLOPS**(论文摘要 "up to 740 TFLOPs/s";博客 620-640 常规流水),约 **~75% 利用率**(相对 H100 BF16 稠密峰值 989);
- FP8:~**1.2-1.3 PFLOPS** 前向,相对 H100 FP8 理论峰值约 **61-66% MFU**(不是 80%);
- **加速基线**:BF16 相对 FA2 实测 **1.5-2×(指 BF16 加速)**,不是 FP8 的收益。

---

## 4. 实现细节与特性

- 基于 **CUTLASS 3.x / CuTe 模板**(C++ 模板,非 DSL;独立包 `flash_attn_3`);
- **Pingpong scheduling(FA3 论文招牌技术)**:CTA 内 **1 个 producer warpgroup + 2 个 consumer warpgroup**,用 `bar.sync` 强制交错,使 WG1 的 softmax 落在 WG2 的 GEMM 影子里(FP16 前向 570→620-640 TFLOPS;论文 §4.2 ablation 同源 570→661 TFLOPS 是 pingpong + warp specialization 合计口径,勿与单独 pingpong 混淆);
- **前向 2-stage / 反向 3-stage pipeline**:论文的 2-stage 特指 **GEMM-softmax 交错流水**(Algorithm 2,跨迭代用寄存器缓冲打破 S→softmax→P 的串行依赖),不是简单的 load-then-compute;反向 3-stage 更复杂 overlap;
- **反向:producer + consumer + dQ writer 三角色**,仍 per-tile 重算 attention matrix + 3-stage overlap 训练读者最易在这踩坑——反向不只是"重叠",多一个写 dQ 的角色参与流水;
- **动机数字(为什么必须隐藏 softmax)**:H100 FP16 Tensor Core matmul 989 TFLOPS vs 特殊函数单元(exp)吞吐极低(论文/博客估算 ≈3.9 TFLOPS 量级,比 matmul **低 2-3 个数量级**),seqlen 8K/hd128 下 exp 可占 ~50% 周期,FP8 下更糟——所以要把 softmax 藏进 GEMM 的影子;
- 配合 `setmaxnreg` 寄存器重分配(producer 少寄存器、consumer 多);WGMMA 用 `fence/commit_group/wait_group` 显式指令流;
- 特性:FP16/BF16 前后向、**FP8(E4M3)前向**、varlen、**paged KV cache**、**SplitKV**(长序列把 KV 维 split,最后 reduce)。

> **当前状态(2026 时效)**:FA3 的 `hopper/` 目录长期停留在 **beta**(至 beta25);**Blackwell SM100 不支持 WGMMA,改用第五代 Tensor Core 指令 tcgen05(其异步矩阵乘指令 UMMA,Unordered Multi-precision Matrix Multiply),故 FA3 无法上 B200**——由 FA4(CuTe DSL)接管(B200 用户走 FA4)。

---

## 5. 实践要点(checklist)

1. **FA3 只在 Hopper 有效(B200 走 FA4)**:依赖 WGMMA/TMA(SM90);Ampere(A100)用 FA2;**Blackwell SM100 不支持 WGMMA(改 tcgen05),FA3 无法上 B200,由 FA4(CuTe DSL)接管**。
2. **FP8 必须做 block quantization + incoherent processing**:FA3 的 FP8 是 Q/K/V 都按块量化 + 对 Q/K 乘 Hadamard 消 outlier(2.6× 误差降低)——不是简单把 Q 换 E4M3;跳过这两项,outlier 会让精度崩。
3. **pipeline 级数要调**:前后向 2/3-stage 是 GEMM-softmax 交错流水的骨架;**改动 barrier/TMA mbarrier、WGMMA 的 commit_group/wait_group 的任何同步次序都可能引入 race**,务必参照 cutlass 流水篇的同步模型。
4. **warp specialization 是核心且是"1 producer + 2 consumer"的 pingpong**:producer/consumer 分工交错让 softmax 藏进 GEMM 影子——读 FA3 kernel 先看这层。
5. **长序列用 SplitKV**:KV 维 split 到多 block 增高并行,注意有最终 reduce(同 cutlass split-K)。
6. **反向也重叠**:FA3 反向靠 3-stage pipeline 重算 attention matrix 来 overlap——训练场景它同样重要。
7. **性能口径别混**:BF16(~75% 利用率)相对 FA2 加速 1.5-2×;FP8 是 ~1.2-1.3 PFLOPS(61-66% MFU)——两条曲线是不同维度。

---

## 6. 溯源

- Tri Dao FA3 博客(overlap、WGMMA/TMA、FP8、pingpong 拓扑;570→620-640 TFLOPS、exp≈3.9 TFLOPS 等博客口径数字): https://tridao.me/blog/2024/flash3/
- FlashAttention-3 论文(§3.2 pingpong scheduling、§3.3 block quantization + incoherent processing 的 128×128/1×128 粒度、§5 性能与 2.6× 误差;NeurIPS 2024): https://arxiv.org/abs/2407.08608
- 相关:CUTLASS 领域的 WGMMA/TMA/epilogue 篇(同硬件特性,跨参考)
