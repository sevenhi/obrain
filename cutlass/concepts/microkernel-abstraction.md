---
title: 两级 Micro-kernel 抽象:BLIS 哲学如何搬到 GPU(CUTLASS 3.x)
type: concept
tags: [cutlass, cute, microkernel, blis, tiledmma, collective, partitioning]
source:
  - "pdf: A Generalized Micro-kernel Abstraction for GPU Linear Algebra (Thakkar/Cecka/Vuduc, BLIS Retreat 2023) | A Generalized Micro kernel Abstraction for GPU Linear Algebra.pdf (primary)"
  - "web: 同名讲稿公开 PDF (UT Austin BLIS Retreat 2023) | https://www.cs.utexas.edu/~flame/BLISRetreat2023/slides/Thakkar_BLISRetreat2023.pdf"
  - "web: CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design (NVIDIA Technical Blog, Thakkar et al., 2025-07) | https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/"
  - "web: CUTLASS: Principled Abstractions for Handling Multidimensional Data Through Tensors and Spatial Microkernels | https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels/"
  - "web: BLIS 上游仓库 | https://github.com/flame/blis"
date: 2026-09
---

# 两级 Micro-kernel 抽象:CUTLASS 3.x 为什么长成这样

## TL;DR

CPU 世界有 BLIS:核心 microkernel 是**单线程 agent 发 SIMD 指令**,多线程切分只动最外层循环,所以"一次写 microkernel,处处能用"。GPU 抄不了这个作业——核心 microkernel 是**多线程 agent(warp/warpgroup,甚至 warp-specialized 生产者消费者)发 3D tensor core 指令**,且 copy 和 compute 的 partitioning 图案不一致、每代架构都变。CUTLASS 3.x 的答案是**两级 micro-kernel 抽象**:

1. **Spatial micro-kernel**(空间级)= `cute::TiledMMA / TiledCopy`:把架构原语(Atom)按线程和数据平铺交织,统一描述"一次数学/拷贝操作的完整空间切分",从 Maxwell SIMT 到 Hopper WGMMA 一套接口;
2. **Temporal micro-kernel**(时间级)= `cutlass::gemm::collective::CollectiveMma/Epilogue`:把多个空间 microkernel 的"节拍"用架构特定的异步同步(mbarrier/TMA/warp specialization)编排成软件流水线,一个 collective 算完一个(可部分的)输出 tile。

**核心判断**:读 CUTLASS 3.x 源码时,先问自己"这段在哪一级"——Atom/TiledMMA 是空间级(数据怎么切给线程),Collective 是时间级(节拍怎么流水),Kernel 层只是两者的组合点 + 调度。两级解耦是"自定义 mainloop 复用官方 schedule、自定义 schedule 复用官方 mainloop"这种可组合性的来源。

---

## 1. 机制原理:GPU 为什么需要自己的 microkernel 概念

### 1.1 BLIS 的成功公式(CPU)

BLIS 的关键设计:

- **核心 microkernel 是单线程 agent** 发 1D SIMD 向量指令;
- **多线程 partitioning 只影响最外层循环**,microkernel 不感知并行;
- 分层暴露(1m/3m/5m level),各层可独立定制;
- 结果是 programming-model aware 但 architecture agnostic 的 CPU 线性代数概念化。

### 1.2 GPU 上三条公式全失效

| 维度 | CPU (BLIS) | GPU |
|---|---|---|
| microkernel 的 agent | 单线程 + 1D SIMD | **多线程 collective**(quadpair/warp/warpgroup),发 **3D tensor core** 指令 |
| partitioning | 只动最外层循环 | **每代 tensor core 都换线程×数据布局**,且 copy 与 compute 的 partitioning 图案**不共享** |
| 执行模型 | 同步为主 | 程序员管理的 scratchpad 上的**深软件流水线**;warp-specialized async producer/consumer;Hopper 连 MMA 本身都是异步的(wgmma) |

**partitioning 的定义**(原文):*Determining which logical coordinates of data map to a given parallel agent*——决定数据的哪些逻辑坐标映射到哪个并行 agent。GPU 上这件事无法跨代泛化,是 CUTLASS 2.x 痛苦的根源。

### 1.3 不变量与变化量:抽象的切分线在哪

论文给出的架构无关概念化的依据——把 GPU 线性代数拆成"跨代不变"与"每代都变"两组:

**跨代不变**(可以做进抽象):
- global memory 的 layout 语义(行/列主序、batch stride)不随代际变化;
- GPU 可建模为**任意层数的嵌套并行 agent + 对应层级的内存空间**;
- 嵌套 tiling 是带宽高效计算的根本策略;通常只有最外层 tiling 是动态的(用户输入 shape/stride 动态),最内层循环全静态(理想指令调度、峰值发射率);
- 用程序员管理的异步做软件流水线来隐藏数据搬运延迟。

**每代都变**(必须封装进架构特定实现):
- 架构原生 MMA 指令集的线程与数据布局;
- 异步流水线的结构;
- **agent-locale 层级的深度本身**(Volta 是 warp 级,Hopper 出了 warpgroup + TMA + cluster)。

CUTLASS 3.x = 对不变量建接口(Atom/TiledMMA/Collective 概念),对变化量建 traits(每代架构的 `MMA_Traits`、dispatch policy)。

### 1.4 两级 microkernel 与五层 API

概念层级(自底向上),对应 API 入口:

| 层 | 概念 | API 入口 | 组合点(可定制什么) |
|---|---|---|---|
| Atom | 架构指令 + 元信息:完成一次 math/copy 所需的**最少线程数与值数** | `cute::MMA_Atom` / `cute::Copy_Atom` | 复数 MMA、semiring GEMM 等自定义算子 |
| Tiled MMA/Copy(**空间 microkernel**) | 一次操作的**完整空间 tiling** | `cute::TiledMma<>` / `TiledCopy<>` | 架构无关外循环 ↔ 架构特定指令之间的稳定接口 |
| Collective(**时间 microkernel**) | 完整**时间 tiling(pipeline)** | `cutlass::gemm::collective::CollectiveMma/Epilogue/Conv<>` | mainloop 融合(如 3xTF32);由 DispatchPolicy 定 pipeline 结构 |
| Kernel | grid 中全部 CTA/cluster 的集合:launch、grid planning、负载均衡 | `cutlass::gemm::kernel::GemmUniversal<>` | kernel 级融合(如 B2B GEMM);mainloop × epilogue × schedule 组合 |
| Device | host 侧装配 | `cutlass::gemm::device::GemmUniversalAdapter<>` | 2.x/3.x kernel 通用,无状态包装 |

"空间/时间"这对词是本文最重要的词汇:**TiledMMA 管"数据怎么切给线程"(空间),Collective 管"节拍怎么流水"(时间)**;每个 temporal microkernel 用架构特定的同步模式编排多个空间 microkernel 的 beat,并可与任意外层循环 schedule(stream-K/split-K、L2 swizzle、persistent)组合。

---

## 2. 可复现细节

### 2.1 MMA Atom:一条指令的完整语义固化

Atom = `MMA_Op`(裸 PTX)+ `MMA_Traits`(PTX 元信息)→ 提供 checked call 接口与 fragment 生成。Traits 编码 ISA 里**不表达**的语义:值类型、collective 的线程布局、`[thr_id, val_id] → 逻辑坐标 (M,N,K)` 的数据布局映射。三个真实 traits(原 PPT 摘录):

```cpp
// Pascal int8 点积 DP4A:单线程,"tensor core 之前"的起点
template <> struct MMA_Traits<SM61_DP4A> {
  using ElementDVal = int32_t;  using ElementAVal = int8_t;
  using ElementBVal = int8_t;   using ElementCVal = int32_t;
  using ThrID   = Layout<Shape<_1>>;
  using ALayout = Layout<Shape<_1,_4>>;   // 1 线程 4 个 A 值
  using BLayout = Layout<Shape<_1,_4>>;
  using CLayout = Layout<Shape<_1,_1>>;
};

// Volta 第一代 tensor core:8 线程 quadpair 协作,8x8x4
template <> struct MMA_Traits<SM70_8x8x4_F32F16F16F32_NT> {
  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = Layout<Shape <_4,_2>, Stride<_1,_16>>;
  // (T8,V4) -> (M8,K4)
  using ALayout = Layout<Shape <Shape <_4,_2>,_4>, Stride<Stride<_8,_4>,_1>>;
  // (T8,V8) -> (M8,N8)
  using CLayout = Layout<Shape <Shape <_2,_2,_2>, Shape <_2,_2,_2>>,
                         Stride<Stride<_1,_16,_4>, Stride<_8,_2,_32>>>;
};

// Ampere fp16:整 warp 协作,16x8x8
template <> struct MMA_Traits<SM80_16x8x8_F32F16F16F32_TN> {
  using Shape_MNK = Shape<_16,_8,_8>;
  using ThrID   = Layout<_32>;
  using CLayout = Layout<Shape <Shape <_4,_8>,_2>, Stride<Stride<_16,_1>,_8>>;
};
```

注意布局方向:traits 里的 layout 是 **(thr,val) 域 → MNK 逻辑坐标**;使用侧一般想要它的逆(MNK → thr,val),由 CuTe 代数的 `right_inverse/left_inverse` 完成。三代指令的 ThrID/值布局完全不同——这正是"没有 trivial 方法把 partitioning 泛化跨代"的具体形态,也是 Atom 抽象要吞掉的复杂度。

### 2.2 TiledMMA:空间 microkernel 的三个旋钮

```cpp
template <class MMA_Atom,
          class AtomLayoutMNK = Layout<Shape<_1,_1,_1>>,  // atom 平铺
          class ValLayoutMNK  = Layout<Shape<_1,_1,_1>>,  // 每线程值复制
          class PermutationsMNK = Tile<Underscore,Underscore,Underscore>> // 数据置换
struct TiledMMA : MMA_Atom {
  using ThrLayoutVMNK = decltype(tiled_product(AtomThrID{}, AtomLayoutMNK{}));
};
```

三旋钮语义:**AtomLayout**(把几个 atom 铺给几个线程组)、**ValLayout**(每线程重复几份值,喂更大 tile)、**Permutation**(置换数据 tiling 换更友好的布局——例如 Volta 需要 2x2 空间交织让一条 128-bit smem load 喂两个 MMA 的高低 64 位)。跨代实例(原 PPT,同一接口):

```cpp
// Hopper warpgroup MMA:atom 本身就大到 64x128x16,一行完事
TiledMma<MMA_Atom<SM90_64x128x16_F32F16F16_SS>>;
// Volta:交织 atom layout(ComposedLayout+Swizzle) + 2x2 值平铺 + M/N 置换
TiledMma<MMA_Atom<SM70_8x8x4_F32F16F16F32_NT>,
         ComposedLayout<Swizzle<1,0,-1>{},
                        Layout<Shape <Shape <_2,_2>,Shape <_2,_2>>,
                               Stride<Stride<_1,_4>,Stride<_2,_8>>>>,
         Layout<Shape<_2,_2>>{},
         Tile<Layout<Shape<_4,_2,_2>,Stride<_1,_32,_8>>,
              Layout<Shape<_4,_2,_2>,Stride<_1,_32,_8>>>>;
// Maxwell SIMT fp32(没有 tensor core 也能表达)
TiledMMA<MMA_Atom<UniversalFMA<float>>,
         Layout<Shape<_8,_16,_1>>, Layout<Shape<_2,_2,_1>>,
         Tile<Underscore, Layout<_2,_16>>>;
```

### 2.3 generalized partitioning:终于能写规范三重循环

`ThrMMA = TiledMMA 按 threadIdx 取的线程切片`,提供 partition + slice 工具。有了它,无论底下的线程×数据切分多复杂,算法层都是 BLIS 式规范循环:

```cpp
Tiled_MMA mma = make_tiled_mma(SM90_16x8x4_F64F64F64F64_TN{},
                               Layout<Shape<_2,_2>>{},   // 2x2 atom tile
                               Layout<Shape<_1,_1>>{},   // 1x1 value tile
                               tuple<Layout<_2,_8>,_,_>{}); // M-perm
Thr_MMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(A)); // A 任意 layout
Tensor tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(B));
Tensor tCrC = thr_mma.partition_C(C);
// 规范三重循环:partitioning 的复杂性全被 TV layout 吸收
for (int k = 0; k < size<1>(tCrA); ++k)
  for (int m = 0; m < size<0>(tCrA); ++m)
    for (int n = 0; n < size<0>(tCrB); ++n)
      cute::gemm(mma, tCrC(m,n), tCrA(m,k), tCrB(n,k));
```

对照 CUTLASS 2.x:同样的事要靠**每代架构手写 iterator/layout functor**,"indexing math 难写更难维护"。TiledCopy 结构完全同构(`Copy_Op → Copy_Traits → Copy_Atom → TiledCopy → ThrCopy`,多出 `retile_D` 把目标 fragment 重排成 copy 视角),并且 `make_tiled_copy_A(copy_atom, tiled_mma)` 可以**直接继承 MMA 的 accumulator 布局**——这就是"copy 与 compute partitioning 图案不一致"问题的工程解:让 copy 按 math 的图案重铺。

### 2.4 Collective:时间 microkernel 与 dispatch policy

```cpp
template <class DispatchPolicy, class TileShape,
          class ElementA, class SmemLayoutA, class ElementB, class SmemLayoutB,
          class ElementC, class ArchTag, class TiledMma,
          class GmemCopyAtomA, class SmemCopyAtomA,
          class GmemCopyAtomB, class SmemCopyAtomB>
struct CollectiveMma;

// dispatch policy 例:policy 名字直接编码 pipeline 结构
struct MainloopSm70TwoStage;               // 2 级流水:1 级 smem + 1 级 rmem,谓词化 gmem load
struct MainloopSm80CpAsyncUnpredicated;    // smem n-buffer(cp.async)+ 寄存器流水
struct MainloopSm90TmaGmmaWarpSpecialized; // TMA n-buffer + Hopper GMMA,warp-specialized 动态调度
```

约束与自由度:每个 collective **只算一个(可部分的)输出 tile**;warp-specialized、producer/consumer 异步都封在 policy 里;外层 schedule(split-K/stream-K、tile swizzling for L2、persistent)通过 dispatch policy 组合。自定义 kernel 的标准姿势就是写一个新 policy struct(Stages/ClusterShape/ArchTag/Schedule 四件套)然后特化 CollectiveMma——"plug and play with dispatch policies"。

Hopper warp-specialized persistent 形态(原文伪码):producer warpgroup 跑 `collective_mainloop.dma()`(TMA 进),consumer warpgroup 跑 `collective_mainloop.mma()`(主循环+epilogue),各自 `scheduler.advance_to_next_work()` 领 tile——TMA、TensorCore、Epilogue 三种异步机器在同一时间轴上流水,这正是 BLIS 内层 microkernel 无法泛化的场景。

---

## 3. 数据/事实对比

- **CUTLASS 2.x 的复杂度账单**:每代架构要求新定义一批 layout/iterator,"dozens of bespoke layouts/iterators";3.x 用**单一 `Layout` 词汇类型 + 形式代数**(composition/complement/left_inverse/right_inverse/product/divide)全部收编——复杂度从 N 套手写 indexing math 收敛为 1 套可组合代数(详见 [cute-layout-algebra.md](cute-layout-algebra.md))。
- **表达力覆盖面**:同一 TiledMMA 接口跨 Maxwell(SIMT FMA)→ Pascal(DP4A)→ Volta(8x8x4 quadpair)→ Turing/Ampere(16x8x8 warp)→ Hopper(64x128x16 warpgroup SS),含 Swizzle 交织与 M/N 置换;GETT(广义张量收缩)在 3.0 里只是"分层 stride 的 GEMM"——rank-3 (M,K,L)/(N,K,L)/(M,N,L) 算法与 batched GEMM 共用同一循环体,Hopper GEMM 被描述为"GETT in disguise"。
- **生态量级**(原 PPT 2023 口径):CUTLASS 近 **200 万次下载/月**,数百应用与框架集成;本库 index 快照为 ~10.3k stars(2026-09)。
- **API 面收窄**:2.x 暴露大量 iterator/ functor 组合;3.x 入口收敛为四层五个模板名(TiledMma/TiledCopy → Collective* → GemmUniversal → GemmUniversalAdapter),配合 static_assert 防非法组合——"correctness by construction"。

---

## 4. 实践要点

1. **读 3.x kernel 源码先定级**:看到 `cute::` 前缀 = 空间级(Atom/TiledMMA/TiledCopy/Layout);看到 `collective::` = 时间级(pipeline/同步);`kernel::GemmUniversal` 只做组合与调度。混着读是 3.x 源码劝退的主因。
2. **自定义 mainloop ≠ 从零写**:写 dispatch policy struct + 特化 CollectiveMma,即可复用官方 tile scheduler/stream-K;反过来自定义 schedule 也能复用官方 mainloop。这是论文反复强调的"组合点"设计,别绕过它直接改 GemmUniversal。
3. **copy 对齐 math**:用 `make_tiled_copy_A/B(atom, tiled_mma)` 让 smem→reg 的搬运直接继承 MMA 的 fragment 布局,再 `retile_D` 重排目标;手写两边布局然后对拍是 Volta 时代的老路,3.x 不要走。
4. **Permutation 不是装饰**:Volta 的 2x2 空间交织(一条 128-bit LDSM 喂两个 MMA)靠 PermutationsMNK 表达;新架构 smem 排布约束(swgemm descriptor 的 layout atom,见 [hopper-wgmma.md](hopper-wgmma.md))同理。遇到"布局对但性能差",先查 permutation/swizzle。
5. **GETT/批处理别绕开 CuTe**:多维模(multi-mode)的 stride 用嵌套 `Stride<...>` 组合表达,`cute::gemm` 的 rank-3 循环直接吃分层 tensor;把 batch 维摊平回 rank-2 会丢掉 L2 swizzle 与 scheduler 的既有支持。
6. **static assert 是朋友**:3.x 在组合处放满 static_assert(非法 layout/alignment/schedule 组合直接编译失败),报错误导信息时优先怀疑组合非法而不是改 assert。

## 溯源

- 主源:Thakkar (GaTech & NVIDIA), Cecka (NVIDIA), Vuduc (GaTech), *A Generalized Micro-kernel Abstraction for GPU Linear Algebra*, BLIS Retreat 2023-08-28 讲稿(库内 PDF,页眉标注 NVIDIA CONFIDENTIAL——只提炼观点与公开代码模式,不扩散原件);公开镜像为 UT Austin flame 组 BLISRetreat2023 slides。
- 交叉源:NVIDIA Technical Blog 两篇 CUTLASS 3.x 系列文(2025-07,五层 hierarchy/Collective 职责与本库描述一致);BLIS 上游 README(https://github.com/flame/blis)。
- 库内互链:[gemm-tiling.md](gemm-tiling.md)(tiling 全景)、[cute-layout-algebra.md](cute-layout-algebra.md)(Layout 代数本体)、[hopper-wgmma.md](hopper-wgmma.md)(时间级在 Hopper 的同步形态)、[tile-scheduler.md](tile-scheduler.md)(外层 schedule)。
