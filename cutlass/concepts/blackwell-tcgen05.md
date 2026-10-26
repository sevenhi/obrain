---
title: Blackwell tcgen05 / UMMA:第五代 Tensor Core 指令与 TMEM
type: concept
tags: [cutlass, blackwell, sm100, sm107, sm120, tcgen05, umma, tmem, cute, mma-atom, cta-group]
source:
  - "code: include/cute/arch/mma_sm100_umma.hpp (main, 2041 行) | github.com/NVIDIA/cutlass ↑primary"
  - "code: include/cute/arch/mma_sm100_desc.hpp (main) | github.com/NVIDIA/cutlass"
  - "code: include/cute/arch/tmem_capacity_sm100.hpp (main) | github.com/NVIDIA/cutlass"
  - "code: include/cute/arch/tmem_allocator_sm100.hpp (main) | github.com/NVIDIA/cutlass"
  - "web: PTX ISA tcgen05 章节 | https://docs.nvidia.com/cuda/parallel-thread-execution/"
date: 2026-10
---

# Blackwell tcgen05 / UMMA:第五代 Tensor Core 指令与 TMEM

## TL;DR

**tcgen05 是 Blackwell(SM100)起的第五代 Tensor Core 指令集**,取代 Hopper 的 `wgmma.mma_async`。它的核心变化:MMA 升级为 **CTA-group 级异步操作**(单 CTA = `cta_group::1`、双 CTA 协作 = `cta_group::2`),累加器不再落寄存器而是落 **TMEM(tensor memory)**。CuTe 把它包装成 **`SM100_MMA_*`** 命名的 MMA atom(真实文件 `include/cute/arch/mma_sm100_umma.hpp`,2041 行),写法与 SM90 的 `SM90_64x64x16_*` 完全不同——**`SM100_MMA_F16BF16_SS` 这种命名,不再是 `SM90_64x64x16_*` 风格**。

**TMEM 规格(tmem_capacity_sm100.hpp)**:每 SM **128 rows × 512 cols(SM107 = 576 cols)**,每 cell 32-bit;分配用 `tcgen05.alloc`(列数 power-of-2 ∈ [32, 512],SM107 可到 576)。tcgen05 已延展到 **SM107(Rubin)/ SM120**(呼应库内 CUTLASS v4.8.0 新增 Rubin SM107 初步支持)。

**与 Hopper 的关键区别一句话**:Hopper 的 wgmma 是「warpgroup 级、累加器在寄存器、B 恒 SMEM」;tcgen05 是「CTA-group 级、累加器 D 在 TMEM、SS/TS 对应 A/B 来源 SMEM/TMEM」——CTA-group 级 + TMEM 是这一代的两个结构性变化。

---

## 1. 机制:与 Hopper wgmma 的关键区别

> 下图并排 Hopper wgmma 与 Blackwell tcgen05,点出「执行粒度升到 CTA-group、累加器搬进 TMEM」两个结构性变化:

<div class="archify-figure">
<iframe src="../../../diagrams/cl-tcgen05-vs-wgmma.workflow.html"
        style="width:100%;height:560px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="tcgen05 vs Hopper wgmma:两个结构性变化"></iframe>
</div>


| 维度 | Hopper wgmma(SM90) | Blackwell tcgen05/UMMA(SM100) |
|---|---|---|
| 执行粒度 | warpgroup(4 warps = 128 线程) | **CTA-group**(单 CTA 或 2-CTA cluster) |
| 累加器 D 位置 | 寄存器(RMEM) | **TMEM(tensor memory)** |
| 异步性 | async proxy 异步 | 异步(同样要显式同步) |
| A 来源 | SMEM(SS)或 RMEM(RS) | SMEM(SS)或 **TMEM**(TS) |
| B 来源 | 恒 SMEM | 恒 SMEM(或 TMEM,见 TS 变体) |
| atom 命名 | `SM90_64x64x16_F16F16F16_SS` | **`SM100_MMA_F16BF16_SS`** |
| PTX 指令 | `wgmma.mma_async...` | **`tcgen05.mma.cta_group::1/2.kind::f16 ...`** |

**`SS` / `TS` 后缀含义**(对照 Hopper 的 SS/RS):

- **SS** = A、B 都来自 **SMEM**(shared memory);
- **TS** = A 来自 **TMEM**、B 来自 SMEM(`SM100_MMA_F16BF16_TS` 的 static_assert 明确:`a_major == UMMA::Major::K`,A 从 TMEM 时**不可转置**)。

**`cta_group::2` = 2-CTA MMA**:两个相邻 CTA 组成 cluster 协同发射同一条 MMA,把 M 维切成上下两半各自算,累加器 D 写进拼接的 TMEM。这是 FA4「2-CTA MMA」的指令基础(应用侧收益见 [../../attention/implementation/flash-attention-implementation.md](../../attention/implementation/flash-attention-implementation.md) §4:TMEM 容量加倍、smem 带宽减半)。

> 下图展示 2-CTA MMA 的协作:两个相邻 CTA 各算 M 维一半,累加器写进拼接 TMEM,这是 FA4 的指令基础:

<div class="archify-figure">
<iframe src="../../../diagrams/cl-tcgen05-2cta.workflow.html"
        style="width:100%;height:560px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="2-CTA MMA:cta_group::2 的协作"></iframe>
</div>

---

## 2. CuTe atom:真实文件与命名

> 下图展开 SM100_MMA_* atom 命名空间:TF32/F16BF16 × SS/TS × SCALED/SPARSE × 2x1SM 的组合:

<div class="archify-figure">
<iframe src="../../../diagrams/cl-tcgen05-atom-naming.tree.html"
        style="width:100%;height:560px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="SM100_MMA_* atom 命名空间"></iframe>
</div>


**文件位置**(不是 SM90 那套 `SM90_64x64x16_*`):

- atom 本体:`include/cute/arch/mma_sm100_umma.hpp`(main,2041 行);
- descriptor:`include/cute/arch/mma_sm100_desc.hpp`(957 行,`UMMA::InstrDescriptor` 等);
- TMEM 容量:`include/cute/arch/tmem_capacity_sm100.hpp`;
- TMEM 分配器:`include/cute/arch/tmem_allocator_sm100.hpp`;
- SM107/SM120 延展:`mma_sm107_umma.hpp` / `mma_sm120.hpp` / `mma_sm120_sparse.hpp`。

**atom 命名样例**(grep 真实文件所得):

```
SM100_MMA_TF32_SS             SM100_MMA_TF32_TS
SM100_MMA_F16BF16_SS          SM100_MMA_F16BF16_TS
SM100_MMA_TF32_SS_SCALED      SM100_MMA_F16BF16_SS_SCALED
SM100_MMA_TF32_SS_SPARSE      SM100_MMA_F16BF16_SS_SPARSE
SM100_MMA_TF32_2x1SM_SS       SM100_MMA_F16BF16_2x1SM_SS   ← 2-CTA(cluster)版本
SM100_MMA_TF32_2x1SM_TS       SM100_MMA_F16BF16_2x1SM_TS
```

### 2.1 可复现:一个 atom 的 PTX 真身(mma_sm100_umma.hpp 节选)

`SM100_MMA_F16BF16_SS` 单 CTA 版的形状约束与内联 PTX:

```cpp
// struct SM100_MMA_F16BF16_SS (mma_sm100_umma.hpp:86,节选)
static_assert(M == 64 || M == 128, "SM100_MMA_F16BF16 M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
static_assert(N 是 8 的倍数且在 [8, 256], "SM100_MMA_F16BF16 N-mode ...");

asm volatile(
  "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
  ...);
```

对照 Hopper atom 的 PTX `wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16`——tcgen05 用 `.cta_group::N.kind::<dtype>` 表达粒度与类型,寄存器操作数 `{%5..%8}` 是被 TMEM 地址指针间接引用的累加器「槽位」,`[%0]` 是 TMEM 目标地址。单 CTA 版 M ∈ {64,128};**2x1SM 版 M ∈ {128,256}**(两个 CTA 各算一半 M)。

### 2.2 descriptor(desc 头文件)

与 Hopper 的 `make_gmma_desc`(64-bit SMEM descriptor)对应,tcgen05 侧有 `UMMA::InstrDescriptor`(mma_sm100_desc.hpp:424,联合体),由 `UMMA::make_instr_desc<...>()` 从操作数布局构造,编码 `a_format_` / `b_format_` / `c_format_`(累加器格式)与 SMEM/TMEM 操作数描述;`UMMA::Layout_MN_XXX_Atom` / `Layout_K_XXX_Atom` 提供 canonical 布局 atom(对应 Hopper 的 8 种 GMMA layout atom)。

---

## 3. TMEM:tensor memory(非通用寄存器文件)

> 下图走一遍 TMEM 的「分配 → 累加 → 释放」生命周期,并给出 128×512 规格与「为什么搬进 TMEM」的理由:

<div class="archify-figure">
<iframe src="../../../diagrams/cl-tcgen05-tmem.dataflow.html"
        style="width:100%;height:560px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="TMEM 分配生命周期"></iframe>
</div>


**TMEM(tensor memory)是 Blackwell 新增的片上存储,专为 tensor core 累加器设计**——精确表述是「tensor memory(非通用寄存器文件)」,不要理解成「另一块通用 shared memory 或寄存器堆」。

**规格(tmem_capacity_sm100.hpp,一手)**:

```cpp
namespace cute::TMEM {
static constexpr int Sm100TmemCapacityRows    = 128;
static constexpr int Sm100TmemCapacityColumns = 512;
static constexpr int Sm107TmemCapacityColumns = 576;   // SM107(Rubin)
// TargetTmemCapacityBits = 128 * TargetTmemCapacityColumns * 32   ← 每 cell 32-bit
}
```

- **128 rows × 512 cols(SM100),SM107 = 576 cols,每 cell 32-bit**;
- SM107/SM120 延展:`__CUDA_ARCH__ == 1070` 取 576 列,`>= 1000 && < 1200` 取 512 列;
- **分配器 `tcgen05.alloc`**(tmem_allocator_sm100.hpp:77):

```
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;
```

- **列数约束**:基础指令 power-of-2 ∈ **[32, 512]**(SM107 可到 576);`ColumnsPerAllocationSlice = 32` 是单次分配片粒度;释放用 `tcgen05.dealloc`、让出许可用 `tcgen05.relinquish_alloc_permit`。

**为什么累加器搬进 TMEM**:Hopper 上 wgmma 累加器占大量寄存器,限制单 CTA 的 tile 尺寸;tcgen05 把累加器放 TMEM,寄存器压力大减 → 单 CTA 能塞更大的 Q/K/V tile,2-CTA 拼接又把可驻留 tile 翻倍(FA4 的首要结构手段,见 §1)。

---

## 4. 代际延展:tcgen05 → SM107 / SM120

- **SM100(Blackwell)**:tcgen05/UMMA 首代,`mma_sm100_umma.hpp` + `mma_sm100_desc.hpp`;
- **SM107(Rubin)**:`mma_sm107_umma.hpp` / `mma_sm107_desc.hpp`,TMEM 576 列——**呼应库内 CUTLASS 域已记 v4.8.0 新增 Rubin SM107 初步支持**(见 [../index.md](../index.md));
- **SM120**:`mma_sm120.hpp` / `mma_sm120_sparse.hpp`——tcgen05 体系继续延展。

**与 FA 代次的衔接**:FA3 依赖 wgmma,SM100 不支持 wgmma → FA3 上不了 B200;FA4 改用 tcgen05/UMMA + TMEM + 2-CTA MMA(见 [../../attention/implementation/flash-attention-implementation.md](../../attention/implementation/flash-attention-implementation.md) §4)。本域 [hopper-wgmma.md](hopper-wgmma.md) 讲 Hopper 指令机制,本篇讲 Blackwell 指令机制,FA4 篇讲应用收益,三层不重叠。

---

## 5. 实践要点(checklist + 坑)

1. **命名认准 `SM100_MMA_*`**:不是 `SM90_64x64x16_*` 风格;TF32/F16BF16 × SS/TS × SCALED/SPARSE × (2x1SM) 组合,查 `mma_sm100_umma.hpp` 真实类名再写。
2. **M 维约束记牢**:单 CTA M ∈ {64,128}(F16BF16);`2x1SM`(2-CTA cluster)M ∈ {128,256}。
3. **TS 模式 A 不可转置**:`SM100_MMA_F16BF16_TS` static_assert `a_major == UMMA::Major::K`——A 从 TMEM 时必须是 K-major。
4. **TMEM 是累加器专属,不是通用存储**:128×512(SM100)、每 cell 32-bit;分配走 `tcgen05.alloc`,列数 power-of-2 ∈ [32,512],别当 shared memory 用。
5. **2-CTA MMA 要成对 CTA 协作**:`cta_group::2` 需要两个相邻 CTA 组成 cluster,各自 M 维一半;单 CTA 发 `cta_group::2` 是错的。
6. **编译目标**:tcgen05 属 architecture-accelerated 特性,SM100 需 `sm_100a`(类比 Hopper 的 `sm_90a`,见 [../frameworks/cutlass-build-and-usage.md](../frameworks/cutlass-build-and-usage.md) §2)。
7. **同步**:异步 MMA 仍需显式同步(mbarrier / tcgen05 的完成等待),别照搬 wgmma 三件套名(`wgmma.fence/commit/wait`),tcgen05 侧用 TMEM 相关的 fence/mbarrier 原语。

---

## 6. 溯源

- **mma_sm100_umma.hpp**(main,2041 行;atom 命名、形状约束、`tcgen05.mma.cta_group::1/2.kind::*` PTX):https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_umma.hpp
- **mma_sm100_desc.hpp**(`UMMA::InstrDescriptor`、`make_instr_desc`、Layout atom):https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_desc.hpp
- **tmem_capacity_sm100.hpp**(128×512/SM107=576/32-bit):https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/tmem_capacity_sm100.hpp
- **tmem_allocator_sm100.hpp**(`tcgen05.alloc/dealloc`、power-of-2 ∈ [32,512]、slice=32):https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/tmem_allocator_sm100.hpp
- **SM107/SM120 延展文件**:`mma_sm107_umma.hpp` / `mma_sm107_desc.hpp` / `mma_sm120.hpp` / `mma_sm120_sparse.hpp`(同仓库 `include/cute/arch/`)
- **PTX ISA**:https://docs.nvidia.com/cuda/parallel-thread-execution/(tcgen05.mma 章节)
- **库内交叉**:[hopper-wgmma.md](hopper-wgmma.md)(Hopper 指令机制,对照)、[cute-layout-algebra.md](cute-layout-algebra.md)(CuTe Layout 通识)、[gemm-tiling.md](gemm-tiling.md)(三层 tiling)、[../../attention/implementation/flash-attention-implementation.md](../../attention/implementation/flash-attention-implementation.md)(FA4 应用收益)、[../index.md](../index.md)(v4.8.0 SM107 初步支持)。
