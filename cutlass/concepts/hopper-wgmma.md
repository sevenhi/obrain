---
title: Hopper WGMMA:warpgroup 级异步矩阵乘指令机制
type: concept
tags: [cutlass, hopper, sm90, wgmma, tensor-core, gmma, swizzle, descriptor, cute]
source:
  - "web: Colfax Research — CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA Hopper GPUs (Part 1, 2024-08-06) | https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/ ↑primary"
  - "code: CUTLASS examples/cute/tutorial/wgmma_sm90.cu(3.5.1 起,Pradeep Ramani 贡献)| github.com/NVIDIA/cutlass"
  - "web: PTX ISA — Asynchronous Multiply-Accumulate Instructions (wgmma.mma_async / core matrix / descriptor) | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async"
  - "web: NVIDIA H100 GPU Datasheet(FP16 TC 2,000 TFLOPS* 稀疏 / 1,000 dense)| https://resources.nvidia.com/en-us-data-center-overview/nvidia-tensor-core-gpu-datasheet"
  - "web: NVIDIA A100 Datasheet(FP16 TC 312 TFLOPS dense / 624* 稀疏)| https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf"
date: 2026-09
---

# Hopper WGMMA:一条指令,一个 warpgroup,64×N×16

## TL;DR

**WGMMA**(`wgmma.mma_async`)是 Hopper(SM90)引入的 **warpgroup 级异步矩阵乘累加**指令:**4 个连续 warp(= 128 连续线程,首 warp 的 rank 是 4 的倍数)协同**执行一次 `C = A*B + C`(或禁用 C 输入的 `C = A*B`),累加器留在寄存器里跨指令复用。

**三条硬约束(不是建议,违反即错误/未定义行为)**:
1. **操作数 B 恒在 SMEM**;**A 可在 SMEM 或 RMEM**;**累加器 C 恒在 RMEM**;
2. SMEM 布局不能随便排——必须用 **8 种 canonical GMMA layout atom + `tile_to_shape`** 构造,否则 `make_gmma_desc` 算不出合法 descriptor;
3. 异步指令,**同步三件套不能省**:`warpgroup_arrive()`(fence)→ `cute::gemm` → `warpgroup_commit_batch()` + `warpgroup_wait<N>()`;普通 ld/st 写过 SMEM 还要补 `fence.proxy.async`(TMA 加载则免)。

**核心判断**:SM80 的 `mma.sync` 是"warp 级同步、操作数全在寄存器";WGMMA 换成"**warpgroup 级异步、操作数直接从 SMEM 喂(经 64-bit matrix descriptor)**",单条指令最大 `m64n256k16`(FP16,524,288 FLOPs,为 Ampere mma.sync 的 128 倍,见 §7)——省掉 SMEM→RMEM 的搬数指令、放大单指令工作量、并让 TC 计算与 softmax/搬运重叠。这是 FA3 等 Hopper kernel 的计算底座。

---

## 1. 定义:warpgroup 级异步 MMA

- **warpgroup** = 4 个连续 warp = **128 连续线程**,且第一个 warp 的 warp-rank 是 4 的倍数(对齐要求)。
- `wgmma.mma_async` 由这 128 线程**共同执行**,形式为 `C = A*B + C` 或 `C = A*B`(禁用累加器输入,对应 PTX 里 `scale_D=0`)。
- **异步**:wgmma 在 **async proxy** 中执行,可与其他操作并发,因此必须显式同步(§5)。
- CUTLASS 代码参考:`examples/cute/tutorial/wgmma_sm90.cu`(3.5.1 版本加入)。

### 操作数位置(雷区级硬约束)

| 操作数 | 允许位置 | 备注 |
|---|---|---|
| **A** | **SMEM 或 RMEM** | SS 模式来自 SMEM;RS 模式来自寄存器 |
| **B** | **恒 SMEM**(无例外) | 所以 atom 名后缀第二位恒为 `S` |
| **C(累加器)** | **恒 RMEM** | 128 线程分摊;64×64 FP16 时每线程 32 个值 |

> ⚠ 这条是 PTX ISA 层面的硬约束,不是 CUTLASS 风格建议。想"把 B 放寄存器"没有任何合法路径。

---

## 2. MMA atom 命名与形状约束

CUTLASS 把每条 wgmma PTX 指令包装成 **MMA Atom**,名字与指令一一对应:

```cpp
// 这个 atom
SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{}
// 包的就是这条 PTX
wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16
```

命名规则 `SM90_MxNxK_XYZ_SS/RS<Major_A, Major_B>`:
- `X`/`Y` = 操作数 A/B 的 dtype,`Z` = 累加器 dtype;
- **形状约束**:`M` 恒为 **64**;`N` ∈ {8, 16, …, 256}(8 的倍数);16-bit dtype 时 `K=16`,一般化规则是 **K × sizeof(dtype) = 32 字节**;
- 后缀 `SS` / `RS`:第一个字母指 **A 的来源**(S=SMEM,R=RMEM),第二个恒 S(B 只能 SMEM);
- 两个模板参数 `GMMA::Major::MN` / `GMMA::Major::K`:A、B 各自在 MN 模还是 K 模上内存连续。BLAS 记法下"A、B 都 K-major"对应 TN gemm。

**布局/dtype 约束(易错)**:
- **16-bit dtype(FP16/BF16)**:MN-major 或 K-major 都合法;
- **非 16-bit dtype(如 TF32、FP8)**:**必须 K-major**——PTX 的 imm-trans 转置操作只对 `.f16/.bf16` 的 SMEM-descriptor 操作数存在(见 PTX ISA `wgmma.mma_async` 一节及 Table 37 附近说明)。

**CTA tile 整除约束**:MMA atom 的 MxNxK 必须整除操作数/累加器 tile:bM 是 64 的倍数、bN 是 N-atom 的倍数、bK 是 16 的倍数(FP16)。

### warpgroup 数量:AtomLayoutMNK

`make_tiled_mma` 第二参数可让**多个 warpgroup 各算输出 tile 的一半**(各自发自己的 wgmma):

```cpp
TiledMMA tiled_mma = make_tiled_mma(
    SM90_64x64x16_F16F16F16_SS{},
    Layout<Shape<_2,_1,_1>>{});   // 2 个 warpgroup 沿 M 切分(bM 须为 128 倍数)
// size(tiled_mma) == 256;单 warpgroup 时 dim3 dimBlock(cute::size(tiled_mma)) == 128
```

`AtomLayoutMNK` 与 `PermutationMNK` 对任何 MMA atom 语义一致(与 [cute-layout-algebra.md](cute-layout-algebra.md) §4 的 TiledMMA 平铺是同一套机制)。

---

## 3. SMEM 布局:8 种 canonical layout atom + tile_to_shape(最大实操坑)

WGMMA 对 sA/sB 的 SMEM **shape 和 stride 都有额外约束**,且随 swizzle 模式变化——`(bM,bK):(1,bM)` 这种"顺手"的布局一般**不合法**。

**实操上唯一保证兼容的构造模式**:从 8 个预定义 layout atom 里选一个,喂给 `cute::tile_to_shape`:

```cpp
// 8 个 canonical GMMA layout atom(MN-major / K-major × 4 种 swizzle):
GMMA::Layout_MN_INTER_Atom<T>   GMMA::Layout_K_INTER_Atom<T>    // 无 swizzle(隐含 16B 边界)
GMMA::Layout_MN_SW32_Atom<T>    GMMA::Layout_K_SW32_Atom<T>     // 32B swizzle(交换 2 个 16B 段)
GMMA::Layout_MN_SW64_Atom<T>    GMMA::Layout_K_SW64_Atom<T>     // 64B swizzle(交换 4 个)
GMMA::Layout_MN_SW128_Atom<T>   GMMA::Layout_K_SW128_Atom<T>    // 128B swizzle(交换 8 个)
```

```cpp
// host 侧(T = cutlass::half_t;MN-major FP16,128B swizzle,3 级流水)
auto bM = Int<128>{}; auto bN = Int<128>{}; auto bK = Int<64>{}; auto bP = Int<3>{};
auto sA = cute::tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},
                              cute::make_shape(bM, bK, bP));
auto sB = cute::tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},
                              cute::make_shape(bN, bK, bP));
// print(sA):
// Sw<3,4,3> o smem_ptr[16b] o ((_64,_2),(_8,_8),_3):((_1,_512),(_64,_1024),_8192)
```

读法:
- `tile_to_shape` 把 atom 布局(这里是 `(64,8):(1,64)`)像 `numpy.tile` 一样**列主序平铺**到 `(128,64,3)`——外层 stride 512 落在 M 模、1024 落在 K 模、8192 落在 stage(P)模;
- **atom 在 contiguous 方向的长度恒等于 swizzle 字节数**(64 × sizeof(half) = 128B = SW128 的命名由来;INTER 则是 16B);
- K-major 版本:`tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(bM,bK,bP))` 打印为 `Sw<3,4,3> o smem_ptr o (_128,_64,_3):(_64,_1,_8192)`(atom `(8,64):(64,1)` 平铺后 coalesce);
- shape 各模要能被 atom 的 tile 整除——这是 **swizzle 模式对 SMEM shape 的约束**,与 §2 的 MMA atom 整除约束是**两条独立约束**,都要满足。

> swizzle 函数 `Sw<B,M,S>` 的代数含义与 TMA 侧的配合,见 [cute-layout-algebra.md](cute-layout-algebra.md) §3 末尾的 Hopper 注记,不在此重讲。

---

## 4. Fragment 语义与 matrix descriptor(SS 模式的"假寄存器"陷阱)

device 侧用 `get_thread_slice` 拿 `ThrMMA`,再 partition:

```cpp
ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);  // 0..127
Tensor tCsA = thr_mma.partition_A(sA);   // (MMA,MMA_M,MMA_K,PIPE)
Tensor tCsB = thr_mma.partition_B(sB);   // (MMA,MMA_N,MMA_K,PIPE)
Tensor tCgC = thr_mma.partition_C(gC);   // (MMA,MMA_M,MMA_N)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // 真·寄存器累加器
Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // ⚠ 不是寄存器数据!
Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // ⚠ 同上
```

打印任意线程的这些 tensor:

```
tCsA: Sw<3,4,3>_smem_ptr[16b](0x7f8800000400) o ((_64,(_8,_2)),_2,_4,_3):((_1,(_64,_1024)),_512,_2048,_8192)
tCrA: GMMA::DescriptorIterator o (_1,_2,_4,_3):(_0,_64,_256,_1024)
tCrC: ptr[16b](0x7feee1fffbe0) o ((_2,_2,_8),_2,_2):((_1,_2,_4),_32,_64)
```

三个关键语义(与 SM80 直觉最不同的地方):
1. **`tCsA` 不是线程切片**:它是**整个 SMEM tensor 换了个重排布局**——因为 wgmma 是 warpgroup 集体指令,操作数不属于任何单线程;shape `(MMA,MMA_M,MMA_K,PIPE)` 中 MMA 模是 atom 的 M×K 形状,MMA_M = bM/64、MMA_K = bK/16;
2. **`tCrA`/`tCrB` 不装数据**:SS 模式下它们是 **`GMMA::DescriptorIterator`**——SMEM 数据**不会**被拷进寄存器,"访问 tCrA 的值"访问的是 **64-bit matrix descriptor**;且是"迭代器":任一时刻寄存器里只持有当前这条 wgmma 用的那 1 个 descriptor(而非全部,如 24 个);
3. **`tCrC` 才是标准寄存器 tensor**:64×64 atom 中每线程持 32 = (2,2,8) 个值,按 PTX 文档的 **Z 字形 replicated 模式**分布(thread 0 持 (0,0),(0,1),(8,0),(8,1),每 8 列向右重复)。

### Core matrix 与 descriptor 的 5 个参数

wgmma 把 atom tile wA(64×K)/wB(N×K)切成 **core matrix**:每个 core matrix 在 **strided 方向长 8、contiguous 方向长 16 字节**;**wA = 8×2 个 core matrix,wB = 2×(N/8) 个**。

SS 模式需要 desc-a / desc-b 两个 matrix descriptor,CUTLASS 的 `make_gmma_desc` 从 SMEM tensor 布局构造 `GmmaDescriptor`,编码 **5 个参数**:

| 参数 | 含义 |
|---|---|
| Start address | 操作数在 SMEM 的起始基址 |
| **LBO**(leading dimension byte offset) | K 方向相邻两个 core matrix 的字节距离 |
| **SBO**(stride dimension byte offset) | M/N 方向相邻两个 core matrix 的字节距离 |
| Swizzling mode | none / 32 / 64 / 128 字节 |
| Matrix base offset | SMEM 地址未按 swizzle 重复模式对齐时的补偿 |

**前提是布局出自 §3 的 canonical atom + tile_to_shape 模式**——满足则 `make_gmma_desc` 自动算对 LBO/SBO/swizzle。K-major FP16 下各 swizzle 的合法布局形如(Colfax 原文,T = 8 元素 = 16B):

```
No swizzle : Swizzle<0,4,3> o smem_ptr o ((8,m),(T,2)):((1T,SBO),(1,LBO))   LBO=128B, SBO=256B
128B swz   : Swizzle<3,4,3> o smem_ptr o ((8,m),(T,2)):((8T,SBO),(1, T ))   SBO=1024B
```

注意 64/128B swizzle 下这些布局**非 compact**:swizzle 把 K 方向逻辑相邻的 2/4/8 个 core matrix 在内存中交错,它们会分属不同 wgmma atom(这正是 `tCsA` K 模打印成 `(_8,_2)` 嵌套的原因)。

### atom 的 fma:PTX 内联真身

`cute::gemm` 最内层调 atom 的 `fma`,就是内联 PTX:

```cpp
// SM90_64x64x16_F16F16F16_SS::fma(节选)
asm volatile(
  "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
  "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
  "{%0,...,%15}, %16, %17, p, %19, %20, %21, %22;\n}\n"
  : "+r"(d00), ..., "+r"(d15)          // 16 个 uint32 = 累加器(RMEM)
  : "l"(desc_a), "l"(desc_b),          // 2 个 uint64 = SMEM descriptor
    "r"(int32_t(scale_D)), ...);       // scale_D∈{0,1}: 是否清零累加器
```

`scaleA/scaleB`(±1 取负)、`tnspA/tnspB`(K-major=0 / MN-major=1)都是编译期模板常量。

---

## 5. 同步机制:三件套 + fence.proxy.async

wgmma 是异步指令,同步不当的后果:**(a) 隐蔽 race、(b) 编译器把 wgmma 串行化(性能大跌)、(c) 未定义行为**。

```cpp
cute::warpgroup_arrive();                 // = wgmma.fence.sync.aligned
cute::gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
cute::warpgroup_commit_batch();           // = wgmma.commit_group.sync.aligned
cute::warpgroup_wait<0>();                // = wgmma.wait_group.sync.aligned 0
```

对照 PTX 文档的 WGMMA 流程逐条理解:

| 原语 | PTX | 作用 |
|---|---|---|
| `warpgroup_arrive()` | `wgmma.fence.sync.aligned` | 保证 wgmma 只在**之前对这些 RMEM 地址的访问全部完成后**才访问它们;缺了是未定义行为 |
| **(条件)** | `fence.proxy.async` | wgmma 在 **async proxy** 执行;若 SMEM 是**普通 ld/st(generic proxy)写入**的,必须补此 fence 才对其可见;**TMA 加载则不需要**(CUTLASS 中由 `cutlass::arch::fence_view_async_shared()` 包装,可 grep 验证) |
| `warpgroup_commit_batch()` | `wgmma.commit_group` | 把此前所有未提交的 wgmma 打包成一个 wgmma-group(本例一批 = MMA_M×MMA_N×MMA_K 条) |
| `warpgroup_wait<N>()` | `wgmma.wait_group N` | 等到**最多还剩 N 个最近的 group 未完成**、更早的全部完成;N=0 即全等完 |

两个进阶要点:
- **同 accumulator shape 的多条 wgmma 可免重复 fence**:Hopper 允许多条 `wgmma.mma_async` 同时在飞,只要累加器 shape 相同、写同一批寄存器地址,中间不需要再插 `wgmma.fence`——所以 `cute::gemm` 内部 MMA_K 循环里不用逐条 fence;
- **`wait<N>` 的 N>0 是 overlap 的抓手**:warpgroup 有独立计算可做时(如 FA3 的 GEMM-softmax 重叠),让上一组 wgmma 还在飞、先干别的,靠 N>0 控制等待深度。FA3 的 pingpong/warp-spec 编排详见 [flash-attention-implementation.md](../../attention/implementation/flash-attention-implementation.md)。

---

## 6. 可复现代码:mainloop 里的完整调用骨架

以下即 Colfax 教程(基于 `wgmma_sm90.cu`)的 kernel 主干,`cute::gemm` 调用点与 SM80 tutorial kernel **语法完全一致**——差别全在参数构造:

```cpp
template <class TiledMMA, ...>
__global__ device_gemm(TiledMMA tiled_mma, ...) {
  // PROLOGUE:TMA 流水加载 sA/sB(略,见 Colfax Part 2)
  ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);        // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);        // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);        // (MMA,MMA_M,MMA_N)
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // 累加器,RMEM
  clear(tCrC);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // DescriptorIterator(SS)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  while (k_tile_count > -K_PIPE_MAX) {           // PIPELINED MAIN LOOP
    // ...(TMA producer/consumer 同步,略)
    cute::warpgroup_arrive();
    // (V,M,K) x (V,N,K) => (V,M,N)
    cute::gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();                   // 等本 K_TILE 的全部 MMA
  }
  // EPILOGUE:tCrC 写回 gC(略)
}
```

host 侧配套(§2 atom + §3 布局):

```cpp
TiledMMA tiled_mma = cute::make_tiled_mma(
    SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});
dim3 dimBlock(cute::size(tiled_mma));            // 128 = 1 warpgroup
```

`cute::gemm` 的重载先循环外层 MMA_M/MMA_N、MMA_K 模,归约到 `(V)x(V)=>(V)` 的 atom 级 dispatch,最终落到 §4 的 `fma` 内联 PTX。

**动手路径**:编译 `examples/cute/tutorial/wgmma_sm90.cu`(需 `-arch=sm_90a`,构建注意见 [../frameworks/cutlass-build-and-usage.md](../frameworks/cutlass-build-and-usage.md) §2),用 `print`/`print_tensor` 对照 §4 的布局打印。

---

## 7. 数据对比:WGMMA 为什么值这套复杂度

Colfax Part 1 本身无性能数字(评论区作者也确认:量化收益看 FA3 论文),以下是可验证的公开数据:

**单指令工作量(vs Ampere warp 级 mma.sync)**:

| 指令 | 形状(FP16) | FLOPs/指令 | 折算 m16n8k16 条数 |
|---|---|---|---|
| Ampere `mma.sync.aligned.m16n8k16.f16` | 16×8×16 | 4,096 | 1 |
| Hopper `wgmma.m64n64k16`(本文 atom) | 64×64×16 | 131,072 | **32** |
| Hopper `wgmma.m64n256k16`(最大 N) | 64×256×16 | 524,288 | **128** |

一条 wgmma 抵 32~128 条 Ampere mma 的计算量,且操作数经 descriptor 直接来自 SMEM(RS/SS),**省掉了 Ampere 模式下 SMEM→寄存器的一整层 `ldmatrix`/`lds` 搬运指令流**;指令发射开销也从"每 warp 各自发"摊薄成"每 warpgroup 发一条"。

**整卡 FP16 Tensor Core 峰值(官方 datasheet,dense)**:

| GPU | FP16 TC(dense) | FP16 TC(稀疏) | 代际 |
|---|---|---|---|
| A100 SXM | 312 TFLOPS | 624 TFLOPS | 3rd-gen TC,`mma.sync`(warp 级) |
| H100 SXM | ~1,000 TFLOPS | 2,000 TFLOPS | 4th-gen TC,`wgmma`(warpgroup 级)+ Transformer Engine FP8 |

≈ **3.2× 代际提升**(H100 FP16 dense 峰值精确口径 989.4 TFLOPS,datasheet 取整 ~1,000);单 TC 每周期算力按杨远航笔记的口径:Volta 1st-gen 128 FLOPs/cycle,每代翻倍 → Hopper 4th-gen **1024 FLOPs/cycle**、Blackwell 5th-gen 2048(Blackwell 已改用 `tcgen05`/UMMA,wgmma 为 Hopper-only)。

**工程实效参照**:Colfax 团队成员 Hieu Pham 在评论区给的估计口径(原文带问号,非 FA3 论文作者本人)——FA3(H100,TMA+WGMMA+warp-spec)相对 FA2(A100,cp_async+WMMA+多级流水)约 **2~3×?**;FA3 论文正式口径为 **BF16 相对 FA2 加速 1.5-2×**(与库内 fa3 篇一致),论文实测 FP16/BF16 前向最高 **740 TFLOPS**,约为 H100 dense 峰值(989)的 **75%**。细节见 [flash-attention-implementation.md](../../attention/implementation/flash-attention-implementation.md) 与 [gemm-tiling.md](gemm-tiling.md) §4。

---

## 8. 实践 checklist

1. **atom 选型**:先定 dtype 与 BLAS 布局(TN/NT/…)→ 16-bit 才有 MN-major 自由,非 16-bit 一律 K-major;N 取 8~256 中整除 bN 的值,大 N 单指令工作量更大。
2. **SMEM 布局只走正道**:`GMMA::Layout_{MN,K}_{INTER,SW32,SW64,SW128}_Atom<T>` + `tile_to_shape(·, make_shape(bM/bN, bK, bP))`。手写 shape/stride"碰巧能跑"= 未定义行为温床;swizzle 模式还要与 TMA 加载侧一致。
3. **thread 数 = `cute::size(tiled_mma)`**:单 warpgroup 128;要 2 warpgroup 用 `AtomLayoutMNK = Layout<Shape<_2,_1,_1>>`(bM 须 128 倍数)。
4. **同步三件套一个不能少**:`warpgroup_arrive` / `commit_batch` / `wait<N>`;性能不对先查是不是 fence 缺失导致编译器串行化。
5. **SMEM 若由普通 ld/st 写入 → 必须 `fence.proxy.async`**(即 `fence_view_async_shared()`);TMA 写入免。
6. **别把 `tCrA/tCrB` 当寄存器数据**:SS 模式下它们是 DescriptorIterator,`print` 出来没有数值;调试数据要看 `tCsA/tCsB`(SMEM)和 `tCrC`(累加器)。
7. **做 overlap 时用 `wait<N>` N>0**:GEMM 与 softmax/epilogue 重叠(FA3 模式)靠"只等到剩 N 组",`wait<0>` 是最保守的全等待。
8. **编译目标必须 `sm_90a`**:wgmma 属 architecture-accelerated 特性,`sm_90` 编译会运行时报错(同 Blackwell `100a`,见 build 篇 §2)。
9. **读布局打印的习惯**:遇到 `((_64,(_8,_2)),_2,_4,_3)` 这类嵌套 shape,先按 `(MMA, MMA_M, MMA_K, PIPE)` 注解对号入座,再对照 §4 的 core matrix 交错解释。

---

## 9. 溯源

- **主源**:Colfax Research — *CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA Hopper GPUs*(GEMM 三部曲 Part 1,**2024-08-06** 发布):https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
  - Part 2(TMA + warp-specialization + ping-pong,GEMM kernel 整体设计):https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/
- **代码参考**:`examples/cute/tutorial/wgmma_sm90.cu`(Pradeep Ramani 贡献,CUTLASS 3.5.1 加入):https://github.com/NVIDIA/cutlass
- **PTX ISA**(wgmma.mma_async 语法、core matrix 图、descriptor 字段、imm-trans 仅 f16/bf16、Table 37 swizzle 与 majorness 独立):https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async
- **性能数据**:H100 datasheet https://resources.nvidia.com/en-us-data-center-overview/nvidia-tensor-core-gpu-datasheet ;A100 datasheet https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf ;FA3 数字见库内 [flash-attention-implementation.md](../../attention/implementation/flash-attention-implementation.md);TC 代际 FLOPs/cycle 口径见杨远航 CUTLASS 笔记导读(知乎专栏 https://www.zhihu.com/column/c_1938664963049763058 )
- **库内交叉**:[cute-layout-algebra.md](cute-layout-algebra.md)(Atom/TiledMMA/TV layout 通识 + Hopper GMMA 注记)、[gemm-tiling.md](gemm-tiling.md)(三层 tiling / warp 特化 / pipelining)、[../frameworks/cutlass-build-and-usage.md](../frameworks/cutlass-build-and-usage.md)(sm_90a 构建)、[../frameworks/cutlass-profiling-tuning.md](../frameworks/cutlass-profiling-tuning.md)(调优)
