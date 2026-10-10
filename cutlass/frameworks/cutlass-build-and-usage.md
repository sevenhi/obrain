---
title: CUTLASS 构建与上手:从 cmake 到第一个 kernel
type: howto
tags: [cutlass, build, cmake, collectivebuilder, cute-dsl, examples, profiler]
source:
  - "web: NVIDIA CUTLASS Quickstart | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html"
  - "code: examples/49_hopper_gemm_with_collective_builder / 71_blackwell_gemm_with_collective_builder | github.com/NVIDIA/cutlass"
  - "web: NVIDIA Tech Blog — CUTLASS 3.x abstractions (CollectiveBuilder 用法) | https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/ (2025-07)"
  - "web: 杨远航《CUTLASS 笔记》系列(导读 + Minimal GEMM,§0/§3.6 主源)| https://www.zhihu.com/column/c_1938664963049763058 (连载中,基于 CUTLASS 4.1.0,目录为 2025 连载快照;微信公众号为转载载体,以知乎专栏为准)"
date: 2026-09
---

# CUTLASS 构建与上手:从 cmake 到第一个 kernel

## TL;DR

**CUTLASS 是 header-only 库**——直接用只需把 `include/` 加进 include path,**根本不需要构建**。构建只是编译**示例 / 测试 / profiler**。上手流程:

1. 依赖:CUDA 11.4+(建议 12.0+/12.8)、CMake 3.18+、C++17(g++≥9)
2. 构建(仅示例/工具):`cmake .. -DCUTLASS_NVCC_ARCHS=<你的架构>` + `make cutlass_profiler`
3. 推荐用法(3.x):`CollectiveBuilder` 给高层准则,底层自动推导
4. 验证:`cutlass_profiler --kernels=sgemm --m=4352 --n=4096 --k=4096`

**最容易踩的坑(`CUTLASS_NVCC_ARCHS` 没带 `a` 后缀)**:Hopper/Blackwell 的 Tensor Core 指令是"architecture-accelerated",必须用 `90a`/`100a` 目标,用 `90` 会运行时报错。

**学习路线**:先用 §0 定位"为什么值得学 CUTLASS"与三条技术路径,再从 §3.6 的最小 CuTe kernel(单条 16x8x8 mma)动手,之后进入 CollectiveBuilder(§4)与 Hopper 特性(见 [../concepts/hopper-wgmma.md](../concepts/hopper-wgmma.md))。

---

## 0. 为什么选 CUTLASS:三条实现路径与学习路线

> 本节主源:杨远航《CUTLASS 笔记》系列(知乎专栏,连载中),基于 CUTLASS 4.1.0,目录为 2025 连载快照。

### 0.1 2025 年写高性能算子的三条路径

| 路径 | 代表 | 特点 |
|---|---|---|
| ① 自动编译 | `torch.compile` | 省事,性能上限与可控性有限 |
| ② Python DSL + PTX 编译器 | triton / CuTe DSL / TileLang / Mojo | 开发效率高;灵活性受 DSL 与编译器 pass 限制 |
| ③ C++ 模板封装 PTX | **CUTLASS** / ThunderKittens | 白盒、可控、可扩展;学习曲线最陡 |

(还有手搓 PTX 的路径,如 LeetCUDA 项目。)

### 0.2 Why CUTLASS

- **完全白盒**:复杂 C++ 模板之下,**CUTLASS 本质就是对 PTX 指令的封装**——不需要学一堆 CUDA Runtime API 再猜它背后做了什么,了解 PTX + 用对应 CUTLASS API 就能近乎完全掌控硬件行为;
- **可控且灵活**:写代码时就能感知硬件会如何执行;新算法/新指令集出来后,可以在已有组件基础上**扩展新 scheduler、新 GEMM 编排、融入新指令**——除 nvcc、PTX、SASS 外没有别的东西约束算子开发;
- **对比 triton**:提升开发效率的同时牺牲了灵活性——要么等 DSL 更新接口,要么自己改编译器写 pass;且随 NV 架构 DSA 化,triton 语法也越来越复杂;
- **学习 CUTLASS 本质就是学 CUDA/PTX**:有了这层知识储备,回头写 triton kernel 也得心应手;只会 triton 的话,遇到性能问题或不符预期的结果,深入排查极其困难;
- 生态背书:PyTorch、vLLM、FA2、FA3 等均用 CUTLASS 开发;CUTLASS 4.0 起的 Python DSL(CuTe DSL)在保留 C++ API 原味之外提供 Python 接口,是高性价比折中(见 §5)。

### 0.3 学习资源与路线

- **@reed**:中文社区最好的 CUTLASS 教程(系列文章的起点);
- **Colfax Research**(https://research.colfax-intl.com/blog/):FA3 主要开发团队,外网最好的 CUTLASS 教程——**GEMM 三部曲全系列针对 Hopper:Part 1 讲 WGMMA、Part 2 讲 pipelining/warp-specialization、Part 3 讲 persistent/Stream-K**,本库 WGMMA 机制篇即出自其 Part 1:[../concepts/hopper-wgmma.md](../concepts/hopper-wgmma.md);
- **杨远航《CUTLASS 笔记》**(https://www.zhihu.com/column/c_1938664963049763058):从 Minimal GEMM 起步 → CuTe 组件 → SMEM Swizzling / Dynamic MMA / Pipelining / GEMM API(Part 1,SM80 及之前)→ TMA / Warpgroup MMA / Warp Specialization(Part 2,SM90),目标是在最新架构上写出 optimized GEMM 融合算子;
- **CUTLASS GitHub Discussions**(https://github.com/NVIDIA/cutlass/discussions):官方开发者回复积极,能学到源码和博客看不到的细节。

**前置知识**(零基础自查):GPU 编程模型与 SIMT(grid/block/warp/thread、GMEM/SMEM/Register)、C++17 模板语法、CUDA 扩展语法(`__global__`/`__device__`、`threadIdx`/`blockIdx`)、能写简单 kernel。

---

## 1. 前提与依赖

| 依赖 | 最低 | 推荐 |
|---|---|---|
| CUDA Toolkit | 11.4+ | 12.0+ / 12.8 |
| CMake | 3.18+ | — |
| Host 编译器(C++17) | g++ 7.5 | GCC ≥ 9(GCC 8.5 有已知回归)|
| Python | 3.6+ | — |
| 可选链接 | — | cuBLAS、cuDNN v7.6+ |

---

## 2. 基本构建(只构建你要的)

```bash
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
mkdir build && cd build

# ⚠ 指定目标架构(关键)
cmake .. -DCUTLASS_NVCC_ARCHS=90a        # Hopper
# cmake .. -DCUTLASS_NVCC_ARCHS=100a     # Blackwell SM100

make cutlass_profiler -j12   # 只构建 profiler(最快)
# make test_unit -j          # 单元测试
```

**`CUTLASS_NVCC_ARCHS` 是必设关键项**:
- CUDA 12.0 引入"architecture-accelerated features",其 PTX **无前向兼容保证**;
- Hopper/Blackwell 的 Tensor Core 指令属此类 → 必须 `sm_90a` / `sm_100a`(**带 a**);
- 误用 `sm_90` 编译含这些特性的 kernel,运行时报错;
- **默认会让 CMake 为该 CUDA 版本支持的全部架构编译(可高达十几个),非常慢**,务必用 `CUTLASS_NVCC_ARCHS` 收窄。

---

## 3. 缩短编译时间(实用技巧)

```bash
# 只构建 profiler + 关测试 + unity build
cmake .. -DCUTLASS_NVCC_ARCHS=90a \
    -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON

# 只编译某类算子(如 conv2d)
cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_LIBRARY_OPERATIONS=conv2d

# 按名称通配过滤 kernel(关键:避免 all)
cmake .. -DCUTLASS_NVCC_ARCHS='75;80' \
    -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s*gemm_f16_*_nt_align8

# 排除 cuBLAS/cuDNN
cmake .. -DCUTLASS_ENABLE_CUBLAS=OFF -DCUTLASS_ENABLE_CUDNN=OFF
```

> 别用 `CUTLASS_LIBRARY_KERNELS=all`——编译慢、链接易失败。用通配符只编需要的子集。

---

## 3.5 编译你自己的第一个 kernel(header-only 用法)

CUTLASS 是 header-only,自己写 kernel 时**只需把 include 路径指对 + 开两个 flags**,不需要整个 CMake 工程:

```bash
# 最小 nvcc 编译命令(三件套:include 路径 + relaxed-constexpr + 目标架构)
nvcc -std=c++17 \
  -I /path/to/cutlass/include \
  -I /path/to/cutlass/tools/util/include \
  --expt-relaxed-constexpr \
  -arch=sm_90a \
  my_gemm.cu -o my_gemm
```

- `-I include` 是 CUTLASS 源码本体,`-I tools/util/include` 是示例/工具用的 host 辅助(写简单 kernel 可不要,但示例基本都依赖);
- `--expt-relaxed-constexpr` 是 CUTLASS 模板大量用 constexpr 的必需品,缺了报编译错;
- `-arch=sm_90a` 必须带 `a`(Hopper 加速特性,见 §2);
- 想"把 CUTLASS 当子目录接进自己工程",官方给了 CMake 示例 **`examples/60_cutlass_import`**——这是本文主题里最容易被忽略的接入方式。

---

## 3.6 用 CuTe 从零手写第一个 kernel(Minimal GEMM,单条 16x8x8 mma)

> 来源:杨远航《CUTLASS 笔记 (1)》Minimal GEMM Kernel(CUTLASS 4.1.0,代码在 SM90 上编译验证)。CuTe API 名以 [../concepts/cute-layout-algebra.md](../concepts/cute-layout-algebra.md) 为准。这是"不用 CollectiveBuilder、直接裸用 CuTe"的最小完整闭环:**写 kernel → PyTorch binding → 精度验证 → 计时 → ncu/PTX 分析**。

### 3.6.1 场景与用到的 CuTe 组件

算子:单条 `SM80_16x8x8_F16F16F16F16_TN` mma 指令,算 16×8×8 的 `C = A*B`(或 `C = A*B + C`);一个 block、32 线程(1 warp)、无 SMEM、无分级 tiling。组件对应关系:

| CuTe 组件 | 作用 |
|---|---|
| `make_tensor(make_gmem_ptr(p), make_shape(...), make_stride(...))` | 建立 GMEM 矩阵的 Tensor 表示(注意 CuTe 缺省 stride 是 left-major,与 PyTorch right-major 相反) |
| `local_tile(mA, tiler, coord, Step<_1,X,_1>{})` | tiling API:按 tile shape 切出本 block 的分块(`Step` 指定在哪些维上切;`make_tile`/`make_coord` 均为 `cute::tuple` 别名) |
| `make_tiled_mma(MMA_op{})` → `get_slice(tid)` → `partition_A/B/C` + `partition_fragment_A/B/C` | MMA API:拿到线程级 GMEM 分片(tCgA)与寄存器 fragment(tCrA);命名规范 t=tiled、C=为算 C 而切、g/r=GMEM/register |
| `AutoVectorizingCopy{}` + `copy` | Copy API:按 MMA 布局自动选最大连续长度的向量化拷贝指令(GMEM↔RMEM) |
| `gemm(tiled_mma, tCrD, tCrA, tCrB, tCrC)` | GEMM API:执行 mma |

### 3.6.2 KernelSpec 参数类 + kernel 主干

常量收进一个 Spec 类(改参数不动 kernel 逻辑):

```cpp
template <typename T_, int kTileM_ = 16, int kTileN_ = 8, int kTileK_ = 8>
struct KernelSpec {
  using T = T_;
  static constexpr int kTileM = kTileM_, kTileN = kTileN_, kTileK = kTileK_;
  using MMA_op  = SM80_16x8x8_F16F16F16F16_TN;
  using TiledMMA = decltype(make_tiled_mma(MMA_op{}));
  static constexpr int kThreadNum = size(TiledMMA{});   // = 32
  static constexpr int kShmSize = 0;
};
```

kernel(三矩阵行连续;`IsGemm=true` 走 `C=A*B`,`false` 走 `C=A*B+C`):

```cpp
template <typename Spec, bool IsGemm>
__global__ void minimal_gemm(void* Cptr, const void* Aptr, const void* Bptr,
                             int m, int n, int k) {
  using T = typename Spec::T;
  // 1) 整个问题规模的 Tensor 表示
  Tensor mA = make_tensor(make_gmem_ptr((T*)Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr((T*)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor mC = make_tensor(make_gmem_ptr((T*)Cptr), make_shape(m, n), make_stride(n, Int<1>{}));
  // 2) local_tile 切出本 block 的分块(单指令场景 tiler=问题规模,只取 (0,0,0))
  auto tiler = make_tile(Int<Spec::kTileM>{}, Int<Spec::kTileN>{}, Int<Spec::kTileK>{});
  auto coord = make_coord(0, 0, 0);
  Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});  // (kTileM, kTileK)
  Tensor gB = local_tile(mB, tiler, coord, Step< X, _1, _1>{});  // (kTileN, kTileK)
  Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{});  // (kTileM, kTileN)
  // 3) MMA API:线程级分片 + 寄存器 fragment,shape 均为 (MMA, MMA_M/N, MMA_K/N)
  TiledMMA tiled_mma;  // typename Spec::TiledMMA
  ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
  Tensor tCgA = thr_mma.partition_A(gA);            Tensor tCrA = thr_mma.partition_fragment_A(gA);
  Tensor tCgB = thr_mma.partition_B(gB);            Tensor tCrB = thr_mma.partition_fragment_B(gB);
  Tensor tCgC = thr_mma.partition_C(gC);            Tensor tCrC = thr_mma.partition_fragment_C(gC);
  // 4) 拷贝 + 计算 + 写回
  auto copy_atom = AutoVectorizingCopy{};
  copy(copy_atom, tCgA, tCrA);
  copy(copy_atom, tCgB, tCrB);
  if constexpr (IsGemm) clear(tCrC);                // C=A*B:累加器清零,不拷 C
  else copy(copy_atom, tCgC, tCrC);                 // C=A*B+C
  gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
  copy(copy_atom, tCrC, tCgC);
}
```

调试利器:`print(tCgA)` 看 layout、`print_tensor(tCgA)` 看数据(配 `thread0()` 只让 0 号线程打)。

### 3.6.3 PyTorch binding 与精度验证

- binding 函数签名 `run_minimal_gemm(a, b, std::optional<c>)`:未传 c → `torch::empty` 新建且 `is_gemm=true`;传了 → 用 `BOOL_SWITCH(is_gemm, IsGemm, ...)` 选 kernel 模板实例,`<<<grid, block, shm, stream>>>` 发射;`PYBIND11_MODULE` 导出;Python 侧 `torch.utils.cpp_extension.load` 即时编译后 `lib.minimal_gemm(a, b, None)` 使用(a: M×K,b: N×K,均行连续 half)。
- **精度验证**:以 PyTorch 结果为 base,比 Max Diff / Mean Diff / **Relative Error**(`‖out−ref‖₂ / ‖ref‖₂`,阈值 1e-3);单指令 FP16 16×8×8 场景应得 RE=0.00%(与 torch 位级一致)。

### 3.6.4 计时:CUDA event 的坑,ncu 更准

```bash
ncu -o ncu_prof_1 --import-source 1 --set full --kernel-name "minimal_gemm" -f python minimal_gemm.py
```

- kernel 前后 `cudaEventRecord` + `cudaEventElapsedTime` 测得 **~8us**,而 ncu 报的 Duration **~3us**,nsys 确认 **ncu 更准**;
- **作者推测的原因**:CUDA event 计时起点是"event 被插入 stream 后开始执行"的时刻——当 kernel launch 到 event launch 之间的 CPU 时间大于 kernel 本身执行时间时,event 计时可能把这段 launch 延迟也算进去了。**微小 kernel 一律以 ncu Duration 为准**;
- ncu 重点观测:Duration、Compute/Memory Throughput(判 compute-bound vs memory-bound)、#Registers、Grid/Block size;访存瓶颈进一步看 Memory Chart / Memory Table(指令数、数据量、transaction 数)。

### 3.6.5 PTX/SASS 对照:整个 kernel 的核心只有 6 条指令

编译时开 `--generate-line-info`,在 Nsight Compute 的 Source 页选 "View PTX and SASS"。`C=A*B` 的核心 load-mma-store 对应 6 条 PTX:

```
ld.global.u32 %r5, [%rd9];      ld.global.u32 %r6, [%rd11];   ld.global.u32 %r7, [%rd15];
mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%r1,%r2},{%r3,%r4},{%r5},{%r6,%r7};
st.global.u32 [%rd11], %r1;     st.global.u32 [%rd15], %r2;
```

SASS(SM90)全程序仅 34 条,核心同样 6 条:`LDG.E`×3 → `HMMA.1688.F16` → `STG.E`×2——**CuTe API 层层展开后就是这几条 PTX**,这正是"CUTLASS 是 PTX 白盒封装"的最直观证据(PTX 前向兼容、SASS 架构相关;triton 等编译器的最终产物也是 PTX)。

### 3.6.6 往哪走

单指令 mma(SM80,warp 级、操作数在寄存器)是起点;放大到真实 GEMM 要加多级 tiling、SMEM、pipelining(§4 CollectiveBuilder / [../concepts/gemm-tiling.md](../concepts/gemm-tiling.md));升到 Hopper 则换成 warpgroup 级 `wgmma`(操作数经 descriptor 直接来自 SMEM)——机制详见 [../concepts/hopper-wgmma.md](../concepts/hopper-wgmma.md),`cute::gemm` 调用点在两代之间语法完全一致,变的只是 TiledMMA/fragment/同步的构造。

---

## 4. 推荐用法(3.x):CollectiveBuilder

CUTLASS 3.x 推荐 `CollectiveBuilder`——**给出高层准则,底层参数自动推导**,避免手拼一堆模板。下面是根据示例 49 源码整理的**真实签名骨架**(`CollectiveBuilder` 的第二个实参是算符类别 `OpClassTensorOp`,不是 DispatchPolicy——后者是 builder 的**推导产物**,这正是"自动推导"的体现):

```cpp
// 依据 examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu 整理(示意)
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
// ...(另有 arch/、gemm/device/ 等 4-5 个头,header-only 用法下"include 哪些头"是第一个 kernel 最大的坑)

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,                       // 目标架构
    cutlass::arch::OpClassTensorOp,            // 算符类别(Tensor Op)
    ElementA, LayoutA, AlignmentA,             // A 类型/布局/对齐
    ElementB, LayoutB, AlignmentB,             // B 类型/布局/对齐
    ElementAccumulator,                        // 累加器类型
    TileShape_MNK, ClusterShape,               // tile 形状 + Hopper 必需的 cluster 形状
    cutlass::gemm::collective::StageCountAuto, // 或 StageCountAutoCarveout<...>(大 epilogue 需从 smem carve out)
    cutlass::gemm::collective::KernelScheduleAuto // kernel 调度自动;注意 mainloop 用 Auto 则 epilogue 也要 EpilogueScheduleAuto,须配套
>::CollectiveOp;
using CollectiveEpilogue = /* 同理可 builder 推导 */;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,                              // ← 第一个实参是 ProblemShape(mm,mn,mk,kl)
    CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::PersistentScheduler>;
using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

> ⚠ 照抄要点:真实 `CollectiveBuilder` 必带 `AlignmentA/B`,缺了编译不过(新手最高频报错);`GemmUniversal` 第一个实参是 `ProblemShape`。以仓库 `49_hopper_gemm_with_collective_builder` 源码 L318 起为准。

参考 `examples/49_hopper_gemm_with_collective_builder`(Hopper)/ `71_blackwell_gemm_with_collective_builder`(Blackwell,含 mainloop+epilogue 的 builder schedule 与 epilogue visitor tree)。

---

## 5. CuTe DSL(Python,4.x)

用 Python 写 kernel,JIT 编译成 PTX/CUBIN,性能对标手写 C++:

```python
# 安装(4.x CuTe DSL):官方包
#   pip install nvidia-cutlass-dsl
@cute.jit
def gemm_kernel(...):
    # 用 CuTe layout 代数描述线程-数据布局
    # 用 TiledMma / TiledCopy 划分计算
    ...
```

入门参考官方 **CuTe DSL Quick Start** 与 **`examples/python/CuTeDSL/`** 目录(Python 示例);`examples/cute/` 下面是 C++ 的 CuTe 原语教程(`tutorial/`),不含 Python DSL。注意:未见官方 FP8/FP4 的 `examples/cute/` 示例,别按旧信息去找。

---

## 6. Profiler 快速验证

```bash
make cutlass_profiler -j16
# GEMM
./tools/profiler/cutlass_profiler --kernels=sgemm --m=4352 --n=4096 --k=4096
# 卷积(需补全尺寸参数,缺了 profiler 报参数错误)
./tools/profiler/cutlass_profiler --kernels=s1688fprop --n=8 --h=224 --w=224 --c=1 --k=3 --r=3 --s=3
```

输出含 Disposition/Status、Bytes、FLOPs、Runtime、**Memory 带宽、Math GFLOP/s**——可直接对比不同 tile/配置。

---

## 7. 示例目录导航(按需抄)

| 示例 | 主题 | 架构 |
|---|---|---|
| `00_basic_gemm` | 最简 GEMM 入门 | 通用 |
| `06_splitK_gemm` | Split-K GEMM | 通用 |
| `24_gemm_grouped` / `38_syr2k_grouped` | Grouped GEMM | Ampere |
| `27_ampere_3xtf32_...` | 3×TF32 仿真 FP32 | Ampere |
| `41_fused_multi_head_attention` | 融合多头注意力 | Ampere |
| `47_ampere_gemm_universal_streamk` | Stream-K 调度 | Ampere |
| `48_hopper_warp_specialized_gemm` | Warp-specialized kernel | Hopper |
| `49_hopper_gemm_with_collective_builder` | **CollectiveBuilder 推荐用法** | Hopper |
| `111_hopper_ssd` / `112_blackwell_ssd` | SSD(Structured State Space Duality,结构化状态空间对偶,Mamba-2/Dao-Gu 谱系的 GEMM 序列形式) | Hopper/Blackwell |
| `examples/cute/` | CuTe 原语 C++ 教程(`tutorial/`,含通用 `sgemm_sm70`/`sgemm_sm80` 等,不只 Hopper/Blackwell) | 通用/Hopper/Blackwell |

---

## 8. 常见坑(FAQ 速查)

1. **运行时报 "architecture-accelerated feature"** → 检查 `CUTLASS_NVCC_ARCHS` 是否带 `a`(`90a` 而非 `90`)。
2. **编译极慢/链接失败** → 别用 `CUTLASS_LIBRARY_KERNELS=all`;用通配符子集或 `CUTLASS_LIBRARY_OPERATIONS` 限定。
3. **RTX 50 跑不了 Blackwell kernel** → RTX 50 是 **SM120**,与数据中心 SM100 不同,`sm_100a` kernel 不兼容 RTX 50。**该用** `-DCUTLASS_NVCC_ARCHS=120a`,并参考 GeForce 专用示例(`79_blackwell_geforce_gemm` / `80_blackwell_geforce_sparse_gemm` / `87_..._blockwise`)。
4. **一些示例只给一次构建+运行**:调试单个示例无需全量,`cmake --build . --target 49_hopper_gemm_with_collective_builder` 即可。
5. **Windows 上 3.x 构建失败** → 已知问题(官方 README 注明 **3.x 在 Windows 上有已知构建问题,团队修复中**,未提 4.x),建议 Linux。

---

## 9. 溯源

> 版本说明:本篇官方文档链接统一用 `latest`(跟随最新版,现为 v4.8.0);需查历史版本时把 URL 中 `latest` 换成如 `4.7.0`。

- Quickstart / 初始构建步骤: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html
- CollectiveBuilder 示例与构建: `examples/49_hopper_gemm_with_collective_builder` / `71_blackwell_gemm_with_collective_builder`
- 3.x blog(CollectiveBuilder 代码语境): https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/
- §0(三条路径 / Why CUTLASS / 学习资源)与 §3.6(Minimal GEMM / binding / 精度验证 / CUDA event vs ncu 计时 / PTX-SASS): 杨远航《CUTLASS 笔记》导读 + 笔记(1),知乎专栏 https://www.zhihu.com/column/c_1938664963049763058 (基于 CUTLASS 4.1.0,2025 连载快照;微信公众号为转载载体,以知乎专栏为主源;代码仓库 cutlass-notes)
