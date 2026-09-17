---
title: CUTLASS 构建与上手:从 cmake 到第一个 kernel
type: howto
tags: [cutlass, build, cmake, collectivebuilder, cute-dsl, examples, profiler]
source:
  - "web: NVIDIA CUTLASS Quickstart | https://docs.nvidia.com/cutlass/4.7.0/media/docs/cpp/quickstart.html"
  - "code: examples/49_hopper_gemm_with_collective_builder / 71_blackwell_gemm_with_collective_builder | github.com/NVIDIA/cutlass"
  - "web: NVIDIA Tech Blog — CUTLASS 3.x abstractions (CollectiveBuilder 用法) | https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/ (2025-07)"
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
| `111_hopper_ssd` / `112_blackwell_ssd` | SSD(State Space Decomposition,SSM/Mamba 类算子) | Hopper/Blackwell |
| `examples/cute/` | CuTe 原语 C++ 教程(`tutorial/`) | Hopper/Blackwell |

---

## 8. 常见坑(FAQ 速查)

1. **运行时报 "architecture-accelerated feature"** → 检查 `CUTLASS_NVCC_ARCHS` 是否带 `a`(`90a` 而非 `90`)。
2. **编译极慢/链接失败** → 别用 `CUTLASS_LIBRARY_KERNELS=all`;用通配符子集或 `CUTLASS_LIBRARY_OPERATIONS` 限定。
3. **RTX 50 跑不了 Blackwell kernel** → RTX 50 是 **SM120**,与数据中心 SM100 不同,`sm_100a` kernel 不兼容 RTX 50。**该用** `-DCUTLASS_NVCC_ARCHS=120a`,并参考 GeForce 专用示例(`79_blackwell_geforce_gemm` / `80_..._sparse` / `87_..._blockwise`)。
4. **一些示例只给一次构建+运行**:调试单个示例无需全量,`cmake --build . --target 49_hopper_gemm_with_collective_builder` 即可。
5. **Windows 上 3.x/4.x 构建失败** → 已知问题(官方 README 注明 3.x 与 4.x 在 Windows 均不可用),建议 Linux。

---

## 9. 溯源

> 版本 pin 说明:本篇正文链接统一 pin 4.7.0 docs(快照稳定、不随 latest 漂移);需最新版时把 URL 中 `4.7.0` 换成 `latest`。

- Quickstart / 初始构建步骤: https://docs.nvidia.com/cutlass/4.7.0/media/docs/cpp/quickstart.html
- CollectiveBuilder 示例与构建: `examples/49_hopper_gemm_with_collective_builder` / `71_blackwell_gemm_with_collective_builder`
- 3.x blog(CollectiveBuilder 代码语境): https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/
