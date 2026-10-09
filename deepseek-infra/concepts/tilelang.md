---
title: TileLang — 高性能算子 DSL 与昇腾后端
type: concept
tags: [tilelang, ascend, dsl, tvm, npu, gemm, tilelang-ascend, npuir]
source:
  - "code: tile-ai/tilelang(上游 DSL,多后端 CUDA/ROCm/Metal)| https://github.com/tile-ai/tilelang ↑primary"
  - "code: tile-ai/tilelang-ascend(昇腾专用后端仓,双路线 Ascend C & PTO / AscendNPU IR)| https://github.com/tile-ai/tilelang-ascend"
  - "web: TileLang 官方文档站 | https://tilelang.com/"
  - "code: tilelang-ascend/docs/TileLang-Ascend Programming Guide.md | https://github.com/tile-ai/tilelang-ascend"
date: 2026-09
---

# TileLang — 高性能算子 DSL 与昇腾后端

## TL;DR

**TileLang**(tile-lang)是一种 **tile(数据块)级 DSL**:Pythonic 语法 + TVM 编译器基础设施,让你显式控制内存分配、数据移动、布局与并行,从而把「算子怎么写」翻译成「硬件怎么跑」——同时抽象掉 CUDA / Ascend C 这些硬件细节,又保留芯片特性的优化空间。**它定位是「算子编写 → 硬件执行」之间的桥**。

**昇腾后端要分三层看,别混**:
1. **上游 `tile-ai/tilelang`**(多后端:CUDA/ROCm/Metal)——注意这是 **tile-ai 组织的第三方 DSL(8.5k+ stars),不是 DeepSeek 的仓库**;DeepSeek 是重度使用者兼昇腾后端贡献方。现支持 **Ascend 950 native backend**(2026-09-30 起)。
2. **独立仓 `tile-ai/tilelang-ascend`**——昇腾专用后端,A2/A3 的 ascendc/pto/npuir(Ecosystem),内部走**两条技术路线**:**Ascend C & PTO**(代码生成目标)与 **AscendNPU IR / MLIR**。
3. **tilelang-mlir-ascend**——MLIR 路线。

昇腾后端提供**三种编程接口**(自上而下):Beginner(硬件无关,当前尚不支持)/ Developer(Tile 库,预定义算子原语)/ Expert(线程原语,直接控 Cube/Vector/MTE/UB)。**实测基准(tilelang-ascend 官方 benchmark,详见 §3)**:FlashAttention 达手写 AscendC 的 80%、mHC vs CANN 5.98×、Sparse Flash MLA 五场景 parity。DeepGEMM-Ascend 的 HC prenorm kernel 就依赖 tilelang。

---

## 1. 机制:为什么是「tile 级 DSL」,以及昇腾后端的三层结构

### 1.1 核心抽象:tile 与编译降级

TileLang 的最小编程单元是 **tile(数据块)**。用户用 Python 描述「把输入切成多大的块、块内怎么搬、块间怎么并行」,编译器(TVM 之上)负责降级到具体硬件指令。相比直接写 CUDA/Ascend C:

- **生产力**:不用手写线程索引、寄存器分配、内存层级搬运;
- **可移植**:同一份 kernel 源码换 target 即可(上游支持 CUDA/ROCm/Metal,昇腾走独立仓/主仓 native backend);
- **可优化**:tile 级抽象保留了「分块大小、流水级数、布局」这些对硬件峰值最关键的旋钮,不会被黑盒库吃掉。

### 1.2 昇腾后端:上游 DSL 与昇腾仓的三层结构

**关键事实(本域 P0)**:TileLang 的昇腾支持是**三层结构**,写作必须分清:

| 层 | 仓库 | 定位 |
|---|---|---|
| 上游 DSL | tile-ai/tilelang | 通用 tile-lang 语言 + TVM 编译器基础设施,多后端;现支持 Ascend 950(2026-09-30 起,native backend) |
| 昇腾后端仓 | tile-ai/tilelang-ascend | A2/A3 的 ascendc/pto/npuir(Ecosystem,developed in tilelang-ascend) |
| MLIR 路线 | tilelang-mlir-ascend | MLIR 路线 |

tilelang-ascend 的编译器后端支持**两条技术路线**(不是同一条):
- **Ascend C & PTO**:以 Ascend C 为代码生成目标,PTO 后端(代码生成目标,全称待核)是 2026-01-23 落地的生成目标;
- **AscendNPU IR / MLIR**:以昇腾 NPU IR 为目标(branch `npuir`)。

### 1.3 三种编程接口(降级流程的三个层级)

| 接口 | 抽象层次 | 适用 | 状态 |
|---|---|---|---|
| **Beginner**(Hardware-Unaware) | 完全不关心硬件细节 | 只想写逻辑 | 当前**尚不支持** |
| **Developer**(Tile 库) | 硬件感知,用预定义算子/模式 | 了解内存层次与性能考量的开发者 | 可用 |
| **Expert**(线程原语) | 直接控 Cube/Vector/MTE/UB、细粒度同步、寄存器设置 | 深度性能调优 | 可用 |

三种接口可在同一 kernel 内混用(从高级描述逐步降级到硬件相关实现)。

---

## 2. 可复现代码:developer-mode GEMM(一手,来自 tilelang-ascend README)

```python
@tilelang.jit(out_idx=[-1], target="npuir")
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def gemm(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
             C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, initC=(k == 0))
            T.copy(C_local, C[by * block_M, bx * block_N])
        return gemm
```

读法:与 CUDA 版 tilelang GEMM 同构——`T.alloc_shared`(片上缓冲)、`T.alloc_fragment`(寄存器累加器)、`T.Pipelined(..., num_stages=2)`(两级软件流水)、`T.gemm`(矩阵乘累加)。差别只在 `is_npu=True` 与 `target="npuir"`,即「同一份抽象,换个后端 target」。

---

## 3. 数据对比(带环境口径)

来源:tilelang-ascend 各 examples 目录下一手 benchmark(不同算子不同基线,勿混用同一口径):

| 算子 | 性能 | 基线 / 口径 |
|---|---|---|
| FlashAttention | **80.52% / 80.77% / 80.95%** | 达手写 AscendC(32K / 64K / 128K 序列)——FA 是 tilelang-ascend 的主要性能短板场景之一 |
| mHC(Manifold HyperConnection) | **5.98×** | vs CANN(PyTorch),kernel-only(CANN 9.0.0 环境) |
| Sparse Flash MLA(DeepSeek V4) | 六场景**五 ≥100% parity**,唯一落后 **hca prefill 95.7%** | vs 对应 baseline(≈0.96 的那个数字实为此行,非 mHC) |

口径:**各算子基线不同**——FA/mHC/Sparse Flash MLA 分别 vs 手写 AscendC / CANN / baseline,不能合并成「统一 vs 手写 AscendC」的比值。

---

## 4. 实践要点 / 坑

- **归属别写错**:TileLang 是 **tile-ai**(第三方)的仓库,不是 `deepseek-ai/tilelang`;DeepSeek 只是重度使用者兼昇腾后端贡献方。DeepSeek 官方 6 件套链接里 TileLang 给的正是 `tile-ai/tilelang`。
- **层别混**:上游 `tile-ai/tilelang`(多后端,现已支持 Ascend 950 native backend)、独立仓 `tile-ai/tilelang-ascend`(A2/A3 的 ascendc/pto/npuir)、`tilelang-mlir-ascend`(MLIR)是三层。
- **两条路线别当一条**:Ascend C & PTO 与 AscendNPU IR / MLIR 是两条编译路线。
- **安装**:上游 DSL 走 `pip install tilelang`;昇腾后端走 `tilelang-ascend` 的 `install_ascend.sh` + `source set_env.sh`(需先 `source {cann}/ascend-toolkit/set_env.sh`)。
- **双后端运行时自动选择**:同一 Python API 在 NVIDIA GPU 与华为 NPU 上由运行时按平台选后端。
- **camodel 软件仿真**:无 NPU 硬件也能验证 kernel **逻辑正确性**(A5 平台软件仿真,仅 PTO 后端)——**只验正确性,不做性能测试**,别把仿真耗时当性能数字。

---

## 溯源

- **上游 DSL**:tile-ai/tilelang README(现支持多后端 CUDA/ROCm/Metal 及 Ascend 950 native backend、TVM 之上)—— https://github.com/tile-ai/tilelang ;官方站 https://tilelang.com/
- **昇腾后端仓**:tile-ai/tilelang-ascend(README:双路线 Ascend C & PTO / AscendNPU IR、三种编程接口、FA/mHC/Sparse Flash MLA 基准、DeepSeek V4 kernels 发布)—— https://github.com/tile-ai/tilelang-ascend
- **编程指南**:docs/TileLang-Ascend Programming Guide.md(Beginner/Developer/Expert 三接口、CANN 8.3.RC1+/torch-npu 2.6.0.RC1+ 环境、elementwise add 示例)—— https://github.com/tile-ai/tilelang-ascend/blob/main/docs/TileLang-Ascend%20Programming%20Guide.md
- **库内交叉**:[../../cutlass/concepts/microkernel-abstraction.md](../../cutlass/concepts/microkernel-abstraction.md)(CuTe 的空间/时间级 microkernel 抽象,与 TileLang 的 tile 级抽象对照)、[../../cutlass/concepts/gemm-tiling.md](../../cutlass/concepts/gemm-tiling.md)(三层 tiling/pipelining 通识)。
