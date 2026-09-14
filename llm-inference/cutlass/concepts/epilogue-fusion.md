---
title: CUTLASS Epilogue:主循环之后的融合与后处理
type: concept
tags: [cutlass, gemm, epilogue, fusion, bias, relu, softmax, operator-api]
source:
  - "web: NVIDIA Custom epilogue fusions for GEMMs | https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/001_gemm_with_fused_epilogue.html"
  - "code: CUTLASS examples 12_gemm_bias_relu / 35_gemm_softmax / 37_gemm_layernorm / 45_dual_gemm | github.com/NVIDIA/cutlass"
  - "web: PyTorch Blog — TorchInductor's CuteDSL backend | https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/ (2026-04)"
date: 2026-09
---

# CUTLASS Epilogue:主循环悄悄干完的那些事

## TL;DR

**Epilogue(收尾)** 是 GEMM/卷积主循环(mainloop)算完 `C = A·B` 之后、把结果写回内存之前的所有后处理:`D = alpha·accumulator + beta·C`,以及和 bias / ReLU / GELU / softmax / layer-norm 等的**融合**。

**为什么重要(一句话)**:在 GPU 上,"把中间结果写回 HBM、再读回来做下一步"极其昂贵。把相邻的 element-wise 操作**融进同一个 kernel 的 epilogue**,一次写出,能省掉整趟内存往返——这是深度学习推理框架(如 PyTorch / TensorRT-LLM)性能的关键来源之一。

**核心判断**:GEMM 不是"算一次矩阵乘",而是"**主循环(计算密集)+ epilogue(写回+融合)**"两部分;优化 epilogue 就是优化融合、优化写回效率。

---

## 1. 为什么 epilogue 是性能关键:Kernel fusion 的动机

先看数据搬移的成本(GPU 上最贵的是访问片外 HBM):

- 主循环把 `C` 的 tile 累加在 **registers**(片上,快)里;
- 若不做融合,要把 `C` 整个写回 HBM,再开**新的 kernel** 读回来做 bias/ReLU;
- 每多一次"写回再读回" = 一整趟 HBM 吞吐。

**融合(fusion)** = 把后续的 element-wise 操作塞进**当前 kernel 的 epilogue**,让它们趁数据还在 registers/片上时就做完,只写回一次。

| 做法 | 内存往返次数 | 说明 |
|---|---|---|
| 不融合 | 2+ 次 | GEMM 写 C → 读 C → elementwise 写 D(每多一层多一趟)|
| 融合(bias+ReLU) | 1 次 | registers 里算完 `D=ReLU(beta·C+alpha·acc)`,一次写 HBM |

代价:kernel 变"胖"(一次干更多),但省了 N 趟内存带宽。**对 compute-bound 的 GEMM 是纯赚**;对 memory-bound 的逐元素算子,融合收益尤其大。

---

## 2. Epilogue 的标准形式

CUTLASS 里 epilogue 的经典语义是**线性组合 + 逐元素变换**:

```
D = alpha * accumulator + beta * C
  → 然后可选: bias, ReLU/GELU, softmax, layer-norm ...
```

- `accumulator` = mainloop 在 registers 里累加的 C 部分和(通常 FP32 累加);
- `C` = 可选的第二个输入(如 bias 张量,或残差连接的输入);
- `alpha` / `beta` = 线性比例(scaling),如 `beta=0` 时退化为 `D=alpha·acc`(纯 GEMM);
- 输出 `D` 通常需要**数据类型转换**(FP32 累加 → FP16/BF16 写出)和**转置/排布适配**(写回时按目标 layout)。

**融合的通用模式(bias 为例)**:`12_gemm_bias_relu` 示例里,bias 与输出 layout 的对应关系(官方源码注释原话):
- 输出 **column-major** → bias 是**每行一个**(per-row,即 M×1 向量);
- 输出 **row-major** → bias 是**每列一个**(per-column)。

> 原理:为了让 bias 加法向量化/连续,偏置应沿**输出最连续的那一维**展开——column-major 输出每行的 M 是连续地址,所以 bias 逐行(M×1)才能与内存访问对齐。示例 12 本身用 ColumnMajor + per-row(M×1)bias。另外注意:示例 12 实际用 `ScaleType::NoBetaScaling` 把 `C` 通道当 bias 向量用,与通用公式 `D=alpha·acc+beta·C` 里"`C` 是残差源矩阵"的语义不同——可别套错公式。

---

## 3. 融合能做什么:常见 epilogue 变换

| 变换 | CUTLASS 示例 | 说明 |
|---|---|---|
| bias | `12_gemm_bias_relu` | 加每行/每列偏置 |
| ReLU / GELU | `12_gemm_bias_relu` | 激活函数 |
| softmax | `35_gemm_softmax` | 整行 softmax(需跨 tile 归约)|
| layer-norm | `37_gemm_layernorm_gemm_fusion` | GEMM + LN + 再 GEMM 融合成一个大 kernel |
| 双 GEMM / 残差 | `45_dual_gemm` | 两个 GEMM 或加残差输入 |
| 输出 swizzle | `50_hopper_gemm_with_epilogue_swizzle` | 写回时重排,配合 TMA store layout |
| **EVT 激活融合(3.x 现代范式)** | `113_hopper_gemm_activation_fusion` | EVT + `AccCastLinCombEltActScale`,含 gated activation(= SwiGLU 类) |

**softmax / layer-norm 这类"全局归约"的融合,要看清它是"多级 kernel"的**:

- softmax 需要对**整行**先做 reduction(求 max/scale)才知道归一化因子——这天然跨多个 tile;
- 实际 `35_gemm_softmax` 是**三次 kernel launch**,不是一次写出:
  1. **GEMM + epilogue visitor** 做 partial reduction:写 `D` + 每 tile 的行最大/部分和;
  2. **最终归约 kernel**(`ApplySoftmaxFinalReduction`,来自 `cutlass/reduction/kernel/reduce_softmax_final.h`):跨 tile 归约出每行的 sum/系数;
  3. **exp 归一化 kernel**(示例内另定义的 `ApplySoftmax`):对 `D` 做 `exp(x-norm)/sum` 写出。
- 所以"融合"省的是**把部分归约融进 GEMM epilogue**,但 `D` 仍要读回做两次后处理——**没有免费午餐**。设计心智模型:**1 个 GEMM + 2 个后处理 kernel(final reduction 与 apply)**,不是一步到位。

> **layer-norm 别按 softmax 猜**:示例 `37_gemm_layernorm` 要求 hidden dim **落在单个 epilogue tile 内**,在 epilogue 内一次算完(不跨 tile 交换统计量),与 softmax 的跨 tile 两级不同。另外它的全融合 LN 用 **single-pass 平方和方差算法**,源码注释**明确警告有数值稳定性问题**(引用 Wikipedia "Algorithms for calculating variance",可能不适合真实场景),官方给替代方案 `tools/util/include/cutlass/util/device_layernorm.h`——**这是最容易踩的精度坑,做 LN 融合前先看这条**。

---

## 4. CUTLASS 里怎么做 epilogue:两代 C++ API + Python API

### 4.1 传统 C++(2.x vs 3.x):Thread functor → Epilogue Visitor Tree(EVT)

**先分清两代机制(容易搞混)**:
- **2.x 的主机制是 thread-level fusion functor**:`cutlass::epilogue::thread::LinearCombination*` 系列(如 `LinearCombinationReLU`)。经典示例 `12_gemm_bias_relu` / `35_gemm_softmax` / `37_gemm_layernorm` 都是这一路——已对照源码确认它们用老式 thread epilogue(如 `cutlass::epilogue::thread::ScaleType::NoBetaScaling`)。所以这些示例**不是用 EVT**。
- **3.x(SM90/Hopper 起)的主机制才是 EVT**(`cutlass::epilogue::fusion` 命名空间、`Sm90EVT`)。

**3.x 的 EVT**:是一棵由 **Load / Compute / Store 节点组成的"逐元素计算图"**,Load 节点取 **accumulator(寄存器累加结果)/ C / bias 等**(注意:mainloop 的输入矩阵 A 不是 epilogue 输入,A 在 mainloop 阶段已消费),Compute 节点做算术(内建 `LinCombEltAct` / `DeEltAct` 等),Store 节点写回输出。**价值在于用组合子表达任意 DAG**(多 Load、多 Store、per-column broadcast)——这是手写 `LinearCombination` 时代做不到的。

> ⚠ **EVT fusion op 的注入位置(易搞错)**:要由 **`cutlass::epilogue::collective::CollectiveBuilder`** 显式作为模板参数传入(不是"gemm builder 有 epilogue 参数")。参照 `113_hopper_gemm_activation_fusion`(Hopper EVT 激活融合,含 gated activation = LLM 的 SwiGLU 类融合):
> ```cpp
> using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
>     cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, /* ... */
>     cutlass::epilogue::fusion::AccCastLinCombEltActScale<...>   // ← fusion op 显式模板参数
> >::CollectiveOp;
> ```

**3.x(尤其 Hopper)epilogue 真实数据通路**:registers 累加结果 → 经 SMEM 分块暂存(方便跨 warp 处理 EPILOGUE tile)→ TMA store 写回。不是单纯"registers 写完就出去",中间可能过 SMEM;写回排布(swizzle)也因此重要。Blackwell(SM100)的 EVT 还要叠 TMEM 累加器路径。

> 一段能读懂 EVT 的最小指引:定位 `cutlass/epilogue/fusion/` 下的 Operation(如 `LinCombEltAct` / `DeEltAct`),再按上面 113 示例把它作为 `cutlass::epilogue::collective::CollectiveBuilder` 的显式模板参数拼进 epilogue(不是走 gemm builder)。想了解 EVT 的现代用例,直接看 `113_hopper_gemm_activation_fusion`;想快速上手老式 thread functor,看 `12_gemm_bias_relu` 即可。

### 4.2 4.x Operator API(Python):把 epilogue 当函数传

> ⚠ **运行前提**:官方这片 Operator API 教程首段即要求 **Blackwell `sm_100f` 家族 GPU**(`device_or_env_supports("100f")`);非 Blackwell 环境跑不了该示例。上手时先确认硬件,别照抄示例发现起不来。

CUTLASS 4.x Operator API 允许**用 torch.Tensor 层级的函数直接表达 epilogue**,框架自动把它融进支持该函数的 GEMM operator:

```python
def my_epilogue(accum, C, alpha, beta, extra_scalar):
    Aux = (alpha * accum) + (beta * C)
    D = extra_scalar * Aux
    return D, Aux   # 可以返回多个输出(如 softmax 的中间量留给反传)

epi_args = ops.EpilogueArguments(
    my_epilogue, C=C, alpha=alpha, beta=beta,
    extra_scalar=extra_scalar, D=D_, Aux=Aux_,
)
args = ops.GemmArguments(A=A, B=B, out=D, accumulator_type=torch.float32,
                         epilogue=epi_args)
operators = ops.get_operators(args, target_sm=...)
```

- 支持**多输出**(如同时出 `D` 和中间量 `Aux` 供 backprop 用);
- 支持 **partial(per-row / per-column)reductions**;
- 与 **legacy CUTLASS Python API 的 epilogue visitor tree 前端**共享接口(注意:是 Python DSL 的 EVT 前端,不是 C++ EVT)。

### 4.3 为什么这受框架欢迎:TorchInductor 的视角

PyTorch 的 TorchInductor 在 GEMM 上有三套 autotune 后端:Triton、CUTLASS C++、cuBLAS。他们的博客(2026-04)把 **CuteDSL**(CUTLASS 4.x 的 Python DSL)作为**第四套后端**引入,理由里明确点了 **epilogue fusion 的优势**:

> "CuteDSL 建立在与 CUTLASS C++ 相同的抽象上，后者在 FP8 GEMM 和 **epilogue fusion** 上都已证明出色性能，但 CuteDSL 是 Python、编译更快、更易维护——有潜力在更新硬件上**取代 CUTLASS C++ 集成**。"

也就是说,epilogue fusion 不只是 CUTLASS 的一个特性,而是它被现代推理框架采纳的**核心竞争力之一**。

---

## 5. 实践要点(写/调 epilogue 时的 checklist)

1. **先判断该不该融合**:compute-bound GEMM 融合几乎总是赚;但若 epilogue 变换复杂且访问模式差,也要实测对比(融合 kernel 更胖,占用率可能降)。
2. **注意寄存器压力**:epilogue 在 registers 里工作,融合太多变换会让寄存器暴涨、占用率下降。要在"融合收益"和"occupancy"间权衡。
3. **per-row / per-column 偏置的 layout 陷阱**:输出 **column-major → bias 逐行(per-row)**;row-major → bias 逐列。搞反了 bias 就算错(官方示例 12 注释原话)。
4. **softmax/LN 这类全局归约是"多级的"**:GEMM epilogue 只做 **partial reduction**,再由独立 kernel 做最终归约 + 归一化(softmax 共 3 个 kernel:GEMM→final reduction→apply)——不是一次写出。**LN 融合另有精度坑**(见 §3):single-pass 平方和方差不稳定。
5. **多输出/反传**:需要中间量留给反向传播的(如 softmax 的 scale),用支持多输出的 epilogue 一次算好,别算完再重算。
6. **写回排布**:写 HBM 时注意读写合并与目标 layout 适配(`epilogue swizzle` 就是为此设计),避免按 tile 边界非连续写。

---

## 6. 溯源

- 官方 Operator API epilogue fusion 教程(Blackwell `sm_100f`,Python 函数式 epilogue): https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/001_gemm_with_fused_epilogue.html(第 2-4 段讲了 `EpilogueArguments` / `GemmArguments` / `get_operators` 的用法)
- 融合示例: `12_gemm_bias_relu` / `35_gemm_softmax` / `37_gemm_layernorm_gemm_fusion` / `45_dual_gemm` / `50_hopper_gemm_with_epilogue_swizzle` / `113_hopper_gemm_activation_fusion`(EVT 现代范式,github.com/NVIDIA/cutlass)
- TorchInductor 引入 CuteDSL 及 epilogue fusion 竞争力说明(2026-04,引文为中文转译): https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/
