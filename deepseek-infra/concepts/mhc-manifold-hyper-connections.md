---
title: mHC(Manifold-Constrained Hyper-Connections)— 把残差流投影到 Birkhoff 多面体,以及 DeepGEMM 的融合内核
type: concept
tags: [mhc, hyper-connections, residual-stream, birkhoff, sinkhorn, doubly-stochastic, deepgemm, cuda-graph, split-k, tma, tmem, sm100, tilelang, ascend]
source:
  - "pdf: Xie et al., mHC — Manifold-Constrained Hyper-Connections (DeepSeek-AI) | arXiv:2512.24880v2 | https://arxiv.org/abs/2512.24880 ↑primary"
  - "code: deepseek-ai/DeepGEMM( mega_mhc 内核 / CUDA graph epoch 修复 #478) | https://github.com/deepseek-ai/DeepGEMM"
  - "code: deepseek-ai/TileKernels(mHC 的 TileLang 参考实现,Sinkhorn + mix 拆分) | https://github.com/deepseek-ai/TileKernels"
  - "code: deepseek-ai/DeepGEMM-Ascend(昇腾侧 HC Prenorm GEMM,带宽口径数据) | https://github.com/deepseek-ai/DeepGEMM-Ascend"
  - "pdf: Zhu et al., Hyper-Connections(mHC 的前作) | arXiv:2409.19606v3 | https://arxiv.org/abs/2409.19606"
date: 2026-10
---

# mHC(Manifold-Constrained Hyper-Connections)

## TL;DR

**残差连接十年没变过形状**;Hyper-Connections(HC)把残差流从 1 条扩成 n 条(典型 n=4)并让它们互相混合,换来明显收益,但代价是**丢掉恒等映射的性质** → 深层训练发散(复合映射增益峰值 ~3000)。**mHC 只加了一条约束**:把 n×n 的残差混合矩阵投影到**双随机矩阵流形(Birkhoff 多面体)**,用 Sinkhorn-Knopp 迭代实现。双随机矩阵谱范数 ≤1、对乘法封闭、是置换矩阵的凸组合——于是前向信号增益与反向梯度增益都钉在 1 附近(27B 30 层复合后偏差最大 ~1.6),**恒等映射在任意深度上被恢复**。

**工程上它是三件事**:① 系数生成(3 个小矩阵乘法 + Sinkhorn 20 轮)全部融进一个 kernel;② 显存用「按 L_r≈√(nL/(n+2)) 层分块、丢掉中间激活、反向重算」压住;③ 与 DualPipe 流水重叠。最终 **n=4 只增加 6.7% 训练时间**,27B 上 8 个下游基准平均比 baseline 高约 4.5 分、比 HC 高约 0.9 分(7 胜 1 负)。

**DeepGEMM 的 `mega_mhc`** 是这套东西在 SM100(Blackwell)上的单 kernel 实现:6 个 warpgroup 静态分工(TMA/MMA、Post×2、workspace epilogue、Mix、Norm)、split-K 工作区、跨 CTA 的 Mix/Norm 屏障。**2026-10-20 的 #478 修掉一个真 bug**:屏障状态原本用 `%gridid` 做代数,而 CUDA graph 重放时 `%gridid` 恒定 → 重放会读到上一次重放的旧 partial;改法是在屏障缓冲区首行的空闲尾部存 **epoch 计数**,最后一个退出的 CTA 推进 epoch。

---

## 1. 问题:HC 拿到了收益,但破坏了恒等映射

### 1.1 HC 是什么

标准残差:$\mathbf{x}_{l+1}=\mathbf{x}_l+\mathcal{F}(\mathbf{x}_l,\mathcal{W}_l)$。HC(Zhu et al., 2024)把 $\mathbf{x}_l$ 沿宽度复制成 $n$ 条流 $\mathbf{x}_l\in\mathbb{R}^{n\times C}$(n 称 expansion rate),再用三个可学习映射管住「读入 / 写出 / 流间混合」:

$$\mathbf{x}_{l+1}=\mathcal{H}^{\mathrm{res}}_l\,\mathbf{x}_l+\mathcal{H}^{\mathrm{post}\top}_l\,\mathcal{F}\!\left(\mathcal{H}^{\mathrm{pre}}_l\mathbf{x}_l,\mathcal{W}_l\right),\qquad
\mathcal{H}^{\mathrm{pre}}_l,\mathcal{H}^{\mathrm{post}}_l\in\mathbb{R}^{1\times n},\ \mathcal{H}^{\mathrm{res}}_l\in\mathbb{R}^{n\times n}$$

- $\mathcal{H}^{\mathrm{pre}}$:把 nC 维流聚合成 C 维层输入(读);
- $\mathcal{H}^{\mathrm{post}}$:把层输出写回 n 条流(写);
- $\mathcal{H}^{\mathrm{res}}$:**流内混合**(每条流 t 的新值 = 上一轮所有流的加权和)。

系数本身是「静态 bias + 依赖输入的动态项」:HC 原论文用 $\alpha\tanh(\theta\tilde{\mathbf{x}}^\top)+\mathbf{b}$;mHC 改为先把 $\mathbf{x}_l$ 展平成 $\vec{\mathbf{x}}_l\in\mathbb{R}^{1\times nC}$ 再做线性投影,以保留完整上下文。消融(论文 Tab.1,绝对 loss gap 对 baseline)很清楚:**$\mathcal{H}^{\mathrm{res}}$ 贡献最大**——只开它 −0.022,再加 pre −0.025,三个全开 −0.027。所以「流间信息交换」是 HC 的主要收益来源,也正是风险来源。

关键点:**HC 增宽的是残差流,不是层输入**($n\ll C$,典型 4 vs 2560),所以 FLOPs 几乎不变;HC 原论文在 OLMoE-1B-7B 上 DHC×4 比 baseline **收敛快 1.8×**、500B token 时 ARC-Challenge +6 分。这正是它吸引人的地方——用几乎免费的拓扑换性能。

### 1.2 代价一:复合映射的增益失控

把 HC 按 $\mathbf{x}_{l+1}=\mathcal{H}^{\mathrm{res}}_l\mathbf{x}_l+\dots$ 递归展开,从第 $l$ 层到第 $L$ 层的有效传播是**矩阵连乘** $\prod_{i=1}^{L-l}\mathcal{H}^{\mathrm{res}}_{L-i}$。单个 $\mathcal{H}^{\mathrm{res}}_l$ 学出来是任意实矩阵(可正可负、行和≠1),连乘几十层后:

- 前向看**行和的最大绝对值**、反向看**列和的最大绝对值**(论文称 Amax Gain Magnitude),理想值都是 1;
- 27B 实测 HC 的复合增益**峰值 ~3000**,且不是个别路径——最大增益大的时候其他元素也一起大,整条传播链路都在抖;
- 现象:HC 在 ~12k step 出现 loss 突增,梯度范数同步异常(论文 Fig.2)。

一句话因果:**identity mapping 是「守恒机制」(跨流平均信号强度不变),HC 的无约束混合把它变成「放大/衰减器」,深度一长就爆。**

### 1.3 代价二:访存与并行通信

HC 的 FLOPs 几乎为零,但**访存成了新瓶颈**。论文 Tab.2 给了每 token、每残差层的元素级 I/O(不含 $\mathcal{F}$ 内部):

| 方案 | 读(元素) | 写(元素) |
|---|---|---|
| 标准残差(残差合并) | $2C$ | $C$ |
| HC(算 pre/post/res + 三次 apply + 残差合并) | $(5n+1)C+n^2+2n$ | $(3n+1)C+n^2+2n$ |

代入 n=4(我按公式重算):**读 21C+24、写 13C+24**,相对标准残差的 2C/1C 是 **~10.5× 读、~13× 写**。再叠加两件事:① pre/post/res 的参数要留中间激活给反向,显存吃紧到必须上 checkpoint;② 流水并行下残差流宽了 n 倍,**PP 通信量也 ~n 倍**(气泡变大)。

融合能大幅回收这笔账:把 $\mathcal{H}^{\mathrm{post}}$ 与 $\mathcal{H}^{\mathrm{res}}$ 的应用与残差合并融成一个 kernel(即 $F_{\mathrm{post,res}}$),读从 $(3n+1)C$ 降到 $(n+1)C$、写从 $3nC$ 降到 $nC$(n=4:13C→5C、12C→4C)。这就是后面 §4 那个 kernel 里的 WG1–WG3 在干的事。

---

## 2. 机制:把 $\mathcal{H}^{\mathrm{res}}$ 投影到 Birkhoff 多面体

### 2.1 约束定义

$$\mathcal{P}_{\mathcal{M}^{\mathrm{res}}}(\mathcal{H}^{\mathrm{res}}):=\left\{H\in\mathbb{R}^{n\times n}\;\middle|\; H\mathbf{1}_n=\mathbf{1}_n,\ \mathbf{1}_n^\top H=\mathbf{1}_n^\top,\ H\ge 0\right\}$$

即**双随机矩阵**(行和、列和都为 1,元素非负),几何上是 **Birkhoff 多面体**——置换矩阵集合的凸包。三条性质直接对应三个工程需求:

1. **非扩张**:双随机矩阵谱范数 $\|H\|_2\le 1$ → 单层增益有界,梯度不会指数放大;
2. **乘法封闭**:两个双随机矩阵相乘仍是双随机 → 复合映射 $\prod_i\mathcal{P}(\mathcal{H}^{\mathrm{res}}_{L-i})$ **在任意深度都保持守恒**,这是「恢复到任意深度」的关键(单纯给每层做归一化没有这个性质);
3. **凸组合语义**:$H\mathbf{x}$ 是 n 条流的凸组合 → 特征均值守恒、信号范数被正则化;反复施加会**单调增加流间混合**,天然是稳健的特征融合机制。

边界情形:$n=1$ 时双随机退化成标量 1,即原始恒等映射——这把「残差」和「HC」统一成同一族的两端。

另外 $\mathcal{H}^{\mathrm{pre}},\mathcal{H}^{\mathrm{post}}$ 只做**非负约束**(sigmoid / 2·sigmoid),防止正负系数相互抵消(也算一种特殊的流形投影)。

### 2.2 参数化:Sinkhorn-Knopp 迭代

先把 $\vec{\mathbf{x}}_l=\mathrm{vec}(\mathbf{x}_l)\in\mathbb{R}^{1\times nC}$ 做 RMSNorm,再一次投影出所有系数(论文 Eq.10–13 把 bias 与投影合并成 $\varphi_l\in\mathbb{R}^{nC\times(n^2+2n)}$、标量 $\mathbf{b}_l\in\mathbb{R}^{1\times(n^2+2n)}$,RMSNorm 权重也吸进 $\varphi_l$):

```
h_tilde = [H_pre~, H_post~, H_res~] = (x_vec / r) * phi + b        # r = ||x_vec||_2 / sqrt(nC)
H_pre   = sigmoid(h_tilde_pre)                                      # 非负
H_post  = 2 * sigmoid(h_tilde_post)                                 # 非负,(0,2)
H_res   = SinkhornKnopp(exp(h_tilde_res))                           # 双随机
```

Sinkhorn 迭代就是「行归一化 / 列归一化交替」:

```python
M = exp(H_tilde_res)                        # 先取指数保证正
for t in range(t_max):                      # 论文 t_max = 20
    M = M / M.sum(dim=-1, keepdim=True)     # 行和 → 1
    M = M / M.sum(dim=-2, keepdim=True)     # 列和 → 1
H_res = M                                   # t_max→∞ 时收敛到双随机
```

论文补了两处工程常数:行归一化后加 `sinkhorn_eps`(代码里的 `+ mix_args.sinkhorn_eps`)避免除零/退化;反向不存迭代中间量,而是**在 kernel 内重算整条迭代**(自定义 backward kernel)。**有限迭代 = 近似解**,这正是后文「复合增益 1.6 而非 1」的来源,也是选择 t_max 的精度/开销折中。

> 数值口径提醒:`num_sinkhorn_iters` 是运行时参数。论文实验取 20;DeepGEMM 的单测取 10。写 benchmark 结论时必须对齐口径,别把 10 轮的结果当 20 轮。

---

## 3. 系统实现(论文 §4.3):三处必须做的优化

| 手段 | 解决什么 | 做法 |
|---|---|---|
| kernel 融合 + RMSNorm 重排 | nC 维 RMSNorm 的延迟 | 把「除以范数」挪到矩阵乘之后(数学等价),三个系数 kernel 合成少数几个;backward 两次 matmul 合成一个,免二次载入 $\vec{\mathbf{x}}_l$ |
| 选择性重算 | n 条流 + 系数中间激活的显存 | 前向丢掉 mHC kernel 中间量,反向重跑 mHC kernel(不含 $\mathcal{F}$);每 $L_r$ 层只留一份 $\mathbf{x}_{l_0}$ |
| DualPipe 扩展 | 跨 stage 通信 ~n 倍 + stage 边界重算开销 | $\mathcal{F}_{\mathrm{post,res}}$ 放独立高优先级 stream;attention 侧不用 persistent kernel 以便被抢占;重算块与 pipeline stage 边界对齐 |

重算块大小有个闭式解(论文 Eq.20):常驻 $\lceil L/L_r\rceil$ 份 $\mathbf{x}_{l_0}$(每份 nC)+ 活跃块暂存 $(n+2)C\times L_r$,最小化总占用得

$$L_r^*=\arg\min_{L_r}\left[nC\left\lceil\frac{L}{L_r}\right\rceil+(n+2)C\,L_r\right]\approx\sqrt{\frac{nL}{n+2}}$$

n=4、L=30(27B 模型层数)代入 ≈ **4.5 层**;而 27B 每 stage 层数与之接近,于是「重算边界 = pipeline stage 边界」既省又不需要跨 stage 依赖。这是把理论最优和并行约束对齐的一个典型手法。

---

## 4. DeepGEMM 的 `mega_mhc`:一个 kernel 干完整层

### 4.1 接口与几何(源码核实)

`deep_gemm.mega_mhc(...)`(SM100 专属,`get_arch_major()==10`)输入输出:

| 张量 | 形状 | dtype | 含义 |
|---|---|---|---|
| `x` | `[T,H]` | bf16 | 当前层输入(单流) |
| `residual` | `[T,n,H]` | bf16 | n 条残差流 |
| `post_mix` / `comb_res_mix` | `[T,n,1]` / `[T,n,n]` | fp32 | 上一轮写回系数 / 流间混合矩阵 |
| `fn` | `[n(n+2), n·H]` | fp32 | 合并后的投影权重($\varphi$),TF32 GEMM |
| `mix_scales` / `mix_bases` | `[3]` / `[n(n+2)]` | fp32 | 三组 $\alpha$ / bias |
| `new_residual` | `[T,n,H]` | bf16 | 新的 n 条流 |
| `new_{post,prev,comb}_mix` | 同输入 | fp32 | 供下一轮(及 shifted 版)用的新系数 |
| `y_bf16` / `y_fp8` + SF | `[T,H]` / `[T,H/128]` | bf16/fp8/uint32 | 融合 RMSNorm 后的层输入,可选 FP8 输出(U E8M0 packed SF) |

编译期常量(源码 `layout/mega_mhc.cuh`):`kNumRoutes=4`、`kNumHCOutputs = n(n+2) = 24`、`BLOCK_M=64`、`BLOCK_N=24`、`BLOCK_K=64`、`kNumIOStages=4`、`kNumAStages=4`、`kNumThreads=6×4×32=768`。**$n(n+2)=n+n+n^2$** 正好是 pre(n 个)+post(n 个)+res(n² 个)系数总数——一个 warp 用 24 条 lane 就能同时持有全部输出系数,这是「Mix 用单 warp 完成」的前提。

一个 loop 里做的事(注意和论文公式的对应):`new_residual = x ⊗ post_mix + comb_res_mix @ residual`(流更新)→ TF32 GEMM `mixes = new_residual @ fnᵀ`(系数 logit)→ 乘 `rsqrt(mean_square)` 做**重排后**的 RMSNorm → sigmoid / Sinkhorn → RMSNorm 输出(可选 FP8 cast + UE8M0 SF)。**Residual merge、系数生成、Sinkhorn、Norm、FP8 量化全在一次 kernel 启动内。**

### 4.2 6 个 warpgroup 的静态分工

persistent kernel,`grid = num_sms`(一 SM 一 CTA),6 个 WG,每个 WG 静态领一个角色,注册器按角色重分配(`setmaxnreg`):

```
WG0  TMA/MMA producer      : 载 residual/x/fn/系数,发 UMMA(TF32),4 级 pipeline
WG1  Post worker           : 流更新残差合并 + 写 new_residual(每 lane 2 行)
WG2  Post worker           : 同上(双 WG 覆盖 BLOCK_M=64 行)
WG3  workspace epilogue    : split partial 落 gmem(GEMM partial + hc_sqr_sum),轻量寄存器
WG4  Mix worker            : warp 内 Reduce split partials → 系数 → Sinkhorn → 存 new_*_mix
WG5  Norm worker           : 融合 RMSNorm(+FP8 cast/SF 打包);shifted 版 WG1/WG2 退休后也并入此队列
```

寄存器预算:control 32、Post 136(shifted)/112(normal)、lightweight 40、Norm 96、normal 80;有静态断言 `kMaxRegisterBudget*128 <= 64512` 守着单 SM 寄存器总量。**每个角色在自己的活干完之后 `warpgroup_reg_dealloc` 交还配额**(Post worker 返回后 → 32;normal 版所有 WG 干完 → 32;WG3/WG4 一开始就降到 40,因为它们是轻量归约),这样 cross-role 的寄存器池可以在 CTA 内被下一个角色复用——这是「一个 CTA 里塞 6 种角色还不溢出」的关键技巧。**注意 dealloc 是 per-role 而非 per-task**:一旦降额,该角色不会再加回去,所以任务循环里的热路径必须用低额度的那套寄存器。

### 4.3 split-K 与跨 CTA 屏障

M 方向按 `BLOCK_M=64` 切 m-block,K 方向再切 `kNumSplits`,task = (m_block, split)。分片数由 host 侧启发式给(`get_num_splits`):

```
若 deterministic_algorithms: splits = 16(kDefaultNumSplits)
否则: max_s = min(64, max(16, num_sms / num_m_blocks))
      splits = ceil_div(num_k_blocks, ceil_div(num_k_blocks, max_s))   # 最少分片数满足最短最长任务
```

按这个公式重算两例(hidden=7168 → k_blocks=112,148 SM;数值为我按源码公式计算,非 README 数据):

| m_blocks | max_s | splits | 每片 k-block |
|---|---|---|---|
| 1 | 64 | 56 | 2 |
| 8 | 18 | 16 | 7 |

也就是说小 batch(T 小、m_block 少)时靠**多切 K** 把 SM 喂满。注意 `kNumMaxSplits=64` 与整除约束:`hidden % (kDefaultNumSplits*BLOCK_K) == 0` 是 host 侧硬断言。

工作区(`Workspace`)布局 = `gemm_partials[num_m_blocks, splits, 64, 24]` + `hc_norm_sqr_sum_partials[..., 64]` + `x1_sqr_sum_partials[..., 64]`,每个 task `64×24 + 2×64 = 1664` 个 float(6656 B)。**Mix 阶段在同一 warp 内做归约**:24 条 lane 各负责一个系数,`#pragma unroll` 沿 split 累加(splits ≤ 64,有界所以能展开);sqr_sum 由 32 lane 分片累加后 `warp_reduce_sum`。

跨 CTA 只同步两件事,用两条 gmem 屏障线:

- `Norm`(barrier 0):shifted 版本里,WG1/WG2 把 X1 与统计量落盘后才 arrive;WG5 等齐才做 Norm;
- `Mix`(barrier 1):WG3 等齐所有 split 的 partial 才让 WG4 归约。

屏障缓冲:2 条 × `kNumMaxMBlocks`(2²⁰ token / 64 = 16384)行 × 128 B = **4 MiB**;按 CUDA stream 缓存(`split_barriers_by_stream`),所以**不同 stream 上的并发调用互不干扰**(单测明确验证这一条)。

---

## 5. 2026-10-20 的 `#478`:CUDA graph 重放时 `%gridid` 不是「这一次」

这是本轮同步要补的核心变化,也是一个**可以当教材的同步 bug**。

### 5.1 症状与根因

原实现用 grid 维的硬件 ID 给屏障状态打「代次」标签:

```cpp
CUTLASS_DEVICE uint64_t get_grid_idx() {          // ptx/utils.cuh
    uint64_t grid_idx; asm volatile("mov.u64 %0, %%gridid;" : "=l"(grid_idx));
    return grid_idx;
}
static uint64_t encode_state(uint64_t grid_idx, uint32_t arrived = 0) {
    return (grid_idx + 1) * kNumSplitBarriers * kNumMaxSplits + arrived;   // 旧
}
```

**CUDA graph 的一个 kernel node 在每次 replay 时 `%gridid` 是同一个值**。于是第 2 次 replay 开始时,屏障线里残留的仍是上一次 replay 的终态 `encode_state(grid_idx, kNumSplits)`——正好等于这一次要找的目标值。还没开始干活的 m-block 初始化 CTA 一旦迟到,Mix/Norm 的 waiter 就直接放行,读到**上一次 replay 的旧 split partial**。表现是偶发数值错误、且只在 graph replay 下出现(即时启动每次 gridid 都变,不复现)。

细节上还有一层:初始化与等待的先后被 `wait_init` 保护(用无符号距离判断 `state - base >= kNumSplits` 拒绝陈旧标签),但当 gridid 恒定,「基线」和「陈旧值」重合,保护失效。

### 5.2 修法:epoch 存在缓冲区自己的空闲尾部

```cpp
// layout/mega_mhc.cuh —— 缓冲不变,只是把首行没用到的后半段征用
static constexpr uint64_t kSplitBarrierEpochOffset     = kSplitBarrierLineBytes / 2;      // 64
static constexpr uint64_t kSplitBarrierExitCountOffset = kSplitBarrierEpochOffset + 8;   // 72
DG_STATIC_ASSERT(kSplitBarrierExitCountOffset + sizeof(uint32_t) <= kSplitBarrierLineBytes, ...);
```

一条屏障线 128 B,但只用了头部 8 B 存状态,尾部的 16 B 空着——**不动 host 侧缓冲区形状**就把 `{epoch: u64, exit_count: u32}` 塞进第 0 行的尾部。代次推进用「最后一个退出的 CTA」负责:

```cpp
struct Epoch {
    static CUTLASS_DEVICE uint64_t get(uint64_t* bar) {          // 每个 CTA 在 grid 依赖同步之后读
        return ptx::ld_acq_gpu(get_epoch_ptr(bar));
    }
    static CUTLASS_DEVICE void retire(uint64_t* bar, const uint32_t& num_ctas) {
        auto cnt = get_exit_count_ptr(bar);
        if (ptx::atomic_add_rel(cnt, 1) == num_ctas - 1) {        // 我是最后一个
            __threadfence();
            *cnt = 0;                                            // 复位计数
            ptx::st_rel(get_epoch_ptr(bar), *get_epoch_ptr(bar) + 1);   // 推进 epoch
        }
    }
};
// kernel 尾部(所有角色 __syncthreads 之后、TMEM 释放之后)
if (threadIdx.x == 0)
    sched::mega_mhc::Epoch::retire(gmem_split_barriers, gridDim.x);
```

关键时序约束(源码注释写得很明确):**epoch 从 `cudaGridDependencySynchronize()` 之后到所有 CTA 退出之前保持稳定**。kernel 里正好在 `cudaGridDependencySynchronize()` 之后才 `init` 屏障:

```cpp
cudaGridDependencySynchronize();                       // PDL:等前一个 grid 真的完成
if (warp_idx == kFirstPostWGIdx*kNumWarpsPerWG + 1 and cute::elect_one_sync())
    for (m_block ... ) Mix::init(gmem_split_barriers, m_block);       // 用 Epoch::get() 编状态
```

`encode_state` 只换了参数名(epoch 代替 grid_idx),状态编码算术不变:`(epoch+1)·2·64 + arrived`,例如 epoch=3、splits=56 → `4·128+56 = 568`。**同一代内 arrival 计数仍在低 7 位,不同代次天然区隔**,且"代次 × 屏障 × 最大分片"的进位关系保证跨代次不撞值。

### 5.3 三个可复用的教训

1. **不要用硬件 ID 给「同一次逻辑执行」打标签**:`%gridid` / `blockIdx` 在单次启动内稳定,但 graph replay 会复用;凡是要跨 replay 判新旧,必须用**由 kernel 自己推进、存在显存里的**计数器。
2. **屏障缓冲的空闲字节可以征用**:128 B 对齐的行只用了 8 B,把元数据塞进尾部既省一次分配,也保证了「屏障与其代次同址」——避免了两个 buffer 生命周期不同步。加静态断言锁住布局。
3. **graph 捕获前必须 warm up**:`get_split_barriers` 里断言 `currentStreamCaptureStatusMayInitCtx() == CaptureStatus::None`,因为首次 `torch.zeros` 只做一次、不能被捕获进 graph;单测 `test_mega_mhc_api_contract` 直接把「未 warm up 就 capture 会抛 `CaptureStatus::None` 断言」写成了契约。用这个 kernel 的人照抄这条测试即可。

---

## 6. 数据

### 6.1 架构收益(论文,27B MoE,基于 DeepSeek-V3 结构)

| 基准(metric,shots) | Baseline | +HC | +mHC | mHC−HC |
|---|---|---|---|---|
| BBH (EM, 3-shot) | 43.8 | 48.9 | **51.0** | +2.1 |
| DROP (F1, 3-shot) | 47.0 | 51.6 | **53.9** | +2.3 |
| GSM8K (EM, 8-shot) | 46.7 | 53.2 | **53.8** | +0.6 |
| HellaSwag (Acc, 10-shot) | 73.7 | 74.3 | **74.7** | +0.4 |
| MATH (EM, 4-shot) | 22.0 | **26.4** | 26.0 | −0.4 |
| MMLU (Acc, 5-shot) | 59.0 | 63.0 | **63.4** | +0.4 |
| PIQA (Acc, 0-shot) | 78.5 | 79.9 | **80.5** | +0.6 |
| TriviaQA (EM, 5-shot) | 54.3 | 56.3 | **57.6** | +1.3 |

(差值是我按表重算的;8 项里 mHC 对 HC 7 胜 1 负,平均 +0.91;对 baseline 平均 +4.49,与论文「surpasses HC on the majority」一致。)

- **稳定性**:mHC 相对 baseline 最终 loss 低 **0.021**,梯度范数曲线与 baseline 同级;HC 在 ~12k step 有 loss 尖峰;
- **传播增益**:HC 复合映射 Amax gain 峰值 **~3000** → mHC **~1.6**(降三个数量级),单层因 Sinkhorn 20 轮近似而略有偏离 1;
- **开销**:n=4 大规模训练**增加 6.7% 时间**(含重算与通信重叠后的净值);
- **规模验证**:3B/9B/27B 算力扩展曲线优势保持(略衰减),3B 在 1T token 上的 token-scaling 曲线同向;
- **模型配置**(论文 Tab.5):n=4,层数 30(27B)/18(9B)/12(3B),MLA 且 KV rank=512、RoPE 维度 64、θ=10000,64/72 routed experts + 2 shared,active 6,`α` 初始化 0.01,`t_max=20`。

### 6.2 内核侧数据(昇腾 HC Prenorm GEMM,DeepGEMM-Ascend README)

同一条「残差合并 + 系数投影」在昇腾侧被单独做成 prenorm GEMM(M=token,N=24,K=n·H=28672),口径是**接近打满 HBM 写带宽**:

| M | N | K | Latency (µs) | TFLOPS | 带宽 (GB/s) |
|---|---|---|---|---|---|
| 13 | 24 | 28672 | 6.7 | 3 | 519 |
| 137 | 24 | 28672 | 9.6 | 20 | 1110 |
| 512 | 24 | 28672 | 15.4 | 46 | 2082 |
| 4096 | 24 | 28672 | 72.7 | 78 | 3276 |
| 8192 | 24 | 28672 | 136.7 | 82 | 3463 |

这组数字说明 mHC 的内核本质是**带宽题不是算力题**:M=8192 时 FLOPs 折算只有 ~82 TFLOPS,带宽却到 3.4 TB/s(接近 HBM 上限)。⚠ 口径提醒:小 M 行(13/137)的带宽若按裸 M 反推只有 ~111/818 GB/s,与表里 519/1110 差 ~5×——说明官方的带宽口径含 padding(按 BLOCK_M 对齐)或读写合计,具体未在 README 说明;要复现这组数字,得用同样的 `bench_msprof` 冷 L2 口径,别用公式反推。

> 口径警告:以上是昇腾 950 上的数字,NVIDIA 侧 `mega_mhc` README 未给独立 benchmark;论文的 6.7% 是整模型训练时间占比。三者不可互相换算。

---

## 7. 实践要点 / 坑

**用 DeepGEMM `mega_mhc` 时**

- 只在 **SM100**(`arch_major==10`)可用;host 侧直接断言,SM90 会挂。`n` 被硬编成 4(`DG_HOST_ASSERT(num_routes == kNumRoutes)`),想要 n=2/8 得改常量重编 JIT。
- `hidden` 必须满足 `hidden % (kDefaultNumSplits*BLOCK_K) == 0`,即 **7168/4096 这类 1024 的倍数**才安全(16×64=1024);`num_tokens ≤ 2²⁰`。
- **shifted 与非 shifted 是两套语义**:`shifted_prev_mix` 与 `new_prev_mix` 必须同时给或不给;只有 shifted 版存 `new_prev_mix`,normal 版(repr 里 pre 系数不写回)省这笔写。用错会出现「上一层的 pre 系数没更新」这类静默错。
- **FP8 输出三条路互斥**:要么 `y_gemm_sf` 独给(TMA 对齐、列主序),要么 `y_routed_sf` + `y_shared_sf` 同给(行主序 / Mega MoE 共享专家布局);`y_bf16`/`y_fp8` 至少一个。`y_shared_sf` 还要求 `shared_sf_block_m > 0` 且底层 storage 足够(按 `ceil(T/block_m)*align(block_m,128)` 行算)。
- **split 数不是常数**:想确认实际分片,开 `DG_PRINT_CONFIGS=1`,启动时会打印 `T/H/splits/stages/threads/smem`。要可复现就 `deep_gemm.use_deterministic_algorithms(True)`(固定 16 splits,单测验证两次调用 bitwise 相同)。
- **CUDA graph**:① 必须先在一个 stream 上 warm up 再 capture(屏障缓冲首次 zero 不能进图);② 屏障按 stream 缓存,同一 stream 串行、不同 stream 并行,别跨 stream 复用同一 graph;③ 升到含 #478 的版本,否则 replay 有读到旧 partial 的风险——这个 bug 不会稳定复现,只会偶发。
- **别把 mHC 当成「加了个归一化」**:收益来自 n 条流的拓扑 + 流间混合,$\mathcal{H}^{\mathrm{res}}$ 才是主力(消融 −0.022/−0.027);而稳定性来自双随机约束的**乘法封闭性**,不是单层归一化能替代的。

**自己实现/复现时**

- Sinkhorn 是**近似**投影:迭代轮数直接决定单层增益偏离 1 的程度;论文用 20,少到 10 会放大后续层的累计偏差。反向要么重算整条迭代,要么存全部中间量(显存换时间)。
- 系数生成用 TF32 GEMM($\varphi$ 是 fp32),不要图省事降到 bf16:系数是 $O(1)$ 量级的 logit,精度损失会被 Sinkhorn 的指数放大。
- 显存按 $L_r^*\approx\sqrt{nL/(n+2)}$ 分块重算,并且**让重算块对齐 pipeline stage 边界**,否则重算会跨 stage 拉长关键路径。
- 流水并行下 $\mathcal{F}_{\mathrm{post,res}}$ 要放**独立高优先级 stream**:它跟通信抢带宽,放默认流会把通信 stream 顶住;attention 侧则相反,不要用 persistent kernel,留出被抢占的余地。

---

## 溯源

- **mHC 论文(主源:机制/公式/Sinkhorn/t_max/重算 L_r*/DualPipe/基准表/模型配置)**:Xie, Wei, Cao, Zhao, … Liang (DeepSeek-AI), *mHC: Manifold-Constrained Hyper-Connections*, arXiv:2512.24880v2(2026-01-05 修订)— https://arxiv.org/abs/2512.24880
- **HC 前作(OLMoE-1B-7B DHC×4 收敛 1.8×、ARC +6、消融基线)**:Zhu et al., *Hyper-Connections*, arXiv:2409.19606v3 — https://arxiv.org/abs/2409.19606
- **DeepGEMM 源码(接口/常量/6-WG 角色/split 启发式/屏障与 epoch,本轮逐文件核对)**:
  - `csrc/apis/mega_mhc.hpp`(host 断言与 scratch 布局)— https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/apis/mega_mhc.hpp
  - `deep_gemm/include/deep_gemm/layout/mega_mhc.cuh`(几何常量、epoch 偏移)— https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/layout/mega_mhc.cuh
  - `deep_gemm/include/deep_gemm/scheduler/mega_mhc.cuh`(Epoch / SplitBarrier)— https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/scheduler/mega_mhc.cuh
  - `deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh`(kernel 主体)— https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh
  - `deep_gemm/include/deep_gemm/epilogue/sm100_mega_mhc.cuh`(Mix 的 warp 内 Sinkhorn / Norm)— https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/epilogue/sm100_mega_mhc.cuh
  - `tests/test_mega_mhc.py`(确定性、CUDA graph 契约、FP8-only、多 stream 隔离)— https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_mega_mhc.py
- **本轮变更**:`Fix mega_mhc split barriers under CUDA graph replay (#478)`,commit `5081f8513`(2026-10-20),改动 3 文件 +48/−8 — https://github.com/deepseek-ai/DeepGEMM/commit/5081f8513
- **昇腾侧**:DeepGEMM-Ascend README「HC Prenorm GEMM」带宽表(950,`tilelang` 作为依赖)— https://github.com/deepseek-ai/DeepGEMM-Ascend
- **TileKernels**(mHC 的 TileLang 版本:Sinkhorn 归一化 + mix 拆分/应用)— https://github.com/deepseek-ai/TileKernels
- **PDL 语义参考**(`cudaGridDependencySynchronize`,epoch 稳定区间的前提)— https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- **库内交叉**:[deepgemm-nvidia-vs-ascend.md](deepgemm-nvidia-vs-ascend.md)(DeepGEMM 算子族全景)、[tilekernels.md](tilekernels.md)(mHC 的 TileLang 实现位置)、[../../model-architecture/concepts/deepseek-arch-v3.md](../../model-architecture/concepts/deepseek-arch-v3.md)(mHC 实验所用的 V3 类 MoE 底座)、[../../model-architecture/concepts/llm-architecture.md](../../model-architecture/concepts/llm-architecture.md)(pre-norm 残差为何是十年标准)
