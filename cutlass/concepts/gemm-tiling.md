---
title: CUTLASS 分层 GEMM 概念篇:从硬件并发到 tiling 分解
type: concept
tags: [cutlass, gemm, tiling, tensor-core, sgemm, pipeline, cute, stream-k]
source:
  - "web: NVIDIA Efficient GEMM in CUDA | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html"
  - "web: CuTe Layouts | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html"
  - "web: CuTe Layout Algebra | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html"
  - "pdf: Stream-K: Work-centric Parallel Decomposition | arXiv:2301.03598"
  - "web: CUTLASS Tutorial: Efficient GEMM Kernel Designs with Pipelining | https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/ (Colfax Research, 2024-09)"
date: 2026-09
---

# CUTLASS 分层 GEMM:把一次矩阵乘拆给 GPU 的每一层硬件

## TL;DR

GPU 上做一次矩阵乘 `C(M×N) = A(M×K) · B(K×N)`,不是把整块矩阵丢给一个"魔法函数"。CUTLASS 的核心是把这次计算**沿硬件并发层级层层分块(tiling)**,每一层 tile 对应一种并发粒度(threadblock / warp / Tensor Core),同时靠 shared memory 与 registers 两层局部性喂饱计算单元。

**判断标准(记住这个就能读懂 CUTLASS 任意 kernel)**:读一段 CUTLASS 代码,先问三层——
1. **哪个 threadblock 负责输出矩阵的哪一块**(CTA tile,决定并行度/占用率)?
2. **CTA 内部多少个 warp 怎么分**(Warp tile,决定 smem→register 数据分布)?
3. **每个 warp 里 Tensor Core 一次算多大的 MMA tile**(决定 register 数据布局与指令吞吐)?

再问两个**数据在哪**:取值时数据从 global memory 流经 shared memory、registers 到 Tensor Core 的哪一层,谁负责搬(主循环 mainloop)、谁负责写回(尾循环 epilogue)。

---

## 1. 为什么不能"一次算完":硬件并发层级 vs 内存局部性两层

先看硬件为什么要 tiling:

| 硬件层 | 并发粒度 | 关注点 |
|---|---|---|
| **Threadblock (CTA)** | GPU 上并行运行的 threadblock 数 | 并行度 / 占用率 / 负载均衡 |
| **Warp** (32 线程) | CTA 内 warp 的排布 | smem 与 register 的分摊 |
| **Tensor Core** (MMA 指令) | 单条 `mma`/`wgmma` 指令算的 tile | 数据在 register 的排布 |

同时,GPU 存储有两层**片上小但快**的局部性结构,计算必须"贴着"它们走:

| 存储 | 位置 | 速度量级(相对) | 容量量级 |
|---|---|---|---|
| registers | SM 内 | 最快(几个 cycle) | ~256KB/SM |
| shared memory (smem) | SM 内 | 快(几十 cycle,可宽带) | ~100-228KB/SM,Hopper 可更大 |
| global memory (HBM) | 片外 | 慢(数百 cycle,带宽瓶颈) | 数十 GB |

**核心张力(Tensor Core 太快、HBM 喂不饱)**:这一代 Tensor Core 算力远超 HBM 带宽。例如 H200 SXM:Tensor Core 峰值 ~989 TFLOPS(FP16 dense)/ ~3,958 TFLOPS(FP8 sparse),而 HBM 带宽仅 ~4.8 TB/s。若不 tiling、把 C/M/N 全放 HBM、算一块读一块,计算单元绝大部分时间在等数据——利用率可能只有个位数百分比。

**tiling 的意义**就是:把输出切成小块,让**计算密集的多次循环发生在片上(smem/register 内)**,把昂贵的 HBM 访问次数压到最低,把数据流动组织成可重叠(mainloop 流水)的形态。

---

## 2. 核心机制:三层 tiling 的嵌套循环

> 下图把原文那段循环嵌套改写成流程图:外层 CTA、中层 warp、内层 MMA 各决定什么,以及 `cta_k` 为什么是流水线的划分单位。


<div class="archify-figure">
<iframe src="../../../diagrams/cl-three-level-tiling.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="三层 tiling 的嵌套循环"></iframe>
</div>


CUTLASS 把"一次 GEMM"映射成下面的循环嵌套(Efficient GEMM in CUDA 官方结构)。**逐层看注释,这就是整个引擎的灵魂**:

```c
// ===== 外层:threadblock 级并发 =====
for (cta_n = 0; cta_n < GemmN; cta_n += CtaTileN)        // 每个 threadblock_y 负责输出一块
  for (cta_m = 0; cta_m < GemmM; cta_m += CtaTileM)      // 每个 threadblock_x
    for (cta_k = 0; cta_k < GemmK; cta_k += CtaTileK) {  // 🟢 "GEMM mainloop"
      //      —— 不展开；一次迭代 = 一个 "stage"（流水线一级）
      //      —— C 的累计在这里面做：每个 stage 累加一部分 M·N

      // ===== 中层:warp 级并发 =====
      for (warp_n = 0; warp_n < CtaTileN; warp_n += WarpTileN)
        for (warp_m = 0; warp_m < CtaTileM; warp_m += WarpTileM)
          for (warp_k = 0; warp_k < CtaTileK; warp_k += WarpTileK) {
            //      —— 完全展开；一次迭代 = 一个 "k Group"

            // ===== 内层:指令级(MMA/Tensor Core)并发 =====
            for (mma_k = 0; mma_k < WarpTileK; mma_k += MmaK)
              for (mma_n = 0; mma_n < WarpTileN; mma_n += MmaN)
                for (mma_m = 0; mma_m < WarpTileM; mma_m += MmaM)
                  mma_instruction(d, a, b, c);   // Tensor Core 算一次性矩阵乘累加
          }
    }
```

**读懂这张图的三个要点**:

1. **三层对应三种并发粒度**(threadblock / warp / MMA 指令),每层 tile 尺寸都是可调模板参数(`CtaTileM/N/K`、`WarpTileM/N/K`、`MmaM/N/K`)。
2. **mainloop(`cta_k` 循环)是分段式(sectioned)的**,每一段叫一个 **stage**,是软件流水线的划分单位。mainloop 只做一件事:取下一块 A/B 的 tile 到片上,用当前块做 MMA 累加。
3. **两层局部性**对应:全局→smem(tile 落在 CTA 级)、smem→register(数据分给 warp)。两层搬移在不同并发层穿插进行,正是 pipelining 要重叠的东西。

### 2.1 典型 tile 尺寸量级(给个"数感")

配置到具体硬件时常见量级(FP16 GEMM on Ampere/Hopper):

- CTA tile:`128 × 128 × 32`(M×N×K)——主流;更大 `256×128` 并行度紧,更小则 HBM 搬运多
- Warp tile:`64 × 64 × 32`(2~4 warp 分 CTA 的 M、N);注意实践中多数配置 **WarpTileK = CtaTileK**,warp 沿 M/N 切分、不切 K($warp\_k$ 循环退化为一次迭代)
- MMA instruction tile:FP16 下 `16 × 8 × 16`(Ampere m16n8k16)/ `64 × 64 × 16`(Hopper wgmma 更大)
- mainloop 流水级数(stages):常 3~6 级,与 smem 容量、占用率相互制约

这些不是定死的——具体值由 kernel generator / 手工调优按 GPU 与问题形状决定,但"CTA~128²、MMA~16×8×16"的量级能帮你快速定位一段 kernel 在干嘛。

---

## 3. 数据流:mainloop(搬+算)与 epilogue(写回)

> 数据在四个存储层之间的走法:mainloop 只做「搬 + 累加」,K 走完才进 epilogue。


<div class="archify-figure">
<iframe src="../../../diagrams/cl-mainloop-epilogue.dataflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="mainloop 与 epilogue 的数据流"></iframe>
</div>


一次 CUTLASS GEMM 的计算体量分两段:

```
global memory(HBM)
   │  A,B 的 tile 被搬到 smem        ←── mainloop 的"取数据"前半
   ▼
shared memory (smem)
   │  每个 warp 从 smem 读自己的子块到 register
   ▼
registers
   │  Tensor Core 执行 mma:reg 里累加 C 的部分和
   ▼
(循环 k 方向) 累加完整个 K 维度后
   ▼
epilogue(尾循环):
   registers → 处理(如 bias/ReLU/scale/转置) → 写回 global memory 的 C
```

- **mainloop**:负责"喂 Tensor Core",把 A/B 的 tile 从 HBM 经 smem 搬进 register,并做 mma 累加。这一段的性能决定 kernel 快不快。
- **epilogue**:负责写回输出,顺带做 element-wise 后处理(activation、bias、scale、做转置/行-列主序适配)。这段通常不是瓶颈,但写 C 时要注意读写合并(避免按 tile 边界的非连续写)。

---

## 4. mainloop 的优化核心:pipelining(让搬和算重叠)

有了上面的结构,最大优化空间在 **mainloop 里让"搬运下一块"和"计算当前块"重叠**——这就是 CUTLASS 的 pipelining。两种主流策略:

### 4.0 先分清搬运指令的代际(Ampere vs Hopper)

mainloop 里"从 global 搬数据"这条腿,不同代 GPU 用不同指令,编程模型差异显著:

| 代际 | 搬运指令 | 要点 |
|---|---|---|
| Ampere (A100) / Ada (L40) | `cp.async` | 单元素/小块异步拷贝,可带 cache-bypass hint(L1 直通),per-thread 粒度;sm_80 引入,sm_86/89(Ada)仍支持 |
| Hopper (H100) | **TMA** `cp.async.bulk.tensor` | 硬件 Tensor Memory Accelerator,整块 bulk 拷贝,带 tensor descriptor(自动处理 strided/多维 tile 的形状),一个指令就把整个 tile 搬进 smem |
| Blackwell (B100/B200) | **TMA + UMMA** `tcgen05` | SM100 改用 tcgen05.mma(UMMA):单线程发射、操作数直接放 TMEM、2-SM MMA 用 cluster 配对;wgmma 为 Hopper-only |
| Volta/Turing (V100/T4) 及更早 | 普通 `cp.global→shared` | 无异步拷贝指令,需显式 barrier,难隐藏延迟 |

> 纠错:`cp.async` 自 Ampere 引入,Ada 属 Ampere 阵营;**没有异步拷贝指令、需显式 barrier** 的一代是 Volta/Turing 及更早。选型时别把 Ada 当"旧代"。

TMA 相对 `cp.async` 的核心优势是**bulk 整块搬 + 硬件协调 + 不占满所有线程**:一个线程即可发起整块拷贝,配合 named barrier 做到达同步,腾出资源给计算。这直接催生了下面的 warp 特化。

### 4.1 多级流水(Multistage,Ampere/Hopper 通用)

> 时序图:`buf0`/`buf1` 交替,一个 stage 的搬运与另一个 stage 的 mma 在同一时间窗里并行——这就是 pipelining 的全部内容。


<div class="archify-figure">
<iframe src="../../../diagrams/cl-multistage-pipeline.waterfall.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="多级流水:搬与算重叠"></iframe>
</div>


给 smem 里的 A/B 开 **N 份 buffer(级数 = stages)**,异步拷贝(`cp.async` 在 Ampere,TMA 在 Hopper)把下一 stage 的数据搬进来,同时 Tensor Core 算当前 stage。

```
stage 1:  [copy A1/B1 → smem buf0]              // 延迟隐藏:等数据
stage 2:  [copy A2/B2 → smem buf1]  [mma B1]    // 算第1块同时搬第2块
stage 3:  [copy A3/B3 → smem buf0]  [mma B2]    // buf0 已被算完,可复用
...
```

**缓冲份数(级数)不是越大越好**:
- 太少 → 搬算无法重叠,等待暴露;
- 太多 → smem 被缓冲区占满,占用率(occupancy)下降,能跑的 threadblock 变少。
- 经验上 3~6 级常见,需针对目标 GPU 的 smem 容量和 tile 大小调。

### 4.2 Warp 特化(Warp Specialization,Hopper)

> 角色分工图:producer 与 consumer 各自循环,barrier 只负责「数据就绪」与「buffer 可复用」两个方向的通知。


<div class="archify-figure">
<iframe src="../../../diagrams/cl-warp-specialization.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="Warp 特化:producer / consumer"></iframe>
</div>


把 warp 分成**两拨角色**:producer warp(专职搬数据:global→smem,用 TMA `cp.async.bulk.tensor` 或 `cp.async`)和 consumer warp(专职算:smem→register,发射 `wgmma`/`mma` 到 Tensor Core)。生产与消费**并发运行**,各自循环,靠 barrier 同步。

> 纠错:别把 `wgmma` 当搬运指令——`wgmma`(warp-group MMA)是 Hopper 的**计算**指令,由 consumer 发射;producer 负责的是 **TMA 异步批量搬数据**(`cp.async.bulk.tensor`),二者分工正是 warp 特化的核心。wgmma 指令本身的机制(操作数位置约束、SMEM 布局、descriptor、同步)详见 [hopper-wgmma.md](hopper-wgmma.md)。

### 4.3 效果参考

Colfax 的教程用 Hopper 手写 pipelined GEMM,单靠这一招就把 **FP16 GEMM 推到 ~65% 利用率**。注意:这已经是不错的结果,但离 Tensor Core 理论峰值仍远——剩余差距来自负载均衡、tile 形状、TMA 效率等(见下节 Stream-K)。

### 4.4 正确性:同步原语

pipelining 让"谁先谁后"变复杂,正确性靠两种同步保证:
- **数据依赖**:一块 smem buffer 必须被本轮 mma 读完,才允许下一轮 copy 覆盖它;
- **barrier / fence / named barrier**:CUTLASS 的异步流水类封装了这些,如 `PipelineTmaAsync` / `PipelineAsync` / `PipelineTmaUmmaAsync`(头文件 `cutlass/pipeline/sm90_pipeline.hpp`)。手写时最容易漏、最容易出 race 的就是这里。

---

## 5. 更深的负载均衡:Stream-K(work-centric 分块)

> 两条分派路径对比:按 tile 分会让尾批早退;按迭代数分则人人均等,但 K 被切开后必须 fixup。


<div class="archify-figure">
<iframe src="../../../diagrams/cl-streamk-vs-tile.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="tile-based vs Stream-K"></iframe>
</div>


传统 tile-based 分解按**输出 tile** 分派工作:每个 threadblock 负责一块完整的 C tile。问题在**边缘不齐**(tail effect):当输出 tile 总数 `⌈M/TileM⌉×⌈N/TileN⌉` 不能被 SM 数整除时,最后一批 threadblock 只干一点点活就退场,GPU 空闲。

**Stream-K(arXiv:2301.03598)** 换了个思路——**不按输出 tile 分,而是按"内层迭代数"均分给所有 processor**(work-centric):

- 把 K 维度的累计工作当成一条"工作流",切成均匀的若干份,分给固定数量的 threadblock;
- 每个 threadblock 可能做**多个 K 分片**(如果分给它的工作 > 一个 tile 的 K),也可以**只做一个 tile 的一部分 K**(当工作 < 一个 tile)。
- 关键:所有 threadblock 几乎同时「下班」,不再有尾部空闲。

**结果(官方论文,32,824 种 GEMM 几何)**:相对 CUTLASS 峰值提速最高 **14×**、相对 cuBLAS **6.7×**,且**平均性能更高、更稳定**——尤其对长而扁(非方形)的 GEMM。且单靠一个 tile 尺寸配置就能覆盖宽范围问题形状,省掉今天 cuBLAS/CUTLASS 那套"大量 kernel 变体 + kernel-selection 启发式"的调优负担。

**一句话**:tile-based 追求"每块算得高效",Stream-K 追求"每个核都别闲着"——后者在大范围非规整几何上更稳。

**⚠ 但 Stream-K 有代价(fixup,上之前先看)**:
- 一个输出 tile 的 K 维度被拆给**多个 CTA** 分头算 → 需要**跨 CTA 归约(fixup)**;**SM90 默认在同一 kernel 内**由持有最后 K-split 的 CTA 用 barrier 轮询合并 partial(不是独立归约 kernel),临时 workspace 是额外开销;
- **默认 `ReductionMode::Deterministic`(确定性归约)**,不是"天然不可复现";想要更高吞吐才选 `Nondeterministic`(放弃位级确定性);
- 仍要评估:fixup 同步与 workspace 占用对性能/实现的额外负担,尤其精度敏感场景。
- 在 CUTLASS 里调度器是**模板参数**(`cutlass::gemm::StreamKScheduler`),经 `GemmUniversal` 注入——要显式选,不是默认行为。

> tail effect:**输出 tile 总数 `⌈M/TileM⌉×⌈N/TileN⌉` 不能被 SM 数整除**时才有尾部空闲(而不是按"网格/H·W"判);H·W 是卷积记法,GEMM 用 M/N。

---

## 6. CuTe:把 tiling 抽象成"Layout 代数"

> 抽象层次图:Layout 是什么、它与 MMA Atom / TV layout 的关系,以及 divide 族在切分里扮演的角色。


<div class="archify-figure">
<iframe src="../../../diagrams/cl-layout-algebra.tree.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="CuTe Layout 代数"></iframe>
</div>


3.x 的架构级重构把上述 tiling / 数据分布**形式化**成 CuTe 的一套核心抽象,核心是 `Layout`:

> **一个 `Layout` 是从坐标空间到索引空间(线性内存地址)的映射**——本质是"怎么把多维数组摊平到一维地址"。

```cpp
// 行主序 4×8(行步长 8,列步长 1):(m,n) → m*8 + n
Layout rowMajor = Layout<Shape<_4,_8>, Stride<_8,_1>>{};
// 列主序 4×8(行步长 1,列步长 4):(m,n) → m*1 + n*4
Layout colMajor = Layout<Shape<_4,_8>, Stride<_1,_4>>{};
```

**为什么抽象成 Layout 很关键**:有了它,同一个算法代码可以**不感知具体内存排布**地写——换行/列主序、换 tile 形状、甚至跨线程/跨 warp 拆分数据,都只是换一个 Layout 组合,而算法主体不变。CuTe 提供一套 **Layout 代数**来组合/切分:`composition`(函数复合)、`product`(按一种 layout 复制另一种)、`divide`(按一种 layout 切分另一种)。切分是核心——**把数据 layout 按线程/线程组 layout 划分**,真实运算名例如:

- `logical_divide(layout, divisor_layout)`:按逻辑模式切分一个 layout
- `zipped_divide(layout, tile)` / `tiled_divide(layout, tile)`:按 tile 把 layout 切成块
- `partitioned_divide(layout, partition)` 等划分系(部分 CUTLASS 版本/分支提供,与 `tiled_divide` 族相关):按 partition 划分,得到"每个线程分到哪块"的映射
- `coalesce(layout)`:化简(把相邻可合并模式合并,得到更少 mode 的等价 layout)
- `flatten` / `repeat` / `composition` 等基础工具

这些运算把"大 Layout"切成"分给 thread/warp/Tensor Core 的小 Layout",是 CUTLASS 抽象数据分布的核心机制。**注意别记错名**:不存在 `left_padding`/`partitioned_into_TV` 这类函数;`TV layout`(thread-value)是真实概念——切分函数以 `divide` 系列(`logical_divide` / `tiled_divide` / `zipped_divide`)为主。

**现代 CUTLASS(3.x/4.x)= "Layout 代数描述数据 + 模板参数描述 tiling + Pipeline 描述流水 + (4.x)CuTe DSL Python 生成 kernel"** 的组合产物。

---

## 7. 实践要点(写/调 CUTLASS 或读别人 kernel 时的 checklist)

1. **从读循环嵌套开始**:拿到一段 kernel,先还原 `cta_tile / warp_tile / mma_tile` 三层,再定位 mainloop 与 epilogue 分界。
2. **判断 tile 尺寸是否合理**:
   - `CtaTileM/N` 选太大 → 并行度不足、占用率低;选太小 → HBM 搬运次数多。
   - `CtaTileK` 决定单 stage 数据量 → 直接影响需要的流水级数与 smem 占用。
3. **带宽瓶颈优先找 copy,算力瓶颈优先找 mma**:性能上不去,先分清是"搬不动"(HBM/smem 带宽)还是"算不满"(Tensor Core 利用率),对症下药。
4. **pipelining 级数要跟 smem 容量、occupancy 一起调**,三者互相拉扯。
5. **异构/边缘形状**:非方阵、K 很短、输出不能被网格整除 → 优先考虑 Stream-K 分块而非盲目加 tile。
6. **正确性红线**:改 pipelining 同步(named barrier / fence)时,等价性最难把握;优先复用 CUTLASS 的异步流水类(`PipelineTmaAsync` 等)语义,别裸写同步。

---

## 8. 溯源

- 官方循环嵌套结构、mainloop/stage 定义、epilogue: NVIDIA Efficient GEMM in CUDA — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
- CuTe Layout 核心抽象与 Layout 代数: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html / 02_layout_algebra.html
- Stream-K work-centric 分块与 14×/6.7× 数据: arXiv:2301.03598 (Osama / Merrill / Cecka / Garland / Owens)
- Hopper pipelining(双缓冲、warp-spec、~65% FP16 利用率、H200 989 TFLOPS FP16 dense / 3,958 TFLOPS FP8 sparse / 4.8 TB/s 数字): Colfax Research CUTLASS Tutorial Part 2, https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/

*注:H200 SXM 的 Tensor Core 峰值 ~989 TFLOPS(FP16 dense)/ ~3,958 TFLOPS(FP8 sparse)/ HBM 4.8 TB/s 数值来自 Colfax 教程(2024-09)与 NVIDIA H200 datasheet(FP16/BF16 dense 989 / sparse 1979;FP8 dense 1979 / sparse 3958),用于说明"算力≫带宽"的张力量级,具体型号规格以 NVIDIA 官网最新为准。*
