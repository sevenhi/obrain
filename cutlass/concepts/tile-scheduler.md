---
title: CUTLASS TileScheduler:谁来决定线程块算哪块
type: concept
tags: [cutlass, tile-scheduler, load-balancing, stream-k, grouped, persistent-kernel, rasterization]
source:
  - "web: NVIDIA Grouped Kernel Schedulers | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html"
  - "web: NVIDIA Tech Blog — CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions | https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/ (2025-07-16)"
  - "pdf: Stream-K: Work-centric Parallel Decomposition | arXiv:2301.03598"
  - "code: cutlass sm90_tile_scheduler_stream_k.hpp / sm100_tile_scheduler_stream_k.hpp | github.com/NVIDIA/cutlass"
date: 2026-09
---

# CUTLASS TileScheduler:谁来决定"哪个 threadblock 算哪块"

## TL;DR

GEMM 的输出矩阵被切成很多 tile。**TileScheduler(调度器)决定每个 threadblock 去算哪块、按什么顺序算**,直接决定负载均衡和 GPU 利用率。

- 朴素做法:launch 的 threadblock 数 = tile 数,每个算一块 → **tile-based**(边缘不齐会尾部空闲)
- **Stream-K** scheduler:把 K 维工作均分给固定数目的 threadblock,无论输出几何多不规整都能近完美利用(但引入 fixup 代价)
- **Grouped** scheduler:一个 kernel launch 服务**多个 problem**(多个 GEMM/SYR2K),persistent threadblock 循环领活
- 现代 CUTLASS(3.x+)把调度器当作 Kernel 层的**可组合组件**,是负载均衡策略的注入点

**核心判断**:调度器选型 = 在「每块算得高效(tile-based)」和「每个核都别闲着(Stream-K)」之间,加上「要不要一个 launch 跑多个 problem(grouped)」来权衡。

---

## 1. 问题背景:输出空间如何分给硬件

一次 GEMM 输出 `C(M×N)` 被切成 `⌈M/TileM⌉×⌈N/TileN⌉` 个 tile。GPU 上有多少 **SM**(计算单元),就得决定:每个 SM 上跑哪些 threadblock、算哪些 tile。

**朴素 tile-based 分解**:
- launch `T = tile 总数` 个 threadblock,每个算一块;
- 当 `T` 不能被 SM 数整除,或某些 tile 的 K 维度短 → 最后一批 threadblock 干很少的活就退,GPU 空闲(tail effect);
- 非规整几何(长扁矩阵、大 K 起伏)下利用率差。

调度器就是来解决「如何分配 tile 给 threadblock」,让空闲最小化。

---

## 2. 从 3.x 架构看调度器的位置

CUTLASS 3.x(NVIDIA 技术博客,2025-07-16)把 GEMM 实现组织成**五层概念层级**,调度器位于 Kernel 层:

| 层 | 抽象 | 职责 |
|---|---|---|
| **Atom** | `cute::Mma_Atom<>` / `Copy_Atom<>` | 架构特定指令(mma/copy)|
| **Tiled MMA/Copy** | `cute::TiledMma<>` / `TiledCopy<>` | 空间微内核:atoms 的平铺/交织 |
| **Collective** | `CollectiveMma` / `CollectiveEpilogue` | **时间**微内核:用架构同步编排一群原子 |
| **Kernel** | tile scheduler + collective 组装 | **调度器注入点**:定义时间的组织方式 |
| **Device** | 最外层调用 | 启动 kernel |

(概括自 NVIDIA 3.x 博客)**Kernel 层也是指定 tile scheduler 的组装点——正如 kernel schedule 定义工作的时间组织,调度器决定线程块算哪块**。

这解释了调度器在现代 CUTLASS 里是**可插拔组件**:换一种负载均衡策略 = 换一个 scheduler,而非重写 kernel。

---

## 3. 调度策略全景

> ⚠ **先纠正一个常见误解**:在 CUTLASS 3.x(Hopper/Blackwell)语境下,**\"每 tile 一 CTA 的朴素 tile-based\"不再是默认**。Hopper+ 的默认调度器是 **persistent tile scheduler**(`PersistentTileSchedulerSm90`,一 SM 一 CTA、循环领 tile)。所谓 "tile-based" 只是最朴素的档位。

### 3.0 先分清三类调度器(3.x Hopper 语境)

| 调度器 | 代码(类/文件) | 机制 |
|---|---|---|
| **Persistent**(默认) | `PersistentTileSchedulerSm90` / `sm90_tile_scheduler.hpp` | launch 固定数 threadblock(=SM 数或倍数),每个**持久驻留**,循环从工作队列领 tile |
| **Stream-K** | `PersistentTileSchedulerSm90StreamK` / `sm90_tile_scheduler_stream_k.hpp` | 均分 K 维内层迭代,跨 CTA 归约(fixup) |
| **Grouped** | `sm90_tile_scheduler_group.hpp` | 一个 launch 服务多个 problem(persistent 循环领活) |
| 朴素 tile-based | (device/threadblock 层最朴素的每块一 CTA) | 已非主流,仅最简档 |

调度器的**两大控制维**:
1. **persistence**:线程块是否驻留循环(还是每 tile 一 launch)——影响 launch 开销与占用;
2. **rasterization order(遍历顺序)**:threadblock 领 tile 的**顺序**,直接影响 **L2 cache 局部性**(见 §3.4)。

### 3.1 Persistent(3.x 默认):一 SM 一 CTA 循环领 tile

- launch 固定数 threadblock(≈SM 数),每个**驻留**,从工作队列循环领下一个 tile;
- 省掉每 tile 一次 kernel launch 的开销,占用率高;
- 配套 **`RasterOrder`**(AlongM / AlongN)+ **`max_swizzle`** 控制领 tile 顺序(见 §3.4)。

### 3.2 Stream-K(work-centric,均分工作)

- **不按输出 tile 分,按"内层(K 方向)迭代总数"均分给固定数目的 threadblock**;
- 每个 threadblock 可能做多个 tile 的多段 K,或只做一个 tile 的一部分 K;
- 所有 threadblock **几乎同时下班**,无尾部空闲;
- **代价**:一个输出 tile 的 K 被拆给多个 CTA → 需要**跨 CTA 归约(fixup)**。**SM90 默认在同一 kernel 内**由持有该 tile 最后一个 K-split 的 CTA 用 barrier(`BarrierManager::wait_eq/wait_lt`)轮询合并 workspace 里的 partial,再直接做 epilogue(不是独立归约 kernel;"separate reduction unit"只是可选模式)。
- **与 2.x split-K 的本质区别(容易混)**:split-K 是 **tile-centric**(先切输出 tile、再拆每个 tile 的 K);Stream-K 是 **work-centric**(把所有 K 迭代打平成一维工作量,再均分给 CTA)——后者在任何几何下都能均匀分配,前者在 tile 边缘仍会不齐。
- **确定性**:`ReductionMode::Deterministic` 是**默认值**(源码 `sm90_tile_scheduler_stream_k.hpp` Arguments 默认)。需要更高吞吐时可选 `Nondeterministic`(放弃位级可复现换性能)。所以"Stream-K 一定不可复现"是误读——默认就是确定的。
- 论文数据(arXiv 报告的多几何评测,研究实现、特定代际硬件上的峰值倍数):峰值最高 14×(vs CUTLASS)/ 6.7×(vs cuBLAS),单 tile 配置覆盖宽范围几何。
- 代码: `cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp`(调度器 `PersistentTileSchedulerSm90StreamK`)。

### 3.3 Grouped(一个 launch 服务多个 problem)

**Grouped kernel = 单个 CUDA launch 里跑多个 problem(多个 GEMM/SYR2K)**:

- 通常 launch 的 threadblock 数 **小于** group 内所有 problem 的 tile 总数;
- 每个 threadblock **persistent** 地循环:查 scheduler 领下一个 tile 来算 → `next_tile()` / `advance(gridDim.x)`;
- 目的:省掉多次 kernel launch 的开销、提高小 problem 场景的利用率(如 batch 里很多小 GEMM)。

```cpp
// 伪代码示意(actual API 方法名见 sm90_tile_scheduler_group.hpp)
ProblemVisitor pv;              // grouped kernel scheduler (code 里也叫 problem visitor)
while (pv.next_tile()) {        // 领下一个 tile
    // 算这个 tile 的 MMA + epilogue
    pv.advance(gridDim.x);      // 通知 scheduler 当前 tile 算完
}
```

- **调度模式**:grouped GEMM 用 **round-robin** 把 group 内各 problem 的 tile 轮流分给 threadblock(Grouped Rank2K/SYR2K 用不同调度方案);
- **负载均衡优化**:先把 problem 按大小**排序(sorting)**,大 problem 先领、小 problem 收尾,减少尾部空闲。

### 3.4 Rasterization order:调度器里最该懂的一根旋钮

**rasterization order 决定 threadblock 沿哪个方向、按什么顺序领 tile**——它直接关系 **L2 cache 局部性**:

- **几何直觉**:相邻 threadblock 若处理**内存相邻**的 tile,它们读 A/B 的 tile 片会在 L2 里**互相命中 → 提高 L2 命中率、减少 DRAM 读**;反之若各读各的远隔片,则 L2 复用差。所以光栅方向不是细节,是访存效率的核心。
- CUTLASS 的 persistent scheduler 提供 **`RasterOrder`**(`AlongM` / `AlongN`)选光栅**主方向**,再用 **`max_swizzle_size`**(取值 1/2/4/8 的组大小)把 tile 索引分组,让同组 CTA 处理的 tile 复用**同一条 A(或 B)数据带**;
- **方向(AlongM/N)要和 operand 复用形状配合调**:沿哪个维度的数据带被复用,取决于 A/B 的 tile 如何被多个 CTA 共享——这是"**日常最常动**"的调度参数,比换 Stream-K 更基础。
- 配置位置:scheduler 的 `Arguments`(如 `PersistentTileSchedulerSm90StreamK::Arguments` 的 `raster_order` / `max_swizzle_size`,example 48 用 `arguments.scheduler.raster_order` / `arguments.scheduler.max_swizzle_size` 访问)。

> swizzle 直观理解:若 tile 按行顺序领,相邻 CTA 同时碰同一行相邻段 → L2 抖动;swizzle 让每个 CTA 领到的 tile 在地址上均匀错开,提高复用。**先调它,再考虑换调度器**。

### 3.5 演进:Blackwell 的 persistent cluster / Cluster Launch Control

Blackwell(SM100)的 **persistent cluster(Cluster Launch Control, CLC)** 机制与直觉相反——它并**不是**静态预留一部分 SM 驻留:像非 persistent kernel 一样 launch **全量 tile 数的 grid**,已启动的 Worker 用 `clusterlaunchcontrol.try_cancel` **"取消掉尚未启动的 block、领走其坐标"** 继续干活。解决的是 **SM 被其他 kernel 挤占时,静态 persistent 无法感知真实可用 SM 数** 的问题(green context / 共享 GPU / MIG 多租户场景)。

- **源码位置**:动态 CLC 版在 `include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp` 的 **`PersistentTileSchedulerSm100`**(内含 `CLCResponse` / `PipelineCLCFetchAsync`);静态版在 `sm100_static_tile_scheduler.hpp` 的 **`StaticPersistentTileScheduler100`**(注:`DynamicPersistentScheduler` / `StaticPersistentScheduler` 是 **tag 名**,不是类名;selector 经 tag 映射到上述类,别按 tag 名去找头文件);
- **机制**:`try_cancel` 是 **device-side 指令式抢占**——被取消的 block 不启动,已启动的 Worker 领走它的 tile 索引继续算(SM100 才有该指令能力);
- **两个 tag 的场景**:`DynamicPersistentScheduler` = 动态感知实际空闲 SM(适合与其他 kernel 共享、SM 数不确定的 green context);`StaticPersistentScheduler` = 静态按声明 SM 数驻留(独占 GPU 时更稳);
- 与 Hopper 静态 persistent 的本质区别:Hopper 假设"启动时的 SM 数",Blackwell CLC 动态适配"实际可用的 SM 数"。

> 一句话:CLC = 动态抢占式持久化——不静态预留,而是取消未启动块来适配真实空闲 SM,区别于 Hopper 的静态 persistent。

---

## 4. 选型决策树(把 "用哪种调度" 变成可判断的问题)

```
输出几何规整(方形、K 均匀、T 整除 SM数)?
├─ 是 → persistent(默认即可,无归约)✅ 推荐
└─ 否 → 需要高利用率?
       ├─ 能接受跨 CTA 归约 fixup 开销?→ Stream-K ✅ (默认即确定性归约)
       └─ 完全不想有 fixup / 要最大简单?→ 保持 persistent,靠 rasterization/tile 形状调优

多个小 problem 要一起跑(batch 小 GEMM)?
└─ 是 → Grouped 调度(round-robin + 按大小排序)✅
```

> ⚠ **Grouped 与 Stream-K 互斥**(Hopper 3.x,见 NVIDIA/cutlass issue #1941):**grouped GEMM 和 stream-K 不能同时用**;cooperative grouped kernel 还有 `static_assert` 限定只用默认调度器。别把两者组合,会编译/运行报错。

**什么时候别用 Stream-K**:
- 不愿承担 **barrier 轮询/额外 workspace/Memset 开销**时(尤其 CUDA Graph 场景,见 §5-2)——默认虽 Deterministic 可复现,但 fixup 引入的同步与清零可能有性能/链路代价:
- 该 GEMM 本身已是瓶颈且几何规整(收益小、纯加 fixup 开销)。

---

## 5. 实践要点(checklist)

1. **先量 tail effect 是否存在**:输出 tile 数能否被 SM 数整除?K 是否均匀?不规整才值得上 Stream-K。
2. **Stream-K fixup 默认在同一 kernel 内完成**:由持有最后 K-split 的 CTA 轮询 barrier 合并 partial(不是独立归约 kernel);workspace 临时存储别漏分配,默认 `ReductionMode::Deterministic` 已是确定性归约。
   - **⚠ 高价值坑(CUDA Graph 场景,issue #2075)**:Stream-K 的 barrier workspace **每次 kernel launch 前必须清零**(`gemm.initialize()` 逐次重调 + 一次 Memset)。CUDA Graph 下这会在图里加 Memset 节点、产生 kernel gap——LLM 推理普遍用 CUDA Graph,这是选型级影响。另注意 `KernelHardwareInfo` 要正确传 SM 数。
3. **Grouped 用前先排序 problem**:按 size 排序能显著改善尾部均衡,尤其 group 里 problem 大小差异大时。
4. **持久化 kernel 注意资源**:persistent threadblock 占住 SM 直到结束,适合 batch 均匀、单 problem 产出稳定的场景;problem 间开销不均会拖慢。
5. **看代码认 scheduler**:`sm90_tile_scheduler_*` / `sm100_tile_scheduler_*` 头文件命名即知架构与类型——`persistent`(默认)、`stream_k`、`group`、`static`(SM100)等;scheduler **tag**(`cutlass::gemm::StreamKScheduler` / `PersistentScheduler` / `GroupScheduler` / `DynamicPersistentScheduler` / `StaticPersistentScheduler`)经 `GemmUniversal` 的模板参数注入。
6. **rasterization 是日常旋钮**:默认 persistent 调度先调 `RasterOrder`(AlongM/AlongN)+ `max_swizzle`(L2 局部性),再考虑换 Stream-K。

**怎么换调度器(最小可复现)**:调度器是 `GemmUniversal` 的模板参数,不是运行时配置:

```cpp
// 换调度器 = 换这个模板参数(Heuristic 还会按 shape 自动回退到合适分解)
using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape_MNK, Mainloop, Epilogue,
    cutlass::gemm::StreamKScheduler   /* 或 PersistentScheduler / GroupScheduler */>;
// DecompositionMode:Heuristic(自动按几何选,Stream-K 不总是最优)/ DataParallel / SplitK / StreamK
```

相关示例:`47_ampere_gemm_universal_streamk` / `49_hopper_gemm_with_collective_builder`(用 `StreamKScheduler` 构造)/ `57_hopper_grouped_gemm` / `74_blackwell_gemm_streamk`。

---

## 6. 溯源

- Grouped kernel scheduler 机制(`next_tile()` / round-robin / problem sorting): https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html
- CUTLASS 3.x 五层抽象与调度器位于 Kernel 层: NVIDIA Tech Blog, https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/ (2025-07)
- Stream-K 论文(work-centric、14×/6.7×): arXiv:2301.03598
- Stream-K 实现与默认 deterministic 归约: `cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp` / `sm100_tile_scheduler_stream_k.hpp`;默认 persistent 在 `sm90_tile_scheduler.hpp`
- Grouped/Stream-K 互斥: NVIDIA/cutlass issue #1941
- Blackwell persistent clusters / Cluster Launch Control: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html(版本说明:本篇官方文档链接统一用 latest 跟随最新版,现为 v4.8.0;需查历史版本时把 URL 中 `latest` 换成如 `4.7.0`)
