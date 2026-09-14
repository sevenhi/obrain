---
title: CuTe 深度:Layout、Atom 与 TiledMMA 的抽象世界
type: concept
tags: [cutlass, cute, layout, tensor, atom, tiledmma, thread-value, mma]
source:
  - "web: NVIDIA CuTe's support for MMA instructions | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html"
  - "web: CuTe Layouts | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html"
  - "web: CuTe Layout Algebra | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html"
  - "web: Lei Mao's Log Book — CuTe Thread-Value Layout | https://leimao.github.io/blog/CuTe-Thread-Value-Layout/ (2025-10)"
  - "web: Lei Mao's Log Book — CuTe Tiled MMA | https://leimao.github.io/blog/CuTe-Tiled-MMA/"
date: 2026-09
---

# CuTe 深度:Layout、Atom 与 TiledMMA——CUTLASS 3.x 的抽象底座

## TL;DR

CuTe(CUTLASS 3.x 的核心库)用三个递进抽象把"硬件上怎么铺数据、怎么算"**描述**出来,让高层算法代码与具体硬件解耦:

1. **Layout**:坐标→索引的映射(= 内存里怎么排),是"形状"与"步长"的形式化;
2. **Atom**:单个硬件指令(如 `mma`) + 它的**元信息**(TV layout),把"这条指令各线程取哪些数据"固化下来;
3. **TiledMMA / TiledCopy**:把多个 Atom 平铺、交织成更大的**微内核**,再往上组装成 Collective / Kernel。

**核心判断**:真正读懂 CUTLASS 3.x+,不是看模板参数,而是看懂 **Layout(怎么排)+ Atom(单条指令怎么取)+ 数据分布的 TV layout(每个线程拿哪些值)** 三者如何叠加。CuTe 的优雅在于:换硬件,只换 Atom 和它的 Traits,上层 Layout 组合代码基本不变。

---

## 1. Layout:一切抽象的地基

> **一个 `Layout` 是从坐标空间到线性内存索引的映射**,本质是"怎么把多维数组摊平成一维地址"。

```cpp
// Shape = 各维大小,Stride = 各维步长
Layout rowMajor = Layout<Shape<_4,_8>, Stride<_8,_1>>{};   // (m,n) → m*8 + n 行主序
Layout colMajor = Layout<Shape<_4,_8>, Stride<_1,_4>>{};   // (m,n) → m + n*4  列主序
```

**为什么抽象成 Layout 很关键**:
- 同一套算法代码**不感知具体内存排布**——换行/列主序只是换一个 Layout 组合;
- Layout 的**代数**(composition / product / divide / coalesce …)用来拼接和切分布局,是后面 TV 分布、tiling 的工具;
- CuTe 里的 **Tensor = 数据 + Layout**(`cute::Tensor<Engine,Layout>`),既管数据又管形态。

> ⚠ 运算名:是 `logical_divide` / `zipped_divide` / `tiled_divide` / `coalesce` / `flatten` / `composition` / `complement`,**不是** `left_padding` / `partitioned_into_TV`。注意 `tiled_divide` / `tiled_product` 是合法名(属于 divide/product 系列本体),别误以为"tiled 系列都不存在"。

**一个可复现的 composition 例子**(官方 02_layout_algebra 的例):

```cpp
// composition(A, B) = A∘B :先按 B 重排索引,再按 A 映射 —— 直觉是"按 B 决定遍历顺序"
// 官方例:A = (6,2):(8,2)   (6行2列,行步长8、列步长2)
//        B = (4,3):(3,1)
// 结果:A∘B = ((2,2),3):((24,2),8)
//  即 3 列,每列先序化：先列(6→拆成 (2,2)),再行(8+8=16... 领 3 个),compose 明细见官方文档
// 手算验证:A∘B 把 B 的坐标 (i,j) 先经 B 映射到 A 的坐标,再经 A 映射到内存索引
```

**验证**:可复现 = 有输入、有运算、有输出。上面给的 `A=(6,2):(8,2)`, `B=(4,3):(3,1)` ⟹ `A∘B = ((2,2),3):((24,2),8)` 是官方文档原例,可在 CuTe 里 `print_layout` 验证。

- **product**(`logical_product` / `tiled_product` / `zipped_product`):按一种 layout 复制另一种,用于用多次复制拼接成大 layout;
- **divide 细分**(`logical_divide` / `zipped_divide` / `tiled_divide`):
  - **`logical_divide` 保留 mode 语义**:M 模仍是 M 模(逻辑切分,不动 mode 结构);
  - **`zipped_divide`** 把 Tile 收进 mode-0、**Rest 收进一个嵌套的 mode-1**(如 `(TileM,TileN),(RestM,RestN,...)`),保留嵌套结构;
  - **`tiled_divide`** 只把 **Rest 部分展平**(`(TileM,TileN),RestM,RestN,...`),不是 flatten 整个结果。
  - 三者关系是 layout 代数里最容易混的一组,按上面这条记:别把 `logical_divide`(保留 mode)和 `zipped`/`tiled` 的展平程度搞反。
- **complement**:构造"补" layout,是把一个 layout 补成完整索引覆盖的关键运算(Lei Mao 的 TV layout 构造大量使用),也是本篇之前漏掉的重要成员。

这些运算把大 Layout 切成/拼成"分给 thread/warp/Tensor Core 的小 Layout",是 CUTLASS 抽象数据分布的核心机制。

---

## 2. MMA Atom:把一条硬件指令变成可组合的组件

硬件 mma 指令是架构相关的(Ampere 的 `m16n8k16`、Hopper 的 `wgmma`…)。CuTe 把每条指令包装成**Atom** = **Operation struct**(具体 PTX 指令)+ **MMA_Traits struct**(元信息):

- **Operation**:封装 PTX 指令的入参/出参,最小软件依赖(不用 layout/tensor);
- **MMA_Traits**:定义逻辑计算类型、逻辑形状、以及这条指令的 **线程-值布局(TV layout)**;
- **Atom** = Operation + Traits:把"线程和数据布局的复杂性"与指令调用点解耦,**一条指令的语义被完整固化**,无论它在哪层粒度工作。

**Atom 可工作在不同硬件粒度**:
- 单线程(FMA 指令);
- quadpair(Volta);
- 单 warp(Ampere,如 `SM80_16x8x16_F16F16F16F16_TN`);
- warpgroup(Hopper `wgmma`)。

---

## 3. TV layout:每个线程到底拿哪些值

**Thread-Value(TV)layout** 描述线程组里**每个线程访问哪些值**,是真正决定数据分布的核心。

- 2D TV layout:第一个 mode = 线程组里的**线程**,第二个 mode = 每个线程访问的**值**;
- 用于 **TiledCopy**(各线程按原子拷贝的模式取多个值)和 **TiledMMA**(各线程从 MMA 子矩阵取值、执行原子 mma);
- **Inverse TV layout**:反过来描述"每个数据元素落在哪个线程、哪个 value 索引",常用于**验证访问模式是否符合预期**。

> ⚠ **Inverse TV 的 1D→2D 折算(关键坑)**:inverse TV 返回的是 **1D 索引**,必须再用 TV layout 的 shape(线程数 × 每线程值数)**手动折算成 2D 的 (thread, value) 坐标**——不折算直接用会拿错线程/值。

**具体例子**(Lei Mao / CUTLASS 官方):`cute::SM80_16x8x16_F16F16F16F16_TN` 是一个 Ampere MMA atom(M16N8K16,FP16,TN 排布):

```
MMA_Atom ThrID: _32:_1   Shape_MNK: (_16,_8,_16)
LayoutA_TV: ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
LayoutB_TV: ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
LayoutC_TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
```

含义:32 个线程(`ThrID _32:_1`),`Shape_MNK: (_16,_8,_16)` 是**整个 MMA 指令**的 (M,N,K) 逻辑形状(不是 A 一个操作数的形状)。三个操作数各按自己的 TV layout 分给线程:
- **A** 是 M×K = 16×16,`LayoutA_TV` 把 16×16 的 A 按线程切成每组 `(4,8)` 线程、每组取 `(2,2,2)` 值;
- **B** 是 K×N = 16×8,`LayoutB_TV` 同理;
- **C/D** 是 M×N = 16×8,`LayoutC_TV` 同理。

**TN 含义(容易讲反)**:MMA atom 命名用两个字母分别描述 A、B(相对各自行/列主序的自然形态):
- `T` = A 是 **transposed**;`N` = B 是 **non-transposed**;
- **在 Ampere mma 约定里"transposed"指 **K-major**:A 是 M×K 沿 K 连续(行主序 M×K)、B 也要求 K-major(K 连续)。
- 所以 **TN atom 要求 A、B 均 K-major(A 行主序 M×K、B 沿 K 连续)——这正是 mma 指令的原生布局**。写 kernel 时 smem/fragment 排布按"两操作数都 K-major"放,别把 T/N 和 行/列主序搞反。

这些 TV layout 由 CuTe 架构师**按底层 PTX mma 的数据访问模式预先设计好**,使用者**不必自己研究 PTX mma 指令**。这正是 CuTe 的"开箱即用"价值。

> ⚠ Hopper wgmma 的 TV layout 形态与 Ampere 的线程级 2D TV layout **不同**:Hopper GMMA 的 `ThrID` 是 `Layout<_128>`(warp-group 级),配 `SM90::GmmaDescriptor`(`Major::K` / `Major::MN`)+ swizzle atom(如 `Layout_SW128_Atom`)表达数据排布——别把 §3 的 Ampere 读法直接套到 Hopper(wgmma 无 Ampere 那种 `WarpsTileMode`,该名在 CUTLASS 全库不存在)。

---

## 4. TiledMMA / TiledCopy:把 Atom 平铺成微内核

单个 Atom 的粒度太细(一条指令)。**TiledMMA** 把多个 Atom **平铺(tile)和交织(interleave)**,构造更大、可复制的微内核:

- **`TiledMMA` 提供 `partition_shape_A/B/C`、`thrfrg_*`、`make_fragment_*`** 等,把数据 layout **按线程分布**;**单线程视角的实际 partition 方法 `partition_A/B/C` 是 `ThrMMA`(`TiledMMA::get_slice()` 的返回物)的成员**——要先 `get_slice()` 拿 ThrMMA 再 partition,这是最易踩的坑**;
- `TiledCopy` 同理,在多个线程/多个 Atom 间编排拷贝;
- 再往上,Collective 层用这些 Tiled 微内核 + 架构同步,组织"一个 stage 的搬算";Kernel 层再组装成整个 kernel。

```cpp
// 最小端到端用法:make_tiled_mma → get_slice → partition_A/B/C
auto tiled_mma = make_tiled_mma(MMA_Atom{}, Layout<Shape<...>, Stride<...>>{}); // 铺 tile
auto thr_mma  = tiled_mma.get_slice(thread_idx);       // 拿到单线程视角 ThrMMA
auto A_frag   = thr_mma.partition_A(global_A);         // 当前线程的 A 子矩阵(fragment)
// cute::gemm(thr_mma, ...) 执行
```

**抽象层级递进**:

```
PTX 指令 → Operation + MMA_Traits → Atom
   → 多 Atom 平铺/交织 → TiledMMA / TiledCopy   (空间微内核)
   → Collective 层同步编排                      (时间微内核)
   → Kernel 层 + TileScheduler → 完整 GEMM
```

---

## 5. 实践要点(checklist)

1. **先认 Layout,再看模板参数**:读 CUTLASS 3.x 代码,先定位 `Layout`(数据怎么排)和 `TiledMMA`(数据怎么分给线程),比盯模板参数更能把握全貌。
2. **TV layout 是用来"免学 PTX"的**:官方 `mma_traits` 已按硬件数据访问模式设计好 TV layout,别自己重发明;要验证分布有没有猜对,用 **inverse TV layout** 反查。
3. **换硬件换 Atom,上层组合不变**:架构升级时,CuTe 的高层 Layout 组合代码通常不动,只换对应 SM 的 Atom/Traits——这正是可移植性的来源。
4. **想读源码**:核心类型在 `include/cute/`(shape/stride/layout/tensor);Operation structs 在 `include/cute/arch/`(`mma_sm*.hpp`),`MMA_Traits` 特化在 `include/cute/atom/mma_traits*.hpp`(如 `mma_traits_sm80.hpp`)。
5. **别混淆两层抽象**:`Layout`(数据在内存怎么排)≠ `TV layout`(线程取哪些值)。前者管存储形态,后者管数据分发,两者通过 `tiled_divide` / `partition_A/B/C` 等运算关联。

---

## 6. 溯源

- CuTe Layout 核心抽象: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html
- CuTe Layout 代数(composition/product/divide/coalesce): https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html
- CuTe MMA 指令支持(Atom / Operation / MMA_Traits,各硬件粒度): https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html
- 官方 Source on GitHub: `include/cute/layout.hpp` / `include/cute/atom/mma_traits.hpp`
- TV layout 与 inverse TV layout(含 `SM80_16x8x16` 完整 TV layout): Lei Mao's Log Book, https://leimao.github.io/blog/CuTe-Thread-Value-Layout/ (2025-10)
- CuTe Tiled MMA: https://leimao.github.io/blog/CuTe-Tiled-MMA/ (Lei Mao's Log Book)
