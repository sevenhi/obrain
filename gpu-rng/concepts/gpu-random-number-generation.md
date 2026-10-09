---
title: GPU 随机数生成:counter-based(Philox)与 F2-linear(MTGP)两条路
type: concept
tags: [gpu, rng, prng, philox, mtgp, curand, counter-based, monte-carlo, cuda]
source:
  - "pdf: Parallel Random Numbers: As Easy as 1, 2, 3 (Salmon/Moraes/Dror/Shaw, SC11 Best Paper) | DOI 10.1145/2063384.2063405 | Parallel Random Numbers: As Easy as 1, 2, 3.pdf (primary, counter-based)"
  - "pdf: Variants of Mersenne Twister Suitable for Graphic Processors (Saito/Matsumoto) | arXiv:1005.4973 | Variants of Mersenne Twister Suitable for Graphic Processors.pdf (primary, MTGP)"
  - "web: Random123 官方站(DEShawResearch) | https://www.thesalmons.org/john/random123/"
  - "web: MTGP 官方站(広島大 松本实验室) | https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/MTGP/"
  - "web: cuRAND Host API / Device API Overview (CUDA Toolkit 文档) | https://docs.nvidia.com/cuda/curand/host-api-overview.html ; https://docs.nvidia.com/cuda/curand/device-api-overview.html"
date: 2026-09
---

# GPU 随机数生成:为什么是 counter-based 与 MTGP

## TL;DR

传统 PRNG(如 Mersenne Twister)是**有状态的顺序递推** `s_{n+1}=f(s_n)`,天生串行,搬到 GPU 上有两难:一线程一 generator → 每线程要存 ~2.5KB 状态,塞爆片上内存;共享一个 generator → 又是串行瓶颈。GPU 上跑出来两条主流路:

1. **Counter-based(CBRNG,代表 Philox)**:第 n 个随机数 = `b_k(n)`,把"计数器 + key"过一个**无状态双射**(block cipher / 类密码学混淆函数)。天然并行、几乎零状态,支持 ≥2⁶⁴ 条独立流、每流周期 ≥2¹²⁸,过 BigCrush。Philox 是 GPU 上最快的 Crush-resistant PRNG(SC11 口径 GTX580 上 202 GB/s/chip)。cuRAND 的 `PHILOX4_32_10` 就是它。
2. **F2-linear 大块并行(MTGP)**:不是一线程一 generator,而是**一个 block 的多线程协作推进一个大状态空间**(周期 2¹¹²¹³−1 等),把 MT 的递推里天然存在的并行度暴露出来。cuRAND 的 `MTGP32` 用它(200 组预制参数集、每 block ≤256 线程)。

**选型一句话**:要极致吞吐/确定性/跨平台可复现 → **Philox(counter-based)**;要超长周期 + 高维等分布的强统计保证、且能吃 block 级协作 → **MTGP**。cuRAND 默认的 XORWOW 快但**过不了 BigCrush**(CollisionOver/SimpPoker/LinearComp 系统性 reject),严肃蒙特卡洛别用。

---

## 1. 机制原理

### 1.1 传统 PRNG 为什么难并行

递推式 PRNG 的统计质量来自状态空间 S 要大、转移函数 f 要复杂,而 `s_{n+1}=f(s_n)` **逐值依赖前值**,本质串行。两种并行化都有坑:

- **multistream**(多流,不同参数跑多实例):只对"有大量已验证好参数"的家族可行,而这样的好参数集很难枚举;
- **substream**(把一条超长序列切成不相交子流):需要能确定性切分状态空间,只有少数 PRNG 支持;否则只能概率性随机选起点(生日问题——周期不够长就可能重叠),且长周期 PRNG 的**状态初始化本身极难做对**,做错会引入缺陷。

再加上:每个并行单元都要存状态,而 GPU 片上内存(寄存器/shared)是稀缺资源。MT19937 的 **2.5 KB/线程** 状态在高度并行 GPU 应用里就是灾难——这是 counter-based 与 MTGP 共同要解决的问题。

### 1.2 Counter-based:把复杂度从 f 挪到 g

Salmon 等人的关键翻转:传统 PRNG 是"复杂 f + 平凡 g",counter-based 反过来用"**平凡 f + 复杂 g**"。

```
传统:   s_{m} = f(s_{m-1})            u_n = g(s_{⌊n/J⌋})   // f 复杂,g 平凡(线性映射到 [0,1))
counter: s_n = (s + 1) mod 2^p          u_{k,n} = h_j(b_k(n))  // f 只是计数器,b_k 是 keyed 双射
```

- **f 退化成一个计数器**:状态就是 p-bit 整数 n;
- **g = h_j ∘ b_k**:`b_k` 是一个 **keyed bijection**(密钥化的双射),`h_j` 从 p-bit 输出里挑第 j 个 r-bit 块(J = 输出重数,如 128-bit 出 4 个 32-bit)。
- `b_k` 从哪来?**密码学**。CTR 模式的 block cipher 就是"对计数器做 keyed 双射"——counter-based PRNG 与 CTR-mode 密码的唯一区别是前者把输出转成数值而非与明文 xor。任何扛住密码学界检验的 block cipher 几乎必然能做高质量 `b_k`(diffusion/avalanche 判据:输入单 bit 翻转 → 每个输出 bit 有 ~50% 概率翻转)。

**并行方式天然**:multistream = 每计算单元分配不同 key k,自增各自 counter;substream = 把 counter 空间按 block/stride 分给各单元,key 相同。更妙的是 **API 把 k 和 n 的控制权完全交给应用**:可以用机器参数(core/rank/thread id)派生,也可以用**应用变量**(粒子 id、体素号、时间步)派生。后者给出**机器无关的随机流**——同样的应用逻辑在不同并行度/不同硬件上产生同样序列,可复现、可调试,这是传统 PRNG 几乎做不到的(要为亿万对象各存一份状态)。

**周期**:双射 f 周期 2^p,每次出 J 个值 → 周期 J·2^p。实测各 PRNG 的 J·2^p 就是最小周期(已验证更小的因子都不是周期)。

### 1.3 Philox:GPU 上最快的 counter-based

三大家族(见 §3 数据表),Philox 是 GPU 冠军:

- **结构**:SP 网络,整体像 Threefry,但 S-box 用**乘法**做扩散。Philox S-box 是一个广义 Feistel(用 xor 作群运算):
  ```
  L' = mullo(R, M)
  R' = mulhi(R, M) ⊕ k ⊕ L
  ```
  `mulhi/mullo` 是"取乘法高/低半"的单指令,每个现代架构都有;对常数奇 M,`mullo(a,M)` 是双射,单 bit 翻转会引发 avalanche(x86 上 mulhi+mullo 可一条指令同时算)。
- **N>2 时**:输入先过 Threefish 的 N-word P-box 置换,再两两喂进 N/2 个 S-box,各有自己的乘子 M 与 key;k 按 Weyl 序列(`key_i = key_{i-1} + 常数`)逐轮更新。
- **乘子怎么定**:量化 avalanche(用 χ² 统计量,测 4 轮),遗传算法式搜索(点突变 + 重组 + 最速下降)最大化 avalanche,再过全套 Crush 定最小轮数。结果:Philox-4×32 用 `0xCD9E8D57` 与 `0xD2511F53`;Philox-2×64 用 `0xD2B74407B1CE6E93`;Crush-resistant 最小轮数为 Philox-4×32-7 / 2×64-6 / 4×64-7。
- **为什么 4×32 适合 GPU**:依赖 32-bit 乘法,GPU(及低功耗嵌入式 CPU)支持更好;SC11 实测 GTX580 上 202 GB/s/chip(已知单芯片最高吞吐),AMD HD6970 上 113 GB/s/chip。cuRAND 用带 3 轮安全裕度的 **Philox-4×32-10**。

### 1.4 MTGP:一个 block 协作推进一个大 generator

Saito/Matsumoto 的路子正相反——**不抛弃 F2-linear 递推,而是把递推内部的并行度暴露出来**:

- **递推结构**:`x_{N+i} = g(x_{M+i}, x_{1+i}, x_{i})`,M 是 middle position。当数组长度 L ≥ 2N−M 时,`i = 0 … N−M−1` 这一段可**并行计算**(到 i=N−M 时才需要串行依赖)。MTGP 就用这 N−M 个并行位喂一个 block 的线程。
- **一线程一 generator vs 一 block 一 generator**:朴素做法(SDK-MT)是 4096 个 MT607,每线程一个,内存 = (607+参数)×线程数;MTGP 是**一 block 共享一个大状态**,内存 = 32-bit×线程数,状态放 shared memory,周期却高达 2¹¹²¹³−1 且高维等分布更好。三种周期 2¹¹²¹³−1 / 2²³²⁰⁹−1 / 2⁴⁴⁴⁹⁷−1 对应线程数 256 / 512 / 1024。
- **recursion 公式**(C 语言版,`^` 为按位异或):
  ```c
  t = X[1+i] ^ (X[i] & BITMASK(w,r));
  t = t ^ (t<<sh1);
  u = t ^ (X[M+i]>>sh2);
  X[N+i] = u ^ rectbl[u & 0xf];   // rectbl = (4×w)矩阵 R 的 16 项查表
  ```
  `R` 的乘法用 16 项查表实现(u4 只有 16 种取值);查表放 **texture memory**(只读缓存)最合适,因为它**无 bank conflict**——若放 shared memory,同 warp 不同线程的 u4 常相等会撞 bank;放寄存器又挤占 occupancy。
- **tempering**(提升 k(v) 高维等分布):
  ```c
  t = X[M-1+i] ^ (X[M-1+i] >> 16);
  t = t ^ (t >> 8);
  return X[N+i] ^ tmptbl[t & 0xf];
  ```
- **MTGPDC(参数集生成器)**:接收 32-bit ID 嵌入递归参数矩阵 R,ID 不同 → 特征多项式不同(用 SHA1 digest 校验)→ 流独立,适合大 grid 每 GPU 要独立流。搜索很慢(p=11213 递归参数平均 338 秒/组),**必须离线预生成、复用**,不能在模拟中现搜。cuRAND MTGP32 用的正是预制 200 组参数。

---

## 2. 可复现细节

### 2.1 Philox-4×32 单轮(S-box)伪码

```
输入 4×32-bit 计数器 (c0,c1,c2,c3),key (k0,k1)
每轮:
  hi0,lo0 = mulhilo32(M0=0xD2511F53, c0)
  hi1,lo1 = mulhilo32(M1=0xCD9E8D57, c2)
  c' = (hi1 ^ c1 ^ k0,  lo1,
        hi0 ^ c3 ^ k1,  lo0)
  key 更新(Weyl): k0 += 0x9E3779B9; k1 += 0xBB67AE85
重复 R 轮(Philox-4×32-10 → R=10),输出 4×32-bit
```

无状态:给定 (counter, key) 直接算出输出,不需要保存中间状态。

### 2.2 cuRAND device API 用法(GPU kernel 内生成)

```cpp
#include <curand_kernel.h>
__global__ void setup(curandStatePhilox4_32_10_t* s, unsigned long long seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Philox:seed/subsequence/offset;subsequence 与 offset 共同定位 2^128 周期序列内的位置
  curand_init(seed, i /*每线程唯一 subsequence*/, 0, &s[i]);
}
__global__ void gen(curandStatePhilox4_32_10_t* s, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = curand_uniform(&s[i]);   // (0,1] 均匀分布
}
```

- **XORWOW/MRG32k3a**:`curand_init(seed, sequence, offset, state)`,state 相当于"从 seed 状态走了 2⁶⁷·sequence + offset 步 curand() 之后"。周期 ≈2¹⁹²。
- **Philox**:周期 2¹²⁸;subsequence 长度 2⁶⁴,一个 subsequence 用尽自动进下一个。
- **MTGP32**:一个 state 结构对应一组参数(一个序列);单 block 内 ≤256 线程 thread-safe 协作,**两个 block 不能操作同一 state**;需 host 侧 `curandMakeMTGP32KernelState` 预置参数集。

### 2.3 性能要点(cuRAND 官方 Performance Notes)

- `curand_init()` 比 `curand()`/`curand_uniform()` **慢得多**;offset 越大越慢;**保存/恢复 state 远快于反复重算初始 state**。
- 推荐把 state 放 global memory,跨 kernel launch 复用:
  ```cpp
  __global__ void example(curandState* global_state) {
    curandState local = global_state[threadIdx.x];  // 载入 local/寄存器
    for (int i = 0; i < 10000; i++) { unsigned x = curand(&local); /* ... */ }
    global_state[threadIdx.x] = local;               // 写回
  }
  ```
- init 通常比生成更吃寄存器/local memory,**把 `curand_init` 与 `curand` 拆进不同 kernel** 可提速。
- 每线程用不同 seed + 常数 sequence=0 能加速 setup,但统计保证更弱(哈希与周期的坏交互理论上可能让某些 seed 下线程输出高度相关,虽未知问题值)。

---

## 3. 数据对比

### 3.1 SC11 counter-based 性能/内存表(节选,cpB=cycles per byte,越低越快)

| 方法 | 类别 | 状态 | Intel CPU cpB | NVIDIA GPU GB/s | AMD GPU GB/s |
|---|---|---|---|---|---|
| AES(hw, AES-NI) | 密码学 | 176 B/流 | 1.7 | – | – |
| ARS-5(hw) | counter, Crush-resistant | 0 | 0.7 | – | – |
| Threefry-4×64-12 | counter, CR | 0 | 1.1 | 84.1 | 90.0 |
| **Philox-4×32-7** | counter, CR | 0 | 2.4 | **201.6** | 113.1 |
| Philox-4×32-10(+裕度) | counter, CR | 0 | 3.6 | 145.3 | 79.1 |
| Mersenne Twister | 传统, **Crushable** | 312×8 B | 2.0 | 18.3 | – |
| **XORWOW(cuRAND 默认)** | 传统, **Crushable** | 6×4 B | 1.6 | 136.7 | 81.1 |

> 推荐组合(SC11 结论):有 AES-NI 且能容 176 B/流 → AES;要极致且不想存 key → ARS-7;无 AES-NI 的 CPU → Threefry-4×64-20;**GPU → Philox-4×32-10**(带安全裕度,仍快于任何 Crush-resistant 传统 PRNG)。

### 3.2 MTGP 与其它 GPU PRNG(5×10⁷ 个 [0,1) float,GTX260)

| PRNG | GTX260 耗时 | 周期 | BigCrush |
|---|---|---|---|
| SDK-MT(4096×MT607) | 18.6 ms | 2⁶⁰⁷−1 | **reject** |
| MTGP11213 | 4.7 ms | 2¹¹²¹³−1 | pass(除 F2-linear 相关测试) |
| HybridTaus | 9.3 ms | ~2¹²¹ | pass(同上) |
| WarpStandard | 3.2 ms | 2¹⁰²⁴−1 | **pass\*(全过)** |
| CURAND(XORWOW) | 3.0 ms | ~2¹⁹² | **reject** |

> XORWOW/CURAND 最快但被 CollisionOver(7)、SimpPoker(27)、LinearComp(81) 系统性 reject;看连续输出差分 y_n=x_n−x_{n−1} 时更多测试失败。MTGP 只在测 F2-线性依赖的 LinearComp/MatrixRank 上失败(这对 MT/WELL 是共性,通常不影响把输出当实数/整数用的随机模拟)。WarpStandard 是唯一全过 BigCrush 的。

---

## 4. 实践要点

1. **默认别用 XORWOW**:cuRAND `CURAND_RNG_PSEUDO_XORWOW` 是默认最快项,但过不了 BigCrush。严肃蒙特卡洛用 **`PHILOX4_32_10`**(counter-based,可复现、零状态)或 **`MTGP32`**(超长周期)。
2. **要跨平台可复现 → counter-based**:用应用变量(粒子 id/体素号/时间步)拼 counter+key,不同并行度、不同硬件都出同一序列。这是传统 PRNG 做不到的。
3. **state 存 global、跨 launch 复用**:反复 `curand_init` 很贵;把 state 缓存在 global memory,kernel 内载入 local 生成后写回。init 与 generate 拆成两个 kernel 更省寄存器。
4. **Philox 选 4×32 不选 4×64**(在 GPU 上):32-bit 乘法 GPU 支持更好、吞吐更高;要 10 轮版本留安全裕度。
5. **MTGP 的硬约束**:一个 state = 一个 block,≤256 线程协作,**跨 block 不能共享 state**;参数集必须离线用 MTGPDC 预生成(搜索以分钟计),用 cuRAND 时直接吃预制 200 组。
6. **F2-linear 的 BigCrush 失败别慌**:MTGP/MT/WELL 在 LinearComp/MatrixRank 上失败是 F2-线性的共性,把输出当均匀实数/整数用的常规模拟一般不受影响;但对**密码学或线性结构敏感**的应用,选 counter-based(Philox/Threefry)而非 F2-linear。
7. **验证工具**:TestU01 的 SmallCrush/Crush/BigCrush(BigCrush 在现代 CPU 上跑几小时、测 ~2³⁸ 样本);"Crush-resistant" 是全过零 Failed。counter-based 还要专门测**流间独立性**(不同 key 的 multistream、不同 counter block/stride 的 substream,含 gray-code/constant-Hamming 对抗性 counter)。

## 溯源

- counter-based 主源:Salmon, Moraes, Dror, Shaw, *Parallel Random Numbers: As Easy as 1, 2, 3*, SC11(Best Paper)。库内 PDF + 官方站 https://www.thesalmons.org/john/random123/ (Random123 库,含 ARS/Threefry/Philox 的 C/C++/CUDA/OpenCL 实现)。
- MTGP 主源:Saito, Matsumoto, *Variants of Mersenne Twister Suitable for Graphic Processors*, arXiv:1005.4973。库内 PDF + 官方站 https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/MTGP/ (MTGP/MTGPDC 源码与参数集)。
- 工程交叉源:NVIDIA cuRAND 文档(Host/Device API、Performance Notes、生成器类型与周期口径),用于把论文的算法结论对齐到 `curandStatePhilox4_32_10_t`/`MTGP32`/`XORWOW` 的实际 API 与推荐用法。
