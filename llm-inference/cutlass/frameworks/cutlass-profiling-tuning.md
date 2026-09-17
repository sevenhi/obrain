---
title: CUTLASS 性能调优:从 Profiler 到关键旋钮
type: howto
tags: [cutlass, gemm, tuning, profiler, roofline, stages, warp-specialization, tma, tmem]
source:
  - "web: NVIDIA CUTLASS Profiler 文档 | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html"
  - "web: NVIDIA Tech Blog — Improving GEMM Kernel Auto-Tuning Efficiency on NVIDIA GPUs with Heuristics and CUTLASS 4.2 (2025-09) | https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/"
  - "pdf: Understanding GEMM Performance and Energy on NVIDIA Ada Lovelace: A Machine Learning-Based Analytical Approach | arXiv:2411.16954"
date: 2026-09
---

# CUTLASS 性能调优:从 Profiler 到关键旋钮

## TL;DR

CUTLASS 性能调优有**清晰的方法论**,不是瞎试:

1. **先测再调**:用 `cutlass_profiler` 拿基线(带宽 / GFLOPs / Runtime),不要凭感觉;
2. **先判受限类型**:用 **roofline** 模型判断是**内存受限**还是**计算受限**,对症下药;
3. **关键旋钮**(按影响力排):**tile 尺寸** → **流水线级数** → **warp-specialization** → **split-K / 调度策略** → **数据类型**;
4. **架构专属优化**:Hopper/Blackwell 用 TMA / WGMMA / clusters / TMEM(Blackwell 的 accumulator-buffer ping-pong);
5. **能自动就自动**:4.x 自动调优分两条路径——`cutlass_library` 路径支持 Hopper+Blackwell;Operator API(`nvMatmulHeuristics`)目前仅 SM100 non-blockscaled GEMM。

---

## 1. 第一步:用 Profiler 拿基线

> 前提:先用 `cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_ENABLE_TESTS=OFF` configure 过,再 `make cutlass_profiler`。

```bash
make cutlass_profiler -j16
# GEMM(完整单条命令,可复制)
./tools/profiler/cutlass_profiler --kernels=sgemm --m=4352 --n=4096 --k=4096

# 通配符过滤只测某类 kernel(嵌进命令)
./tools/profiler/cutlass_profiler --kernels='cutlass_tensorop_s*gemm_*' --m=4352 --n=4096 --k=4096

# 横向对比可靠化:增大 sleep 隔离 cache 状态 + 导出 CSV(选项是 --output,不是 --export-csv)
./tools/profiler/cutlass_profiler --kernels=sgemm --m=4352 --n=4096 --k=4096 --sleep-duration=100 --output=out.csv
```

> ⚠ **测速前先锁频**(最容易踩的坑):`nvidia-smi -lgc <clk>` 锁定 GPU 时钟,否则跨轮次/跨 kernel 对比会被动态降频噪声污染,数据不可信。这是"拿 profiler 做可靠横比"的头号陷阱。

**输出字段**:Disposition/Status、Bytes、FLOPs、Runtime、**Memory 带宽、Math GFLOP/s**。**先看 `Disposition=Passed` 再谈性能**——错结果也能跑出漂亮的 GFLOP/s。关键用途:同问题不同 tile 配置**横向对比**,量化每次改动收益。

---

## 2. 第二步:判受限类型(roofline)

调优第一步不是调,是**判断瓶颈**。用 roofline 模型:

- **内存受限(memory-bound)**:带宽是瓶颈 → 优化访存(布局、TMA、减少搬运);加算力没用;
- **计算受限(compute-bound)**:算力是瓶颈 → 优化 tile/流水/数据类型。

**脊点(ridge point)判据**:某 GPU 的 FLOPs-per-Byte 阈值 = 算力/带宽。**关键:必须注明口径**——如 RTX 4070 在 **FP32 CUDA-core 口径**下 ridge ≈ 29 TFLOPS ÷ 504 GB/s ≈ **57.5** FLOPs/Byte(论文 arXiv:2411.16954 报 ~59,但那是按其不同实验口径,别混进同一条算式);在 **Tensor Core 口径**下 ridge 高得多。不注明口径,拿 FP32 脊点去判 TF32/FP16 GEMM 会得出"永远计算受限"的错误结论。

**算术强度公式(先算再判)**:`算术强度 = 2·M·N·K / 总访存字节`,其中总访存 = **读 A+B + 写回 C/D**。对 M、N 大而 K 小的 shape,写回占比不可忽略,漏掉会系统性高估算术强度、误判为计算受限;K 较大时 C 写回相对可忽略。算出来与脊点比:低于 → 内存受限,高于 → 计算受限。

---

## 3. 关键旋钮(按影响力排序)

### 3.1 Tile 尺寸(CtaTile / WarpTile / MmaTile)
- 在**并行度**(tile 小→多 CTA)与**寄存器/共享内存占用**(tile 大→占用高)间权衡;
- **区分三层,别混**:`CtaTile`(并行度/占用,起步 **128×128×32** 量级)、`WarpTile`(smem→reg 分布,64×64 量级)、`MmaTile`(指令吞吐,如 Ampere 16×8×16)。tensor-core GEMM 起步别再往 16×16 的 CTA 上试(那是某些研究中 warp/指令级 tile 的结论);
- 往往某个具体 tile 配置能带来数倍收益(见溯源注:具体倍数因版本/硬件/shape 差异大,不引单一数字以免误导)。

### 3.2 流水线级数(stages)
- mainloop 软件流水线深度,Hopper 上可叠 **TMA 异步拷贝**;
- 太少→搬算无法重叠;太多→smem 被占、occupancy 降。3~6 级常为经验窗口;
- **可算的 smem 预算判据**:`stages × (cta_m×cta_k + cta_k×cta_n) × sizeof(dtype) ≤ SMEM 上限`(Hopper 228KB)——用当前 tile 反推最多几级,别盲试。

### 3.3 Warp-specialization
- 生产者/消费者 warp 分工(Hopper 主流范式,`48_hopper_warp_specialized_gemm`);
- 提高**占用率与指令级并行**,把搬运与计算真正并发。

### 3.4 调度策略 + split-K
- 大 tile / 少 tile / 输出不规整 → 用 **Stream-K** 解决负载均衡(见 tile-scheduler 篇);多小 problem → **Grouped**;
- **split-K(别漏)**:把 K 维拆给多个 CTA、各自做部分和再归约,增加并行度;**LLM 推理 decode 的 skinny GEMM(M 很小)第一瓶颈正是并行度不足,split-K 是首选旋钮**。`nvMatmulHeuristics` 官方将其列为关键调优维度之一(CTA tile、split-K factor、cluster 尺寸等元参数)。

### 3.5 数据类型 / 混合精度
- **TF32**(3×TF32 仿真 FP32,精度换吞吐);
- **FP8 / FP4 blockscaled** 按精度需求与吞吐权衡。

---

## 4. Hopper / Blackwell 专属优化

| 特性 | 架构 | 作用 |
|---|---|---|
| **TMA**(Tensor Memory Accelerator) | Hopper+ | 异步批量拷贝,卸载地址计算与搬运,CuTe Tiled copy 表达 |
| **WGMMA**(Warpgroup MMA) | Hopper | warp-group 级异步矩阵乘累加,collective 层封装 |
| **Thread Block Clusters** | Hopper+ | 跨 CTA 协作、共享 SMEM |
| **TMEM**(Tensor Memory) | Blackwell(SM100) | 512 COL;**accumulator-buffer 规划(ping-pong 重叠)** 是 Blackwell 关键手法 |

**Blackwell 王牌范式**:warp-specialized + TMA + UMMA(tcgen05),加 TMEM 的 accumulator ping-pong——这是读 Blackwell kernel 时最该盯的优化点。

---

## 5. 自动调优

- **自动调优分两条集成路径(别搞混)**:
  - **`cutlass_library` 路径**(CMake `CUTLASS_LIBRARY_HEURISTICS_*` + cutlass_profiler):官方 Coverage 支持 **Hopper(sm9x)+ Blackwell(sm10x)** 的 f8/f16/f32 dense GEMM——上面 96%(H100)那组数据就走这条路径;
  - **CUTLASS 4.x Operator API 路径**:`nvMatmulHeuristics` 按估算性能对 kernel 排名,目前**仅支持 SM100(Blackwell)的 non-blockscaled GEMM**。
- **4.2 官方博客硬数据(2025-09,注意负载/GPU 归属)**:**Llama 3 405B 训练负载、H100 SXM**(library 路径)上,穷举耗时 >700 分钟,启发式选 16 个候选 ≈150 分钟即达峰值 **96%**;**B200 上 DeepSeek-R1 训练负载**达穷举性能 **99%**,编译+调优时间 **>5× 加速**(两组数字别混,96% 是 H100/Llama,99% 才是 B200);
- 学界:扫描上万配置 + **Random Forest 建模**预测最优(runtime R²=0.98,GEMM 性能建模详见 arXiv:2411.16954 一系工作)——说明自动调优在该领域很成熟。

---

## 6. 调优 checklist(按顺序走)

1. [ ] `cmake` configure + `make cutlass_profiler`;`nvidia-smi -lgc` **锁频**
2. [ ] 算算术强度 `2MNK/(A+B读 + C写回)`,用 roofline(注明算力口径)判**内存 vs 计算受限**
3. [ ] 调 tile(CTA 128×128 量级起步,区分三层),观察并行度与占用
4. [ ] 调 stages(用 smem 预算公式定上限,3~6 级)看搬算重叠
5. [ ] 尝试 warp-specialization(尤其 Hopper)
6. [ ] 并行度不足(如 decode skinny GEMM)→ split-K;几何不规整 → Stream-K/Grouped
7. [ ] 精度允许 → 试 TF32 / FP8
8. [ ] 每次改动用 profiler(增 `--sleep-duration` 隔离 cache + `--output=out.csv`)量化,先看 `Disposition=Passed` 再记性能

---

## 7. 溯源

- CUTLASS Profiler 用法与输出字段: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html
- tile 选择收益量化:不建议引单一数字——canonical 的 3.2×/22% 源自 arXiv:2411.16954("GEMM 性能建模 / RF",特定于 RTX 4070 + naive tiled kernel,不可泛化);真实收益以 profiler 本机实测为准
- 自动调优硬数据(穷举 >700min、启发式 16 候选 96%，H100/Llama 3 405B；DeepSeek-R1 99%、>5×编译加速,B200): NVIDIA 官方博客 CUTLASS 4.2 autotuning, https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/ (2025-09)
