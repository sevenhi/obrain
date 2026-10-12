---
title: FP8 / FP4 硬件原生低比特格式:E4M3、E5M2、NVFP4 与 scale 粒度
type: concept
tags: [quantization, fp8, fp4, e4m3, e5m2, nvfp4, mxfp4, blackwell, hopper, transformer-engine, microscaling, deepseek-v3]
source:
  - "pdf: DeepSeek-V3 Technical Report | arXiv:2412.19437 (DeepSeek-AI, 细粒度 FP8 训练 recipe, ↑primary)"
  - "web: NVIDIA Transformer Engine Documentation | https://docs.nvidia.com/deeplearning/transformer-engine/ | 阅读日期 2026-10"
  - "web: OCP Microscaling Formats (MX) Specification v1.0 | https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf (2024-02)"
  - "web: DeepGEMM | https://github.com/deepseek-ai/DeepGEMM (细粒度 FP8 GEMM 库)"
  - "web: MS-AMP | https://github.com/Azure/MS-AMP (微软 FP8 训练框架)"
date: 2026-10
---

# FP8 / FP4 硬件原生低比特格式:E4M3、E5M2、NVFP4 与 scale 粒度

> **TL;DR**:FP8/FP4 是**硬件原生**低比特格式(浮点数制 + Tensor Core 直通路径),与 INT8/AWQ 的「自定义 scale + 整数 kernel」路线、QuIP#/AQLM 的「算法压缩 + 码本 kernel」路线都不同。Hopper 起 FP8 成为默认服务/训练格式;Blackwell 引入 FP4、NVFP4、MXFP4。真正决定可用性的**不是位宽本身,而是 scale 粒度**(per-tensor / per-token / block-wise)与配套 kernel 成熟度——粗粒度 scale 会被 outlier 击穿,细粒度 scale 才是近年训练 recipe 的核心变量。

## 1. 为什么是「格式」而不只是「位宽」

低比特领域至少有三条正交路线,混用概念会导致选型错误:

| 路线 | 数值格式 | 典型实现 | 硬件路径 | 代表 |
|---|---|---|---|---|
| 硬件原生浮点 | FP8(E4M3/E5M2)、FP4(E2M1) | Tensor Core / tcgen05 低精度 GEMM | 有官方 dense 路径 | Hopper FP8、Blackwell FP4 |
| 整数量化 | INT8/INT4 + scale | 自定义 quant kernel + INT8 Tensor Core | 需自定义 kernel 或 cuBLASLt | SmoothQuant、AWQ |
| 算法压缩 | 2-bit 码本 / 格量化 | 私有权重布局 + 专用 kernel | 一般无原生 dense 路径 | QuIP#、AQLM |

「FP8 比 INT8 省事」的原因不是精度更好,而是 **GEMM 可以几乎原样跑在 FP8 Tensor Core 上**,不需要为量化重写一套 int8 kernel,也不需要每个后端各自约定 zero-point 布局。代价是浮点表示的量化网格不均匀(指数位决定动态范围),对 scale 约定比 INT8 更敏感。

## 2. 浮点格式解剖:E4M3 与 E5M2

IEEE 754 风格的位布局:`(-1)^S × 2^(E-bias) × (1.M)`(正规数)。FP8 的两个 OCP/NVIDIA 格式只是在 8 bit 里重新分配指数与尾数:

```
E4M3:  [S | E E E E | M M M]      1 符号 + 4 指数 + 3 尾数   (bias = 7)
E5M2:  [S | E E E E E | M M]      1 符号 + 5 指数 + 2 尾数   (bias = 15)
E2M1:  [S | E E | M]              FP4: 1 符号 + 2 指数 + 1 尾数 (bias = 1)
INT8:  [s | d d d d d d d]        定点,均匀网格,scale 外置
```

**统一的量化/还原形式**(下文所有粒度讨论都套这个模板):

```
前向:  q = cast_fp8( x / s )          # 除 scale → 落入 FP8 可表示区 → 舍入
还原:  x̂ = dequant(q) * s
优化:  s* = argmin_s  E[ (x − s · cast(x/s))² ]   # 实践中用 amax 或 MSE 搜索近似
```

- scale 的角色不是「把数变小」,而是**把张量的数值分布对齐到 FP8 的非均匀网格**:E4M3 的小指数区更密、大指数区更疏,分布落点决定有效精度。
- 舍入模式(就近、向零)、是否 saturate 到最大值(448)、NaN 语义(E4M3FN 无 Inf)都属于格式约定的一部分,checkpoint 与 kernel 必须一致。

**关键取舍:指数位数 ↔ 动态范围;尾数位数 ↔ 相对精度。**

- **E4M3**:尾数 3 bit → 相对精度约 2⁻⁴ = 6.25%;指数 4 bit → 有限范围很小。NVIDIA 的 E4M3FN(Finite Numbers)把「指数全 1 + 尾数全 1」定义为 NaN,去掉 ±Inf,最大有限值 **448 = 1.75 × 2⁸**。适合**权重与激活**(前向):数值已被 scale 归一到 O(1),需要的是精度而非范围。
- **E2M1(FP4)取值闭集**:非零幅值只有 `{0.5, 1, 1.5, 2, 3, 4, 6}`(含 0 共 16 个码点,含 −0)。任意两个相邻档之间的相对间隔都在 25%~100% 量级,所以 FP4 **没有 scale 几乎不可用**——它天然就是「微缩放」格式。
- **E5M2**:尾数 2 bit → 相对精度约 2⁻³ = 12.5%;指数 5 bit,沿用 IEEE 的 Inf/NaN 语义,最大有限值 **57344**。适合**梯度**(反向):梯度动态范围极大、对精度更不敏感,过小的范围会直接下溢成 0。
- **实践口径**(Transformer Engine 默认习惯):前向 GEMM 用 E4M3 存权重/激活,反向用 E5M2 或 E4M3 存梯度;DeepSeek-V3 则在前向全程用 E4M3 + 细粒度 scale,反向对激活/梯度另设策略(见 §6)。

### 2.1 表示范围对照

| 格式 | 位宽 | 最大有限值 | 最小正规数 | 相对精度(数量级) | 典型用途 |
|---|---|---|---|---|---|
| FP32 | 32 | ≈3.4e38 | ≈1.2e-38 | 2⁻²⁴ | master weights、optimizer |
| BF16 | 16 | ≈3.4e38 | ≈1.2e-38 | 2⁻⁸ | 训练默认激活/梯度 |
| FP16 | 16 | 65504 | ≈6e-5 | 2⁻¹¹ | 推理/混合精度 |
| **E4M3** | 8 | **448** | **2⁻⁶ ≈ 0.0156** | 2⁻⁴ | FP8 前向、权重、激活 |
| **E5M2** | 8 | **57344** | **2⁻¹⁴ ≈ 6.1e-5** | 2⁻³ | FP8 反向梯度 |
| **E2M1(FP4)** | 4 | **6** | **1.0**(最小正规数;0.5 为最小子正规) | 2⁻¹ | Blackwell 4-bit 权重/激活 |
| INT8 + scale | 8 | 由 scale 决定 | 由 scale 决定 | 2⁻⁷(均匀) | W8A8、KV8 |

要点:E4M3 的 448 听起来「够大」,但**没有 scale 时** LLM 激活的 outlier 动辄上百,普通值会被压到 2⁻⁶ 以下的下溢区。这就是 scale 粒度成为一等公民的原因。

### 2.2 与 INT8 的对比

| 维度 | FP8(E4M3/E5M2) | INT8 |
|---|---|---|
| 网格 | 浮点,非均匀(小值密、大值疏) | 均匀 256 档 |
| 动态范围 | 由指数位决定,scale 只需粗对齐 | 完全由 scale/zero-point 决定 |
| Tensor Core | Hopper+ 原生 FP8 路径 | Turing+ 原生 INT8 路径 |
| 额外约定 | scale 粒度、amax 统计、cast 时机 | zero-point、对称/非对称、group layout |
| 典型误差源 | 大值挤占精度、下溢 | 离群值撑大 scale、舍入平坦化 |
| 工程成本 | TE/cuBLASLt/DeepGEMM 开箱可用 | 常需 SmoothQuant 级预处理 + 定制 kernel |

两者不是替代关系:INT8 在纯推理吞吐上仍常见;FP8 的优势是**训练+推理同一格式**,且和 BF16 master weight 的混合精度梯子天然兼容。

## 3. Scale 粒度:决定成败的那个变量

设量化算子 `Q(x) = cast(x / s)`,还原 `x̂ = dequant(Q(x)) * s`。误差取决于 **s 覆盖多大的元素集合**:

| 粒度 | scale 个数 | 优点 | 风险 | 代表用法 |
|---|---|---|---|---|
| per-tensor | 1 | 开销最低,易融合 | 一个 outlier 决定全张量 | 早期 TE delayed scaling |
| per-channel / per-row | C 或 K | 适配权重通道差异 | 激活的 token 维 outlier 管不到 | 权重量化常用 |
| per-token | T(每个 token 一个) | 激活 outlier 逐 token 隔离 | 需要 row-wise reduce | 推理激活量化 |
| block-wise / tile-wise | M/128 × N/128 之类 | 细粒度,抗 outlier | scale 元数据、GEMM 需专用 kernel | **DeepSeek-V3:权重 per-block 128×128 + 激活 per-token-per-group 1×128** |
| microscaling(MX) | 每 32 个元素共享 | 标准化、带硬件 scale 表示 | 粒度固定,依赖配套硬件 | OCP MXFP8/MXFP4 |

**块量化的显式形式**(以 tile 为单位;DeepSeek-V3 权重用 128×128 block,激活按 per-token-per-group 1×128 分组):

```
对权重 W 的每个 tile  T_ij ∈ R^{128×128}:
    s_ij = amax(T_ij) / 448                 # E4M3 上限 448,amax 归一
    Ŵ_ij = s_ij · cast_fp8( T_ij / s_ij )
激活 X 则按 per-token-per-group(1×128,即每 token 每 128 channels)分组:
    s_t  = amax(X 的每个 1×128 组) / 448
GEMM 在 FP8 上计算,输出用高精度累加器承接:
    Y ≈ Σ_ij  Ŵ_ij · X̂_ij   (accum 精度 FP16/FP32)
```

- per-tensor 相当于「整网只有一个 s_ij」;block-wise 则是把上式里 `(i,j)` 的覆盖范围缩小到 tile。**scale 越细,单个 outlier 污染的元素越少**,但 scale 元数据与计算也越多。
- 元数据开销可估:权重侧每 128×128=16384 个 FP8 元素配 1 个 FP32 scale(激活侧每 1×128 组 1 个),开销约 0.02%,可忽略;真正的代价在 kernel 要按 tile 读 scale 并在寄存器里完成还原。

**DeepSeek-V3 的细粒度方案**:对 GEMM 的两个操作数分设粒度——**权重按 128×128 block、激活按 1×128 tile(per token per 128 channels)**——各自求 scale,配合在线量化(online quantization,不缓存过期 amax)与更高精度累加。其工程实现见 DeepGEMM(库内见 [DeepGEMM(NVIDIA)](../../deepseek-infra/concepts/deepgemm-nvidia-vs-ascend.md))。这是「精度不够,粒度来凑」的典型:用更多 scale 元数据 + 定制 GEMM,换回 FP8 的硬件吞吐。

### 3.1 Delayed scaling vs Just-in-time scaling

Transformer Engine 里两种 scale 策略是服务侧常踩的坑:

- **Delayed scaling**:维护历史 amax(绝对值最大)缓冲,用近若干步的 amax 推出 scale。优点是量化与 GEMM 之间几乎零额外同步;缺点是 scale **滞后于当前张量**,遇到分布突变(如长上下文、领域切换)会短暂失配。
- **Just-in-time(JIT)scaling**:每个张量当场算 amax 再定 scale。数值最准,但多一次全张量 reduction,且 scale 生成容易成为 kernel 边界上的同步点。
- **MS-AMP**(微软)给出了另一条工程路线:在保留 FP8 GEMM 的同时,把 optimizer state 也下放到 FP8(O2/O3 级),用分层 scale 管理梯度/动量/方差——说明「scale 策略」本身可以是一个框架级组件,而不只是 kernel 参数。

**经验**:推理服务优先 JIT(数值稳定可复现);超大 batch 训练可用 delayed 换带宽,但要在数据分布切换处强制刷新 amax。

工程上还要区分「scale 存在哪里」:

- **打包进 checkpoint**:转换工具把 `s` 与 FP8 权重一并写出,推理端直接加载,版本与 layout 必须冻结;
- **运行时计算**:激活/梯度的 scale 天然只能 JIT,由 kernel 或框架在 cast 时现算;
- **两者混用**最常见(权重 scale 静态、激活 scale 动态),此时日志里要能区分两类 scale,否则数值问题无法归因。

Transformer Engine 的浮点类型接口(`Float8CurrentScaling` / delayed scaling recipe 等)本质上就是把「格式 + 粒度 + 统计策略」绑成一个可替换配方;MS-AMP 则把同一思想扩展到 optimizer state。框架选择决定你能调哪些旋钮,而不是反过来。

## 4. FP4 时代:NVFP4 / MXFP4 / MXFP8 与 Blackwell

4-bit 浮点的元素格式几乎都是 **E2M1**:可表示的非零幅值只有 `{0.5, 1, 1.5, 2, 3, 4, 6}`。单元素表达力极低,因此 **FP4 的全部希望都押在 scale 粒度与累加精度上**。

### 4.1 三种常听到的「FP4/FP8 微缩放」口径

| 名称 | 元素 | scale 形式 | 块大小 | 出处 |
|---|---|---|---|---|
| **MXFP4 / MXFP8** | E2M1 / E4M3 | 共享指数(E8M0,2 的幂) | **32** elements | OCP Microscaling 规范 |
| **NVFP4** | E2M1 | 块内 FP8(E4M3)scale + per-tensor 双层 scale | **16** elements | NVIDIA Blackwell 推理格式 |
| DeepSeek 细粒度 FP8 | E4M3 | 权重每 block FP32 scale;激活 per-token-per-group | 权重 128×128 / 激活 1×128 | DeepSeek-V3 训练 |

- **OCP MX** 是开放规范:同一层里「共享 scale」被显式写入格式定义,便于跨厂商硬件对齐。scale 用 E8M0(纯指数)意味着缩放只能是 2 的幂,硬件只需移位。
- **NVFP4** 是 NVIDIA 实现取向的变体:块更细(16),scale 本身是 FP8,能更贴合实际分布;外层再放一个 per-tensor scale 解决 FP8 scale 动态范围不足。落地时以 Blackwell 的 kernel/文档为准,不要把 MXFP4 与 NVFP4 的 layout 混用。
- **MXFP8** 不是「更小的 FP8」,而是**给 FP8 再套一层每 32 元素共享指数**的微缩放容器,面向需要跨硬件统一语义的场景。

### 4.2 Blackwell tcgen05 低精度路径

Blackwell 的 Tensor Core(架构代号 tcgen05)把 FP4 纳入 dense GEMM 指令集,与 FP8/FP16/BF16/TF32 同一条编程模型演进线。工程含义:

- GEMM 算子(含 warpgroup 级流水)可以在 FP4 输入下给出**与 FP8 同构的 tiling/epilogue 写法**,而不是走「反量化到 FP16 再 GEMM」的模拟路径;
- 累加器仍需高精度(通常 FP16/FP32 accum),低比特只作用于操作数——这是所有 FP8/FP4 GEMM 的共同前提;
- 具体指令、smem 布局与 atom 规则见 CUTLASS 的 Blackwell 部分([CUTLASS tcgen05](../../cutlass/concepts/blackwell-tcgen05.md))与 [Hopper WGMMA](../../cutlass/concepts/hopper-wgmma.md) 的对照演进。

### 4.3 与 QuIP# / AQLM 的分野(务必分清)

| 维度 | 硬件原生 FP4(NVFP4/MXFP4) | 算法压缩 2-bit(QuIP#/AQLM) |
|---|---|---|
| 权重表示 | E2M1 + 共享 scale,规则稠密 | 码本/格 + 索引,非规则布局 |
| 计算路径 | Tensor Core dense GEMM | 专用 CUDA kernel / 查表反量化 |
| 目标 | 训练+推理全链路吞吐 | 极限权重显存压缩(weight-only) |
| 精度下限 | 4-bit,配合细 scale 可逼近 8-bit 体感 | 2-bit,PPL 退化可感知(3-bit 才是甜点) |
| 生态 | CUDA/cuBLAS/TE/推理引擎主流路径 | 依赖仓库自带 kernel,引擎集成有限 |

一句话:**FP4 是「硬件给了新数制」,QuIP#/AQLM 是「算法把 2-bit 塞进旧硬件」**。两者解决的问题不同,不是同一条升级路径。细节见 [2-bit 权重量化前沿](two-bit-weight-frontier.md)。

## 5. 训练 vs 推理

### 5.1 训练:DeepSeek-V3 FP8 recipe 要点

DeepSeek-V3(arXiv:2412.19437)是「大规模 FP8 训练可行」的标志性公开 recipe,可概括为四条互相咬合的设计:

1. **细粒度量化**(权重 block 128×128、激活 per-token-per-group 1×128):避免 per-tensor scale 被 outlier 击穿;
2. **在线量化**(online quantization):每个 GEMM 当场从 BF16 激活 cast 到 FP8,不依赖陈旧的离线 scale;
3. **提高累加精度**:Tensor Core 低比特累加容易在长 K 维上丢精度,通过更高精度累加/更细的累加分段缓解;
4. **调度与格式分工**:不同张量(权重、激活、梯度、KV)按敏感度选格式与粒度,而不是全局一刀切。

> 这套 recipe 的工程验证在 DeepGEMM 与配套通信/并行栈里;它证明的不是「FP8 无损」,而是「**scale 粒度 + 累加精度 + 在线统计** 三件套可以支撑生产级训练」。

训练侧的混合精度梯子通常长这样,不要指望某一层单独降位:

```
optimizer state / master weights : FP32
前向权重与激活                   : FP8 (E4M3) + 细粒度 scale
反向梯度                         : FP8 (E5M2/E4M3) 或 BF16
累加器                           : FP16/FP32
```

每一层的「降位」都由上下层的高精度托底;去掉 master weights 或累加精度,FP8 训练会迅速发散。这也是 FP8 训练与「FP8 推理权重量化」的分界:前者是全程受控的混合精度,后者才是 checkpoint 级的一次性转换。

MS-AMP 的贡献在另一侧:把 FP8 从 GEMM 延伸到 optimizer state(动量、方差、master weights 分层),对显存吃紧的训练是独立的加分项。

### 5.2 推理:FP8 权重与 FP8/FP4 KV cache

- **权重 FP8**:主流推理引擎可直接加载 FP8 checkpoint 或在线从 BF16 量化;kernel 成熟度远高于 4-bit weight-only 路线,适合高吞吐服务。
- **KV cache FP8/FP4**:长上下文瓶颈在 KV 显存与带宽,而不是权重。FP8 KV(E4M3 或 E5M2,取决于对下溢的敏感度)已是 vLLM 等引擎的标准选项;更激进的 FP4/混合精度 KV 属于前沿,注意与注意力 kernel 的累加路径匹配。参见 [vLLM GPTQ/AWQ/FP8](../../vllm/concepts/quantization-gptq-awq-fp8.md)、[KV Cache 量化](kv-cache-quantization.md)、[FlashMLA](../../deepseek-infra/concepts/flashmla.md)(其低精度 KV 路径)。
- **服务侧易错点**:FP8 权重 + BF16 KV、FP8 权重 + FP8 KV 是两条不同精度曲线;不要在没有质量回退数据时把两者一起打满。

## 6. 选型 checklist

按顺序问六个问题:

1. **硬件代际**:Hopper(H100/H200/…)→ FP8 是默认加速路径;Blackwell(B100/B200/GB200/…)→ 可评估 FP4/NVFP4;Ampere 及更早 → FP8 只能软模拟,优先 INT8/INT4。
2. **训练还是推理**:训练 → FP8 + BF16 master + 细粒度 scale;推理吞吐 → FP8 权重/激活;推理显存极限 → 4-bit weight-only 或 KV 量化。
3. **Scale 粒度是否匹配 outlier**:激活离群明显 → 至少 per-token 或 block-wise;权重 → per-channel/group 已够。
4. **Kernel 成熟度**(最重要的一条):
   - FP8:cuBLASLt / Transformer Engine / DeepGEMM / 引擎内置 FP8,基本可用;
   - FP4:确认目标卡是 Blackwell 且引擎确实编译了 FP4 路径,不要只看 dtype 选项;
   - INT8/INT4:确认 backend 的 zero-point、group size、layout 约定与 checkpoint 一致。
5. **质量回退路径**:保留 BF16 对照;记录 PPL/下游指标、TTFT、ITL、显存峰值;FP8→FP4 通常是精度拐点,需要单独实验而非「再省一半」的线性外推。
6. **累加与融合**:GEMM 累加精度、epilogue 融合、KV 累加路径三处任一用错,低比特收益会被精度噪声吞掉。

**一句话决策**:

- 想要「今天就能上线的吞吐」→ **FP8**(Hopper+),scale 至少 per-channel/per-token;
- 想要「Blackwell 上的下一跳」→ **FP4/NVFP4**,先确认 kernel 与 scale layout;
- 只为塞显存、且卡上没有低比特 Tensor Core → GPTQ/AWQ 或 2-bit 算法压缩,不指望 FP4;
- 长上下文瓶颈 → 单独评估 KV 量化,不要与权重格式绑死。

## 7. 常见误区

- **「FP8 就是 8-bit 的一半精度,FP4 再减半」**——错误。浮点网格非均匀,scale 粒度不同的话,FP4 与 FP8 的质量差距既不是线性也不是固定的;FP4 的有效精度几乎完全由块大小和累加路径决定。
- **「E5M2 范围大所以更安全,前后向都用它」**——方向反了。E5M2 只有 2 bit 尾数,前向权重/激活的相对误差会明显放大;它适合的是动态范围大、精度要求低的梯度。
- **「checkpoint 转成 FP8 就走上了 FP8 加速」**——不一定。若 engine 回退到反量化+BF16 GEMM,或 scale layout 不匹配,你只是换了一个更大的文件。必须 profile 确认 GEMM 指令路径。
- **「MXFP4 与 NVFP4 可以互转」**——不能默认。块大小(32 vs 16)、scale 数值格式(E8M0 vs FP8)、双层 scale 约定都不同,跨格式转换等于重量化。
- **「有了 FP8 就不需要管 KV cache」**——权重与 KV 是两套预算。长上下文下 KV 显存/带宽仍可能是主导项,需要单独选 FP8/低比特 KV 策略。
- **「细粒度 scale 免费」**——不免费。它要求 GEMM kernel 内联按块还原,且训练时的 scale 同步/在线 cast 会占用额外带宽;用通用 cuBLAS 模拟细粒度路径通常得不偿失。

## 8. 实践要点

1. **先测 scale,再测 bit**:同为 FP8,per-tensor 与权重 128×128 block + 激活 1×128 分组的质量差距可能大于 E4M3/E5M2 的格式差距。
2. **amax 统计是隐蔽的正确性来源**:delayed scaling 的历史长度、是否在序列边界刷新、是否用 FP32 记录 amax,都要写进实验记录。
3. **格式与 kernel 必须成对出现**:checkpoint 是 FP8 不等于服务走了 FP8 Tensor Core;用 profiler 确认 GEMM 真的走低精度指令。
4. **溢出/下溢监控**:E4M3 上限 448、下限 2⁻⁶,日志里保留 scale 与被 clip 元素比例,比事后看 PPL 更早发现问题。
5. **不要混用规范**:OCP MXFP4(块 32、E8M0 scale)与 NVFP4(块 16、FP8 scale)layout 不同;转换工具、引擎、硬件三者对齐前,先做逐元素往返测试。
6. **累加精度单独验收**:在长 K 维 GEMM(注意力、大 hidden 的 MLP)上做 FP8 与 BF16 的数值对齐实验;若误差随 K 线性放大,优先修累加而不是修 scale。
7. **保留「格式回退」开关**:服务侧至少保留 BF16 与 FP8 两条路径;FP4 上线前用同一套请求流量做 A/B,不要只看离线 benchmark。
8. **实验记录最小集**:卡型号与 SM 版本、格式(E4M3/E5M2/E2M1)、scale 粒度、amax 策略、GEMM 库与版本、质量指标、TTFT/ITL/吞吐、显存峰值。

## 相关条目

- [PTQ:GPTQ / AWQ / SmoothQuant](ptq-gptq-awq-smoothquant.md)——INT8/4-bit 自定义 scale 路线,与 FP8 硬件原生路线互补
- [2-bit 权重量化前沿:QuIP# / AQLM](two-bit-weight-frontier.md)——算法压缩路线,与 FP4 的分野见本文 §4.3
- [vLLM GPTQ / AWQ / FP8](../../vllm/concepts/quantization-gptq-awq-fp8.md)——引擎侧 FP8 权重/KV 的支持矩阵
- [KV Cache 量化](kv-cache-quantization.md)——FP8/低比特 KV 的系统收益
- [CUTLASS Blackwell tcgen05](../../cutlass/concepts/blackwell-tcgen05.md)——FP4 dense GEMM 的指令层落点
- [DeepGEMM(NVIDIA vs Ascend)](../../deepseek-infra/concepts/deepgemm-nvidia-vs-ascend.md)——DeepSeek 细粒度 FP8 GEMM 实现
- [FlashMLA](../../deepseek-infra/concepts/flashmla.md)——低精度 KV 注意力路径

## 溯源

- 主源 ↑primary:DeepSeek-V3 Technical Report(arXiv:2412.19437)§5.3 实际表述:激活按 1×128 tile(per token per 128 channels)分组缩放、权重按 128×128 block 分组缩放,KV cache scale 块 128×128——细粒度 FP8 训练 recipe 依此展开。
- 格式规范:OCP Microscaling Formats (MX) Specification v1.0(2024-02),MXFP8/MXFP4/MXINT8 的共享指数与块定义。
- 官方文档:NVIDIA Transformer Engine(docs.nvidia.com/deeplearning/transformer-engine/),E4M3/E5M2 分工、delayed/JIT scaling、FP8 训练接口。
- 代码:DeepGEMM(github.com/deepseek-ai/DeepGEMM)、MS-AMP(github.com/Azure/MS-AMP)——细粒度 FP8 GEMM 与 FP8 optimizer state 的工程实现。
- 硬件路径:Blackwell tcgen05 / FP4 Tensor Core 以 CUTLASS 官方文档与硬件手册为准;NVFP4 与 OCP MXFP4 的 layout 差异以厂商规范为准,本文不做数值互换假设。
- 发表状态:DeepSeek-V3 技术报告 2024-12;OCP MX 规范 2024-02 定稿;FP8/FP4 生态仍在快速演进,涉及 kernel 的结论以 2026-10 实测为准。
