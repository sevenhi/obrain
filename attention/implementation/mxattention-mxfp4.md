---
title: MXAttention：MXFP4 注意力的 UOS 缩放与 PNQ 归一化
type: concept
tags: [attention, flashattention, mxfp4, fp4, microscaling, online-softmax, quantization, uos, pnq, hadamard, video-diffusion]
source:
  - "arXiv: MXAttention: Data-Free Optimal Scaling and Pre-Normalization Quantization for MXFP4 Attention | https://arxiv.org/abs/2607.24377 | 2026-11-19 | primary"
  - "code: Huawei MindIE-SD（MXAttention 实现） | https://gitcode.com/Ascend/MindIE-SD/tree/master/mindiesd | 2026-11-19 | implementation"
  - "pdf: FlashAttention-2 | https://arxiv.org/abs/2307.08691 | 2026-11-19 | online-softmax baseline"
  - "web: OCP Microscaling Formats (MX) Specification v1.0 | https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf | 2026-11-19 | format"
date: 2026-11
reviewed: 2026-11-19
---

# MXAttention：MXFP4 注意力的 UOS 缩放与 PNQ 归一化

> **速览**：MXAttention 是面向视频扩散 Transformer 注意力的 data-free MXFP4 PTQ。它把两个问题拆开解决：UOS（Universal Optimal Scaling）利用 E8M0 二次幂缩放的 log-periodic 性质，把 E2M1 的全局 MSE 目标化成与数据分布无关的固定边界 `Qmax = 7.25`；PNQ（Pre-Normalization Quantization）在 FlashAttention 的 online-softmax 中，先量化未归一化指数，再用同一份量化 tile 更新行和与输出累加器，从结构上保持行归一化。论文在 Wan2.2、HunyuanVideo 上报告至少闭合 95% 的 OCP MXFP4→FP16 Imaging Quality 差距，但 Wan2.2 主比较保留了 Block 0 与最后两步的高精度回退，不能把该数字解读为“所有步骤全 4-bit 无损”。

## 1. 背景：为什么量化 attention

缩放点积注意力为

\[
S=QK^\top/\sqrt d,\qquad P=\operatorname{softmax}(S),\qquad O=PV.
\]

`QKᵀ` 和 `PV` 是两个大 GEMM：前者生成注意力分数，后者把概率权重乘到 `V`；因此在支持原生 FP4 Tensor Core 的设备上，它们是最直接的低精度加速目标。FlashAttention 不把 `S`、`P` 写入 HBM，而是按 `Q_i`、`(K_j,V_j)` tile 流式计算。

### 1.1 MXFP4 与 NVFP4

- **MXFP4（OCP MX）**：每 32 个 E2M1 元素共享一个 E8M0 scale `σ=2^e`。E2M1 的有限非负网格为 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`，最大有限值是 6。
- **NVFP4（NVIDIA Blackwell）**：每 16 个元素共享 E4M3 block scale，另有 tensor-level scale。粒度更细、scale 不是二次幂，代价是元数据和格式路径不同。

两者都不是“4 bit 只少一半精度”的简单关系：块大小、scale 表示和 kernel layout 共同决定误差。MXAttention 针对的是 OCP MXFP4 的 32 元素/E8M0 约束，不能把 `Qmax=7.25` 直接移植到 NVFP4。

### 1.2 FlashAttention online-softmax

对 query tile `Q_i` 和 KV tile `K_j,V_j`，令

\[
S_{ij}=Q_iK_j^\top/\sqrt d,
\quad m_i^{(j)}=\max(m_i^{(j-1)},\operatorname{rowmax}(S_{ij})),
\]
\[
\alpha_i^{(j)}=\exp(m_i^{(j-1)}-m_i^{(j)}),\quad
\widetilde P_{ij}=\exp(S_{ij}-m_i^{(j)}\mathbf1^\top).
\]

FlashAttention-2 维护行和 `ℓ` 与**未归一化**输出累加器 `\widetilde O`：

\[
\ell_i^{(j)}=\alpha_i^{(j)}\odot\ell_i^{(j-1)}+\widetilde P_{ij}\mathbf1,
\]
\[
\widetilde O_i^{(j)}=\operatorname{Diag}(\alpha_i^{(j)})\widetilde O_i^{(j-1)}+\widetilde P_{ij}V_j,
\qquad O_i=\operatorname{Diag}(\ell_i^{(T)})^{-1}\widetilde O_i^{(T)}.
\]

这套递归把归一化推迟到末尾，既省掉 `N×N` 中间矩阵，也为低精度 tile 融合留下了插入点。

## 2. 直接 MXFP4 的两个失效模式

### 2.1 幂次共享 scale 的 clipping–underflow trade-off

块最大值 `M=max|v|` 按 OCP floor 规则选 scale 后，归一化最大值 `x=M/σ` 落在 `[4,8)`。但 E2M1 的最大有限值是 6；在 `(7,8)` 中，未截断的 E2M1 梯子会向 8 舍入，有限格式只能饱和到 6。若 log-domain wrapped phase 近似为 `p(x)=1/(x ln2)`，溢出舍入概率为

\[
\Pr[x\ge7]=\int_7^8\frac{dx}{x\ln2}=\log_2(8/7)\approx19.27\%.
\]

把边界降到 `Qmax=6` 的 TFS（Truncation-Free Scaling）可避免饱和，但会把整个块的有效值除以更大的 scale：小值更容易 rounding-to-zero/underflow。OCP 偏向保留小值、接受少量 clipping；TFS 偏向不溢出、牺牲小值分辨率。这个冲突是 MXFP4 特有的 scale 选择问题，不是简单换一个舍入模式即可解决。

### 2.2 量化插入 online-softmax 破坏行和

直接 baseline 用未量化 `\widetilde P` 更新 `\ell`，却用 `\widehat P=Q(\widetilde P)` 更新输出：

\[
\ell\leftarrow\alpha\ell+\widetilde P\mathbf1,\qquad
\widetilde O\leftarrow\operatorname{Diag}(\alpha)\widetilde O+\widehat P V.
\]

两条路径代表的权重不同。把所有 tile 按后续 max-rescale 合并成有效整行 `\widetilde P_{\rm eff}` 与 `\widehat P_{\rm eff}`，最终诱导权重为

\[
P_{\rm direct}=\operatorname{Diag}(\widetilde P_{\rm eff}\mathbf1)^{-1}\widehat P_{\rm eff},\quad
P_{\rm direct}\mathbf1=\widehat P_{\rm eff}\mathbf1\oslash(\widetilde P_{\rm eff}\mathbf1)\ne\mathbf1.
\]

因此 zeroing、saturation、上下舍入会变成**行相关的输出缩放误差**，而不是一个可用全局因子修正的问题。论文在 Wan2.2 的直接在线 baseline 实测平均行和为 `0.9336`（5 个 denoising steps、2 prompts、40 layers；单行范围 `0.7181–1.1127`）。另一个 materialized-probability 诊断的 80 个 tensor 平均为 `0.9266`；两者统计口径不同，不能混为同一个数字。

## 3. UOS：不校准的最优边界

### 3.1 目标函数与周期结构

对非负 E2M1 网格 `G⁺={0,0.5,1,1.5,2,3,4,6}`，令 `Π` 为带饱和的 round-to-nearest 投影，定义累计投影误差及归一化误差：

\[
E(x)=\int_0^x(v-\Pi_{G^+}(v))^2dv,\qquad D(x)=E(x)/x^3.
\]

采用 ceiling scale family，`q=Qmax`，

\[
e_q(M)=\left\lceil\log_2(M/q)\right\rceil,\quad \sigma_q=2^{e_q(M)},\quad X_q=M/\sigma_q\in(q/2,q].
\]

全局 data-free 目标是 `J(q)=E[D(X_q)] = ∫_{q/2}^q D(x)g_q(x)dx`。令 `U=log₂M`，幂次折叠得到周期化密度 `H(t)=Σ_k f_U(t+k)`，满足 `H(t+1)=H(t)`；`q/2` 与 `q` 的 log₂ 坐标恰好相差一个周期。因此边界密度满足

\[
\boxed{g(q/2)=2g(q)}\qquad\text{(Lemma 1)}.
\]

### 3.2 分布消去与闭式解

对 `J` 用 Leibniz 法则并代入 Lemma 1：

\[
J'(q)=D(q)g(q)-\tfrac12D(q/2)g(q/2)
      =g(q)[D(q)-D(q/2)].
\]

所以 block-maximum 的实际分布只影响导数幅度，不影响符号；最优点由 E2M1 网格决定。令 `Δ(q)=E(q)-8E(q/2)`，则 `D(q)-D(q/2)=Δ(q)/q^3`。在关键区间 `[6,8]` 精确积分得

\[
\Delta(q)=
\begin{cases}
-1/8,&6\le q\le7,\\
(4q-29)(4q-27)/8,&7\le q\le8.
\end{cases}
\]

唯一有效变号点是 `q*=29/4=7.25`；附录把符号延拓到 `q>0`，从而得到全局最优（在周期密度几乎处处为正时唯一）。注意：这是对解析的归一化投影误差目标的 distribution-independent 最优，不等于每个有限样本 tensor 的 MSE 都严格在 7.25；论文的 per-block sweep 只是经验支持。

### 3.3 UOS 算法

对每个 32 元素块，`M=max|v|`：全零块用 `σ=1`；否则

```text
e = ceil(log2(M / 7.25));  σ = 2^e
for v in block:  p = round_to_E2M1(v / σ)  # 超过 6 饱和
```

归一化最大值范围为 `(3.625,7.25]`，只保留窄的 `(7,7.25]` overflow-rounding 区间，相比 OCP `[4,8)` 大幅缩小，又比 TFS `(3,6]` 给小值更多分辨率。UOS 只改变 exponent 选择，不增加校准、逐层搜索或额外 block traversal。

## 4. PNQ：先量化，再共同归一化

PNQ 使用同一个 UOS quantizer `Q*`，但改变量化**位置**：先算 `\widehat P=Q*(\widetilde P)`，再把它同时送入两个状态更新：

\[
\ell\leftarrow\alpha\ell+\widehat P\mathbf1,\qquad
\widetilde O\leftarrow\operatorname{Diag}(\alpha)\widetilde O+\widehat P V.
\]

完整 tile 递归为

\[
\ell_i^{(j)}=\alpha_i^{(j)}\odot\ell_i^{(j-1)}+\widehat P_{ij}\mathbf1,\quad
\widetilde O_i^{(j)}=\operatorname{Diag}(\alpha_i^{(j)})\widetilde O_i^{(j-1)}+\widehat P_{ij}V_j.
\]

把每个 tile 按后续 rescale 因子 `\beta_i^{(j)}=\exp(m_i^{(j)}-m_i^{(T)})` 展开，概念上的整行量化权重 `\widehat P_{\rm eff}` 满足

\[
\ell=\widehat P_{\rm eff}\mathbf1,\quad \widetilde O=\widehat P_{\rm eff}V,\quad
P_{\rm PNQ}=\operatorname{Diag}(\widehat P_{\rm eff}\mathbf1)^{-1}\widehat P_{\rm eff}.
\]

于是 `P_PNQ 1=1`（精确算术）。这不是说量化权重等于 FP16 softmax，而是保证量化后的**自身**归一化，不再叠加“行和走未量化、输出走量化”的结构性比例误差。论文附录还说明：包含行最大值的 tile 中有 `\widetilde P=1`；在 `q=7.25` 时 `1/σ=4` 可精确表示，故 `\ell>0`。真实浮点累加和最终除法仍有普通舍入误差。

## 5. 融合流水线与工程边界

1. 对 Q、K 应用固定正交 Hadamard rotation `R_H`（`Q←QR_H,K←KR_H`）摊平 outlier；V 不旋转。对应逆变换折入模型权重。
2. Q/K/V 用 UOS (`q=7.25`) 量化；计算 `S=Q̂K̂ᵀ/√d`，按 FlashAttention tile 更新 running max。
3. 生成未归一化指数 `\widetilde P`，立即 UOS 量化为 `P̂`；PNQ 用该 tile 同时更新 `ℓ` 与 output accumulator。
4. 所有 KV tile 结束后做 `O=Diag(ℓ)^{-1}\widetilde O`，输出回 FP16/BF16。

UOS 不改变 block reduction、tile traversal 或 GEMM 形状；PNQ 复用本来就要给 `PV` 的量化 tile，不增加第二次量化或 attention-matrix materialization。Hadamard 与量化可在 Q/K 预处理阶段融合，论文称算法开销可忽略、适合 fused kernel。实现仍需确认目标设备原生 MXFP4 指令、E8M0 layout、累加精度和 fallback 策略；论文没有在本机复现性能。

## 6. 实验：质量收益如何解读

模型为 Wan2.2-14B（81 frames、720p、40 steps）与 HunyuanVideo-13B（129 frames、720p、50 steps），指标为 VBench Subject/Imaging/Aesthetic，以及相对 FP16 的 cosine、SSIM、PSNR。

| 模型/方法 | Imaging | Aesthetic | Cosine | SSIM | PSNR |
|---|---:|---:|---:|---:|---:|
| Wan FP16 | 0.7085 | 0.6042 | — | — | — |
| Wan OCP MXFP4 | 0.6414 | 0.6202 | 0.9290 | 0.5076 | 15.58 |
| Wan MXAttention | 0.7054 | 0.6229 | 0.9536 | 0.6319 | 17.92 |
| Hunyuan FP16 | 0.6185 | 0.6242 | — | — | — |
| Hunyuan OCP MXFP4 | 0.4459 | 0.5928 | 0.9489 | 0.5954 | 16.25 |
| Hunyuan MXAttention | 0.6380 | 0.6344 | 0.9745 | 0.7061 | 19.23 |

主表中 Wan 的所有 4-bit 方法共享 hybrid policy：Block 0 和 denoising steps 38–39 保持高精度；HunyuanVideo 主比较是全 attention 4-bit。按 Imaging gap 计算，Wan 恢复 `(0.7054−0.6414)/(0.7085−0.6414)=95.4%`，Hunyuan 超过 FP16（因此“闭合”超过 100%）。全 4-bit sequential ablation 仍显示三组件互补：Wan full 的 Imaging 为 0.6842，去 PNQ/UOS 为 0.6352；Hunyuan 为 0.6380 对 0.5321。不能把主表的高精度回退隐藏掉，也不能把 VBench 结果当成硬件吞吐实测。

## 7. 与相关工作的关系

- **SageAttention**：做 Q/K smoothing、softmax-path scaling 和 kernel 优化；MXAttention 不依赖校准 scale，核心是 MXFP4 固有的幂次边界与 online-softmax 一致性。
- **QuaRot / SpinQuant**：Hadamard/正交旋转抑制激活 outlier；MXAttention 借用固定 Q/K rotation，但进一步处理 E2M1+E8M0 的 scaling 与 PNQ，旋转本身不能解决 clipping/underflow 或行和错配。
- **QAT**：通过训练适应量化；MXAttention 是 post-training、data-free，不需要 calibration、QAT 或逐层搜索。
- **其他 MXFP4 改进**：overflow-aware scaling、adaptive scale selection、metadata augmentation 可按 block 做经验决策；UOS 则给出固定、可证明的边界。OAS 在同一 32 元素粒度下对应 ceiling `q=7`，与 UOS 的 `7.25` 不同。
- **FlashAttention 谱系**：本文继承 online-softmax 的 `(m,ℓ,acc)` 状态；可与 [FA 四代实现](flash-attention-implementation.md) 的 IO-aware tiling、FA3 FP8 pipeline 对照。MXAttention 的改变在量化位置和 scale 规则，不是另一种 attention 数学。

## 8. 复习与验收清单

- 区分三种数：OCP 溢出概率约 19.27%、直接 online baseline 行和均值 0.9336、materialized diagnostic 均值 0.9266。
- 推导 UOS 时先写 `H(t+1)=H(t)` 与 `g(q/2)=2g(q)`，再看 `J'(q)`；不要假设 log-uniform 或忽略分布 harmonic。
- 代码检查 PNQ 是否让**同一** `P̂` 更新 `ℓ` 和 `acc`；只在 `PV` 路径量化仍是 direct mismatch。
- 把“FP16-level quality”与“低延迟”分开验收：论文质量实验不等于本地 kernel benchmark，且 Wan 主表有高精度 fallback。

## 溯源

- 论文正文与附录给出 E2M1 决策边界、`Δ(q)` 分段积分、PNQ full-row lemma、实验统计及限制；本篇公式按 arXiv v1 转写。
- 代码入口：[MindIE-SD/mindiesd](https://gitcode.com/Ascend/MindIE-SD/tree/master/mindiesd)。
- 交叉阅读：[FP8/FP4 硬件原生格式](../../quantization/concepts/fp8-fp4-formats.md)（格式总览）与 [FlashAttention 实现](flash-attention-implementation.md)（online-softmax/内核谱系）。
