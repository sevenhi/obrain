---
title: LSTM 低秩压缩 — 低秩分解 vs 剪枝的两条压缩路线与"该压哪个矩阵"
type: concept
tags: [compression, low-rank, svd, semi-nmf, pruning, lstm, elmo, matrix-factorization]
source:
  - "pdf: On the Effectiveness of Low-Rank Matrix Factorization for LSTM Model Compression(Winata/Madotto/Shin/Barezi/Fung, HKUST CAiRE, ↑primary) | arXiv:1908.09982v1 (PACLIC 2019,arXiv comments 栏实测标注) | ~/zhanbin/library/papers/On the Effectiveness of Low-Rank Matrix Factorization for LSTM Model Compression.pdf"
  - "web: arXiv 摘要页 | https://arxiv.org/abs/1908.09982 (2026-09 实测 200)"
  - "web: AWD-LSTM 基线代码(Salesforce,论文实验用) | https://github.com/salesforce/awd-lstm-lm (2026-09 实测 200)"
  - "web: 方法脉络交叉核对 — Deep Compression(剪枝+量化+Huffman, Han et al.) | arXiv:1510.00149;Jaderberg et al. 低秩卷积加速 | BMVC 2014"
date: 2026-09
---

# LSTM 低秩压缩:低秩分解 vs 剪枝,该压哪个矩阵

> 定位:本篇是 quantization 域"量化与压缩"扩展的**第二条压缩路线**——量化(SpQR/HyQuant:改数值表示)之外,**低秩分解/剪枝(改矩阵结构)**。LSTM 时代的这套实证结论(哪个矩阵天生低秩、MF 和剪枝怎么选)是理解后来 LLM 权重压缩(SpQR 的 outlier 隔离、LoRA 的低秩假设)的前史。

## TL;DR

Winata et al.(PACLIC 2019)对 LSTM 的两个大矩阵做后处理压缩(不重训、可选微调):**W_i(输入矩阵,加法性 recurrence)和 W_h(隐状态矩阵,乘法性 recurrence)**,比较截断 SVD、Semi-NMF 与剪枝。三个可迁移的结论:①**低秩分解总体优于剪枝**(除非矩阵本身极稀疏);②**压 W_h 远比压 W_i 安全**——W_h 天生低秩(nuclear norm 约为 W_i 一半),r=10 的极端压缩下 W_h 只掉少量 PPL 而 W_i 直接崩(PTB 58→462);③**核范数(nuclear norm)可作"该不该低秩压缩"的先验指标**。配合微调,~98% 压缩率下 PTB 推理提速 ~2×,精度甚至略超基线;常规配置平均 1.5× 提速只掉 ~1 个点。

一句话心法:**压缩不是均匀砍——先问矩阵的谱结构(核范数/奇异值衰减),把预算砍在"天生低秩"的部件上;MF 保 salient 信息、剪枝保稀疏性,按矩阵特性选工具**。

---

## 1. 机制原理(为什么 W_h 比 W_i 好压)

### 1.1 LSTM 的两个大矩阵

```
(i_t, f_t, o_t, ĉ_t)ᵀ = (σ,σ,σ,tanh)( W_i·x_t + W_h·h_{t-1} )
c_t = f_t ⊙ c_{t-1} + i_t ⊙ ĉ_t          # 加法性通路(W_i 主导)
h_t = o_t ⊙ tanh(c_t)
W_i ∈ R^{4n_inp×4n_dim}(输入),W_h ∈ R^{4n_dim×4n_dim}(隐状态)
```
术语:W_i = **additive recurrence**,W_h = **multiplicative recurrence**(沿用 Quasi-RNN 一文叫法)。

### 1.2 低秩分解为什么省

W(m×n)mn 参数/mn flops → U(m×r)·V(r×n)只要 r(m+n) 参数与 flops;r≪m,n 时线性级下降。**与剪枝的关键区别:分解后的 U·V 是稠密小矩阵,普通 GEMM 直接吃,不需要稀疏矩阵专用 kernel**——这是作者主张 MF"有保底算法加速"的理由(剪枝的实际加速依赖稀疏实现)。

### 1.3 两种 MF + 一种剪枝

- **截断 SVD**:W=USV 取 top-r,U/V 正交、S 对角;Frobenius 误差意义下最优(Eckart-Young),对符号无约束;
- **Semi-NMF**:W± ≈ U±·V⁺,只约束 V 非负(U、W 可混号),坐标下降交替更新;比 NMF 适配神经网络的混号权重;
- **剪枝**:按幅值 mask 小权重为零,剪枝率对齐低秩的压缩比 r(m+n)/mn,便于公平比较。

### 1.4 因果解释:核范数 → 可压缩性

作者收集 L1 norm(及其 std,刻画稀疏性)与 nuclear norm(≈秩的凸代理)做归因:
- **W_h 的核范数一致低于 W_i**(PTB 上约为其一半)→ W_h 天生更接近低秩 → 低秩近似丢的信息少;即使 r=10,W_h 的 L1 也几乎不掉;
- **MF vs 剪枝在 W_i 上分化**:r≤200 时剪枝使 σ(‖W_i‖₁)低于未压缩基线(丢 salient 信息),MF 则单调保留(把大值推更大、噪声推更零)→ 高压缩比下 MF 胜;
- **W_h 上剪枝反而常更好(ELMo 场景)**:W_h 本身稀疏、值近零者多,剪枝"零掉本来就接近零的",L1 稳、std 升(保住 salient);
- 推论:**核范数小 → 各种压缩方法都表现更好**;选方法前先看谱,不是先跑实验。

## 2. 可复现细节

### 2.1 因子化 LSTM 单元结构

每个门的 (W_i, W_h) 对各自换成 (U_i·V_i, U_h·V_h):
```
前向:x_t → V^g_i → U^g_i ─┐
      h_{t-1} → V^g_h → U^g_h ─┴→ 求和 → 门激活(σ/tanh)→ 标准 LSTM 组合
g ∈ {i, f, o, c};r 为共享秩超参
```

### 2.2 压缩效率度量(论文式 10)

```
E(r) = R(M, M_r) / R(P, P_r),  R(a,b) = (a-b)/max(a,b)
# 性能损失比 ÷ 参数压缩比;越小越好,负值 = 压缩后反而更好
```
(PPL 场景分子取反方向,因为越低越好。)

### 2.3 实验配置

- LM:AWD-LSTM 3 层(Merity et al. 2018 官方代码),PTB + WikiText-2;PTB 压缩后微调若干 epoch;
- 下游:预训练 **ELMo(2 层 biLSTM)只做 ELMo 层压缩**,接 BiDAF(SQuAD)/ESIM(SNLI)/BCN(SST-5);ELMo 占各模型参数 55-67%;
- 秩档位 r=10(极端)与 r=400(温和),5 次平均;测速环境 Intel Xeon E5-2620 v4(CPU 推理)。

## 3. 数据对比

**PTB 语言模型(基线 AWD-LSTM 24M 参数,PPL 58.3)**:

| 方法 | 参数 | PPL(无微调) | PPL(微调) |
|---|---|---|---|
| SVD W_h, r=10 | 9M | 78.07 | 58.18 |
| Pruning W_h, r=10 | 9M | 83.62 | **57.94**(超基线) |
| SVD W_h, r=400 | 18M | **59.34** | 57.81 |
| SVD W_i, r=10 | 15M | 462.19(崩) | 88.12 |
| SVD W_i, r=400 | 20M | 60.59 | 58.04 |

- **W_h vs W_i 差距悬殊**:r=10 时压 W_h PPL 78(微调回 58),压 W_i PPL 462——同压缩比下差 6 倍;
- WikiText-2 同趋势(W_h r=10:SVD 99.92 vs W_i r=10:SVD 443.49);
- 全部方法优于 TT-LSTM(Tensor-Train,12M 参数 PPL 168.6);
- **ELMo 下游**(r=10 极端压缩):SNLI 上压 W_h 仅 88.6→87.2(-1.4);SQuAD 反而"压 W_i 更好"(F1 81.78 ≈ 未压 ELMo 的 81.75)——**任务会翻转 W_i/W_h 的相对重要性,不是普适定律**;SST-5 剪枝 W_h r=400 达 54.66,E(r)=-0.03(超过基线 53.7);
- 速度:常规配置(压加法+乘法 recurrence)平均 **1.5× 推理提速、掉 ~1 点指标**;PTB 微调后 ~98% 压缩率 **2.13× 提速且 PPL 略优**;
- 对比后来者:此结论"结构先验决定可压缩性"在 Transformer 时代由 LoRA(微调侧低秩)、SpQR(敏感度驱动的非均匀预算,见同域 spqr 篇)分别继承。

## 4. 实践要点

1. **压缩前先测谱**:算目标矩阵的核范数/奇异值衰减(或 σ(‖W‖₁) 随 r 的曲线),核范数小、衰减快 → 低秩分解优先;矩阵本身稀疏、近零值多 → 剪枝可能更优(ELMo W_h 实例)。
2. **别均匀压所有部件**:同一网络里不同角色的矩阵可压缩性差数倍(W_i vs W_h);先找"天生低秩"的部件下手,压缩比按敏感度分配——与 SpQR 把 bit 预算按敏感度分配是同一哲学。
3. **微调是低成本放大器**:极端压缩(r=10)后短暂微调可完全恢复甚至超过基线(PTB 57.94 < 58.3),代价远小于重训;对预训练大模型(当年 ELMo、今天 LLM)这是后处理压缩可行的前提。
4. **MF 的部署优势**:U·V 是稠密 GEMM,无稀疏 kernel 依赖,加速有保底;剪枝的理论压缩比要靠稀疏实现兑现(呼应 SpQR:outlier 稀疏部分必须自己写 CSR kernel 才有速度)。
5. **结论的边界**:任务可翻转 W_i/W_h 优劣(SQuAD);本文只压 ELMo/单层 LM,全模型联合分解与"MF+剪枝混合"作者列为 future work。年代注意:2019 年 CPU 推理场景,数字不代表今天 GPU 上的延迟结构。

## 5. 溯源与多源综合说明

- 主源 ↑primary:本地 PDF(arXiv:1908.09982v1, 2019-08-27);发表状态以 arXiv comments 栏为准:**"Accepted in PACLIC 2019"**(2026-09 实测 arXiv 摘要页;PACLIC 33 = 2019-09 于日本函馆,会议 program 存活核对)——网传"EMNLP-IJCNLP 2019"无据,勿引用;
- 基线代码:github.com/salesforce/awd-lstm-lm(2026-09 实测存活);ELMo 权重来自 allennlp.org/elmo;
- 方法脉络交叉核对:剪枝+量化+Huffman 的"压缩三件套"出自 Han et al. Deep Compression(arXiv:1510.00149,本文引其剪枝方法);低秩加速 CNN 出自 Jaderberg et al. BMVC 2014;Semi-NMF 出自 Ding et al. TPAMI 2010;TT-LSTM 对比数字出自 Grachev et al. 2017(本文 Table 1 转录)。
