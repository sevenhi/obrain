---
title: CCNN — 连续卷积核:一个 CNN 通吃任意长度/分辨率/维度
type: concept
tags: [cnn, continuous-convolution, long-range-dependency, kernel-generator, implicit-representation, s4]
source:
  - "pdf: CCNN workshop 论文全文 | arXiv:2206.03398 ICML 2022 Workshop on Continuous Time Methods (Romero, Knigge, Gu et al.) | Towards a General Purpose CNN for Long Range Dependencies in ND.pdf ↑primary"
  - "web: CCNN 正式版论文 | arXiv:2301.10540 ICLR 2023 (Modelling Long Range Dependencies in ND) (抓取 2026-09-21)"
  - "web: CCNN 官方 PyTorch 实现 | https://github.com/david-knigge/ccnn (抓取 2026-09-21)"
  - "web: FlexConv(前驱,可微 kernel 大小) | arXiv:2110.08059 ICLR 2022 (抓取 2026-09-21)"
  - "web: S4(对照的长序列 SOTA) | arXiv:2111.00396 ICLR 2022 (抓取 2026-09-21)"
date: 2026-09
---

# CCNN:连续卷积核与「通用 CNN」

> 论文阅读笔记(arXiv:2206.03398,ICML 2022 Continuous Time Methods Workshop;扩展版 arXiv:2301.10540,ICLR 2023)。作者阵容含 S4 一作 Albert Gu,定位是「CNN 路线对 S4 路线的回应」。

## TL;DR

标准 CNN 的卷积核是**离散权重表**:核大小=参数量,想建模长程依赖只能堆深度+pooling,而且核绑死在特定分辨率/维度上——换个任务(音频 16k 长序列 vs 32×32 图像 vs 点云)就得重新设计架构。CCNN(Continuous CNN)把卷积核换成**坐标的连续函数**:一个小 MLP(kernel generator,GKernel: R^D → R^{Nout×Nin})输入坐标 c_i、输出该位置的核值。核大小与参数量**解耦** → 每层都能用「与输入同长」的大核直接建模长程依赖(FFT 卷积算),不需要 downsampling 和任务定制深度;坐标维度 D 改一改,同一个网络就能处理 1D/2D/3D;分辨率变了只需换一套坐标采样,核函数不变(分辨率无关)。两个关键工程改进:①深度可分离版连续卷积(通道数从 30 提到 110/380 不加复杂度);②**kernel generator 的方差修正初始化**——普通 MLP 初始化让核方差=1,而卷积要求 Var(K)=gain²/(in_ch·kernel_size),否则逐层方差爆炸(CKCNN/FlexNet 初始 logits 达 1e19 量级)。结果:同一架构在 sMNIST 99.72、Speech Commands raw 98.44(超 S4 的 98.32)、LRA PathFinder 2D 96.00(比之前 SOTA 高近 10 个点)。

## 机制原理(为什么)

### 1. 为什么离散核逼出「任务定制架构」

离散核每个位置一组独立权重 → 核长 K 的参数量 O(K·C²),大核不可行;于是标准 CNN 用 3×3 小核 + stride/pooling 逐层扩感受野。后果:
- **感受野与输入长度耦合**:784 步的 sMNIST、16000 点的 1 秒音频、32×32 的 PathFinder,各自需要不同深度/stride 配方;
- **分辨率绑定**:同一语义在 8kHz vs 16kHz 音频、低清 vs 高清图上,离散核覆盖的物理范围不同,响应就不同;
- **维度绑定**:1D/2D/3D 核是不同形状的权重张量,架构不能复用。

### 2. 连续核如何一次解掉三个绑定

核由网络生成:K = [GKernel(c_i)],c_i 是坐标。
- **参数量与核长解耦** → 核可以长到 len(x),每层直接全局感受野;大核卷积用卷积定理(时域卷积=频域逐点乘)FFT 算,O(N log N);
- **核是坐标的函数** → 分辨率变化只是坐标网格变密:Conv(K^{r2}, x^{r2}) ≈ (r1/r2)^D · Conv(K^{r1}, x^{r1})(相差一个与分辨率变化成正比的因子),同一核函数跨分辨率语义一致;非均匀采样数据(如医疗时序)也能直接算——离散核只在预设格点有值,连续核处处有定义;
- **坐标维度即数据维度** → GKernel 输入换成 R^1/R^2/R^3,同一网络处理序列/图像/点云;
- 附带能力:核大小本身可学——FlexConv 用可学习高斯掩码 K(c)=GKernel(c)·Gauss_{μ,σ}(c) 把「核多大」从超参数变成可训练参数。

这一套是 **implicit neural representation 思想用在卷积核上**(与 NeRF 用 MLP 表示场景是同一谱系:坐标→值的 MLP;见本域 [nerf-view-synthesis](nerf-view-synthesis.md))。

### 3. 关键坑:kernel generator 的初始化(为什么之前的 CKCNN/FlexNet 不稳)

卷积层要保方差:Var(y)=Var(x) 要求 Var(K)=gain²/(in_channels·kernel_size)(He 初始化)。但**MLP 的标准初始化保的是「输出方差=输入方差」**,用作 kernel generator 时给出 Var(K)=1 → 卷积输出方差逐层放大 in_ch·K 倍。实测:CKCNN/FlexNet 初始 logits ~1e19,训练不稳、被迫用小学习率。CCNN 的修法(借鉴 hypernetwork 初始化,Chang et al. 2020):**把 GKernel 最后一层权重乘 gain/√(in_channels·kernel_size)**,让核方差回到卷积初始化应有的量级,logits 初始方差归一。

### 4. 结构:向 S4 的 residual block 看齐

CCNN block = S4 式 modernized residual block(pre-norm、GELU 类非线性,Xiong et al. 2020 / ConvNeXt 路线),把 FlexNet 的块换掉;卷积用**深度可分离连续卷积**:每通道一个由 GKernel: R^D → R^{Nin} 生成的连续核(depthwise)+ 1×1 pointwise 混合通道。参数/计算复杂度与传统可分离卷积同阶,但核长任意 → **同样预算下网络可以做宽**(30 → 110/380 通道)。坐标归一化到 [-1,1]^D(否则 16000 长的音频坐标数值不稳),GKernel 用 3 层 MAGNet(多周期激活,ω0 是数据方差先验、按任务调)。

### 5. 与 S4 的路线对比

S4(结构化状态空间)同样每层全局感受野,但把序列**展平成 1D** 处理(多维数据丢失空间结构;定义多维状态空间困难)。CCNN 保留数据的原生维度(2D PathFinder 直接用 2D 核)。实验证明维度信息值钱:LRA PathFinder 当 1D 序列做,S4 86.05 / CCNN 91.51;当 2D 图像做,CCNN 96.00(且 2D 训练收敛更快)。而小核+pooling 的 ResNet-18 做不了 PathFinder(中间 pooling 毁掉细粒度全局连接)。

## 可复现细节

```
连续核卷积层(伪代码):
  # GKernel: 3 层 MAGNet(隐藏 32/64),最后一层乘 gain/sqrt(C_in·K) 初始化
  coords = linspace(-1, 1, K)          # K = len(x);坐标归一化到 [-1,1]^D
  kernel = GKernel(coords)             # [K, C_in](depthwise)或 [K, C_out, C_in]
  kernel = kernel * Gauss_{mu,sigma}(coords)   # FlexConv 可学习掩码(可选)
  y = fft_conv(kernel, x)              # 卷积定理:FFT 逐点乘,O(N log N)
  y = pointwise_1x1(y)                 # depthwise separable:再混通道

CCNN block(S4 式):
  x → Norm → 连续核 depthwise conv → GELU → Norm → pointwise conv → GELU → (+x)

模型规格:CCNN4,110(4 block/110 ch,200K 参数)/ CCNN6,380(6/380,2M 参数)
训练:AdamW + cosine + 10 epoch warmup;每任务搜 lr/dropout/wd/ω0
     (ω0 例:sMNIST 2976,MFCC 语音 750,CIFAR10-2D 1436——量级差 4 倍,必须调)
代码:github.com/david-knigge/ccnn(PyTorch Lightning + Hydra)
```

## 数据对比

同一架构(CCNN6,380,2M 参数)跨任务(论文 Table 1-4):

| 任务 | CCNN6,380 | 最强对手 |
|---|---|---|
| sMNIST(784 步像素序列) | **99.72** | S4 99.63 / LSSL 99.53 |
| sCIFAR10(1024 步) | **93.08** | S4 91.13 |
| Speech Commands raw(16k 波形) | **98.44** | S4 98.32 / WaveGAN-D(26.3M 参数)96.25 |
| Speech Commands MFCC | 97.98 | FlexTCN-6 **97.67→**CCNN 更高;S4 93.96 |
| CIFAR10(2D) | 95.20 | ResNet-18(11.2M 参数)94.92 |
| STL10 | **83.00** | ResNet-18 81.04 |
| LRA PathFinder(1D 序列) | 91.51 | S4 86.05 |
| LRA PathFinder(2D 图像) | **96.00** | (之前 SOTA ≈86,高近 10 个点) |
| LRA Image(2D) | 88.90 | S4(1D)87.26 |
| LRA 平均(可比子集) | 77.02 | S4 80.48(ListOps/Retrieval CCNN 弱) |

- **参数量对照**:CCNN4,110 只有 200K 参数,已超 500K 的 HiPPO-RNN(sMNIST 98.4)和 7.8M 的 LSSL(sCIFAR10 84.65 → CCNN 90.30)。
- **弱项要诚实**:ListOps(43.60 vs S4 58.35)和 Retrieval 上 CCNN 明显落后——连续核对「纯符号/组合推理」类序列不如状态空间模型;「平均超 S4」不成立,只在多数感知类任务占优。
- **维度红利**:2D PathFinder 96.00 vs 同任务展平 1D 的 91.51,证明保留原生维度结构有直接收益。

## 实践要点

- [ ] **跨分辨率/跨采样率/跨维度复用同一模型**是连续核的杀手级场景(医疗不规则采样、多采样率音频、1D/2D/3D 统一栈);离散核 CNN 做不到。
- [ ] **kernel generator 必须做方差修正初始化**(最后一层乘 gain/√(C_in·K)),否则 logits 爆 1e19、训练发散——这是 CKCNN→CCNN 最重要的修复,自造 hypernetwork 生成权重时同理。
- [ ] 坐标归一化到 [-1,1]^D;ω0(激活频率先验)按任务数量级调,不能一套用到底。
- [ ] 大核走 FFT 卷积;depthwise separable 是把网络做宽的前提(否则参数随核长线性涨)。
- [ ] 需要 downsampling 的任务(大规模高分辨率图像)此路线未验证;它的甜区是「长度/分辨率本身就是难点」的中小规模任务。
- [ ] 选型对照:纯长序列建模(尤其符号类 ListOps/Retrieval)优先 S4 系;数据有原生多维结构(PathFinder 型)优先 CCNN 型 2D 连续核。
- [ ] 与本域关联:连续核 = 隐式坐标表示思想在卷积上的应用,与 [NeRF 的坐标 MLP](nerf-view-synthesis.md)(位置编码/频谱偏置问题同源——GKernel 用多周期 MAGNet 正是为了生成高频核)互为印证。

## 溯源

- 主源:workshop 版论文论文原文(链接见 frontmatter);全部数字取自论文 Table 1-6。
- 网络综合:官方代码库 david-knigge/ccnn(README 指向 ICLR 2023 正式版 arXiv:2301.10540,标题 *Modelling Long Range Dependencies in ND: From Task-Specific to a General Purpose CNN*,确认 workshop→会议版扩展关系及 3D 点云新增实验);前驱 FlexConv(arXiv:2110.08059,可微核大小/高斯掩码)与对照 S4(arXiv:2111.00396,*Efficiently Modeling Long Sequences with Structured State Spaces*)arXiv abs 页核对定位。全部 URL 于 2026-09-21 curl 实测存活(200)。
- 注意:本地 PDF 是 workshop 短版(8 页),ICLR 2023 正式版含点云 3D 实验等增量,本篇结论以 workshop 版为准、正式版仅登记链接。
