---
title: OREPA — 在线卷积重参数化(训练期即合并)
type: concept
tags: [reparameterization, cnn, repvgg, dbb, training-efficiency, orepa]
source:
  - "pdf: OREPA 论文全文 | arXiv:2204.00826 CVPR 2022 (浙大 + 阿里云) | Online Convolutional Re-parameterization.pdf ↑primary"
  - "web: OREPA 官方 PyTorch 实现 | https://github.com/JUGGHM/OREPA_CVPR2022 (抓取 2026-09-21)"
  - "web: RepVGG(重参数化前驱,VGG 式训练多分支/推理单 3×3) | arXiv:2101.03697 CVPR 2021 (抓取 2026-09-21)"
  - "web: DBB/Diverse Branch Block(重参数化前驱,Inception 式多分支) | arXiv:2103.13425 CVPR 2021 (抓取 2026-09-21)"
  - "web: ACNet(非对称卷积分支前驱) | arXiv:1908.03930 ICCV 2019 (抓取 2026-09-21)"
date: 2026-09
---

# OREPA:在线卷积重参数化——训练时就把多分支块合并成单卷积

> 论文阅读笔记(arXiv:2204.00826,CVPR 2022,Hu et al.,浙大+阿里云)。重参数化(re-param)家族:ACNet(2019)→ RepVGG / DBB(2021)→ **OREPA(2022,本篇)** → RepLKNet 等大核路线。

## TL;DR

结构重参数化 = **训练用复杂多分支块、部署时等价折叠成单个 3×3 卷积**,推理零额外成本换精度(RepVGG/DBB 路线)。但它的死穴在训练端:块内每条分支后的 BN 是**非线性**的(要除以 batch 统计量),阻止训练期合并 → 大量中间特征图既费 FLOPs 又费显存,块越复杂训练越贵。OREPA 的洞见:**BN 里真正起作用的是缩放因子 γ,不是统计归一化**——γ 让各分支的优化方向多样化(否则多分支会退化成单分支)。于是把分支内 BN 全部换成**线性 scaling 层**(channel-wise 可学习向量),整块变纯线性 → **训练每一步都能先在 kernel 层面把整块合并成单卷积再前向**(online),额外训练开销从 O(H×W) 特征图级降到 O(KH×KW) kernel 级。结果:比 DBB 省约 70% 额外训练显存、快 1.5~2.3×,ImageNet 精度还高 +0.2~0.6%(OREPA-ResNet-50 77.31% vs DBB 76.89%);省下的预算还能加更强分支(频率先验滤波、线性 DW 可分离卷积、Rep-1×1、线性 deep stem)。

## 机制原理(为什么)

### 1. 重参数化为什么免费涨点:优化多样化(不是表达力)

推理时多分支折叠成单卷积,**表达力与单卷积完全等价**——涨点全部来自「训练动力学被改变」。OREPA §3.4 把这点形式化(接续 Arora et al. 2018 的多层 over-parameterization 分析,扩展到多分支):

- 端到端权重 W_e = Σⱼ γⱼWⱼ,SGD 一步更新后:
  `ΔW_e = -η·Σⱼ(vec(diag(Wⱼ)²) + ‖γⱼ‖²)·xᵀ∂L/∂y`
- **共享 γ 的多分支**(Φ_M):更新式与单分支完全相同 → 多分支白搭,无任何优化改变。
- **分支各自有 γⱼ**(Φ_{M,2}):只要 ①至少两条分支活跃、②各分支初始化不同,更新式就与单分支不同(交叉项 Σᵢ≠ⱼ γᵢγⱼ·diag(Wᵢ)diag(Wⱼ) 无法消掉)→ **Proposition 1:重参数化到「超过两层或多分支」拓扑才改变优化;单层多分支(共享尺度)不变**。
- 这解释了 BN 的作用本质:提供**分支各异的缩放**。DBB/RepVGG 去掉分支 BN 会显著掉点(DBB-18 71.77→71.35,RepVGG-A0 72.41→71.15),而 OREPA 证明:换成纯线性 scaling 层同样提供分支尺度差异(ResNet-18 上 norm 与 scaling 的增益几乎一样:0.42 vs 0.40),**统计归一化部分可以不要**。

### 2. 为什么 BN 挡住在线合并、scaling 不挡

BN 前向要除以 √(Var[x]+ε),Var 是 batch 统计量——非线性、依赖数据,块内各层无法先合并再算。channel-wise scaling(y=γ⊙Wx)是纯线性算子,满足结合律/分配律 → 任何由「线性层序列 + 并行分支」构成的块都能在 **kernel 空间**先折叠:

- **串行**:W_N∗(W_{N-1}∗⋯(W_1∗X)) = (W_N∗W_{N-1}∗⋯∗W_1)∗X(kernel 间卷积,注意核卷积时索引反向)
- **并行**:Σₘ W_m∗X = (Σₘ W_m)∗X(不同尺寸核对齐空间中心,如 1×1 补零到 3×3)
- 一切线性层(conv/group conv/DW conv/avg pooling/scaling/固定频率滤波)都可写成「退化卷积核」,统一进上述两条规则。

于是训练时每个 step 先把整块 squeeze 成一个 3×3 kernel、只卷积一次——**中间特征图根本不存在**,额外开销从 O(H×W)(特征图尺寸)降到 O(K_H×K_W)(kernel 尺寸,如 3×3),计算和显存同量级下降。推理时结构与普通卷积完全相同。

### 3. 三步 Block Linearization(图 3 流程)

1. **去掉块内所有分支 BN**(非线性源);
2. **每条分支末端加 scaling 层**(补回优化多样性,channel-wise 向量;向量 > 标量 > 不加:72.13 / 72.04 / 71.87);
3. **块加和后补一个 post-addition BN**(稳定训练——去掉它梯度直接爆炸 NaN;这是全块唯一的 BN,在块外,不挡内部合并)。

## 可复现细节

```
OREPA block(对应一个 3×3 卷积,六条并行分支,全部线性):
  ① Conv3×3 → Scaling
  ② Conv1×1 → Conv3×3 → Scaling            (串行,可 kernel 级合并)
  ③ Conv1×1 → Pooling3×3 → Scaling
  ④ Conv1×1 → Freq3×3(频率先验滤波,固定 DCT 式基) → Scaling
  ⑤ Conv1×1 → Conv1×1 → Scaling            (Rep-1×1,用于 bottleneck 的 1×1 层)
  ⑥ PW Conv1×1 → DW Conv3×3 → Scaling      (线性 DW 可分离卷积,去掉中间非线性)
  Σ 六分支 → post-addition BN → (块外)ReLU

scaling 初始化(论文附录,经验值,平衡各分支输出方差):
  {1×1: 1.0, k×k: 0.25, 1×1-k×k: 0.5, 1×1-pooling: 0.5, 1×1-filtering: 0.0, dw-pw: 0.5}
  理由:1×1 卷出的特征方差小 → γ 给最大;pooling/filtering 分支易困浅层局部最优 → 压低;
  filtering 干脆初始化为 0(先关闭)

kernel 级合并(每次前向):
  并行:W_unified = Σ_branch align_center(W_branch)      # 小核 zero-pad 到大核
  串行:W_seq = conv2d(W_2, flip?…)                      # kernel 间卷积,索引反向(论文 Eq.2/附录 Eq.2)
  成本 O(K²·C²) 一次,替代整块特征图运算 O(HW·C²·K²)

训练配方(ImageNet):SGD lr=0.1 cosine,120 epoch,warmup 5 epoch,batch 256,4×V100
```

## 数据对比

ImageNet-1K(120 epoch,同机复测,Top-1 / 训练显存 / 每 batch 秒):

| 模型 | 无重参 | DBB(offline) | OREPA(online) |
|---|---|---|---|
| ResNet-18 | 71.21 / 4.2G / 0.086s | 71.77 / 7.5G / 0.241s | **72.13 / 5.2G / 0.120s** |
| ResNet-50 | 76.70 / 10.3G / 0.204s | 76.89 / 13.7G / 0.380s | **77.31 / 11.2G / 0.252s** |
| ResNet-101 | 77.71 / 14.5G / 0.382s | 78.02 / 19.7G / 0.641s | **78.29 / 16.1G / 0.445s** |
| RepVGG-A0 | — | 72.41 / 3.8G / 0.100s | **73.04 / 4.2G / 0.136s** |
| RepVGG-A2 | — | 76.48 / 5.8G / 0.170s | **76.72 / 7.4G / 0.210s** |

- 相对 DBB:**额外训练显存省 65~75%,训练提速 1.5~2.3×,精度反而 +0.2~0.6%**;组件逐个加满时(含 deep stem)offline 路线显存爆掉(ResNet-18 加 deep stem 都放不下),online 提速可达 4×、省 96%+ 额外显存,总开销≈vanilla ResNet。
- 组件消融(ResNet-18/50 Top-1):baseline 71.21/76.70 → +online DBB 71.75/76.86 → +频率先验 71.78/76.92 → +线性 DW sepconv 71.82/76.98 → +线性 deep stem 72.13/77.09 → +Rep-1×1(仅 bottleneck)—/77.31。
- 下游(ResNet-50 骨干):COCO Faster R-CNN mAP 36.5(none)/36.5(DBB)/**37.0(OREPA)**,RetinaNet 36.2/36.6/**36.9**;Cityscapes PSPNet mIoU 74.47/74.50/**75.47**,DeepLabV3+ 76.63/77.15/**77.59**——且训练显存/时间都比 DBB 低。
- 已知局限(论文 §4.5):VGG 式无残差结构上直接套 ResNet 配方不一致,OREPA-VGG 保留了 RepVGG 的全部分支,代价略增。

## 实践要点

- [ ] **想给已定型的部署模型免费涨点**:用重参数化——训练块随便复杂,推理折叠回单卷积;这是「训练-推理解耦」最干净的实例。
- [ ] **判断一个重参数化设计是否有效的准则**(Proposition 1):拓扑必须「超两层 或 多分支且分支各有尺度」;共享尺度的单层多分支 = 无效。
- [ ] 去 BN 必须**补分支级 scaling**(channel-wise 向量,不是标量)+ **块后 post-BN**(否则 NaN);scaling 初始值要按分支输出方差手工配(见上)。
- [ ] 训练预算紧、又想用 DBB 类复杂块 → 直接上 OREPA 路线(kernel 级在线合并),显存/时间开销≈普通网络。
- [ ] 合并实现注意:**不同尺寸核要对齐空间中心**(1×1 pad 到 3×3 中央);kernel 间卷积与常规卷积索引方向相反。
- [ ] 该思想与部署强相关:折叠后的 VGG 式单路 3×3 网络对推理引擎极友好(纯 conv+BN+ReLU 直筒),与本域 [TRT-ViT](trt-vit-inference.md)「以部署延迟为设计准绳」同一哲学——一个在训练端做减法、一个在架构端做取舍。

## 溯源

- 主源:论文论文原文(含补充材料，链接见 frontmatter)。全部数字取自论文 Table 1-5 与附录 Table 2。
- 网络综合:官方代码 JUGGHM/OREPA_CVPR2022(README 与论文互证:两步 pipeline = Linearization + Squeezing,「理论上分支无尺度会退化」的表述一致);前驱三篇 arXiv abs 页核对定位——RepVGG(2101.03697,VGG-style 多分支训练/单路推理)、DBB(2103.13425,Inception-like 多分支,ImageNet 最高 +1.9%)、ACNet(1908.03930,非对称 1×3+3×1+3×3 分支)。全部 URL 于 2026-09-21 curl 实测存活(200)。
- 理论前驱:Arora et al., *On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization*, ICML 2018(多层拓扑改变优化的 Lemma 1,OREPA 将其扩展到多分支)——登记备查,未单独抓全文。
