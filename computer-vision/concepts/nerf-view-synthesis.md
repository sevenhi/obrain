---
title: NeRF — 神经辐射场视图合成(NeRF + NeRF++)
type: concept
tags: [nerf, view-synthesis, volume-rendering, implicit-representation, 3d-reconstruction]
source:
  - "pdf: NeRF 论文全文 | arXiv:2003.08934 ECCV 2020 (Best Paper Honorable Mention) | ~/zhanbin/library/papers/NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.pdf ↑primary"
  - "pdf: NeRF++ 技术报告全文 | arXiv:2010.07492 | ~/zhanbin/library/papers/NeRF++: Analyzing and Improving Neural Radiance Fields.pdf ↑primary"
  - "web: NeRF 官方项目页 | https://www.matthewtancik.com/nerf (抓取 2026-09-21)"
  - "web: NeRF 官方 TensorFlow 实现 | https://github.com/bmild/nerf (抓取 2026-09-21)"
  - "web: NeRF++ 官方 PyTorch 实现 | https://github.com/Kai-46/nerfplusplus (抓取 2026-09-21)"
  - "web: Fourier Features Let Networks Learn High Frequency Functions(位置编码理论基础) | arXiv:2006.10739 (抓取 2026-09-21)"
date: 2026-09
---

# NeRF:神经辐射场视图合成(NeRF + NeRF++ 合并篇)

> 两篇论文内容强重叠(NeRF++ 是 NeRF 的分析与改进),按 SOURCE-POLICY 合并为一篇,source 列全部出处。

## TL;DR

NeRF 用**一个 8 层 MLP(约 5MB 权重)表示一个完整场景**:输入 5D 坐标 (x,y,z,θ,φ),输出体密度 σ 与视角相关颜色 c;渲染时沿相机光线采样、用经典体渲染积分(离散化为 alpha 合成)得到像素颜色;整个管线可微,所以只用「带位姿的 RGB 照片 + 光度重建损失」就能优化。三个关键设计缺一不可:**位置编码**(对抗 MLP 频谱偏置,学出高频细节)、**σ 只依赖位置 / c 依赖位置+方向**的解耦(多视角一致性)、**coarse-to-fine 分层采样**(把采样点集中到表面附近)。原版 NeRF 处理不了 360° 非有界大场景,NeRF++ 用**倒球面参数化**把场景拆成内球(前景,欧氏空间)+ 外球补集(背景,以 1/r 为深度),两个 NeRF 分别 raycast 再合成,在 Tanks&Temples/LF 数据集全面超过原版(Truck 场景 PSNR 20.85→22.77,LPIPS 0.513→0.298)。

## 机制原理(为什么)

### 1. 为什么是 MLP + 体渲染,而不是 voxel grid / mesh

- **voxel grid 的空间复杂度是死的**:高分辨率场景要 1024³ 级网格,存储与采样成本爆炸;LLFF 为每张输入图存 3D voxel grid,一个场景超 **15GB**,而 NeRF 的 MLP 权重只要 **5MB(约 3000× 压缩)**。
- **mesh 优化难收敛**:基于图像重投影的梯度优化 mesh 容易卡在局部极小,且需要固定拓扑的模板 mesh 作初始化,真实场景没有。
- **体渲染天然可微**:渲染积分对 (σ,c) 可导 → 只需最小化「渲染图 vs 真实图」的光度误差,就能端到端反推场景。这是「用可微渲染做优化」路线的奠基之一。

### 2. 体渲染方程与离散化

对相机光线 r(t)=o+td,期望颜色为经典体渲染积分(Kajiya & Von Herzen 1984):

```
C(r) = ∫[tn,tf] T(t)·σ(r(t))·c(r(t),d) dt,  T(t) = exp(-∫[tn,t] σ(r(s)) ds)
```

T(t) 是累积透射率(光线从 tn 走到 t 不碰任何粒子的概率)。数值上用分层采样 + quadrature 离散:

```
Ĉ(r) = Σᵢ Tᵢ·(1 - exp(-σᵢδᵢ))·cᵢ,   Tᵢ = exp(-Σⱼ<ᵢ σⱼδⱼ),   δᵢ = tᵢ₊₁ - tᵢ
```

退化为标准 alpha 合成,αᵢ = 1-exp(-σᵢδᵢ)。分层采样(每 bin 随机取一点)让 MLP 在优化过程中被查询到**连续位置**,等效支持连续表示。

### 3. 为什么必须有位置编码(positional encoding)

MLP 有**频谱偏置**(spectral bias,Rahaman et al. 2018):直接回归低维坐标时偏向低频函数,渲染结果过平滑、丢几何/纹理细节。解法是把输入先映射到高维高频空间(源自 Fourier Features 理论,Tancik et al. arXiv:2006.10739,用 NTK 分析证明该映射把有效核变成带宽可调的平稳核):

```
γ(p) = (sin(2⁰πp), cos(2⁰πp), …, sin(2^(L-1)πp), cos(2^(L-1)πp))
```

坐标 x 用 L=10(60 维),方向 d 用 L=4(24 维)。消融:去掉 PE,PSNR 31.01→28.77(Realistic Synthetic 360°)。

### 4. 为什么 σ 与 c 要分开预测(多视角一致性)

密度 σ 只依赖位置 x(视角无关)→ 几何在空间里唯一确定,强制多视角一致;颜色 c 依赖 (x,d) → 允许镜面反射等非朗伯效应。实现上:MLP 前 8 层(256ch,ReLU,第 5 层 skip connection)从 γ(x) 输出 σ + 256 维特征;特征拼 γ(d) 再过 1 层(128ch)输出 RGB。消融:去掉视角依赖,PSNR 31.01→27.66(比去 PE 还狠)。

### 5. NeRF++ 分析:shape-radiance ambiguity 与 NeRF 为什么没掉坑

理论上,任意错误几何(如把场景设成单位球)都存在一族视角相关 radiance 完美拟合训练视图(训练 PSNR 25.4),但测试视图崩(10.7)。NeRF 靠两点隐式正则躲开:① 错误几何迫使 c 成为关于 d 的高频函数,有限容量 MLP 学不动;② NeRF 的 MLP 结构对 x/d **不对称**(d 在末层才注入、γ(d) 只用低频 L=4),隐式编码了「BRDF 平滑」先验。验证:把 c 换成对称的 vanilla MLP(x、d 都用 γ10 进第一层),DTU 上泛化显著变差(Scan 65 外插 PSNR 28.41→21.65)。

### 6. NeRF++ 改进:倒球面参数化(非有界 360° 场景)

原版 NeRF 的包围盒参数化两难:盒子只罩前景 → 背景崩坏;盒子罩全场景 → 有限采样分辨率处处糊。前向场景用的 NDC(视差坐标)又限制视角必须在参考视锥内。NeRF++ 把空间切成**内单位球(前景+相机,欧氏参数化)** 和**外域(倒球面参数化)**:外域点用四元组 (x′,y′,z′,1/r) 表示,r=‖(x,y,z)‖>1,全部有界(x′,y′,z′∈[-1,1], 1/r∈[0,1]),既数值稳定又天然符合「远处物体分辨率低」。渲染积分按单位球交点 t′ 拆成两段,内段在欧氏空间算、外段以 1/r 为积分变量在倒球空间算;两个独立 NeRF(σ_in/c_in 与 σ_out/c_out)分别 raycast 后合成。物理直觉:以单位球为像面的虚拟球面相机,1/r 充当视差深度(与 multi-sphere images 概念同源)。

## 可复现细节

```
NeRF 网络结构(Fig.7):
γ(x)[60维] → 8×FC(256, ReLU) ──skip: γ(x) 拼到第5层输入
           → σ(ReLU 保证非负) + 256维特征
特征 ⊕ γ(d)[24维] → FC(128, ReLU) → RGB(sigmoid)

训练循环:
每次迭代随机抽 4096 条光线 → coarse 网络分层采样 Nc=64 点
→ 归一化权重 ŵᵢ=wᵢ/Σw 得分段常数 PDF → 逆变换采样 Nf=128 点
→ fine 网络在 64+128 点并集上求值 → 双损失
L = Σ_r ‖Ĉc(r)-C(r)‖² + ‖Ĉf(r)-C(r)‖²   (coarse 也监督,才能给 fine 供权重分布)
Adam,lr 5e-4 → 指数衰减到 5e-5,100-300k iter(单 V100 约 1-2 天)

NeRF++ 外域 raycast:
光线交单位球于 a=o+t_a·d(解 |o+t·d|=1),弦中点 b=o+t_b·d(解 dᵀ(o+t·d)=0)
给定 1/r,把向量 a 绕 b×d 旋转 ω=arcsin|b| - arcsin(|b|·(1/r)) 得 (x′,y′,z′)
→ σ_out(x′,y′,z′,1/r), c_out(x′,y′,z′,1/r,d);1/r 在 [0,1] 均匀采样
真实场景位姿:COLMAP SfM 估计;T&T 场景归一化到半径 1/8 球内
```

## 数据对比

| 指标 | NeRF(原版) | 对照 |
|---|---|---|
| Realistic Synthetic 360° PSNR | **31.01** | LLFF 24.88 / SRN 22.26 / NV 26.05 |
| Diffuse Synthetic 360° PSNR | **40.15** | LLFF 34.38 / SRN 33.20 |
| Real Forward-Facing PSNR | **26.50** | LLFF 24.13(LLFF 的 LPIPS 略优 0.212 vs 0.250) |
| 消融(Realistic Synthetic) | 完整 31.01 | 无PE 28.77 / 无视角依赖 27.66 / 无分层采样 30.06 / 全裸 26.67 |
| 模型体积 | **5MB** | LLFF 单场景 15GB+(约 3000×) |
| 训练 | 100-300k iter,1-2 天/场景(V100) | NeRF++:4×RTX 2080Ti 约 24h |
| 渲染 | 每光线 256 次 MLP 查询,150-200M 查询/帧,V100 约 30s/帧 | NeRF++:1280×720 约 30s/帧(2080Ti) |

| 指标(非有界 360°) | NeRF | NeRF++ |
|---|---|---|
| T&T Truck PSNR / LPIPS | 20.85 / 0.513 | **22.77 / 0.298** |
| T&T Train PSNR / LPIPS | 16.64 / 0.651 | **17.17 / 0.523** |
| LF Africa PSNR / LPIPS | 26.16 / 0.217 | **27.41 / 0.163** |

NeRF++ 全部 8 个场景(T&T 4 + LF 4)三项指标(PSNR/SSIM/LPIPS)一致优于 NeRF;且为公平对比 NeRF 采样数已加倍(coarse 128 + fine 256),计算/显存与 NeRF++ 同量级。

## 实践要点

- [ ] **三大件一个不能少**:位置编码(L=10/4)、σ/c 解耦、分层采样(Nc=64, Nf=128);消融显示 PE 与视角依赖各值 2-3dB PSNR。
- [ ] **别把 c(x,d) 换成对称 MLP**:d 必须在末层低维注入 + 低频编码,否则 shape-radiance ambiguity 会让新视角泛化崩坏(DTU 外插 -7dB)。
- [ ] 360° 非有界场景直接用 **NeRF++ 倒球面参数化**(内球欧氏 + 外球 1/r),别试图放大包围盒。
- [ ] 渲染慢是硬伤:每像素 256 次 MLP 查询,V100 单帧约 30s;需要实时/大场景时看后续加速线(NeRF 系生态见 nerfstudio:https://docs.nerf.studio/)。
- [ ] 真实数据管线:COLMAP SfM 位姿 → 场景归一化(相机中心进单位球)→ 位姿误差会直接伤渲染质量;小标定误差可考虑鲁棒损失/每图自动曝光(NeRF++ 官方库带 experimental 的 autoexposure)。
- [ ] NeRF 的局限(NeRF++ §6 明言):训练贵、相机标定敏感、不建模曝光/渐晕等光度效应。

## 溯源

- 主源:两篇论文本地 PDF(见 frontmatter source,`~/zhanbin/library/papers/`,只读存档)。
- 网络综合:arXiv abs 页(2003.08934 / 2010.07492)核对标题、作者、版本历史;官方项目页 matthewtancik.com/nerf(ECCV 2020 Oral, Best Paper Honorable Mention);官方代码 bmild/nerf(TensorFlow)、Kai-46/nerfplusplus(PyTorch DDP,数据约定:位姿是 camera-to-world、OpenCV 坐标系、归一化把相机放进单位球——与本文「实践要点」互证);arXiv:2006.10739(Fourier Features/NTK 理论,位置编码的「为什么」)。全部 URL 于 2026-09-21 curl 实测存活(200)。
- 所有数字直接取自两篇论文正文/表格,未做外推。
