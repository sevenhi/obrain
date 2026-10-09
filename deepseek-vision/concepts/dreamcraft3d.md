---
title: DreamCraft3D — 图生 3D(分层生成 + 引导式扩散先验)
type: concept
tags: [deepseek, dreamcraft3d, 3d-generation, sds, score-distillation, diffusion-prior, dreambooth, image-to-3d]
series: deepseek-vision
source:
  - "web: DreamCraft3D GitHub README | https://github.com/deepseek-ai/DreamCraft3D | 2026-10 抓取"
  - "arXiv: DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior | 2310.16818"
date: 2026-10
---

# DreamCraft3D — 图生 3D(分层生成 + 引导式扩散先验)

> 本篇是 deepseek-vision 域「图生 3D 谱系」的深度篇。与 VL/OCR/Janus 不同,DreamCraft3D 的「vision」不在编码器而在**扩散先验(diffusion prior)**,是 DeepSeek 视觉家族里唯一走 3D 生成路线的仓库。

## TL;DR(30 秒结论)

**DreamCraft3D 用「一张 2D 参考图」分层生成高保真 3D 物体:先「几何雕刻(geometry sculpting)」用视角依赖的扩散模型做分数蒸馏(SDS)把几何捏一致,再「纹理增强(texture boosting)」用引导式分数蒸馏(Bootstrapped Score Distillation, BSD)把纹理提上去——两级用不同的 3D 先验,交替优化「扩散先验 ↔ 3D 场景表示」互相增强。** 一句话:**DreamFusion 用单一 SDS 出几何,纹理糊;DreamCraft3D 把「几何一致性」与「纹理保真」拆成两级、各配专属 3D 先验,并用「引导式」交替优化把纹理拉到照片级。**

---

## 一、背景:单级 SDS 的纹理困境

图生 3D 的主流范式由 **DreamFusion(Poole et al. 2022)开创**:用预训练 text-to-image 扩散模型对 3D 场景(NeRF/3DGS)做 **Score Distillation Sampling(SDS,分数蒸馏采样)** 损失,把扩散模型当「3D 教师」。但单级 SDS 有个核心矛盾:

- **要几何一致**,需要「视角依赖」的扩散先验,让不同角度渲染出来是同一个物体;
- **但几何优先会牺牲纹理保真**——先验把力气花在「形状别变」上,纹理就糊了。

DreamCraft3D 的切入点是「**一致性**」这个既有工作没做好的问题(论文摘要):把生成拆成两个层级,各配一个定制的 3D 先验。

---

## 二、分层生成机制(为什么分两层)

### 阶段一:几何雕刻(Geometry Sculpting)

- 用**视角依赖(view-dependent)的扩散模型**做 SDS,把 3D 表示(体渲染/NeRF 类)雕出**多视角渲染一致**的几何。
- 这个 3D 先验 + 若干训练策略**优先保证几何一致性,牺牲纹理保真**(论文原话)——所以需要第二阶段补纹理。

### 阶段二:纹理增强(Texture Boosting)+ Bootstrapped Score Distillation

- 在优化出的场景渲染图上,训练一个**个性化扩散模型(用 DreamBooth 微调)**,把「这个特定 3D 场景的知识」注入扩散模型,得到 **3D-aware 的扩散先验**。
- 用这个 3D-aware 先验对场景做 SDS,提供**视角一致**的纹理指导。
- **引导式(Bootstrapped)**:交替优化「扩散先验」与「3D 场景表示」——优化后的 3D 场景反过来帮助训练场景专属扩散模型,扩散模型再提供越来越视角一致的指导,二者互相强化,纹理大幅提升。

### 融合方式总结(文字图)

```
2D 参考图 ────────────────────────────────┐
                                          ▼
阶段一 几何雕刻:视角依赖扩散模型 ──SDS──→ 3D 场景(几何一致,纹理糊)
                                          │
阶段二 纹理增强:                          ▼
  在渲染图上 DreamBooth 微调 → 3D-aware 扩散先验
  交替优化:3D 场景 ↔ 扩散先验(引导式 BSD)
                                          ▼
                              高保真、照片级、视角一致的 3D 物体
```

### 可复现细节

- 官方实现基于 **threestudio** 框架(README:安装部分即装 threestudio,或 Docker);需要 NVIDIA 显卡 ≥20GB VRAM,CUDA + Python≥3.8。
- 依赖:PyTorch≥1.12(实测 torch1.12.1+cu113 / torch2.0.0+cu118);DreamBooth 用于阶段二个性化微调。
- 训练配置(threestudio 约定):`configs/` 下分阶段配置;先跑几何阶段 checkpoint,再喂给纹理阶段。
- 2024-10 发布 **DreamCraft3D++**,生成质量与效率显著提升(README News)。

---

## 三、关键概念:两条蒸馏路线的对比

| 维度 | DreamFusion(单级) | DreamCraft3D(分层) |
|---|---|---|
| 先验 | 一个 text-to-image 扩散模型 | 视角依赖扩散(几何)+ 3D-aware 个性化扩散(纹理) |
| 蒸馏 | 单级 SDS | 几何 SDS + 引导式 BSD |
| 几何一致性 | 一般 | 强(阶段一专攻) |
| 纹理保真 | 糊 | 照片级(阶段二专攻) |
| 核心机制 | 扩散模型当 3D 教师 | 交替优化「先验 ↔ 场景」互相增强 |

**演进逻辑三句话**:

1. **「一致性与保真是一对矛盾」**:单先验无法同时满足——视角一致要「牺牲细节」、纹理保真要「容忍视角漂移」。DreamCraft3D 用「分层 + 专属先验」把矛盾拆开,而不是在单一损失里权衡。
2. **「引导式」是纹理增强的引擎**:DreamBooth 把通用扩散模型个性化成「见过这个 3D 场景」的先验,再用它蒸馏——先验越懂场景、指导越一致,场景优化得越好、先验又训得越准,形成正反馈(bootstrapped)。
3. **「扩散先验」是 DeepSeek 视觉家族在 3D 侧的「编码器」**:VL/OCR/Janus 用 vision encoder 把像素压成 token,这里用扩散先验把 2D 图「蒸馏」成 3D 表示——同一主线(视觉信号 → 可优化的表示)的不同实现。

---

## 实践要点(checklist)

- **环境**:先装 threestudio(或 Docker),显卡 ≥20GB VRAM;两阶段训练:几何 checkpoint → 纹理阶段。
- **坑**:两阶段是「先几何后纹理」,纹理阶段的输入是几何阶段渲染的多视角图,几何没收敛就直接进纹理阶段会导致视角漂移放大。
- **DreamBooth 个性化**:阶段二要对「本场景」微调扩散模型,不同物体要分别微调(场景专属先验),不是复用同一个先验。
- **选型**:要「照片级纹理 + 强一致性」→ DreamCraft3D/++;要极简快速原型 → DreamFusion 单级 SDS;要「生成基础」而非「理解/压缩」→ 本域其他仓库都不相关,这是唯一的 3D 路线。

---

## 溯源

- DreamCraft3D 论文:https://arxiv.org/abs/2310.16818
- 仓库:https://github.com/deepseek-ai/DreamCraft3D
- 项目页:https://mrtornado24.github.io/DreamCraft3D/
- 前置工作 DreamFusion(Poole et al. 2022,SDS 原始):https://dreamfusion3d.github.io/

## 相关条目(库内互链)

- 同域生成侧(Janus 文生图):janus.md
- 3D 表示的通用背景(NeRF 体渲染):../computer-vision/concepts/nerf-view-synthesis.md
