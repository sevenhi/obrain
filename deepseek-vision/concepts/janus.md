---
title: Janus 系列 — 统一多模态理解与生成(双编码解耦 + 单 transformer)
type: concept
tags: [deepseek, janus, janus-pro, janusflow, unified-multimodal, decoupled-encoding, vq, siglip, rectified-flow]
series: deepseek-vision
source:
  - "web: Janus GitHub README | https://github.com/deepseek-ai/Janus | 2026-10 抓取"
  - "arXiv: Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation | 2410.13848 | https://arxiv.org/abs/2410.13848"
  - "arXiv: Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling | 2501.17811 | https://arxiv.org/abs/2501.17811"
  - "arXiv: JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation | 2411.07975 | https://arxiv.org/abs/2411.07975"
  - "code: Janus-1.3B / Janus-Pro-7B config.json | https://huggingface.co/deepseek-ai/Janus*/raw/main/config.json"
date: 2026-10
---

# Janus 系列 — 统一多模态理解与生成

> 本篇是 deepseek-vision 域「统一理解+生成谱系」的深度篇:Janus(2024-10)、JanusFlow(2024-11)、Janus-Pro(2025-01)三篇合一篇,讲清「一个模型同时看懂图和生成图」这条主线。

## TL;DR(30 秒结论)

**Janus 用一个「解耦视觉编码」的核心决策统一了多模态理解与生成:理解用一套编码器(SigLIP-L)、生成用另一套编码器(VQ-16 tokenizer),两条通路各自最优、但共享同一个 transformer 骨干。** 一句话:**Chameleon 用「同一个视觉编码器」通吃理解与生成(信息粒度冲突、理解受损),Janus 把它拆成两条通路(理解/生成各选最合适的编码方式)再放进一个 transformer——「可解耦」正是其灵活性与效果的来源。** Janus-Pro 用「训练策略优化 + 数据扩增 + 规模放大」把 Janus 推到 7B;JanusFlow 把生成侧的 AR 换成整流流(rectified flow),证明整流流能在 LLM 框架内直接训练。

---

## 一、Janus:解耦视觉编码 + 统一 transformer

### 机制原理(为什么「解耦」是关键)

统一多模态模型(如 Chameleon)通常**用同一个视觉编码器同时服务理解与生成**。Janus 的观察是:理解与生成所需的**信息粒度不同**——

- **理解**:需要高层语义(SigLIP 这类图文对齐编码器最合适),token 粒度粗(一个 token 对应一个 patch 的语义);
- **生成**:需要逐像素、低层细节(VQ 量化把连续像素离散成码本 token 最合适),token 粒度细(一个 token 对应一个码本条目)。

同一个编码器被迫兼顾两者,导致「理解任务效果次优」(论文原话)。Janus 的解法:**把视觉编码拆成两条独立通路,但只用一个统一 transformer 处理**。解耦既缓解了「视觉编码器角色冲突」,又让理解/生成各自独立选择最合适的编码方式(例如理解侧可用更快的编码器、生成侧可用更细的 VQ),极大提升灵活性。

**可复现细节(config.json 实测,Janus-1.3B / Janus-Pro-7B)**:

```jsonc
// Janus-Pro-7B
"vision_config":    { "cls": "CLIPVisionTower", "params": { "model_name": "siglip_large_patch16_384", "image_size": 384 } },  // 理解侧
"gen_vision_config":{ "cls": "VQ-16", "params": { "image_token_size": 16384, "n_embed": 8 } },                              // 生成侧
"aligner_config":   { "cls": "MlpProjector", "params": { "depth": 2, "input_dim": 1024, "n_embed": 4096, "projector_type": "mlp_gelu" } },  // 理解 token → LLM
"gen_aligner_config":{ "cls": "MlpProjector", "params": { "depth": 2, "input_dim": 8, "n_embed": 4096, "projector_type": "mlp_gelu" } },        // 生成 token → LLM
"gen_head_config":  { "cls": "vision_head", "params": { "image_token_embed": 4096, "image_token_size": 16384 } },         // 生成头
"language_config":  { "model_type": "llama", "num_hidden_layers": 30, "vocab_size": 102400 }                              // 1.3B/7B dense LLM
```

关键数字:**生成侧 VQ-16 码本 16384 条目、每条目 8 维(n_embed=8)**——图像被 VQ 量化成 16384 类的离散 token,这正是「生成」能用自回归(AR)transformer 建模的原因。理解侧 SigLIP-L@384 输出 1024 维,经 2 层 MLP 投影到 4096(LLM 维度);生成侧 8 维经 2 层 MLP 投影到 4096。**两条通路投影到同一 embedding 空间后,汇入同一个 LLM**。

### 融合方式总结

```
理解通路:图像 ─→ SigLIP-L@384 ─→ MLP 投影(1024→4096) ─┐
                                                          ├─→ 统一 LLM(1.3B/7B)
生成通路:图像 ─→ VQ-16 量化(16384 码本,8 维) ─→ MLP 投影(8→4096) ─┘
                                                          │
                                     生成时 LLM → vision_head(4096→16384 码本)→ 反量化成图
```

**生成是自回归的**:LLM 逐 token 预测 VQ 码本 id,再经 vision head 解码成像素——这就是「统一 transformer 通吃理解与生成」的实质。

### 性能定位(论文)

- Janus 在**统一模型里超 SOTA,且追上或超过某些「任务专精」模型**(多模态理解);生成侧也强。分辨率 384×384。
- 命名来源:Janus(雅努斯)是罗马双面门神,隐喻「一张模型两张脸(理解 + 生成)」。

---

## 二、Janus-Pro:数据与规模放大

### 机制原理(三个升级,摘要明示)

Janus-Pro 是 Janus 的增强版,不改变核心结构,而是:

1. **优化训练策略**;
2. **扩增训练数据**;
3. **放大模型规模**(1B → 7B)。

收益:**多模态理解与文本生成指令遵循(text-to-image instruction following)双提升,且文生图稳定性增强**——尤其改善短 prompt 的生成质量与简单文本渲染能力。Janus 初代因数据少、容量小,在短 prompt 生成与文本渲染上较弱,Pro 用规模补上了。

### 性能定位

- 理解侧:POPE、MME-Perception、GQA、MMMU 平均,超此前的统一模型与部分专精模型。
- 生成侧:GenEval、DPG-Bench(指令遵循)提升;短 prompt 输出更稳定、细节更丰富、能生成简单文本。

---

## 三、JanusFlow:自回归 + 整流流

### 机制原理(为什么「整流流能在 LLM 框架内训」)

JanusFlow 探索**极简架构**统一理解与生成:把 AR 语言模型与 **rectified flow(整流流,一种 SOTA 生成建模方法)** 集成进同一个模型。关键发现:**整流流可以直接在 LLM 框架内训练,无需复杂的架构改造**。两大策略:

1. **解耦理解与生成编码器**(沿用 Janus 思想);
2. **统一训练中对齐两者表示(representation alignment)**。

效果:各领域追上或超过专精模型,统一方法里显著领先。分辨率 384×384。这是「扩散/流式生成也能融进自回归 LLM」的一个轻量证据。

---

## 四、三篇对比与演进逻辑(核心结论)

| 版本 | 时间 | 结构增量 | 生成侧 | 规模 |
|---|---|---|---|---|
| Janus | 2024-10 | 双编码解耦 + 单 transformer | VQ-16 + AR | 1.3B |
| JanusFlow | 2024-11 | + 整流流替代 AR 生成 | VQ + rectified flow | 1.3B |
| Janus-Pro | 2025-01 | + 训练策略/数据/规模 | VQ-16 + AR(不变) | 1B/7B |

**演进逻辑三句话**:

1. **「解耦」是贯穿三篇的核心思想**:理解(SigLIP)与生成(VQ)各走各路、共享一个 transformer——这是 Janus 家族对「统一模型」的独特回答,与 Chameleon 的「单编码器」形成对照。
2. **两条升级路线正交**:Pro 走「规模 + 数据」(结构不动、能力涨),Flow 走「生成范式」(AR→整流流,验证扩散/流可融进 LLM)——**一个扩规模、一个换生成引擎**,可叠加。
3. **「统一」的收益是灵活性**:解耦让理解/生成组件各自独立选型、独立演进,同时共享骨干减冗余——这正是「下一代统一多模态模型」的候选形态。

---

## 实践要点(checklist)

- **复现**:`transformers` 直接 `from_pretrained("deepseek-ai/Janus-Pro-7B")`;理解走 SigLIP、生成走 VQ-16,两路投影层独立。
- **坑(README 明示)**:2024-10-20 修了 `tokenizer_config.json` 的 bug——旧版导致 **classifier-free guidance(CFG)失效**,生成质量受损;复现务必用修复后的 tokenizer。
- **生成分辨率 384×384**:Janus 系生成分辨率固定 384²(不像扩散模型可任意放大),选型时注意上限。
- **选型**:要最强文生图指令遵循 → Janus-Pro-7B;要研究「流式生成融进 AR LLM」→ JanusFlow;要轻量统一模型 → Janus-1.3B。
- **评测**:理解侧用 VLMEvalKit(2024-10-23 已加评测代码,PR #541);生成侧用 GenEval/DPG-Bench。

---

## 溯源

- Janus 论文:https://arxiv.org/abs/2410.13848
- Janus-Pro 论文:https://arxiv.org/abs/2501.17811
- JanusFlow 论文:https://arxiv.org/abs/2411.07975
- 仓库:https://github.com/deepseek-ai/Janus
- config 口径:https://huggingface.co/deepseek-ai/Janus-1.3B 、https://huggingface.co/deepseek-ai/Janus-Pro-7B

## 相关条目(库内互链)

- 同域前代(VL/VL2 理解侧、OCR 视觉压缩):deepseek-vl-vl2.md、deepseek-ocr-ocr2.md
- 图生 3D(生成侧延伸):dreamcraft3d.md
- 统一多模态的对立面(单编码器路线 Chameleon):../model-architecture/concepts/open-model-landscape-2025-2026.md
