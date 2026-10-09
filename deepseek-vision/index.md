---
title: deepseek-vision — DeepSeek 多模态/视觉家族(VL/VL2/OCR/OCR2/Janus/DreamCraft3D)
type: domain
tags: [deepseek, vision, multimodal, vl, vl2, ocr, janus, dreamcraft3d, siglip, sam, moe, vq, sds, 3d]
date: 2026-10
---

# deepseek-vision — DeepSeek 多模态/视觉家族

> 本页是 deepseek-vision 领域的**导航起始页**。定位与关键事实、条目索引、更新维护。
> **真正的可读正文在 `concepts/` 子目录**,每篇一个专题、可独立打开读懂。本页只做导航,不重复正文。

---

## 一句话定位

**deepseek-vision** 记录 DeepSeek 的多模态/视觉模型谱系(6 仓库):从**图文理解**(DeepSeek-VL → VL2)、**文档 OCR 与视觉压缩**(DeepSeek-OCR → OCR-2),到**统一理解+生成**(Janus 系列)与**图生 3D**(DreamCraft3D)。这是 DeepSeek「专属多模态模型仓库」,成系列、成谱系,与库内 computer-vision(通用视觉技术:NeRF/ViT/CNN 表示与部署)、model-architecture(LLM 架构)正交互补。

**主线一句话**:DeepSeek 的视觉模型一直在回答同一个问题——**「vision encoder 怎么把像素压缩成 LLM 能高效消费的 token」**——答案从「Hybrid 双塔编码 + MLP 投影」(VL)、「SO400M 单塔 + 动态分块 + MoE 骨干」(VL2)、「纯视觉大编码器做光学压缩」(OCR)、「编码器带因果推理重排 token」(OCR-2),演进到「理解/生成双编码解耦 + 统一 transformer」(Janus)与「扩散先验驱动 3D」(DreamCraft3D)。

---

## 关键事实速记(快照,2026-10)

- **六个仓库、四条谱系**:VL→VL2(图文理解)、OCR→OCR-2(视觉压缩/文档)、Janus/Janus-Pro/JanusFlow(统一理解+生成)、DreamCraft3D/++(图生 3D)。
- **融合方式演进(用户关注的核心)**:Hybrid encoder + 投影(VL)→ MoE + 多图联合编码(VL2)→ 纯视觉光学压缩(OCR)/ 视觉因果流(OCR-2)→ 统一理解+生成可解耦(Janus)→ 扩散先验(3D)。
- **视觉编码器谱系**:VL 用 **SigLIP-L@384 + SAM-B@1024 双塔(都冻结)**;VL2 换 **SigLIP SO400M@384 单塔(27 层,可训)**;OCR/OCR-2 用 **CLIP-L + SAM-B 混合 "DeepEncoder"**;Janus 理解侧用 **SigLIP-L**,生成侧用 **VQ-16 tokenizer(码本 16384)**。
- **语言骨干**:VL 用 dense LLM(1.3B/7B);VL2 换 **DeepSeekMoE**(Tiny/Small/full = 1.0B/2.8B/4.5B 激活);OCR 用 **DeepSeek3B-MoE-A570M**(64 专家 + 2 共享,top-6,570M 激活);Janus 用 dense LLM(1.3B/7B)。
- **OCR 压缩口径**:DeepSeek-OCR 是「把文本装进图像再让 LLM 解压」的光学压缩——压缩比 <10× 时 97% 精度,20× 时 ~60%;OmniDocBench 上 100 vision token 就超 GOT-OCR2.0(256 token)。
- **归档日期**:2026-10;全部 arXiv 号与 GitHub 仓库 URL 均已 curl 实测存活(HTTP 200)。

### 六仓库 ↔ 谱系映射表(已核实)

| 仓库 | arXiv | 定位 | 视觉编码 | 语言骨干 | 一句话增量 |
|---|---|---|---|---|---|
| DeepSeek-VL | 2403.05525 | 图文理解 | SigLIP-L + SAM-B(冻结双塔) | dense LLM 1.3B/7B | Hybrid encoder + 投影,高分辨率 1024² 固定 token 预算 |
| DeepSeek-VL2 | 2412.10302 | 图文理解(升级) | SigLIP SO400M(单塔) | DeepSeekMoE 1.0/2.8/4.5B 激活 | MoE 骨干 + 动态分块多图联合编码 |
| DeepSeek-OCR | 2510.18234 | 文档 OCR/视觉压缩 | DeepEncoder(CLIP-L + SAM-B) | DeepSeek3B-MoE-A570M | Contextual Optical Compression,100 token 超 GOT-OCR2.0 |
| DeepSeek-OCR-2 | 2601.20552 | 文档 OCR(升级) | DeepEncoder V2(Qwen2-0.5B + SAM-B) | 同上 3B-MoE | Visual Causal Flow,编码器因果重排 token |
| Janus(+Pro/Flow) | 2410.13848 / 2501.17811 / 2411.07975 | 统一理解+生成 | 理解 SigLIP-L / 生成 VQ-16 | dense LLM 1.3B/7B | 双编码解耦,单 transformer 通吃理解与生成 |
| DreamCraft3D(++) | 2310.16818 | 图生 3D | (扩散先验,非 VL 编码器) | — | 分层 3D 生成 + 引导式分数蒸馏(BSD) |

---

## 条目索引

> 每条一个专题,按谱系排(图文理解 → 视觉压缩 → 统一生成 → 3D)。

| 专题 | 文件 | 一句话 | 状态 |
|---|---|---|---|
| DeepSeek-VL & VL2 | [concepts/deepseek-vl-vl2.md](concepts/deepseek-vl-vl2.md) | 图文理解双代:Hybrid 双塔编码 + 投影(VL)→ SO400M 单塔 + 动态分块 + DeepSeekMoE 骨干(VL2) | ✅ 已入库 |
| DeepSeek-OCR & OCR-2 | [concepts/deepseek-ocr-ocr2.md](concepts/deepseek-ocr-ocr2.md) | 视觉压缩双代:Contextual Optical Compression(光学压缩)→ Visual Causal Flow(因果重排 token) | ✅ 已入库 |
| Janus 系列 | [concepts/janus.md](concepts/janus.md) | 统一理解+生成:双编码解耦(SigLIP-L + VQ-16)+ 单 transformer;Janus-Pro 规模化、JanusFlow 整流流 | ✅ 已入库 |
| DreamCraft3D | [concepts/dreamcraft3d.md](concepts/dreamcraft3d.md) | 图生 3D:分层生成(几何雕刻 + 纹理增强)+ Bootstrapped Score Distillation 引导式扩散先验 | ✅ 已入库 |

---

## 相关条目(跨域互链)

- **[`model-architecture/`](../model-architecture/index.md)**:VL2/OCR 的语言骨干是 DeepSeekMoE/MLA,见 [deepseek-arch-v1-moe-v2](../model-architecture/concepts/deepseek-arch-v1-moe-v2.md)(细粒度+共享专家)与 [open-model-landscape](../model-architecture/concepts/open-model-landscape.md)。DeepSeek 网络结构主线(V1→V3.2)见 [deepseek-model-evolution](../model-architecture/concepts/deepseek-model-evolution.md)。
- **[`computer-vision/`](../computer-vision/index.md)**:本域的视觉编码器(SigLIP/SAM/VQ)与 computer-vision 的表示(NeRF)/部署(TRT-ViT)是「理解内容」与「跑得高效」两件事;CCNN 连续卷积核([cnn-long-range-nd](../computer-vision/concepts/cnn-long-range-nd.md))与 VL 的 hybrid encoder 同为「怎么高效表示视觉信号」的两个答案。
- **[`vllm/`](../vllm/index.md)**:DeepSeek-OCR 已被上游 vLLM 官方支持([quantization-gptq-awq-fp8](../vllm/concepts/quantization-gptq-awq-fp8.md) 同属推理侧)。

---

## 溯源

- 外部链接索引见 [references.md](references.md)。
- 六仓库 GitHub + 七篇 arXiv 论文的完整链接清单见 [references.md](references.md)。
