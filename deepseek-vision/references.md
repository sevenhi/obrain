---
title: deepseek-vision 外部链接索引
type: reference
tags: [deepseek, vision, multimodal, vl, vl2, ocr, janus, dreamcraft3d, references]
date: 2026-10
---

# References — deepseek-vision

> 只放链接索引,不复制全文(库规约)。聚合自各篇 frontmatter 的 source;条目级溯源见各篇文末『溯源』。
> 全部 GitHub 仓库与 arXiv 链接于 2026-10 curl 实测存活(HTTP 200)。

## 六仓库

| 仓库 | GitHub | 一句话 |
|---|---|---|
| DeepSeek-VL | https://github.com/deepseek-ai/DeepSeek-VL | 图文理解:Hybrid 双塔 + 投影 |
| DeepSeek-VL2 | https://github.com/deepseek-ai/DeepSeek-VL2 | 图文理解:SO400M + 动态分块 + MoE |
| DeepSeek-OCR | https://github.com/deepseek-ai/DeepSeek-OCR | 光学压缩:DeepEncoder + 3B-MoE |
| DeepSeek-OCR-2 | https://github.com/deepseek-ai/DeepSeek-OCR-2 | 视觉因果流:DeepEncoder V2 重排 token |
| Janus | https://github.com/deepseek-ai/Janus | 统一理解+生成:双编码解耦(Janus/Flow/Pro 同仓) |
| DreamCraft3D | https://github.com/deepseek-ai/DreamCraft3D | 图生 3D:分层生成 + BSD |

## 论文(arXiv)

| 主题 | arXiv | 标题 |
|---|---|---|
| DeepSeek-VL | https://arxiv.org/abs/2403.05525 | DeepSeek-VL: Towards Real-World Vision-Language Understanding |
| DeepSeek-VL2 | https://arxiv.org/abs/2412.10302 | DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding |
| DeepSeek-OCR | https://arxiv.org/abs/2510.18234 | DeepSeek-OCR: Contexts Optical Compression |
| DeepSeek-OCR-2 | https://arxiv.org/abs/2601.20552 | DeepSeek-OCR 2: Visual Causal Flow |
| Janus | https://arxiv.org/abs/2410.13848 | Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation |
| Janus-Pro | https://arxiv.org/abs/2501.17811 | Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling |
| JanusFlow | https://arxiv.org/abs/2411.07975 | JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation |
| DreamCraft3D | https://arxiv.org/abs/2310.16818 | DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior |
| DreamCraft3D++ | https://arxiv.org/abs/2410.12928 | DreamCraft3D++: Efficient Hierarchical 3D Generation with Multi-Plane Reconstruction Model |

> 注:OCR/OCR-2 的 arXiv 号注意 2410.06571 为无关论文号,已排除。正确号经 web_search 核对为 **2510.18234**(OCR)与 **2601.20552**(OCR-2)。

## 模型卡 / config 口径(HF raw)

- DeepSeek-VL-7B-chat:https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat
- DeepSeek-VL2(full/tiny/small):https://huggingface.co/deepseek-ai/deepseek-vl2 、-tiny 、-small
- DeepSeek-OCR:https://huggingface.co/deepseek-ai/DeepSeek-OCR
- DeepSeek-OCR-2:https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
- Janus-1.3B / Janus-Pro-7B:https://huggingface.co/deepseek-ai/Janus-1.3B 、https://huggingface.co/deepseek-ai/Janus-Pro-7B

## 生态 / 工具

- vLLM 官方 DeepSeek-OCR 支持:https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html
- VLMEvalKit(Janus 理解评测):https://github.com/open-compass/VLMEvalKit
- DreamCraft3D 项目页:https://mrtornado24.github.io/DreamCraft3D/
- DreamFusion(SDS 前置):https://dreamfusion3d.github.io/
- threestudio(DreamCraft3D 基础框架):https://github.com/threestudio-project/threestudio
