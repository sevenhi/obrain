---
title: DeepSeekMoE — 细粒度专家 + 共享专家(数据/评测侧)
type: concept
tags: [deepseek, moe, fine-grained-expert, shared-expert, efficiency, domain-model]
series: deepseek-moe
source:
  - "pdf: DeepSeekMoE | arXiv:2401.06066 | https://arxiv.org/abs/2401.06066 | 2026-10-09 | ↑primary"
  - "code: DeepSeek-MoE | https://github.com/deepseek-ai/DeepSeek-MoE"
date: 2026-10
---

# DeepSeekMoE — 细粒度专家 + 共享专家(数据/评测侧)

> **TL;DR**:DeepSeekMoE 16B 是 DeepSeek 第一个稀疏 MoE 基座,总参 16.4B、激活 2.8B,从零训 2T 中英 token,**仅 40% 算力就打平 DeepSeek 7B 与 LLaMA2 7B**(后两者激活参数是它的 ~2.5×)。它贡献了 DeepSeek MoE 家族的两个支柱性设计——**细粒度专家分割**(fine-grained expert segmentation)与**共享专家隔离**(shared expert isolation),这两个设计被后续 V2/V3/Coder-V2 一路沿用。**本篇侧重数据与评测;结构机制细节见网络结构篇,此处不重复。**

> 归属说明:本仓(DeepSeek-MoE)是「独立发布的 16B 基座模型 + 论文」,与网络结构篇 [deepseek-arch-v1-moe-v2](../../model-architecture/concepts/deepseek-arch-v1-moe-v2.md) 里讲的 DeepSeekMoE 结构是同一东西。按简报分工:结构/源码级机制在那篇,本域只补「**数据规模 + 评测结论 + 40% 算力这个核心卖点**」,不复制结构。

---

## 1. 定位

DeepSeekMoE 16B 是 DeepSeek 的第一个开源 MoE 模型,解决经典 MoE(GShard 式 top-K)**「专家专精不足」**的问题——标准 MoE 的专家会学到大量重复知识,浪费容量。

## 2. 两大机制(一句话回顾,细节引结构篇)

1. **细粒度专家分割**:把 N 个标准专家切小成 mN 个小专家、激活 mK 个,激活组合更灵活、知识更聚焦。
2. **共享专家隔离**:设 Ks 个「共享专家」恒激活(不参与路由),捕获跨 token 共性知识;路由专家专注差异化知识,减少专家间冗余。

> 这两条是 DeepSeek 后续所有 MoE(V2 160 专家、V3 256 专家、Coder-V2)的设计底座,详见 [deepseek-arch-v1-moe-v2](../../model-architecture/concepts/deepseek-arch-v1-moe-v2.md)。

## 3. 数据与训练(本域侧重点)

- **从零训练 2T tokens**,中英双语(不是续训,是独立基座)。
- 16B 规模可**单卡 40GB 显存部署,无需量化**——这是它作为「研究用基座」的卖点。
- 配置:28 层、hidden 2048、16 头、64 路由专家 + 2 共享专家、激活 6 路由专家,激活参数 2.8B。

## 4. 评测结论(有据数字,核心卖点)

**「40% 算力打平」是这篇论文最值得记住的一个数字**,三条对照:

| 对照 | 算力占比 | 结论 |
|---|---|---|
| DeepSeekMoE 16B vs DeepSeek 7B(dense,同语料) | **40.5%** | 效果相当 |
| DeepSeekMoE 16B vs LLaMA2 7B | **39.6%** | 多数基准反超 |
| DeepSeekMoE 16B vs 同激活参数开源模型 | — | Open LLM Leaderboard 上大幅领先 |

关键解读:**激活参数 2.8B vs LLaMA2 7B 的 7B(约 2.5 倍),用不到一半的算力打平甚至反超**——这是「稀疏 MoE 用更少算力逼近同容量 dense 上限」的直接实证,也是 DeepSeek 之后押注 MoE 路线的底气。

- 16B Chat 版在相同微调设置/数据下,同样 ~40% 算力打平 DeepSeek 7B Chat 与 LLaMA2 7B SFT。

## 5. 为什么独立成篇(与结构篇的分工)

本仓是「模型发布 + 论文」,网络结构篇是「源码级结构剖析」。为避免内容重叠(库规约:一条知识只收一次),分工如下:

- **结构/源码/路由公式** → [deepseek-arch-v1-moe-v2](../../model-architecture/concepts/deepseek-arch-v1-moe-v2.md)(含 MoE 层代码、路由公式、共享专家实现)。
- **数据规模 + 评测 + 40% 算力卖点** → 本篇。
- 两者互链,读者按需跳。

## 实践要点

- **「激活参数 vs 总参数」是选 MoE 模型的第一问**:显存按总参数(16.4B)、速度按激活参数(2.8B)——部署规划时两个数字都要看,别只盯着总参。
- **细粒度+共享专家是 DeepSeek MoE 的固定配方**:后续所有 DeepSeek MoE 模型都沿用,遇到 DeepSeek 系 MoE 先想到这两条。
- **40% 算力打平是稀疏化的「甜点」示例**:它证明稀疏 MoE 在 7B 量级就已划算(不必等 70B+ 大模型),是 MoE 普惠化的早期证据。

## 溯源

- DeepSeekMoE 论文:https://arxiv.org/abs/2401.06066
- 仓库:https://github.com/deepseek-ai/DeepSeek-MoE

## 相关条目(库内互链)

- **结构/源码/路由机制**(本域不重复的部分):[model-architecture/concepts/deepseek-arch-v1-moe-v2.md](../../model-architecture/concepts/deepseek-arch-v1-moe-v2.md)。
- DeepSeek MoE 路线总览:[model-architecture/concepts/deepseek-model-evolution.md](../../model-architecture/concepts/deepseek-model-evolution.md)。
- 用 MoE 底座做的领域模型:Coder-V2 见 [deepseek-coder.md](deepseek-coder.md)。
