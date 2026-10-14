---
title: DeepSeek-VL & DeepSeek-VL2 — 图文理解双代(Hybrid 双塔 → MoE 动态分块)
type: concept
tags: [deepseek, vision-language, vl, vl2, hybrid-encoder, siglip, sam, moe, dynamic-tiling, projector]
series: deepseek-vision
source:
  - "web: DeepSeek-VL GitHub README | https://github.com/deepseek-ai/DeepSeek-VL | 2026-10 抓取"
  - "web: DeepSeek-VL2 GitHub README | https://github.com/deepseek-ai/DeepSeek-VL2 | 2026-10 抓取"
  - "arXiv: DeepSeek-VL: Towards Real-World Vision-Language Understanding | 2403.05525 | https://arxiv.org/abs/2403.05525"
  - "arXiv: DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding | 2412.10302 | https://arxiv.org/abs/2412.10302"
  - "code: deepseek-vl-7b-chat config.json | https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat/raw/main/config.json"
  - "code: deepseek-vl2(-tiny/-small) config.json | https://huggingface.co/deepseek-ai/deepseek-vl2*/raw/main/config.json"
date: 2026-10
---

# DeepSeek-VL & DeepSeek-VL2 — 图文理解双代

> 本篇是 deepseek-vision 域「图文理解谱系」的深度篇:DeepSeek-VL(2024-03)与 DeepSeek-VL2(2024-12)两代,合一篇讲清「vision encoder 怎么接 LLM」这条主线的第一段。

## TL;DR(30 秒结论)

**DeepSeek-VL 到 VL2,是「视觉编码器怎么高效产出 LLM 能消费的 token」这条主线的第一步演进:VL 用「冻结的 SigLIP + SAM 双塔(Hybrid)」各自编码、MLP 投影拼接进 LLM;VL2 把视觉侧收敛成「单个可训的 SigLIP SO400M」+「动态分块 tiling」处理任意分辨率,把语言侧换成「DeepSeekMoE(MHA,config 实测)」的稀疏骨干。** 一句话:**VL 解决「高分辨率图怎么在固定 token 预算下编码」,VL2 解决「怎么用稀疏算力 + 动态分辨率同时吃掉性能与效率」。**

两代共享同一个融合范式:**vision tower → projector(MLP)→ LLM**。区别只在 vision tower 的构造(双塔冻结 vs 单塔可训+分块)与 LLM 的稀疏化(dense vs MoE)。

---

## 一、DeepSeek-VL:Hybrid 双塔 + 冻结 + 投影

### 机制原理(为什么是「Hybrid 双塔」)

DeepSeek-VL 的核心设计动机写在论文标题里——「Towards Real-World Vision-Language Understanding」:真实场景的输入是**高分辨率、信息密集**的(网页截图、PDF、图表、公式、科学文献),而 vision encoder 面临一个两难:

- **低分辨率塔(SigLIP-L@384)**语义理解强,但 384² 会把 1024² 的密集文档糊掉,细节(小字、公式)丢光;
- **高分辨率塔(SAM-B@1024)**能保留细节,但 1024² 的 patch 序列太长(1024/16)² = 4096 token,直接进 LLM 会撑爆 token 预算、算力暴涨。

DeepSeek-VL 的解法是**「低塔出语义、高塔出细节」的 Hybrid(混合)双塔**,两个塔都**冻结**(freeze_high / freeze_low = true),只训练中间的投影层与 LLM。冻结的理由:两个塔都是现成的强预训练模型(SigLIP 是图文对齐的、SAM 是分割预训练的),冻结既省算力又避免灾难性遗忘,还能保证 encoder 输出稳定。

**可复现细节(config.json 实测,deepseek-vl-7b-chat)**:

```jsonc
"vision_config": {
  "cls": "HybridVisionTower",          // 双塔容器,concat_type="tuple"
  "params": {
    "low_res_cfg":  { "model_name": "siglip_large_patch16_384", "image_size": 384, "output_dim": 1024 },
    "high_res_cfg": { "model_name": "sam_b_downsample",         "image_size": 1024, "output_dim": 1024 },
    "freeze_low": true, "freeze_high": true                     // 双塔都冻结
  }
},
"aligner_config": {
  "cls": "MlpProjector",
  "params": { "depth": 2, "input_dim": 1024, "n_embed": 4096,
              "projector_type": "low_high_hybrid_split_mlp_gelu" }   // 高低两路分开投影再融合
},
"language_config": { "model_type": "llama", "num_hidden_layers": 30, "vocab_size": 102400 }
```

关键点:**高塔 SAM-B 是「downsample」变体**(降采样后的 SAM 视觉编码器,全局注意力集中在 4 层),输出 1024 维,与 SigLIP-L 的 1024 维对齐,这样投影层的 `input_dim=1024` 可以统一处理两路。投影器是 `low_high_hybrid_split_mlp_gelu`——**高低两路先各自投影、再拼接融合**,而不是先 concat 再一个 MLP,这是 Hybrid 命名在代码层的体现。

### 融合方式总结

```
1024² 图像 ──→ SAM-B(downsample) ──→ 高分辨率特征(细节:小字/公式)
384²  图像 ──→ SigLIP-L@384      ──→ 低分辨率特征(语义)
                    │                      │
                    ▼                      ▼
            split MLP 投影(各自投影 + 融合,4096 维)
                    │
                    ▼
           dense LLM(1.3B/7B) → 文本
```

### 训练策略:先保语言、再平衡模态

论文强调一个「竞争动态」:**VL 预训练从 LLM 训练开始,视觉模态逐步加入、ratio 渐增**。这回答「为什么多模态模型常丢语言能力」——vision 与 language 在共享参数里抢容量,先让模型站稳语言底座,再逐步混入视觉,是保护 LLM 能力的关键。这是 VL 区别于「直接把视觉 token 塞进冻结 LLM」的早期工作(如 LLaVA 冻结 LLM)的一个务实选择。

### 数据与规模

- 数据覆盖真实场景:web 截图、PDF、OCR、图表、知识内容(教科书/专家知识),并构建「use case 分类学」做指令微调,直接改善 chatbot 体验。
- 四尺寸:1.3B-base/chat、7B-base/chat,seq len 4096。

---

## 二、DeepSeek-VL2:SO400M 单塔 + 动态分块 + MoE 骨干

### 机制原理(三大升级,论文 §1 明示)

VL2 相对 VL 有两大(论文原文)到三大升级:

1. **视觉侧:动态分块(dynamic tiling)视觉编码**——处理不同宽高比的高分辨率图。不是固定 384²/1024²,而是把图切成**候选分辨率网格里的多个 tile**,每个 tile 单独过 encoder,再拼成 token 序列。config.json 里的 `candidate_resolutions` 有 23 种(384² → 384×3456、1152² 等),`global_view_pos="head"` 表示在序列头部放一个全局视图(整图下采样的缩略图)token,实现「全局 + 局部」联合编码。`tile_tag="2D"` 表示 2D 分块(带行列位置)。
2. **视觉塔收敛成单个 SigLIP SO400M**:`model_name="siglip_so400m_patch14_384"`,`width=1152`,`layers=27`——比 VL 的 SigLIP-L 更大,而且**可训**(不再是冻结双塔)。「多图联合编码」的「多图」即多个 tile + 全局视图,统一由这一个塔编码,不再需要 SAM 高塔。
3. **语言侧:DeepSeekMoE(MHA)**——把 dense LLM 换成 DeepSeek 自家的 MoE 稀疏骨干。注意口径:论文写 MLA,但开源 checkpoint config 实测 `use_mla=false`(tiny 显式 false,small 缺省即 false)、`num_key_value_heads=num_attention_heads`(MHA),论文与 config 两口径不一;按 config 部署勿套 MLA 假设(KV cache 估算会失真)。稀疏激活(1.0/2.8/4.5B)来自 MoE 细粒度专家,与注意力形式无关。

### 三个变体(总参 / 激活参数,README + config 实测)

| 变体 | 激活参数 | hidden_size | 专家数 | 每 token 专家 | 共享专家 |
|---|---|---|---|---|---|
| VL2-Tiny | 1.0B | 1280 | 64 | 6 | 2 |
| VL2-Small | 2.8B | 2048 | 64 | 6 | 2 |
| VL2(full) | 4.5B | 2560 | 72 | 6 | 2 |

> ⚠ **VL2 full 激活参数上游自相矛盾**:README intro 写 **4.5B**,但同一 README 的 web_demo 段写 `activated 4.2B`——保留 4.5B 口径并标注这一矛盾(以 README intro 为准)。

> 📌 **激活参数字段两口径辨析(README vs 论文 Table)**:README intro 的 **1.0B/2.8B/4.5B** 是「总激活」(LLM + 视觉塔);论文 Table 另列「仅 LLM 侧激活」**0.57B/2.4B/4.1B**(总参 3B/16B/27B;README web_demo 同口径精确写 total 3.37B/16.1B/27.5B)。两者不冲突——分别指「含视觉塔的整模型激活」与「语言骨干激活」。

> MoE 路由分变体:full 版 `scoring_func="sigmoid"`、`topk_method="noaux_tc"`(与 DeepSeek-V3 的 aux-loss-free + sigmoid 路由同源,见 model-architecture 域);Tiny/Small 实测 `topk_method="greedy"`。full 版 72 专家 + 2 共享,激活 6。

### 融合方式总结

```
高分辨率图 ──→ 切成候选分辨率网格的多个 tile + 1 个全局视图
                │(每块独立过同一塔)
                ▼
          SigLIP SO400M@384(27 层,可训)→ tile tokens(带 2D 位置)
                │
                ▼
          MLP projector(n_embed=2560)
                │
                ▼
     DeepSeekMoE 骨干(MHA,见上文注;Tiny/Small/full)→ 文本
```

### 数据与规模

- README:VQA、OCR、文档/表格/图表理解、视觉定位(visual grounding)全面领先,「以更少或相当的激活参数」达到 SOTA 或竞争水平。
- 论文 Fig.1 口径:横轴激活参数、纵轴 MMBench/MMStar/MMMU/MathVista/AI2D/OCRBench 平均——VL2 在激活参数-性能曲线上明显优于同档 dense 与 MoE 开源模型。
- 三变体 seq len 4096;推理 deepseek-vl2-small 约需 80GB 显存(README 提示)。

---

## 三、两代对比与演进逻辑(核心结论)

| 维度 | DeepSeek-VL(2024-03) | DeepSeek-VL2(2024-12) |
|---|---|---|
| 视觉塔 | SigLIP-L@384 + SAM-B@1024(双塔,冻结) | SigLIP SO400M@384(单塔,可训) |
| 高分辨率处理 | 固定双塔分别出低/高特征 | 动态分块 tiling + 全局视图 |
| 投影 | split MLP(低高两路分投再融) | 单 MLP projector |
| 语言骨干 | dense LLM 1.3B/7B | DeepSeekMoE(1.0/2.8/4.5B 激活,MHA,见注) |
| 设计主线 | 冻结强编码器 + 保护语言能力 | 稀疏算力 + 动态分辨率同时吃性能/效率 |

**演进逻辑三句话**:

1. **「双塔冻结」→「单塔可训」**:VL 借 SAM 高塔补细节、借 SigLIP 出语义,靠「冻结现成强编码器」省成本;VL2 发现单个更大的 SO400M + 分块就能同时满足语义与细节,于是把「用两个冻结塔」简化成「一个可训塔 + 分块」——**复杂度下降,表达力反而更强**。
2. **「固定分辨率」→「动态分块」**:VL 的 1024² 是写死的;VL2 的 23 种候选分辨率 + 全局视图让 encoder 适应任意宽高比文档,这是「多图联合编码」的实质——不是多张图,是**一张图切多块联合编码**。
3. **「dense」→「MoE」**:语言侧承接 DeepSeek 主线的稀疏化(细粒度专家 + 共享专家;注意力按 config 实测为 MHA,未沿用主线 MLA),把图文模型的「性能/激活参数」比拉到开源第一梯队——与库内 model-architecture 域的 DeepSeekMoE 同源。

---

## 实践要点(checklist)

- **复现 VL**:`pip install -e .`(Python≥3.8),`VLChatProcessor.from_pretrained` + `AutoModelForCausalLM`;双塔权重加载需 `ckpt_path` 指向 SigLIP/SAM 预训练权重。
- **复现 VL2**:三变体从 HF 直接 `from_pretrained`;`candidate_resolutions` 决定分块方案,长文档/宽表格场景可自定义分辨率组合。
- **显存**:VL2-small 推理约 80GB、full 更大(README 明示)——先评估显卡再选变体。
- **坑**:VL 双塔冻结时若 `ckpt_path` 留空会尝试随机初始化,务必对齐 SigLIP-L@384 / SAM-B 的官方权重路径;VL2 的 `global_view_pos="head"` 是把全局视图放序列头部,下游若手工拼 token 顺序要对齐这个约定。

---

## 溯源

- DeepSeek-VL 论文:https://arxiv.org/abs/2403.05525
- DeepSeek-VL 仓库:https://github.com/deepseek-ai/DeepSeek-VL
- DeepSeek-VL2 论文:https://arxiv.org/abs/2412.10302
- DeepSeek-VL2 仓库:https://github.com/deepseek-ai/DeepSeek-VL2
- config 口径:https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat 、https://huggingface.co/deepseek-ai/deepseek-vl2(-tiny/-small)

## 相关条目(库内互链)

- DeepSeekMoE 机制与注意力形式(MHA/MQA/GQA/MLA 主线):../model-architecture/concepts/deepseek-arch-v1-moe-v2.md、../model-architecture/concepts/attention-heads-mha-mqa-gqa.md
- 视觉编码器后续谱系(OCR/Janus):deepseek-ocr-ocr2.md、janus.md
