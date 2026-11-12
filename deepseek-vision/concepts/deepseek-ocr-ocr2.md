---
title: DeepSeek-OCR & DeepSeek-OCR-2 — 视觉压缩双代(光学压缩 → 视觉因果流)
type: concept
tags: [deepseek, ocr, optical-compression, deepencoder, vision-token, causal-flow, moe, vllm]
series: deepseek-vision
source:
  - "code: DeepSeek-OCR 固定快照（forward 数据流与 Gundam 模式） | https://github.com/deepseek-ai/DeepSeek-OCR/tree/09eaf526153e7a01ed16c9dea8c96282aaea29c0 | reviewed 2026-11-04"
  - "web: DeepSeek-OCR GitHub README | https://github.com/deepseek-ai/DeepSeek-OCR | 2026-10 抓取"
  - "web: DeepSeek-OCR-2 GitHub README | https://github.com/deepseek-ai/DeepSeek-OCR-2 | 2026-10 抓取"
  - "arXiv: DeepSeek-OCR: Contexts Optical Compression | 2510.18234 | https://arxiv.org/abs/2510.18234"
  - "arXiv: DeepSeek-OCR 2: Visual Causal Flow | 2601.20552 | https://arxiv.org/abs/2601.20552"
  - "code: DeepSeek-OCR / DeepSeek-OCR-2 config.json | https://huggingface.co/deepseek-ai/DeepSeek-OCR*/raw/main/config.json"
date: 2026-10
reviewed: 2026-11-04
---

# DeepSeek-OCR & DeepSeek-OCR-2 — 视觉压缩双代

> 本篇是 deepseek-vision 域「视觉压缩谱系」的深度篇:DeepSeek-OCR(2025-10)与 DeepSeek-OCR-2(2026-01)两代,合一篇讲清「vision encoder 把像素压成 LLM token」这条主线的第二段——**把图像当作文本的压缩容器**。

## TL;DR(30 秒结论)

**DeepSeek-OCR 用「光学压缩(Optical Compression)」重新定义 OCR:不把 OCR 当「看图识字」,而把图像当「文本的高压缩比载体」——一张文档图用几十到几百个 vision token 就能等价上千个 text token,LLM 负责「解压」。DeepSeek-OCR-2 再往前走一步:让 encoder 自己带「因果推理」,按语义(而非固定光栅顺序)重排 visual token,模拟人类扫读文档的因果顺序。** 一句话:**OCR 回答「vision token 能压多狠仍不失真」,OCR-2 回答「压完的 token 该按什么顺序喂给 LLM」。**

两代共享同一个**纯视觉大编码器(DeepEncoder)→ 小 MoE 解码器**范式,解码器都是 **DeepSeek3B-MoE-A570M**(64 专家 + 2 共享,570M 激活)。

下图概括两代编码器组件变化。图已于 2026-11-04 按固定 README 修正为一代 Native 四档及 Gundam 动态分块,并通过自动图检查与浏览器检查。编码器组件名相同也不意味着所有预处理与连接细节相同;本图不是完整配置差分或性能复现。

<div class="archify-figure">
<iframe src="../../../diagrams/dv-ocr-two-gen.tree.html"
        style="width:100%;height:660px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="DeepSeek-OCR 与 OCR-2 的结构对照"></iframe>
</div>

---

## 一、DeepSeek-OCR:Contextual Optical Compression

### 机制原理(为什么「纯视觉大模型」能压文本)

实际数据流不是独立双塔:图像先过 SAM-B,其特征作为 CLIP-L 的 patch embeddings,再把 CLIP 输出与 SAM 特征拼接后投影进语言模型。压缩主要来自编码/下采样后的 token 预算,不能归因于保持 token 数的逐 token 线性投影。**下图已于 2026-11-04 按固定源码修正 SAM→CLIP 及 SAM→拼接的旁路连线,并通过源码出处校验、自动图检查与浏览器检查;图止于视觉 embedding 排列,未展开语言模型解码。**

<div class="archify-figure">
<iframe src="../../../diagrams/dv-ocr-optical-compression.dataflow.html"
        style="width:100%;height:560px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="DeepSeek-OCR 的光学压缩数据流"></iframe>
</div>

论文的起点是 LLM 长文本处理的 O(n²) 账单:**一段文字用 text token 表示很长(千 token 级),但放进一张图用 vision token 表示很短(几十到几百 token)**。一张包含整页文档的图,vision token 数远少于等价文本 token 数——所以「用视觉当压缩介质」能拿到远高于文本的压缩比。这就是标题 **Contexts Optical Compression(上下文的光学压缩)** 的含义。

DeepSeek-OCR 的结构只有两个组件(论文原话):**DeepEncoder + DeepSeek3B-MoE-A570M 解码器**。

- **DeepEncoder 是核心引擎**:由 SAM-ViT-B 与 CLIP-L 构成,但 config 名称不能独自说明连接方式。固定快照 `DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py` 先执行 `sam_model`、再将结果传入 `vision_model`、最后拼接两级特征;`deepencoder/clip_sdpa.py` 在外部 patch embeddings 非空时直接消费它们。投影层为 `linear`(`2048 → 1280`);`image_size=1024` 是配置值之一,不代表只支持单分辨率。
- **压缩比是关键指标**:压缩比 = ground-truth text token 数 / 模型实际用的 vision token 数。

**可复现细节(config.json 实测,DeepSeek-OCR)**:

```jsonc
"vision_config": {
  "image_size": 1024,
  "model_name": "deeplip_b_l",          // DeepEncoder 容器名
  "width": {
    "clip-l-14-224": { "layers": 24, "width": 1024, "patch_size": 14 },
    "sam_vit_b":     { "layers": 12, "width": 768, "global_attn_indexes": [2,5,8,11] }
  }
},
"projector_config": { "input_dim": 2048, "model_type": "mlp_projector",
                       "n_embed": 1280, "projector_type": "linear" },
"language_config": {
  "model_type": "deepseek_v2", "hidden_size": 1280, "num_hidden_layers": 12,
  "n_routed_experts": 64, "n_shared_experts": 2, "num_experts_per_tok": 6,
  "moe_intermediate_size": 896, "use_mla": false
}
```

> 注意:`use_mla=false`——OCR 解码器虽是 DeepSeekMoE 结构,但**不用 MLA**(它要的是 token 吞吐,MLA 的低秩 KV 压缩在这里非必需)。`vocab_size=129280`、`max_position_embeddings=8192`。

### 压缩比的量化结论(论文摘要 + Fig.1)

下图把「压缩比 → 精度」这条曲线摊成三段,并标出两个档位各自能干什么活——这也是评估 OCR 系模型该看的主指标:

<div class="archify-figure">
<iframe src="../../../diagrams/dv-ocr-compression-curve.workflow.html"
        style="width:100%;height:660px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="压缩比与解码精度的对应关系"></iframe>
</div>

- **压缩比 <10× 时,解码(OCR)精度 97%**;压缩比 20× 时仍 ~60%——即「1 个 vision token 顶 20 个 text token」仍能保住 60% 精度,这是 LLM 长文本压缩/记忆遗忘研究的重要信号。
- **OmniDocBench 上,仅用 100 vision token 就超过 GOT-OCR2.0(256 token/页)**,且以**少于 800 vision token** 超过 MinerU2.0(平均 6000+ token/页)——「最少 vision token 的端到端 SOTA」。
- **生产吞吐**:单卡 A100-40G,每天可生成 200k+ 页 LLM/VLM 训练数据;PDF 并发 ~2500 tokens/s(README)。
- **生态**:2025-10-23 被上游 vLLM 官方支持(需 nightly,`NGramPerReqLogitsProcessor` 做重复 token 抑制)。

### 融合方式总结

```
文档图像/分块 ──→ SAM-B 特征 ──→ CLIP-L（消费 SAM 特征）
                      │             │
                      └── 两级特征拼接 ──┘
                      │  高压缩比:一张页 → 几十~几百 vision token
                      ▼
                linear 投影(2048 → 1280)
                      │
                      ▼
          DeepSeek3B-MoE-A570M(64 专家 top-6)→ 文本「解压」
```

---

## 二、DeepSeek-OCR-2:Visual Causal Flow

### 机制原理(为什么「因果流」能改 token 顺序)

下图把「光栅顺序 vs 因果重排」两条路并排画出来:左侧一代按坐标排序,跨栏句子会被图表 token 隔断;右侧二代让 SAM-B 出细节后交给 Qwen2-0.5B 按语义重排,再投影给同一个解码器:

<div class="archify-figure">
<iframe src="../../../diagrams/dv-ocr2-causal-flow.workflow.html"
        style="width:100%;height:660px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="Visual Causal Flow:光栅顺序 vs 因果重排"></iframe>
</div>

OCR-2 的动机是一个认知观察:**传统 VLM 把 visual token 按光栅顺序(左上→右下)固定送入 LLM,配固定位置编码**;但人类看复杂版面的文档不是光栅扫描,而是**遵循语义、受逻辑结构驱动、有因果性的顺序扫读**(先标题、再正文块、再图表…)。OCR-2 让 encoder 自己模拟这种「因果顺序」。

核心创新 **DeepEncoder V2**:赋予 encoder **因果推理能力,在 LLM 解读之前智能地重排 visual token**。论文提出的范式问题是:**「2D 图像理解能否用两级级联的 1D 因果推理结构达成」**——先由 encoder 做 1D 因果重排(决定看哪、按什么顺序),再由 LLM 做 1D 因果解码,逼近真正的 2D 推理。

**可复现细节(config.json 实测,DeepSeek-OCR-2)**:

```jsonc
"vision_config": {
  "image_size": 1024,
  "model_name": "deepencoderv2",        // 二代编码器
  "width": {
    "qwen2-0-5b": { "dim": 896 },        // 新增:Qwen2-0.5B 做因果重排底座
    "sam_vit_b":  { "global_attn_indexes": [2,5,8,11], "layers": 12, "width": 768 }
  }
}
// 解码器与 OCR 一代相同:DeepSeek3B-MoE-A570M(64 专家 top-6,use_mla=false)
```

关键差异:**DeepEncoder V2 在 SAM-B 之外引入了 Qwen2-0.5B(dim 896)**——用一个小型因果语言模型作为重排 token 的「因果底座」,这正是「Visual Causal Flow」在权重层的落点。相比一代的 CLIP-L,二代改用 Qwen2-0.5B 来提供因果/顺序推理能力。projector 相应适配:config 实测 `input_dim=896`(一代为 2048),把 Qwen2-0.5B 的 896 维因果特征线性投影进 MoE 解码器。

### 支持模式(README)

- **动态分辨率**:默认 (0–6)×768×768 + 1×1024×1024,对应 (0–6)×144 + 256 个 visual token——同 VL2 的分块思路,但服务于「重排」而非「语义编码」。
- **主提示词**:文档 → `<image>\n<|grounding|>Convert the document to markdown.`;无版面 → `<image>\nFree OCR.`。
- **速度**:PDF 并发与 DeepSeek-OCR 持平(README:on-par speed);OmniDocBench v1.5 有批评估脚本。

### 融合方式总结

```
文档图像(1024²) ──→ DeepEncoder V2(SAM-B 细节 + Qwen2-0.5B 因果重排)
                      │  视觉 token 按语义/因果顺序重排(非光栅)
                      ▼
                linear 投影(→1280)
                      │
                      ▼
          DeepSeek3B-MoE-A570M(64 专家 top-6)→ 文本
```

---

## 三、两代对比与演进逻辑(核心结论)

| 维度 | DeepSeek-OCR(2025-10) | DeepSeek-OCR-2(2026-01) |
|---|---|---|
| 编码器 | DeepEncoder = CLIP-L + SAM-B | DeepEncoder V2 = Qwen2-0.5B + SAM-B |
| 核心命题 | 光学压缩:图 = 文本高压缩载体 | 视觉因果流:按语义重排 token |
| token 顺序 | 光栅(默认) | 因果重排 |
| 解码器 | DeepSeek3B-MoE-A570M(同一) | 同一 |
| 分辨率模式 | Native 多档;Gundam: n×640² + 1×1024² | (0–6)×768² + 1×1024²(README 默认分块口径) |

**演进逻辑三句话**:

1. **「压缩多少」→「按什么顺序解压」**:一代解决「vision token 能压多狠」(压缩比 10× 仍 97%),二代解决「压完的 token 顺序对不对」——因为复杂版面里,光栅顺序本身就违背人类解读逻辑,重排顺序直接决定 LLM 解压质量。
2. **「CLIP-L 语义」→「Qwen2-0.5B 因果」**:两代 encoder 都在 SAM-B(细节)之上配一个「语义/顺序」塔,但一代用 CLIP(图文对齐,无因果),二代换成 Qwen2-0.5B(因果 LM)——**从「对齐语义」升级到「推理顺序」**,这是「Visual Causal Flow」的本质。
3. **范式野心**:OCR-2 真正想验证的是「两级级联 1D 因果推理能否替代真正的 2D 推理」——如果成立,就不需要设计复杂的 2D 注意力,这对所有 VLM 都有架构启示,而不仅是 OCR。

---

## 实践要点(checklist)

下图把部署链路和三条必踩的坑串起来:固定版本 → 选推理栈(vLLM 要 nightly)→ 关前缀缓存 → 挂重复抑制 → 按版面选 prompt;下方是三个不报错但会算错的点:

<div class="archify-figure">
<iframe src="../../../diagrams/dv-ocr-deploy-path.workflow.html"
        style="width:100%;height:700px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="DeepSeek-OCR 系部署链路与坑"></iframe>
</div>

- **环境**:cuda11.8 + torch2.6.0 + vllm-0.8.5(README 指定);transformers 推理用 `trust_remote_code=True` + `flash_attention_2`。
- **vLLM 推理(历史接入说明)**:2025-10 接入时曾要求 nightly(至 v0.11.1 前);不能沿用为当前所有发行版的安装要求。本文命令对应原 README 环境,升级后按目标 vLLM 版本核对 OCR 支持、prefix caching 与 `NGramPerReqLogitsProcessor` 配置。
- **OCR-2 推理**:`model.infer(..., base_size=1024, image_size=768, crop_mode=True)`;prompt 分「转 markdown(带 grounding)」与「Free OCR(无版面)」两种。
- **坑**:`config.py` 里 INPUT_PATH/OUTPUT_PATH 需先改;OCR 与 OCR-2 的 vLLM/transformers 同环境会报 transformers 版本冲突(README 明示无需处理);两代解码器都是 `use_mla=false`,别误套 DeepSeek-V3 的 MLA 配置。
- **压缩比是硬指标**:评估 OCR 系模型看「压缩比-精度曲线」,别只看榜单分数——一代的价值正在于「100 token 超 GOT-OCR2.0 256 token」。

---

## 溯源

- 2026-11-04 固定源码核验:上述 `09eaf526...` 快照中 `deepseek_ocr.py:394` 起的局部分块路径、`:404` 起的全图路径均为 SAM→CLIP→拼接;`clip_sdpa.py:131` 起确认 patch embeddings 输入语义;README 的 Native/Gundam 模式证明一代已有动态分块。本次未复现 OCR 精度/吞吐,未将 config 中模型名称当作独立双塔的证据。

- DeepSeek-OCR 论文:https://arxiv.org/abs/2510.18234
- DeepSeek-OCR 仓库:https://github.com/deepseek-ai/DeepSeek-OCR
- DeepSeek-OCR-2 论文:https://arxiv.org/abs/2601.20552
- DeepSeek-OCR-2 仓库:https://github.com/deepseek-ai/DeepSeek-OCR-2
- config 口径:https://huggingface.co/deepseek-ai/DeepSeek-OCR 、https://huggingface.co/deepseek-ai/DeepSeek-OCR-2

## 相关条目(库内互链)

- 图文理解前代(VL/VL2):deepseek-vl-vl2.md
- DeepSeekMoE 机制:../model-architecture/concepts/deepseek-arch-v1-moe-v2.md
- vLLM 官方支持(推理侧):../vllm/index.md
