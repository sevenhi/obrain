---
title: Vision Transformer 基础机制 — patchify、CLS token 与 ViT 为什么需要大数据
type: concept
tags: [vit, vision-transformer, patch-embedding, cls-token, position-embedding, self-attention, transformer, image-classification, DeiT]
source:
  - "pdf: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2020, ICLR 2021) | https://arxiv.org/abs/2010.11929 | 阅读日期 2026-10-14 ↑primary"
  - "pdf: ViT 论文 HTML 全文(Table 1 模型变体 / Table 2 & 5 精度) | https://arxiv.org/html/2010.11929v2 | 抓取日期 2026-10-14"
  - "code: ViT 官方仓库(google-research/vision_transformer) | https://github.com/google-research/vision_transformer | 论文 abstract 注明 finetune 代码与预训练模型发布处"
  - "pdf: Training data-efficient image transformers & distillation through attention (DeiT, Touvron et al., 2021, ICML 2021) | https://arxiv.org/abs/2012.12877 | 阅读日期 2026-10-14 ↑primary"
  - "pdf: End-to-End Object Detection with Transformers (DETR, 交叉注意力用于视觉的先例,ViT 论文引用) | https://arxiv.org/abs/2005.12872 | 阅读日期 2026-10-14"
date: 2026-10
---

# ViT 基础机制:把图像当「16×16 个词的序列」

> 新手友好篇。ViT(arXiv:2010.11929, Google Brain, ICLR 2021)证明:**不需要卷积**,纯 transformer 直接吃图像 patch 序列就能做分类。本域三篇(trt-vit-inference 硬件视角、[检测/分割篇](object-detection-segmentation.md) 的 DETR/SAM)都以此为机制底座。数字均核自论文 Table 1/2/5 与 DeiT abstract。

## TL;DR

ViT 的全部改动只有三步:①**patchify**——把 224×224 图像切成 16×16 的小块,每块拉平后线性投影成一个向量,得到 196 个「视觉词」;②**加 [CLS] token + 可学习 1D 位置编码**——拼一个可学习的分类 token 在序列头,让标准 transformer encoder 处理;③**预训练-微调范式**——在大私有集 JFT-300M(3 亿图)上预训练,迁移到 ImageNet 等下游。结论两半:数据够大时,ViT-H/14 达 **88.55% ImageNet top-1**,超越所有 CNN 且**训练算力更省**;数据不够时(只有 ImageNet-1k),ViT 输给 ResNet——**ViT 缺的是 CNN 自带的归纳偏置(局部性/平移等变),要靠大数据「学回来」**。DeiT(arXiv:2012.12877)随后证明只用 ImageNet-1k 训练 + 蒸馏 token 也能到 83.1%(86M 参数、单机 3 天内),把 ViT 平民化。落地:直接用 torchvision 的 `vit_b_16` 或 timm;硬件效率问题见 [trt-vit-inference](trt-vit-inference.md)。

## 核心思想:为什么能「把图像当句子」

一个直觉例子:读一句话时,transformer 不关心「词」长什么样,只把它们当向量序列做自注意力——每个词和其他所有词算相关性。图像其实也能这样切:一张 224×224 的图,按 16×16 切格,得 14×14=**196 个 patch**;每个 patch 是 16×16×3=768 维的像素块,拉平再乘一个可学习矩阵投影到 D 维,就成了一个「视觉词」。于是「图像分类」变成「读一个 196 词(+1 个 [CLS])的句子,输出 [CLS] 位置的类别」。

为什么这事 2020 年才成?ViT 论文 §1 说得很直白:此前不是没人试过(论文引用了 DETR 等把注意力用于视觉的工作),而是**专用 attention 模式难以在现代加速器上规模化**;ViT 的赌注是「用最标准的 transformer、最少的改动」,换来 NLP 十年积累的整条 scaling 基础设施(数据并行/优化器/预训练配方)直接可用。**代价是归纳偏置**:CNN 天生假设「邻近像素相关(局部性)、图案在哪都一样(平移等变)」,ViT 什么都没有,必须从数据里学——这就是它对数据量极度敏感的原因(下文 §数据敏感性)。

## 机制

### 1. 输入流水线:patchify + 位置编码 + [CLS]

论文 §3.1(Fig.1)的公式化流程,对 224×224 输入、P=16:

```text
x ∈ R^(224×224×3)
  → 切 patch:   N = (224/16)² = 196 个,每个 16×16×3 = 768 维
  → 线性投影:   乘可学习 E ∈ R^(768×D) → N 个 D 维向量
  → 拼 [CLS]:   序列变 N+1 = 197,头部加可学习分类向量 z⁰_class
  → 位置编码:   加可学习 1D 位置嵌入 E_pos ∈ R^((N+1)×D)(不是 2D!)
  → 标准 transformer encoder × L 层(LN → MHSA → LN → MLP,残差连接)
  → 分类头:     取 [CLS] 输出接 MLP → 类别 logits
```

三个容易误解的细节:

- **patch embedding = 卷积**:「拉平+线性投影」在实现上就是一个 `kernel_size=stride=16` 的 Conv2d,一步完成切块与投影(torchvision/timm 均如此)。
- **位置编码是 1D 可学习的**,不是 sin-cos 也不是 2D 网格。论文 §3.2 实验:去掉位置编码精度明显掉,但 1D 与 2D 初始化几乎无差——模型自己从数据里学会了空间结构(论文 Fig.5 显示学到的位置嵌入相似度矩阵自然呈现同行/同列带状结构,且邻近 patch 比远距离更相关)。
- **[CLS] 不是必须的**:论文附录讨论了 CLS vs GAP(全局平均池化)两种分类头,后续 timm 等实现多用「token pooling」抽象;初学按「[CLS] 是序列级答案的收集槽」理解即可——它没有像素含义,全靠注意力从 196 个 patch 收集信息。

### 2. 模型变体(论文 Table 1 原数)

| 变体 | Layers | Hidden size D | MLP size | Heads | Params |
|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | **86M** |
| ViT-Large | 24 | 1024 | 4096 | 16 | **307M** |
| ViT-Huge | 32 | 1280 | 5120 | 16 | **632M** |

命名规则 `ViT-L/16`:L=模型大小,/16=patch 边长。**patch 越小 → 序列越长 → attention 二次方越贵但精度越高**(L/16 比 L/32 序列长 4 倍)。这一「分辨率-序列长度-算力」三角是本域 [trt-vit-inference](trt-vit-inference.md) 全部效率分析的出发点:attention 的 token 数随分辨率平方增长,高分辨率下访存与 O(n²) 主导,所以 TRT-ViT 把 Transformer 只放低分辨率后期 stage。

### 3. 数据敏感性:ViT 的成败分水岭(论文 Table 2/5 原数)

| 预训练数据 | 模型 | ImageNet top-1 |
|---|---|---|
| JFT-300M(私有 3 亿图) | ViT-H/14 | **88.55%**(论文最佳;ImageNet-ReaL 90.72%、CIFAR-100 94.55%) |
| JFT-300M | ViT-B/16 | 84.15% |
| JFT-300M | ViT-B/32 | 80.73% |
| ImageNet-21k(公开 1400 万) | ViT-L/16 | 85.30%(Table 2;低于 BiT-L 87.54、Noisy Student 88.4/88.5) |
| ImageNet-1k(130 万) | ViT | **低于同规模 ResNet**(论文 Fig.4:小数据区 CNN 全面占优) |

论文 Fig.4 的横轴是预训练数据量:小数据端 CNN 在上,大数据端 ViT 反超且**没有饱和迹象**。因果解释(论文 §4.1):CNN 的局部性/等变先验在小数据上是「免费午餐」,大数据上反而成天花板;ViT 无先验,小数据吃亏、大数据受益。另一条正交结论(Table 2 原数):ViT-H/14 预训练只花 **2.5k TPUv3-core-days**,不到 BiT-L(9.9k)的 1/3、Noisy Student(12.3k)的 1/5——大数据下 ViT 不仅更准还更省训练算力。

**DeiT(arXiv:2012.12877)把门槛打下来**:只用 ImageNet-1k,86M 参数(结构=ViT-B)训 **3 天内单机**达 **83.1% top-1**(single-crop);外加一个**蒸馏 token**([CLS] 旁再加一个 [DIST] token 学教师网络,CNN 当教师也行),最高 **85.2%**。训练配方(强增强 RandAug/Mixup + 正则 Repeated Aug/stochastic depth)比架构更重要——「数据饥饿」可以靠配方缓解,不必真有 3 亿图。

### 4. ViT 与本库其它篇的关系(跨域一致性)

- **[trt-vit-inference](trt-vit-inference.md)**:同一 ViT 骨干的**硬件效率视角**——本篇讲「patchify/CLS/位置编码是什么」,它讲「这些 attention 算子在 T4+TensorRT 上为什么慢、怎么重排 stage」。两篇的 ViT block 结构(LN+MHSA+LN+MLP,GeLU)一致。
- **[检测/分割篇](object-detection-segmentation.md)**:DETR 用 transformer encoder 处理 CNN 特征图位置序列(思想同 patchify,粒度是特征图 cell);SAM 的图像编码器**就是 MAE 预训练的 ViT-H(本篇 Table 1 的 632M 变体)**。
- **[deepseek-vision 域](../../deepseek-vision/index.md)**:DeepSeek-VL/Janus 的视觉编码器 SigLIP-L 与 SAM-B 均为 ViT 家族(patchify+CLS/池化 机制同本篇);理解「视觉 token 怎么进入 LLM」先要有本篇的 token 概念。
- **[attention 域 AIANY](../../attention/algorithm/attention-is-all-you-need.md)**:ViT 的 encoder 层与 AIANY 的 encoder block 逐组件对应(self-attn/FFN/残差/LN),差别只在输入是 patch 而非词、位置编码可学习而非 sin-cos、用 pre-LN。

## 关键代码

最小实现:不用任何库,~30 行 PyTorch 写出 ViT 前向(torch>=2.0 可运行;教学用,省略 drop_path 等训练技巧):

```python
import torch, torch.nn as nn

class ViT(nn.Module):
    def __init__(self, img=224, patch=16, in_ch=3, dim=768, depth=12,
                 heads=12, mlp=3072, num_classes=1000):
        super().__init__()
        self.n = (img // patch) ** 2                       # 196 个 patch
        self.proj = nn.Conv2d(in_ch, dim, patch, stride=patch)  # patchify+线性投影一步完成
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))    # [CLS] token
        self.pos = nn.Parameter(torch.zeros(1, self.n + 1, dim))  # 可学习 1D 位置编码
        layer = nn.TransformerEncoderLayer(
            dim, heads, mlp, activation="gelu",
            batch_first=True, norm_first=True)             # pre-LN,与论文一致
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):                                  # x: (B,3,224,224)
        x = self.proj(x).flatten(2).transpose(1, 2)        # (B,196,768)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)  # (B,197,768)
        x = self.enc(x + self.pos)
        return self.head(self.norm(x[:, 0]))               # 取 [CLS] 输出分类

vit = ViT()
print(sum(p.numel() for p in vit.parameters()) / 1e6)      # ≈86M,对应论文 Table 1 ViT-Base
print(vit(torch.rand(2, 3, 224, 224)).shape)               # torch.Size([2, 1000])
```

工程上不要自己造轮子,用官方权重:

```python
# torchvision(>=0.13 起内置 ViT;IMAGENET1K_V1 权重源自 google-research 官方 JFT/IN21k 预训练体系——torchvision 复现了论文发布的权重,非独立私训)
import torchvision
m = torchvision.models.vit_b_16(weights="IMAGENET1K_V1").eval()

# 或 timm(模型花园最全,含 DeiT 全系列)
import timm
m = timm.create_model("deit_base_patch16_224", pretrained=True).eval()
```

## 落地上手

- [ ] **微调而非从头训**:没有 JFT 级数据就永远从预训练权重起步(ImageNet-21k 预训练的 ViT-B/16 是常见性价比点);输入分辨率可以改,位置编码需插值(官方仓库与 timm 均内置)。
- [ ] **patch 大小按算力预算选**:/32 快但糙,/16 是标准,/14 精度最高算力最贵;下游密集任务(检测/分割)输入分辨率大时优先 /16 起步,并用 [trt-vit-inference](trt-vit-inference.md) 的方法在目标硬件实测延迟。
- [ ] **小数据集(<1M 图)**:按论文 Fig.4 与 DeiT 的结论,优先 CNN 或蒸馏版 DeiT + 强增强配方;纯 ViT 从头训基本必输。
- [ ] **导出部署**:ViT 的 LayerNorm/GELU/动态 shape 导出细节走 [ONNX→TensorRT 部署篇](../frameworks/onnx-tensorrt-deployment.md);TRT-ViT 的教训是**导出前先在 TensorRT 上实测延迟**,别信 FLOPs。
- [ ] 训练配方参考 DeiT:AdamW、强增强(RandAugment+Mixup)、正则(Repeated Augmentation+stochastic depth)、长训练;ViT 对增广/正则比 CNN 敏感得多(DeiT 论文 §3 消融)。

## 局限

- **归纳偏置缺失的账单**:小数据/低分辨率场景不如 CNN(论文 Fig.4);对输入扰动与分布偏移的鲁棒性研究(如 adversarial robustness)显示与 CNN 各有胜负,不是免费升级。
- **O(n²) 注意力**:序列长度随分辨率平方增长,高清输入(≥512px)算力/显存爆炸——这正是 Swin(窗口注意力)、TRT-ViT(stage 混合)、乃至 [attention 域 FA 谱系](../../attention/index.md)(内核级精确解)各自要解决的问题。
- **[CLS] 头 vs 密集任务**:分类靠 [CLS] 一个向量,但检测/分割需要每个 patch 的输出特征——下游任务里 CLS 常被丢弃,「ViT 当骨干」实际用的是 patch token 特征(见检测篇 DETR/SAM 用法)。
- **可解释性**:attention 图能看个大概但不忠实(attention ≠ 归因),别拿它当因果解释。
- 本篇所有精度数字绑定论文的预训练数据与微调设置(JFT-300M/ImageNet-21k,高分辨率微调 512/518),复现时缺同等数据必然对不上。

## 溯源

- 论文一手:arXiv:2010.11929v2 HTML 全文(2026-10-14 抓取核对)——Table 1 模型变体全量数字(86M/307M/632M、层数/hidden/MLP/heads)、abstract+§4 最佳成绩 88.55% ImageNet / 90.72% ReaL / 94.55% CIFAR-100 / 77.63% VTAB、Table 5 JFT 预训练 ViT-B/16 84.15% 与 ViT-B/32 80.73%、§3.1 patchify/1D 可学习位置编码/[CLS]、§3.2 位置编码消融与 Fig.5 相似度结构、Fig.4 数据量-精度曲线(CNN 小数据占优)、Table 2 TPUv3-core-days 训练算力对比、附录 CLS vs GAP 讨论;官方仓库 google-research/vision_transformer(论文 abstract 注明的发布地址)。
- DeiT 一手:arXiv:2012.12877 abstract(2026-10-14 核对)——86M 参数、83.1% single-crop、单机 <3 天、蒸馏至 85.2%、「DeiT-B 结构=ViT-B」;训练配方细节出自其 §3(未逐项抄录)。
- 交叉引用:DETR(arXiv:2005.12872)为 ViT 论文 §1 引用的「注意力用于视觉」先例,本库有独立篇([检测/分割篇](object-detection-segmentation.md));SAM 的 ViT-H 编码器见该篇 §4 与 SAM 论文(arXiv:2304.02643)。
- Table 2 训练算力(2026-10-14 核对):ViT-H/14(JFT)2.5k、ViT-L/16(JFT)0.68k、ViT-L/16(I21k)0.23k TPUv3-core-days,对照 BiT-L 9.9k、Noisy Student 12.3k。
- 待核项:torchvision `IMAGENET1K_V1` 权重与论文原始 checkpoint 的对应关系(来自 Google 发布体系,细节未逐一核对)。
- 全库链接索引见 [references.md](../references.md)。
