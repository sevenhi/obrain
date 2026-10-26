---
title: 目标检测与实例分割 — 从两阶段 anchor 到端到端 set prediction 再到可提示分割(SAM)
type: concept
tags: [object-detection, instance-segmentation, faster-rcnn, mask-rcnn, detr, yolo, sam, nms, anchor, bipartite-matching]
source:
  - "pdf: Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (Ren et al., 2015) | https://arxiv.org/abs/1506.01497 | 阅读日期 2026-10-14 ↑primary"
  - "pdf: Mask R-CNN (He et al., 2017, ICCV 2017) | https://arxiv.org/abs/1703.06870 | 阅读日期 2026-10-14 ↑primary"
  - "pdf: End-to-End Object Detection with Transformers (DETR, Carion et al., 2020, ECCV 2020) | https://arxiv.org/abs/2005.12872 | 阅读日期 2026-10-14 ↑primary"
  - "code: DETR 官方仓库 README + Model Zoo | https://github.com/facebookresearch/detr | 抓取日期 2026-10-14"
  - "pdf: You Only Look Once: Unified, Real-Time Object Detection (YOLOv1, Redmon et al., 2016, CVPR 2016) | https://arxiv.org/abs/1506.02640 | 阅读日期 2026-10-14"
  - "web: Ultralytics YOLO11 官方文档(性能表/任务表/引用) | https://docs.ultralytics.com/models/yolo11/ | 抓取日期 2026-10-14"
  - "pdf: Segment Anything (SAM, Kirillov et al., 2023, ICCV 2023) | https://arxiv.org/abs/2304.02643 | 阅读日期 2026-10-14 ↑primary"
  - "code: SAM 官方仓库 README(checkpoints/SA-1B/API) | https://github.com/facebookresearch/segment-anything | 抓取日期 2026-10-14"
  - "code: SAM 2 官方仓库(视频扩展) | https://github.com/facebookresearch/segment-anything-2 | 抓取日期 2026-10-14"
date: 2026-10
---

# 目标检测与实例分割:模型怎么知道「图里有什么、在哪里、边界到哪」

> 新手友好综述篇。三条主线:两阶段(Faster/Mask R-CNN)、单阶段(YOLO)、端到端 set prediction(DETR),外加把分割变成「可提示基础模型」的 SAM。所有数字均来自论文原文或官方仓库/文档(见 frontmatter source 与文末溯源),查不到的标「待核」。

## TL;DR

**目标检测** = 分类(是什么)+ 定位(框在哪);**实例分割** = 再进一步给每个物体一个像素级掩码。技术演进三跳:①**两阶段**(Faster R-CNN):先用 RPN 提候选框再分类回归,准但慢(VGG-16 全流水线 5fps);Mask R-CNN 加一条并行掩码分支就把实例分割做出来了(ResNet-101 mask AP 35.7,5fps)。②**单阶段**(YOLO):把检测直接当回归问题,一次前向出框+类别,快(原版 45fps;Ultralytics YOLO11x 在 T4+TensorRT10 上 11.3ms,54.7 COCO mAP)。③**端到端**(DETR):transformer 编码-解码 + 100 个可学习 object query + 匈牙利二分图匹配,**删掉 anchor/NMS 全部手工组件**,R50 42 AP 与 Faster R-CNN 打平但 FLOPs 减半。**SAM** 换了个玩法:不学「类别」,学「按提示(点/框)分割任何东西」,靠 11M 图 / 11 亿掩码的 SA-1B 数据引擎练成零样本基础模型。落地首选:通用检测/分割用 Ultralytics YOLO11 三行上手;需要精细交互式掩码用 SAM;研究端到端范式看 DETR。

## 核心思想:为什么要专门一套「检测」机制

一个直觉例子:给分类网络(如 ResNet)看一张街景,它只能说「图里有车」;但你想要的是「三辆车,分别在这三个框里,其中一辆还被行人挡住一半」。难点在于**输出数量不定、位置任意、大小悬殊**:

下图把这三个难点和三条主线对起来看——上排是「难在哪」,中排是「为什么难」,下排是「三种回答」:

<div class="archify-figure">
<iframe src="../../../diagrams/cv-detection-core-difficulty.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="检测难在哪:输出数量不定、位置任意、大小悬殊"></iframe>
</div>

- 分类输出固定 K 维;检测输出是「不定长集合」(每张图 0~几百个物体)→ 必须设计「怎么把不定数量的预测对齐到不定数量的真值」。
- 物体可大可小(远处行人 20px,近处卡车 800px)→ 要么多尺度特征(FPN),要么预设不同大小的 anchor。
- 同一物体被多个位置重复预测 → 需要去重(NMS),或者干脆设计成「一个物体只许一个预测」(DETR 的二分图匹配)。

三条主线就是对「怎么对齐预测与真值」的三种回答:**anchor + NMS 后处理**(Faster R-CNN/YOLO)、**query + 匈牙利匹配**(DETR)、**提示 + 多掩码输出**(SAM)。

## 机制

### 1. 两阶段:Faster R-CNN → Mask R-CNN

**Faster R-CNN(arXiv:1506.01497)** 的关键发明是 **RPN(Region Proposal Network)**:一个全卷积网络在共享特征图上每个位置同时预测「候选框边界 + objectness 分数」,与检测头共享全部卷积特征,使候选框生成几乎零成本(论文原话 nearly cost-free)。RPN+Fast R-CNN 合并为单网络——用论文的说法,RPN 就是告诉网络「往哪看」的 attention 机制。VGG-16 下全流水线 5fps,每图仅 300 个 proposal 即达 PASCAL VOC/COCO SOTA。

下图是这条流水线的数据路径,注意末尾的分叉:Mask R-CNN 的掩码分支是从 RoI 特征**并行**长出来的,不是接在检测头后面:

<div class="archify-figure">
<iframe src="../../../diagrams/cv-faster-mask-rcnn.dataflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="Faster R-CNN 到 Mask R-CNN"></iframe>
</div>

**Mask R-CNN(arXiv:1703.06870)** 在 Faster R-CNN 上**并行**加一条掩码分支(不是串行——先框后掩码会伤精度),为每个实例输出像素级掩码;同框架还能直接扩展人体关键点。ResNet-101 下 mask AP 35.7、5fps,拿下当年 COCO 三项全冠。一个工程细节:掩码分支用 RoIAlign 替代 RoIPool 消除量化误差,这是掩码质量的关键(论文 §3)。

### 2. 单阶段:YOLO 家族

**YOLOv1(arXiv:1506.02640)** 把检测重构为**回归问题**:图像划成 S×S 网格,每格直接回归 B 个框+类别概率,单次前向完成(You Only Look Once),基础版 45fps、Fast YOLO 155fps。此后的 YOLOv2/v3/…/v8/v11 演进(多尺度 anchor→anchor-free、CSPNet 骨干、解耦头等)没有单一论文脉络;**Ultralytics 官方明确说明 YOLO11 未发表正式论文**,以官方仓库与文档为准(引用格式为 `@software{yolo11_ultralytics, version=11.0.0}`)。

官方文档的 COCO 检测表(640×640 输入,val5k,mAP 50-95):

| 模型 | mAP | T4 TensorRT10 (ms) | Params (M) | FLOPs (B) |
|---|---|---|---|---|
| YOLO11n | 39.5 | 1.5 | 2.6 | 6.5 |
| YOLO11s | 47.0 | 2.5 | 9.4 | 21.6 |
| YOLO11m | 51.5 | 4.7 | 20.1 | 68.1 |
| YOLO11l | 53.4 | 6.2 | 25.3 | 87.2 |
| YOLO11x | 54.7 | 11.3 | 56.9 | 195.3 |

YOLO11 一个仓库通吃检测/实例分割(-seg)/分类(-cls)/姿态(-pose)/旋转框(-obb)五任务;官方称 YOLO11m 比 YOLOv8m 参数少 22% 而 mAP 更高。NMS 仍在后处理里(YOLO11 需要;Ultralytics 2026 年的 YOLO26 主打端到端 NMS-free,官方文档提及,细节待核)。

### 3. 端到端:DETR(arXiv:2005.12872)

DETR(DEtection TRansformer)把检测视为**直接集合预测(set prediction)**,三个组件替掉全部手工设计:

1. **CNN 骨干 + transformer 编码器**:对特征图所有位置做全局自注意力,物体间关系与全局上下文显式建模(「去重」由注意力隐式完成——重复预测会被同一物体的 query 竞争掉)。
2. **固定 N=100 个可学习 object query + 解码器**:query 通过**交叉注意力**(概念同 [attention 域](../../attention/algorithm/attention-is-all-you-need.md) 的 encoder-decoder attention)从编码特征里「收集」自己负责的物体,并行输出 N 个预测。
3. **匈牙利二分图匹配损失**:预测集合与真值集合一对一最优匹配后再算损失——一个真值只配一个预测,因此**不需要 NMS,不需要 anchor**。

下图是这三件事的因果链:编码器提供全局上下文,query 通过交叉注意力各收集一个物体,训练时一对一匹配把「去重」学进权重,推理只剩阈值过滤:

<div class="archify-figure">
<iframe src="../../../diagrams/cv-detr-set-prediction.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="DETR 用匈牙利匹配替掉 anchor 与 NMS"></iframe>
</div>

官方 README:R50 骨干 **42.0 box AP(COCO val5k)**,与 Faster R-CNN 打平但 **FLOPs 减半、参数量相同**;推理只需 50 行 PyTorch。代价:训练收敛慢(500 epoch,是 Faster R-CNN 的十几倍),小目标弱(后续 Deformable DETR 缓解,本篇不展开)。掩码侧 DETR 还能统一做全景分割(R50 PQ 43.4,论文 Table 5 / 官方 panoptic 权重)。

### 4. 可提示分割基础模型:SAM(arXiv:2304.02643)

SAM 回答的问题不是「这是什么类」而是「你指哪我割哪」。三段式架构(论文 §2/Fig.1b):

下图按「运行频次 / 组件 / 成本」三行拆开这个架构——关键在于重编码器只跑一次,交互成本全落在轻解码器上:

<div class="archify-figure">
<iframe src="../../../diagrams/cv-sam-three-parts.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="SAM 的三段式:重编码器跑一次,轻解码器交互多次"></iframe>
</div>

- **重图像编码器**:MAE 预训练的 **ViT-H**(632M 参数的 ViT,见 [ViT 基础篇](vision-transformer-basics.md)),对整图算一次 embedding——贵但可摊销;
- **轻提示编码器**:点/框/掩码/文本提示,极便宜;
- **轻掩码解码器**:双向交叉注意力融合两路,给定 embedding 后**在浏览器 CPU 上 ~50ms 出一个掩码**(论文引言演示),支持交互。
- **模糊感知**:一个提示可输出多个候选掩码(如「衬衫上的人」vs「整个人」),训练时允许 3 个掩码头分别监督不同有效答案。

数据引擎(model-in-the-loop)三轮迭代攒出 **SA-1B:11M 张授权图像、11 亿掩码**,最终模型零样本迁移到各种分割任务,常胜过全监督基线。官方提供 vit_b/vit_l/vit_h 三档 checkpoint(Apache 2.0);**SAM 2**(arXiv:2408.00714)扩展到视频流。

## 关键代码/伪代码

四条流水线各给最小可运行入口(版本注明在注释;均只需 CPU 可跑通逻辑,GPU 更快):

```python
# ① 两阶段:torchvision 内置 Faster/Mask R-CNN(torchvision>=0.14,COCO 预训练)
import torchvision
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
preds = model([torch.rand(3, 480, 640)])   # NCHW, 0~1 float
# preds[0]: boxes(N,4) / labels(N) / scores(N) / masks(N,1,H,W) —— NMS 已在模型内做完

# ② 单阶段:Ultralytics YOLO11(ultralytics==11.x,首次运行自动下载权重)
from ultralytics import YOLO
model = YOLO("yolo11n.pt")                  # n/s/m/l/x;检测
results = model("bus.jpg")                  # 推理即一行;results[0].boxes / .masks(-seg 模型)
# 实例分割换 YOLO("yolo11n-seg.pt");导出部署: model.export(format="onnx")

# ③ 端到端:DETR 官方 torch.hub(README 原文用法)
import torch
model = torch.hub.load("facebookresearch/detr:main", "detr_resnet50", pretrained=True).eval()
# 注意:DETR 输出是 100 个 (score, label, cxcywh 归一化框),需自己按 score 阈值过滤,无 NMS

# ④ 可提示分割:SAM(官方 README 原文 API;pip install git+https://github.com/facebookresearch/segment-anything.git)
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
predictor.set_image(rgb_image)                            # 重编码器只跑一次
masks, _, _ = predictor.predict(point_coords=pts, point_labels=lbls)  # 交互提示 ~50ms 级
```

伪代码看对齐机制的本质差别:

下图把三条路线的推理流程并排放,看「谁负责去重」这一列——这是三种范式最本质的分歧点:

<div class="archify-figure">
<iframe src="../../../diagrams/cv-three-paradigms-inference.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="推理期三条路线的对齐流程"></iframe>
</div>

```text
Faster R-CNN / YOLO(带 NMS):
  dense 预测(几千~几万个框)→ 阈值过滤 → NMS 去重 → 输出
DETR(set prediction):
  训练: 100 预测 ←匈牙利匹配→ M 真值(一对一)→ 匹配对算损失,未匹配的 query 监督为 ∅
  推理: 100 预测 → score 阈值过滤 → 输出(无 NMS)
SAM(promptable):
  image → ViT-H embedding(一次)
  loop: prompt(点/框)→ 轻解码器 → 每提示 ≤3 个候选掩码 + IoU 预测分数
```

## 落地上手

- [ ] **任务选型**:通用检测/实例分割/姿态/旋转框 → YOLO11(生态最全、部署最顺);交互式精细掩码/数据标注 → SAM;要研究端到端范式或全景分割 → DETR 系。
- [ ] **评价指标**:检测用 COCO mAP(AP^box, IoU 0.5:0.95 均值);分割用 mask AP 或 mIoU(语义分割);SAM 类零样本模型看论文的多 benchmark 迁移表。别只看 mAP@0.5,它对小目标和定位精度远不如 mAP 50-95 敏感。
- [ ] **部署链路**:上述模型全部可走 [ONNX→TensorRT 部署篇](../frameworks/onnx-tensorrt-deployment.md) 的流程;注意 DETR/SAM 的 transformer 部分导出时动态 shape 与算子覆盖问题(该篇 §3 的输入输出契约同样适用),YOLO11 的 NMS 后处理要决定「导出进图内还是留在 CPU」。
- [ ] **延迟预算**:边缘/实时 → YOLO11n-s(T4 TRT10 1.5-2.5ms);精度优先 → YOLO11x 或 DETR-R101;SAM ViT-H 图像编码重(交互场景靠 embedding 复用摊销,视频用 SAM 2 的流式 memory)。
- [ ] 两阶段模型(anchor/NMS 超参多)调优空间大但工程复杂;DETR 无手工组件但训练贵(500 epoch);YOLO 是「工程完成度」路线,官方文档/导出/量化生态一条龙。

## 局限

- **NMS 系**(Faster/Mask R-CNN、YOLO11):NMS 是启发式后处理,密集/遮挡场景会误删真框;anchor 设计引入先验偏置。DETR 用匹配消掉它,代价是收敛慢、小目标弱。
- **DETR**:100 query 固定上限(单图 >100 物体截断);高分辨率下 attention 二次复杂度贵(与 [trt-vit-inference](trt-vit-inference.md) 准则 1 同一根因)。
- **SAM**:不输出类别(只管「分割什么形状」,「是什么」需外接分类器/CLIP);细结构(发丝/栅栏)与低对比度边界仍是论文自认的 limitation;文本提示仅初步结果(论文 §2.4)。
- **数据面**:SA-1B 以「自然图像里的可分割物体」为主,域外(医学/遥感)零样本性能论文标注为参差,需自测。
- 本篇数字均绑定论文/官方发布的硬件与配置(如 YOLO11 表为 T4+TensorRT10、640px;DETR 为 R50/500ep;Mask R-CNN 为 R101),换环境必须重测——同一教训见 [trt-vit-inference](trt-vit-inference.md)。

## 与既有篇互链

- [ViT 基础机制](vision-transformer-basics.md) —— DETR 的编解码器与 SAM 的图像编码器(ViT-H)都建立在它之上;DETR 的交叉注意力即标准 transformer decoder attention。
- [trt-vit-inference](trt-vit-inference.md) —— 「Transformer 放低分辨率后期」准则直接决定检测骨干怎么混 CNN/Transformer(TRT-ViT 的 COCO/ADE20K 下游实验就是检测/分割任务)。
- [ONNX/TensorRT 部署](../frameworks/onnx-tensorrt-deployment.md) —— 本篇模型的导出/量化/数值回归/engine 兼容性全流程。
- [attention 域:AIANY](../../attention/algorithm/attention-is-all-you-need.md) —— DETR 所用自注意力/交叉注意力的机制源头。
- [deepseek-vision 域](../../deepseek-vision/index.md) —— DeepSeek-VL 用 **SAM-B 作为视觉编码器之一**(冻结双塔 SigLIP-L+SAM-B),SAM 的 embedding 在那边服务于图文理解;OCR 系 DeepEncoder 也含 SAM-B。

## 溯源

- 论文一手:Faster R-CNN(1506.01497,abstract 核 5fps/300 proposals/RPN 共享特征)、Mask R-CNN(1703.06870,abstract+Fig.2 核 5fps/mask AP 35.7/R101、§3 RoIAlign)、DETR(2005.12872,abstract+§3 核 set prediction/二分图匹配/100 query/无 NMS;官方 README Model Zoo 核 R50 42.0 AP、FLOPs 减半;panoptic PQ 43.4 实自论文 Table 5,官方 README 附 panoptic checkpoint)、YOLOv1(1506.02640,abstract 核 45fps/155fps)、SAM(2304.02643,§1-2 核三段架构/ViT-H MAE 预训练/浏览器 CPU ~50ms/多掩码模糊感知/SA-1B 11M 图 11 亿掩码/数据引擎 34→14 秒每掩码)。
- 官方仓库/文档一手:facebookresearch/detr README(2026-10-14 抓取,核 Model Zoo 数字与 torch.hub 用法)、ultralytics YOLO11 文档(2026-10-14 抓取,核 COCO 检测性能表五档全量数字、YOLO11m 少 22% 参数、未发表论文说明与 @software 引用、YOLO26 NMS-free 提示)、facebookresearch/segment-anything README(2026-10-14 抓取,核 vit_b/l/h checkpoint、SA-1B 规模、SamPredictor/SamAutomaticMaskGenerator API、Apache 2.0、SAM 2 链接与 arXiv:2408.00714)。
- 待核项:YOLO26 端到端细节(仅官方文档一句提及);DETR 500 epoch 收敛成本的定量对比(论文 Fig.4 训练曲线,本篇未抄录具体数值)。
- 全库链接索引见 [references.md](../references.md)。
