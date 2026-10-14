---
title: Computer Vision - 参考资料索引
type: reference
tags: [nerf, vit, cnn, reparameterization, tensorrt, object-detection, segmentation, detr, yolo, sam]
source: "web: 本域论文来源索引 | 见下方 arXiv/官方链接"
date: 2026-10
---

# 参考资料索引

> 溯源用。只放链接与「看什么」,不复制全文。**全部 URL 于 2026-09-21 curl 实测存活(HTTP 200);2026-10-14 新增 ViT/DeiT/Faster R-CNN/Mask R-CNN/DETR/YOLOv1/SAM 相关条目经 web_extract/curl 重新实测存活**;paperswithcode 的 TRT-ViT 页为软 404(重定向 HF trending),弃用;bytedance/TRT-ViT 官方仓库不存在(404),论文未开源。

## 论文(arXiv)

| 主题 | 链接 | 看什么 |
|---|---|---|
| NeRF | https://arxiv.org/abs/2003.08934 | ECCV 2020;体渲染离散化 Eq.3、位置编码 Eq.4、消融 Table 2、NDC 推导附录 C |
| NeRF++ | https://arxiv.org/abs/2010.07492 | shape-radiance ambiguity 分析、倒球面参数化 Eq.4、T&T/LF 对比 Table 2 |
| Fourier Features(位置编码理论) | https://arxiv.org/abs/2006.10739 | NTK 视角解释为什么坐标 MLP 需要高频映射(频谱偏置) |
| TRT-ViT | https://arxiv.org/abs/2205.09579 | 4 条准则的受控实验(Table 1-3)、MixBlock 结构 Fig.3、延迟测量方法 |
| OREPA | https://arxiv.org/abs/2204.00826 | CVPR 2022;Proposition 1(优化多样化)、block squeezing 推导、scaling 初始化附录 |
| RepVGG | https://arxiv.org/abs/2101.03697 | 重参数化前驱:VGG 式单路推理 + 多分支训练 |
| DBB(Diverse Branch Block) | https://arxiv.org/abs/2103.13425 | 重参数化前驱:Inception 式多分支,OREPA 的直接对比对象 |
| ACNet | https://arxiv.org/abs/1908.03930 | 最早的非对称卷积分支重参数化(1×3+3×1+3×3) |
| CCNN(workshop 版) | https://arxiv.org/abs/2206.03398 | ICML 2022 Continuous Time Methods Workshop;对应论文链接见上表 |
| CCNN(ICLR 2023 正式版) | https://arxiv.org/abs/2301.10540 | 扩展版:新增点云 3D 实验,标题改为 Modelling Long Range Dependencies in ND |
| FlexConv | https://arxiv.org/abs/2110.08059 | 前驱:可微 kernel 大小(高斯掩码) |
| S4 | https://arxiv.org/abs/2111.00396 | 对照路线:结构化状态空间,长序列 1D 处理 |
| RepLKNet(大核设计) | https://arxiv.org/abs/2203.06717 | TRT-ViT 引用的 TeraFLOPS 度量出处;大核 CNN 与重参数化交汇 |
| ViT(An Image is Worth 16x16 Words) | https://arxiv.org/abs/2010.11929 | ICLR 2021;patchify/CLS/1D 位置编码 §3.1-3.2、模型变体 Table 1、JFT vs IN21k vs IN1k 数据敏感性 Fig.4/Table 2&5、训练算力对比 Table 2 |
| DeiT | https://arxiv.org/abs/2012.12877 | ICML 2021;ImageNet-1k only 训练配方、蒸馏 token、83.1%/85.2% 结果 |
| Faster R-CNN | https://arxiv.org/abs/1506.01497 | NIPS 2015;RPN 共享特征、anchor、5fps/300 proposals |
| Mask R-CNN | https://arxiv.org/abs/1703.06870 | ICCV 2017;并行掩码分支、RoIAlign、mask AP 35.7/5fps、COCO 三冠 |
| DETR | https://arxiv.org/abs/2005.12872 | ECCV 2020;set prediction、object query+匈牙利匹配、无 NMS/anchor;R50 42 AP,FLOPs 减半 |
| YOLOv1 | https://arxiv.org/abs/1506.02640 | CVPR 2016;单阶段回归范式源头、45fps/155fps |
| SAM(Segment Anything) | https://arxiv.org/abs/2304.02643 | ICCV 2023;可提示分割、三段架构、SA-1B 数据引擎、~50ms 交互延迟 |

## 官方项目页 / 代码

| 主题 | 链接 | 看什么 |
|---|---|---|
| NeRF 项目页 | https://www.matthewtancik.com/nerf | 结果视频、数据集(ECCV 2020 Best Paper Honorable Mention) |
| NeRF 官方代码(TF) | https://github.com/bmild/nerf | 原版 TensorFlow 实现 |
| NeRF++ 官方代码(PyTorch) | https://github.com/Kai-46/nerfplusplus | DDP 训练、数据约定(camera-to-world/OpenCV 系/场景归一化) |
| nerfstudio 文档 | https://docs.nerf.studio/ | NeRF 系后续生态(实时/大场景方法入口) |
| OREPA 官方代码 | https://github.com/JUGGHM/OREPA_CVPR2022 | PyTorch 实现、ImageNet/COCO/Cityscapes 训练脚本 |
| CCNN 官方代码 | https://github.com/david-knigge/ccnn | PyTorch Lightning + Hydra;models/resnet.py 是块定义 |
| S4 官方代码 | https://github.com/HazyResearch/state-spaces | 对照实现 |
| TensorRT 文档 | https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html | TRT-ViT 延迟测量的推理框架 |
| TensorRT GitHub | https://github.com/NVIDIA/TensorRT | 开源组件 |
| TRT-ViT 社区复现(非官方) | https://github.com/Phoenix8215/TRT-ViT | 论文官方未开源时的参考实现 |
| ViT 官方代码(google-research) | https://github.com/google-research/vision_transformer | 论文 abstract 注明的微调代码与预训练权重发布处(JAX/TF) |
| DETR 官方代码(facebookresearch) | https://github.com/facebookresearch/detr | Model Zoo(R50/R101,DC5 变体)、torch.hub 用法、全景分割权重 |
| SAM 官方代码(facebookresearch) | https://github.com/facebookresearch/segment-anything | vit_b/l/h checkpoint、SamPredictor/SamAutomaticMaskGenerator API、ONNX 导出脚本 |
| SAM 2 官方代码(facebookresearch) | https://github.com/facebookresearch/segment-anything-2 | 视频扩展(arXiv:2408.00714)、流式 memory |
| Ultralytics YOLO 官方仓库/文档 | https://github.com/ultralytics/ultralytics | YOLO11 检测/分割/分类/姿态/OBB 五任务、COCO 性能表、导出部署 |

## 阅读顺序建议

入门:先读本域 index.md → 基础线(vision-transformer-basics → object-detection-segmentation)→ 表示线(nerf-view-synthesis → cnn-long-range-nd)→ 效率线(trt-vit-inference → cnn-reparameterization);每篇文末「溯源」可回原始文献。
