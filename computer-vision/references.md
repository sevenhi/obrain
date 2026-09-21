---
title: Computer Vision - 参考资料索引
type: reference
tags: [nerf, vit, cnn, reparameterization, tensorrt]
source: "pdf: 本域 5 篇源论文 | ~/zhanbin/library/papers/(只读存档)"
date: 2026-09
---

# 参考资料索引

> 溯源用。只放链接与「看什么」,不复制全文。**全部 URL 于 2026-09-21 curl 实测存活(HTTP 200)**;paperswithcode 的 TRT-ViT 页为软 404(重定向 HF trending),弃用;bytedance/TRT-ViT 官方仓库不存在(404),论文未开源。

## 本地原始资料(library 只读,引用不复制)

| 文件 | 对应条目 |
|---|---|
| ~/zhanbin/library/papers/NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.pdf | nerf-view-synthesis |
| ~/zhanbin/library/papers/NeRF++: Analyzing and Improving Neural Radiance Fields.pdf | nerf-view-synthesis |
| ~/zhanbin/library/papers/TRT-ViT: TensorRT-oriented Vision Transformer.pdf | trt-vit-inference |
| ~/zhanbin/library/papers/Online Convolutional Re-parameterization.pdf | cnn-reparameterization |
| ~/zhanbin/library/papers/Towards a General Purpose CNN for Long Range Dependencies in ND.pdf | cnn-long-range-nd |

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
| CCNN(workshop 版) | https://arxiv.org/abs/2206.03398 | ICML 2022 Continuous Time Methods Workshop;本地 PDF 即此版 |
| CCNN(ICLR 2023 正式版) | https://arxiv.org/abs/2301.10540 | 扩展版:新增点云 3D 实验,标题改为 Modelling Long Range Dependencies in ND |
| FlexConv | https://arxiv.org/abs/2110.08059 | 前驱:可微 kernel 大小(高斯掩码) |
| S4 | https://arxiv.org/abs/2111.00396 | 对照路线:结构化状态空间,长序列 1D 处理 |
| RepLKNet(大核设计) | https://arxiv.org/abs/2203.06717 | TRT-ViT 引用的 TeraFLOPS 度量出处;大核 CNN 与重参数化交汇 |

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

## 阅读顺序建议

入门:先读本域 index.md → 表示线(nerf-view-synthesis → cnn-long-range-nd)→ 效率线(trt-vit-inference → cnn-reparameterization);每篇文末「溯源」可回原始文献。
