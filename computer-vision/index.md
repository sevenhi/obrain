---
title: Computer Vision
type: domain
tags: [computer-vision, nerf, vit, cnn, reparameterization, inference, onnx, tensorrt, deployment]
source:
  - "web: ONNX Runtime 官方文档 | https://onnxruntime.ai/docs/ | 阅读日期 2026-10-23(最新发布 1.31.0)"
  - "web: NVIDIA TensorRT 官方文档 | https://docs.nvidia.com/deeplearning/tensorrt/latest/ | 阅读日期 2026-10-23(当前版 11.4.0)"
  - "code: NVIDIA Model Optimizer | github.com/NVIDIA/Model-Optimizer(原仓库名 TensorRT-Model-Optimizer 会 301 重定向到此) + 文档站 https://nvidia.github.io/Model-Optimizer/ | 阅读日期 2026-10-23"
date: 2026-10
---

# Computer Vision

> 领域索引(导航起始页)。定位:**视觉模型的三条主线——基础机制(ViT patchify/CLS、检测/分割三大范式)、表示(NeRF 隐式场景表示、CCNN 连续卷积核)与部署效率(TRT-ViT 硬件感知架构、OREPA 重参数化)**。2026-09 新建,2026-10 补齐 ViT 基础与检测/分割两篇核心支柱,由多篇论文(ViT/DeiT、Faster/Mask R-CNN/DETR/YOLO/SAM、NeRF/NeRF++、TRT-ViT、OREPA、CCNN)+ 1 篇官方文档综合的部署 howto 提炼为 **7 个 concept/howto**。**正文在 `concepts/`,每篇可独立读懂**;本页只导航。

## 一句话定位

从「模型怎么看懂图」到「怎么在真实硬件上跑得快」:ViT 基础篇讲纯 transformer 吃图像 patch 的机制,检测/分割篇讲定位与像素级掩码的三大范式(两阶段/单阶段/端到端)+ SAM 可提示分割;表示线 NeRF 用 MLP+体渲染表示场景,CCNN 把卷积核连续化;效率线 TRT-ViT 与 OREPA 分别从架构设计与训练-部署解耦回答「精度不掉、延迟最低」。

## 条目索引(7 concept/howto)

| 篇目 | 类型 | 一句话 | 状态 |
|---|---|---|---|
| [ViT 基础机制](concepts/vision-transformer-basics.md) | concept | patchify/CLS/1D 可学习位置编码;ViT-B/L/H 变体与「大数据才赢」的数据敏感性;DeiT 蒸馏平民化 | ✅ 已入库 |
| [目标检测与实例分割](concepts/object-detection-segmentation.md) | concept | Faster/Mask R-CNN(anchor+NMS)→ YOLO(单阶段回归)→ DETR(query+匈牙利匹配,无 NMS)→ SAM(可提示分割基础模型) | ✅ 已入库 |
| [NeRF 视图合成(NeRF+NeRF++)](concepts/nerf-view-synthesis.md) | concept | 5D 坐标 MLP + 可微体渲染;位置编码/分层采样/倒球面参数化(非有界场景) | ✅ 已入库 |
| [TRT-ViT 推理导向架构](concepts/trt-vit-inference.md) | concept | 以 TensorRT 延迟(非 FLOPs)为设计准绳;4 条准则 + MixBlockC 混合 CNN/Transformer | ✅ 已入库 |
| [CNN 重参数化(OREPA)](concepts/cnn-reparameterization.md) | concept | 训练多分支/推理单卷积;线性 scaling 替代 BN 使训练期 kernel 级在线合并 | ✅ 已入库 |
| [CCNN 连续卷积核](concepts/cnn-long-range-nd.md) | concept | 核=坐标的 MLP 函数,核长与参数解耦;一个 CNN 通吃 1D/2D/任意分辨率 | ✅ 已入库 |
| [视觉模型部署：PyTorch → ONNX → TensorRT / ONNX Runtime](frameworks/onnx-tensorrt-deployment.md) | howto | 从导出、图验证、动态 shape、量化、数值回归到 engine 兼容性和线上 benchmark 的完整部署流程 | ✅ 已入库 |

## 阅读顺序建议

- 基础线(新手先读):[ViT 基础机制](concepts/vision-transformer-basics.md)(图像怎么变成 token)→ [目标检测与实例分割](concepts/object-detection-segmentation.md)(定位与掩码的三大范式)——两篇是理解本域其余各篇与 deepseek-vision/attention 域的机制底座。
- 表示线:NeRF(隐式场景)→ CCNN(隐式卷积核)——两者同谱系(坐标→值的 MLP),连读能看到 implicit representation 的两个应用面。
- 效率线:TRT-ViT(架构层:什么算子在硬件上真快)→ OREPA(训练层:复杂结构免费折叠)→ [ONNX/TensorRT 部署](frameworks/onnx-tensorrt-deployment.md)(工程层:导出、验证、engine 与线上观测)——同一「部署优先」哲学的三端。

## 相关条目(跨域互链)

- [cutlass/](../cutlass/index.md)——TRT-ViT 的效率分析根因(GEMM/attention 在硬件上的真实利用率)与 CUTLASS 的 tiling/Tensor Core 机制同源;TensorRT 内部以 CUTLASS 生成 GEMM kernel。互链见 [trt-vit-inference.md](concepts/trt-vit-inference.md) 实践要点。
- [deepseek-vision/](../deepseek-vision/index.md)——本域 NeRF 体渲染是 DreamCraft3D 图生 3D 的通用背景([dreamcraft3d.md](../deepseek-vision/concepts/dreamcraft3d.md));TRT-ViT/ViT 与 deepseek-vision 的 SigLIP/SAM 视觉编码器同属「视觉表示 + 部署效率」两端,见其 index 互链。VL 的 SAM-B 编码器机制见本域[检测/分割篇](concepts/object-detection-segmentation.md) §SAM,SigLIP/Janus 所用 ViT patchify/CLS 机制见 [ViT 基础篇](concepts/vision-transformer-basics.md)。
- [attention/](../attention/index.md)——NeRF/TRT-ViT 都涉及「访存而非算力是瓶颈」的判断,与 FA 的 IO-aware 分析同一方法论;ViT encoder 层与 DETR 交叉注意力的机制源头是 [attention-is-all-you-need](../attention/algorithm/attention-is-all-you-need.md)。
- [quantization/](../quantization/index.md)——与 OREPA 同属「模型压缩/加速」工具箱(重参数化 vs 量化,正交可叠加)。

## 溯源

- 外部链接索引见 [references.md](references.md)。
