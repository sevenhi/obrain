---
title: Computer Vision
type: domain
tags: [computer-vision, nerf, vit, cnn, reparameterization, inference, onnx, tensorrt, deployment]
source:
  - "web: ONNX Runtime 官方文档 | https://onnxruntime.ai/docs/ | 阅读日期 2026-09-27"
  - "web: NVIDIA TensorRT 官方文档 | https://docs.nvidia.com/deeplearning/tensorrt/latest/ | 阅读日期 2026-09-27"
  - "web: NVIDIA Model Optimizer | https://nvidia.github.io/TensorRT-Model-Optimizer/ | 阅读日期 2026-09-27"
date: 2026-09
---

# Computer Vision

> 领域索引(导航起始页)。定位:**视觉模型的两条主线——表示(NeRF 隐式场景表示、CCNN 连续卷积核)与部署效率(TRT-ViT 硬件感知架构、OREPA 重参数化)**。2026-09 新建,由 5 篇论文提炼合并为 5 个 concept/howto(NeRF+NeRF++ 合并一篇)。**正文在 `concepts/`,每篇可独立读懂**;本页只导航。

## 一句话定位

从「怎么表示视觉信号」到「怎么在真实硬件上跑得快」:NeRF 用 MLP+体渲染表示场景(表示端极限),CCNN 把卷积核连续化(结构端创新),TRT-ViT 与 OREPA 则分别从架构设计与训练-部署解耦两个角度回答「精度不掉、延迟最低」。

## 条目索引(5 concept/howto)

| 篇目 | 类型 | 一句话 | 状态 |
|---|---|---|---|
| [NeRF 视图合成(NeRF+NeRF++)](concepts/nerf-view-synthesis.md) | concept | 5D 坐标 MLP + 可微体渲染;位置编码/分层采样/倒球面参数化(非有界场景) | ✅ 已入库 |
| [TRT-ViT 推理导向架构](concepts/trt-vit-inference.md) | concept | 以 TensorRT 延迟(非 FLOPs)为设计准绳;4 条准则 + MixBlockC 混合 CNN/Transformer | ✅ 已入库 |
| [CNN 重参数化(OREPA)](concepts/cnn-reparameterization.md) | concept | 训练多分支/推理单卷积;线性 scaling 替代 BN 使训练期 kernel 级在线合并 | ✅ 已入库 |
| [CCNN 连续卷积核](concepts/cnn-long-range-nd.md) | concept | 核=坐标的 MLP 函数,核长与参数解耦;一个 CNN 通吃 1D/2D/任意分辨率 | ✅ 已入库 |
| [视觉模型部署：PyTorch → ONNX → TensorRT / ONNX Runtime](frameworks/onnx-tensorrt-deployment.md) | howto | 从导出、图验证、动态 shape、量化、数值回归到 engine 兼容性和线上 benchmark 的完整部署流程 | ✅ 已入库 |

## 阅读顺序建议

- 表示线:NeRF(隐式场景)→ CCNN(隐式卷积核)——两者同谱系(坐标→值的 MLP),连读能看到 implicit representation 的两个应用面。
- 效率线:TRT-ViT(架构层:什么算子在硬件上真快)→ OREPA(训练层:复杂结构免费折叠)→ [ONNX/TensorRT 部署](frameworks/onnx-tensorrt-deployment.md)(工程层:导出、验证、engine 与线上观测)——同一「部署优先」哲学的三端。

## 相关条目(跨域互链)

- [cutlass/](../cutlass/index.md)——TRT-ViT 的效率分析根因(GEMM/attention 在硬件上的真实利用率)与 CUTLASS 的 tiling/Tensor Core 机制同源;TensorRT 内部以 CUTLASS 生成 GEMM kernel。互链见 [trt-vit-inference.md](concepts/trt-vit-inference.md) 实践要点。
- [attention/](../attention/index.md)——NeRF/TRT-ViT 都涉及「访存而非算力是瓶颈」的判断,与 FA 的 IO-aware 分析同一方法论。
- [quantization/](../quantization/index.md)——与 OREPA 同属「模型压缩/加速」工具箱(重参数化 vs 量化,正交可叠加)。

## 溯源

- 外部链接索引见 [references.md](references.md)。
