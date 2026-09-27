---
title: 视觉模型部署：PyTorch → ONNX → TensorRT / ONNX Runtime
type: howto
tags: [computer-vision, inference, onnx, tensorrt, onnx-runtime, deployment, benchmark]
source:
  - "web: ONNX Runtime 官方文档 | https://onnxruntime.ai/docs/ | 阅读日期 2026-09-27"
  - "web: NVIDIA TensorRT 官方文档 | https://docs.nvidia.com/deeplearning/tensorrt/latest/ | 阅读日期 2026-09-27"
  - "web: TensorRT Architecture Overview | https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html | 阅读日期 2026-09-27"
  - "web: NVIDIA Model Optimizer | https://nvidia.github.io/TensorRT-Model-Optimizer/ | 阅读日期 2026-09-27"
date: 2026-09
---

# 视觉模型部署：PyTorch → ONNX → TensorRT / ONNX Runtime

> **TL;DR**：部署不是“导出 ONNX 后就结束”。可靠流程是：固定模型与预处理 → 导出 → ONNX 图和数值验证 → 选择 runtime/backend → 构建并缓存 engine → 在目标输入形状上测准确率、延迟、吞吐、显存和稳定性。ONNX Runtime 适合作为跨平台基线和 Execution Provider 抽象；TensorRT 适合 NVIDIA GPU 上的深度优化，但 engine 与 TensorRT/CUDA/GPU 兼容性必须纳入发布物。

## 1. 两条部署路线

| 路线 | 定位 | 优点 | 注意事项 |
|---|---|---|---|
| ONNX Runtime | 跨平台 runtime + Execution Provider | API 稳定、可在 CPU/各类加速器间切换、适合作为基线 | 性能取决于 graph partition、EP 和算子覆盖；可能出现 CPU fallback |
| TensorRT | NVIDIA GPU 优化编译器和 runtime | FP16/BF16/FP8/INT8/FP4/INT4、动态 shape、融合、专用 kernel | engine 构建和兼容矩阵更复杂；不能把 plan 文件当成跨机器通用产物 |

ONNX Runtime 会先做图优化，再按可用硬件的 Execution Provider 对图进行分区；如果某些算子不能由目标 EP 执行，可能落回 CPU 或其他 provider，必须在日志和性能中确认，而不是只看“加载成功”。

TensorRT 的典型入口是 PyTorch/其他框架导出 ONNX，再由 ONNX parser 导入并构建 engine；也可以通过 TensorRT API 或 Torch-TensorRT 等路线接入。选择路线时优先看模型算子、动态 shape、量化需求和目标硬件，而不是只看导出格式。

## 2. 可复现部署流水线

```text
训练 checkpoint + 预处理配置
          ↓
固定 export config / opset / dynamic axes
          ↓
导出 ONNX
          ↓
ONNX checker + shape inference + graph inspection
          ↓
PyTorch vs ONNX 数值回归
          ↓
ONNX Runtime baseline
          ↓
TensorRT parse/build/profile
          ↓
TensorRT 数值回归 + 目标硬件 benchmark
          ↓
打包 model / engine / config / runtime manifest
```

每个阶段都保存版本信息：模型 revision、opset、导出脚本 commit、Python/PyTorch/ONNX/TensorRT/CUDA 版本、GPU 型号、输入 shape 和预处理参数。

## 3. 导出阶段的关键决策

### 输入输出契约

先写清楚：

- 输入张量名称、dtype、layout(NCHW/NHWC)、颜色顺序(RGB/BGR)
- 是否包含 resize、normalize、letterbox、padding 和后处理
- 输出是 logits、boxes、masks、embeddings 还是多尺度 feature maps
- batch、height、width 是否动态
- 检测/分割后处理由模型图、runtime 还是业务代码完成

最常见的“精度下降”不是 TensorRT 数学问题，而是预处理、坐标变换、NMS 阈值、颜色通道或 resize 对齐方式不一致。

### Dynamic shape

动态输入需要为 TensorRT optimization profile 指定 min/opt/max shape。`opt` 应靠近生产流量中心，而不是随便取中间值；过大的 max shape 会增加 build 搜索空间和 workspace 压力。若线上 shape 很离散，考虑按常见分辨率构建多个 profile 或多个 engine。

### 算子与控制流

遇到 parser 不支持的算子时，先区分：

1. 能否改写模型，使其落在 ONNX/TensorRT 常见算子集合内。
2. 是否可以在导出前做 constant folding、shape 推导或图简化。
3. 是否应写 TensorRT plugin。
4. 是否让该子图由 ONNX Runtime 或其他 backend 执行。

不要为了“全图 TensorRT”把一个简单、稳定的后处理强行改成复杂 plugin；端到端收益要用 benchmark 证明。

## 4. 数值正确性验证

### 分层比较

按以下顺序定位误差：

1. 原始 PyTorch eager vs 导出 ONNX。
2. ONNX Runtime CPU vs ONNX Runtime GPU/目标 EP。
3. ONNX Runtime vs TensorRT FP32/FP16。
4. FP16/INT8/FP8/FP4 与高精度基线。
5. 单算子、单 batch、整图和真实数据集指标。

### 推荐指标

- 最大绝对误差、平均绝对误差、相对误差
- logits/embedding 的 cosine similarity
- 检测：mAP、precision/recall、NMS 后框一致性
- 分割：mIoU、边界质量
- 分类：top-1/top-5
- 端到端任务：真实业务成功率，而不是只比较中间 tensor

量化或低精度之后，少量 tensor 的误差变大不一定意味着任务失败；相反，平均误差很小也可能导致 NMS 或 top-k 排序翻转，所以必须加入任务级指标。

## 5. 性能测量规范

每次 benchmark 固定：

- GPU、驱动、CUDA、TensorRT/ORT 版本
- engine build 参数、workspace、precision 和 tactic cache
- 输入 shape、batch、预热次数、测量次数
- 是否包含 H2D/D2H、预处理、后处理和同步
- 单流/多流、并发数和功耗状态

至少记录：

| 指标 | 作用 |
|---|---|
| p50/p95/p99 latency | 观察平均和尾延迟 |
| throughput | 检查 batch/并发收益 |
| GPU memory peak | 判断部署容量和并发上限 |
| build time / engine size | 评估发布成本 |
| GPU utilization / power | 判断是否真正吃满硬件 |
| accuracy regression | 防止速度换来不可接受质量损失 |

预热后的 kernel latency 不能代表首次请求延迟；engine build 时间也不能混进在线请求延迟。两者都应该单独记录。

## 6. 量化与图优化

- FP16/BF16：通常先作为低风险基线。
- INT8/FP8：需要校准数据、scale 和硬件/算子支持；校准集必须接近真实输入分布。
- INT4/FP4：压缩收益更大，但算子覆盖、精度回归和 kernel 支持更敏感。
- Model Optimizer 可用于 PTQ、QAT、剪枝、蒸馏和 ONNX 相关优化；使用时要保存配置和校准数据版本。
- 图融合和 constant folding 可能改变数值误差，优化前后都要跑回归。

量化不是部署的最后一个开关，而是模型格式、校准、kernel、输入分布和评测协议的共同结果。和 [PTQ：GPTQ / AWQ / SmoothQuant](../../quantization/concepts/ptq-gptq-awq-smoothquant.md) 一起阅读。

## 7. Engine 发布与兼容性

TensorRT serialized engine 对版本和硬件有兼容限制；不要默认一个 plan 文件可以跨 GPU、CUDA、TensorRT 版本复用。发布包至少包含：

- engine 或可重建 engine 的构建配置
- 原始 ONNX/checkpoint 的 revision
- TensorRT/CUDA/driver/GPU manifest
- dynamic shape profiles
- precision、workspace、plugins 和 tactic cache 信息
- 数值回归样本及结果
- build 命令和回滚用旧 engine

升级 TensorRT 时重新检查 API、plugin、engine 序列化和 calibration cache 兼容性。线上升级应先在独立端口完成 smoke test，再切换流量。

## 8. 实践检查清单

- [ ] 预处理和后处理已版本化，不依赖手工 notebook。
- [ ] PyTorch、ONNX Runtime、TensorRT 输出完成数值回归。
- [ ] 已验证真实输入 shape，而非只有随机 tensor。
- [ ] 已检查 ONNX Runtime 是否发生 CPU fallback。
- [ ] 已为动态 shape 配置合理 profile。
- [ ] 已记录 engine 的构建环境和硬件兼容性。
- [ ] 已分别测 warm-up、kernel、端到端和并发延迟。
- [ ] 已完成精度、显存、吞吐和尾延迟验收。
- [ ] 已保留旧 engine、旧配置和回滚路径。

## 相关条目

- [TRT-ViT：以 TensorRT 延迟为准绳](../concepts/trt-vit-inference.md)
- [CNN 重参数化 OREPA](../concepts/cnn-reparameterization.md)
- [CUDA GPU 性能基础](../../cutlass/concepts/cuda-performance-basics.md)
- [PTQ：GPTQ / AWQ / SmoothQuant](../../quantization/concepts/ptq-gptq-awq-smoothquant.md)
- [LLM 推理引擎选型](../../vllm/practices/inference-engine-selection.md)
