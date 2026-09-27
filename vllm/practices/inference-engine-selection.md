---
title: LLM 推理引擎选型：vLLM / SGLang / TensorRT-LLM / llama.cpp / LMDeploy
type: comparison
tags: [llm-inference, serving, vllm, sglang, tensorrt-llm, llama-cpp, lmdeploy, comparison]
source:
  - "web: vLLM 官方文档 | https://docs.vllm.ai/en/latest/ | 阅读日期 2026-09-27"
  - "web: SGLang 官方文档 | https://docs.sglang.ai/ | 阅读日期 2026-09-27"
  - "web: TensorRT-LLM 官方文档 | https://nvidia.github.io/TensorRT-LLM/ | 阅读日期 2026-09-27"
  - "web: llama.cpp 官方仓库 | https://github.com/ggml-org/llama.cpp | 阅读日期 2026-09-27"
  - "web: LMDeploy 官方文档 | https://lmdeploy.readthedocs.io/en/latest/ | 阅读日期 2026-09-27"
date: 2026-09
---

# LLM 推理引擎选型

## TL;DR

这篇对比页的默认结论是：引擎选型必须和模型、GPU、并发、量化格式以及运维成本一起评估；vLLM 是通用在线服务基线，但不是所有场景的最优答案。

## 1. 能力地图

| 引擎 | 主要定位 | 强项 | 主要代价/边界 |
|---|---|---|---|
| **vLLM** | 通用高吞吐在线推理 | PagedAttention、连续批处理、prefix caching、OpenAI 兼容、并行部署、模型覆盖广 | 版本迭代快；性能依赖 CUDA/attention backend/量化 kernel 组合 |
| **SGLang** | 低延迟、高吞吐和结构化程序化生成 | RadixAttention、prefix caching、OpenAI API、广泛硬件与模型适配 | 高级特性和模型后端变化快；需要按具体模型验证支持度 |
| **TensorRT-LLM** | NVIDIA 硬件优化的生产推理 | TensorRT/CUDA kernel、量化、并行和 NVIDIA 生态整合 | 更依赖 NVIDIA；构建、版本、engine 和硬件兼容矩阵更复杂 |
| **llama.cpp** | 轻量本地/边缘推理 | C/C++、GGUF、CPU 与多种 GPU backend、低依赖、便于单机分发 | 大规模在线服务的调度和生态不以它为主要优势；性能需按 backend 验证 |
| **LMDeploy** | 高效部署与量化工具链 | TurboMind、PyTorch backend、AWQ/GPTQ、KV quant、OpenAI 兼容服务 | 后端选择会改变行为和调优方式；社区资料与模型支持要按版本核对 |

## 2. 按使用场景选择

### 家庭服务器 / 单卡 / 本地模型

优先考虑 **llama.cpp** 或 **vLLM**：

- GGUF、CPU 推理、低依赖和小显存：llama.cpp
- NVIDIA GPU、OpenAI API、连续批处理和多个客户端：vLLM
- 不要仅凭模型文件大小选择；量化格式必须和 backend、GPU、上下文长度一起验证。

### 多用户在线 API

优先从 **vLLM** 或 **SGLang** 开始：

- vLLM 的知识库和部署文档更适合作为通用基线
- SGLang 在 prefix caching、结构化生成和特定模型路径上可能更有优势
- 两者都应使用真实请求分布测试 TTFT、ITL、吞吐、排队和 KV cache 使用率

### NVIDIA 生产集群

评估 **TensorRT-LLM**，但把工程成本计入总成本：

- 需要固定 GPU、CUDA、TensorRT-LLM 和模型版本
- 需要管理 engine 构建、量化校准、并行策略和升级回滚
- 适合性能收益足以覆盖构建和维护成本的稳定模型服务

### 量化和国产/多平台部署

评估 **LMDeploy**，尤其是：

- AWQ/GPTQ 或 KV cache 量化已经是部署约束
- 需要在 TurboMind 与 PyTorch 后端之间切换
- 目标硬件不只是一种 NVIDIA GPU

## 3. 统一评测方法

不要比较项目官网上的不同硬件、不同模型和不同 batch 的单个数字。建议固定：

1. 同一模型 revision、tokenizer、量化格式和 prompt 数据集。
2. 同一 GPU、驱动、CUDA、容器基础镜像和功耗限制。
3. 同一输入长度/输出长度分布和并发阶梯。
4. 同一 API 行为：流式、停止词、采样参数、超时和最大上下文。
5. 同时记录：
   - TTFT p50/p95/p99
   - ITL p50/p95/p99
   - 端到端延迟
   - input/output tokens/s
   - 请求吞吐和排队时间
   - 峰值显存、KV cache 使用率、CPU/RAM
   - 首次加载时间、错误率、重启恢复时间

### 最小测试矩阵

| 测试 | 目的 |
|---|---|
| 单请求短上下文 | 检查启动、正确性和基础延迟 |
| 单请求长上下文 | 暴露 prefill、KV cache 和显存问题 |
| 并发 1/2/4/8/... | 找到吞吐拐点与排队拐点 |
| 混合输入长度 | 检查调度公平性和长请求对短请求的影响 |
| 长时间 soak test | 发现显存碎片、泄漏、连接和重启问题 |
| 量化前后回归 | 同时比较质量、显存和延迟，避免只看压缩率 |

## 4. 选择时容易犯的错误

- 把“模型支持”当成“所有特性都支持”：工具调用、多模态、LoRA、speculative decoding、prefix cache 和量化往往有独立限制。
- 把静态 benchmark 当成在线服务结论：真实请求的长度分布和并发会改变结论。
- 只比较 token/s：高吞吐可能牺牲 TTFT、尾延迟或公平性。
- 忽略运维成本：镜像、engine、模型缓存、监控、回滚和故障恢复都属于系统的一部分。
- 让 `latest` 镜像或未锁定的模型 revision 进入生产，导致结果不可复现。

## 5. 给本知识库的默认决策树

```text
需要在线 OpenAI 兼容 API？
├─ 否：CPU/边缘/GGUF → llama.cpp
└─ 是：
   ├─ NVIDIA 固定硬件 + 愿意构建专用 engine → 评估 TensorRT-LLM
   ├─ 特别依赖结构化生成/prefix cache/特定模型优化 → 对比 SGLang
   ├─ 量化、多后端或 TurboMind → 评估 LMDeploy
   └─ 通用默认基线 → vLLM
```

默认基线不是最终答案；在进入实际服务前必须用 [生产部署与观测清单](production-serving.md) 完成容量和回归验证。

## 相关条目

- [vLLM 生产部署与观测](production-serving.md)
- [vLLM V1 引擎架构](../concepts/v1-architecture.md)
- [PTQ：GPTQ / AWQ / SmoothQuant](../../quantization/concepts/ptq-gptq-awq-smoothquant.md)
- [KV Cache 量化](../../quantization/concepts/kv-cache-quantization.md)
- [PagedAttention](../concepts/paged-attention.md)
