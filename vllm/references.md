---
title: vllm 外部链接索引
type: reference
tags: [vllm, llm-serving, references]
date: 2026-09
---

# References — vllm

> 只放链接索引,不复制全文(库规约)。聚合自各篇 frontmatter 的 source;源码实读基准:旧篇 commit `b74ac0bc6`(v0.28.1rc0-40),投机解码/注意力后端选型两篇为 2026-10 main 快照。

## 论文 / 博客

- PagedAttention (SOSP 2023) — arXiv:2309.06180
- vLLM Blog — Easy, Fast, and Cheap LLM Serving with PagedAttention — https://vllm-project.github.io/2023/06/20/vllm.html
- NVIDIA Blog — TensorRT-LLM Chunked Prefill — https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/
- HuggingFace — How Long Prompts Block Other Requests — https://huggingface.co/blog/tngtech/llm-performance-blocked-by-long-prompts
- Deadline-Aware Adaptive Prefill Chunking (SLOWeave) — arXiv:2609.07883
- zeroentropy — PagedAttention 概念解读 — https://zeroentropy.dev/concepts/paged-attention/
- 微信 — LLM 推理优化 Chunk Pre-fill — https://mp.weixin.qq.com/s/6YUgjMfWYdvryRP-gzppMQ
- Fast Inference from Transformers via Speculative Decoding (Leviathan et al., ICML 2023) — arXiv:2211.17192
- Accelerating LLM Decoding with Speculative Sampling (Chen et al., DeepMind) — arXiv:2302.01318
- EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty — arXiv:2401.15077
- EAGLE-2: Faster Inference with Dynamic Draft Trees — arXiv:2406.16858
- EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test — arXiv:2503.01840
- Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads — arXiv:2401.10774
- Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding (Heming Xia et al.) — arXiv:2401.07851

## 官方文档

- vLLM 文档首页 — https://docs.vllm.ai/en/latest/
- Architecture Overview — https://docs.vllm.ai/en/latest/design/arch_overview/
- V1 Guide — https://docs.vllm.ai/en/latest/usage/v1_guide/
- Speculative Decoding(spec_decode 子页入口)— https://docs.vllm.ai/en/latest/features/spec_decode.html
- OpenAI-Compatible Server — https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/
- Parallelism and Scaling — https://docs.vllm.ai/en/latest/serving/parallelism_scaling/
- Quantization — https://docs.vllm.ai/en/latest/features/quantization/
- SamplingParams — https://docs.vllm.ai/en/latest/api/vllm/sampling_params/
- Production Metrics — https://docs.vllm.ai/en/latest/usage/metrics.html
- Model Runner V2 Design — https://docs.vllm.ai/en/latest/design/model_runner_v2/
- Optimization(Chunked Prefill) — https://docs.vllm.ai/en/latest/configuration/optimization/
- Basic Model 贡献指南 — https://docs.vllm.ai/en/latest/contributing/model/basic/

## 源码(一手实读,commit b74ac0bc6)

- vLLM GitHub — https://github.com/vllm-project/vllm
- 实读路径:vllm/entrypoints/、vllm/v1/engine/、vllm/v1/executor/、vllm/distributed/、vllm/v1/sample/、vllm/model_executor/models/registry.py、vllm/model_executor/layers/quantization/
- 相关 PR:#52131(api_server.py 移出 openai/)、#52309(入口 middleware 整合)
- 实读路径(2026-10 main,注意力后端选型篇):vllm/v1/attention/backends/registry.py、vllm/v1/attention/backend.py、vllm/v1/attention/selector.py、vllm/platforms/cuda.py、vllm/config/attention.py

## 其他引擎(选型对照)

- SGLang — https://docs.sglang.ai/
- TensorRT-LLM — https://nvidia.github.io/TensorRT-LLM/
- llama.cpp — https://github.com/ggml-org/llama.cpp
- LMDeploy — https://lmdeploy.readthedocs.io/en/latest/
- prefill-chunking-profiler — https://github.com/JohnScheuer/prefill-chunking-profiler

### PD 分离(2026-10 补)

- DistServe (OSDI 2024) — https://arxiv.org/abs/2401.09670
- Splitwise (ISCA 2024) — https://arxiv.org/abs/2401.13081
- Mooncake — https://arxiv.org/abs/2407.00079

## 生态仓库(原 vllm-project 域并入,2026-10)

- vLLM-Omni — https://github.com/vllm-project/vllm-omni | arXiv:2610.09307 / arXiv:2602.02204
- Semantic Router — https://github.com/vllm-project/semantic-router | arXiv:2510.08731 / arXiv:2510.26835
- AIBrix — https://github.com/vllm-project/aibrix | arXiv:2504.03648
- Production Stack — https://github.com/vllm-project/production-stack
- vLLM Router — https://github.com/vllm-project/router
- LLM Compressor — https://github.com/vllm-project/llm-compressor
- compressed-tensors — https://github.com/vllm-project/compressed-tensors
- Speculators — https://github.com/vllm-project/speculators
- DeepSpec(deepseek-ai 草稿模型训练/评测全栈,speculators 篇对照源)— https://github.com/deepseek-ai/DeepSpec | arXiv:2607.05147
- Vime — https://github.com/vllm-project/vime
- GuideLLM — https://github.com/vllm-project/guidellm
- recipes — https://github.com/vllm-project/recipes
- vllm-ascend — https://github.com/vllm-project/vllm-ascend
- vllm-metal — https://github.com/vllm-project/vllm-metal
- vllm-gaudi / vllm-neuron / tpu-inference / vllm-xpu-kernels / vllm-openvino — https://github.com/vllm-project
- agentic-api / humming / afd-plugin / dllm-plugin / gguf-plugin / bnb-plugin — https://github.com/vllm-project
