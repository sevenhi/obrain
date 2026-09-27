---
title: vllm 外部链接索引
type: reference
tags: [vllm, llm-serving, references]
date: 2026-09
---

# References — vllm

> 只放链接索引,不复制全文(库规约)。聚合自各篇 frontmatter 的 source;源码一手实读基准 commit `b74ac0bc6`(v0.28.1rc0-40)。

## 论文 / 博客

- PagedAttention (SOSP 2023) — arXiv:2309.06180
- vLLM Blog — Easy, Fast, and Cheap LLM Serving with PagedAttention — https://vllm-project.github.io/2023/06/20/vllm.html
- NVIDIA Blog — TensorRT-LLM Chunked Prefill — https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/
- HuggingFace — How Long Prompts Block Other Requests — https://huggingface.co/blog/tngtech/llm-performance-blocked-by-long-prompts
- Deadline-Aware Adaptive Prefill Chunking (SLOWeave) — arXiv:2609.07883
- zeroentropy — PagedAttention 概念解读 — https://zeroentropy.dev/concepts/paged-attention/
- 微信 — LLM 推理优化 Chunk Pre-fill — https://mp.weixin.qq.com/s/6YUgjMfWYdvryRP-gzppMQ

## 官方文档

- vLLM 文档首页 — https://docs.vllm.ai/en/latest/
- Architecture Overview — https://docs.vllm.ai/en/latest/design/arch_overview/
- V1 Guide — https://docs.vllm.ai/en/latest/usage/v1_guide/
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

## 其他引擎(选型对照)

- SGLang — https://docs.sglang.ai/
- TensorRT-LLM — https://nvidia.github.io/TensorRT-LLM/
- llama.cpp — https://github.com/ggml-org/llama.cpp
- LMDeploy — https://lmdeploy.readthedocs.io/en/latest/
- prefill-chunking-profiler — https://github.com/JohnScheuer/prefill-chunking-profiler
