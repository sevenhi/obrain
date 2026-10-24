---
title: quantization 外部链接索引
type: reference
tags: [quantization, compression, references]
date: 2026-09
---

# References — quantization

> 只放链接索引,不复制全文(库规约)。聚合自各篇 frontmatter 的 source;条目级溯源见各篇文末『溯源』。

## 论文(arXiv)

- HyQuant — arXiv:2608.27875v3(EMNLP 2026 Main)
- SpQR — arXiv:2306.03078(ICLR 2024)
- GPTQ — arXiv:2210.17323
- AWQ — arXiv:2306.00978
- SmoothQuant — arXiv:2211.10438
- KIVI — arXiv:2402.02750
- QuIP# — arXiv:2402.04396(ICML 2024)
- QuIP(前作)— arXiv:2307.13304
- AQLM — arXiv:2401.06118(ICML 2024)
- KVQuant — arXiv:2401.18079(NeurIPS 2024)
- KVTuner — arXiv:2502.04420(ICML 2025)
- Minitron(结构化剪枝 + 蒸馏;15B→8B/4B)— arXiv:2407.14679(NeurIPS 2024)
- LLM Pruning and Distillation in Practice: The Minitron Approach(Llama-3.1-8B→4B、Mistral-NeMo-12B→8B)— arXiv:2408.11796
- Puzzle: Distillation-Based NAS for Inference-Optimized LLMs(异构逐层剪枝 + MIP,Puzzletron 的论文)— arXiv:2411.19146(v5 2025-06)
- Minitron-SSM(Hybrid Mamba 的 group-aware SSM 剪枝)— arXiv:2504.11409
- NVIDIA Nemotron Nano 2(hybrid Mamba-Transformer 剪枝 + 两阶段蒸馏)— arXiv:2508.14444
- ShortGPT(Block Influence 深度剪枝指标)— arXiv:2403.03853
- LaCo(Layer Collapse,剪层时把被删层信息并入相邻层)— arXiv:2402.11187
- Low-Rank Matrix Factorization for LSTM Compression — arXiv:1908.09982(PACLIC 2019)
- LLM.int8() outlier features — arXiv:2208.07339
- Deep Compression — arXiv:1510.00149
- Jaderberg et al. 低秩卷积加速 — BMVC 2014

## 官方代码 / 项目

- NVIDIA Model Optimizer(剪枝三 mode:minitron / puzzletron / fastnas)— https://github.com/NVIDIA/Model-Optimizer
- ModelOpt MoE 逐层 GPTQ 链:PR #2840(逐专家 Hessian,fused MoE)— https://github.com/NVIDIA/Model-Optimizer/pull/2840
- ModelOpt PR #2864(逐层 GPTQ 加速三修,2026-10-23 合并)— https://github.com/NVIDIA/Model-Optimizer/pull/2864
- ModelOpt PR #2877(fused NVFP4 GPTQ kernel 按张量所在 GPU 启动,2026-10-23 合并)— https://github.com/NVIDIA/Model-Optimizer/pull/2877
- ModelOpt PR #2945(Hessian 按实测 headroom 放置 + 逐层 batch 探测,**open 未合并**)— https://github.com/NVIDIA/Model-Optimizer/pull/2945
- ModelOpt PR #2943(AutoQuant 自动 batch 探测复现打分负载,2026-10-23 合并)— https://github.com/NVIDIA/Model-Optimizer/pull/2943
- ModelOpt 公告《Quantizing a 4.9 TB Qwen3.8 Model on a Single GPU》(2026-09-28)— https://nvidia.github.io/Model-Optimizer/announcements/20260928-single-gpu-ptq.html
- 逐层导出示例 recipe(nvfp4_experts_only-kv_fp8_layerwise_export.yaml)— https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/general/ptq/nvfp4_experts_only-kv_fp8_layerwise_export.yaml
- ModelOpt model_calib API(gptq / imatrix / local_hessian / layerwise_calibrate)— https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_calib.html
- ModelOpt 源码 calib_utils.py(update_hessian / GPTQHelper / _hessian_offload_device)— https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/utils/calib_utils.py
- ModelOpt 源码 hf_ptq.py(LAYERWISE_CALIB_TOKENS=8192)— https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/hf_ptq.py
- ModelOpt DP=EP 专家并行 PTQ 传输(PR #2738,21 PR 系列第 1 块)— https://github.com/NVIDIA/Model-Optimizer/pull/2738
- ModelOpt `modelopt/torch/utils/moe.py`(a2a_moe / torch_a2a 注册)— https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/utils/moe.py
- transformers Expert parallelism 文档 — https://huggingface.co/docs/transformers/en/expert_parallelism
- ModelOpt 剪枝示例(支持矩阵、超参建议、端到端教程)— https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/pruning/README.md
- ModelOpt Puzzletron 示例(MIP sweep、runtime 延迟约束、AnyModel)— https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/puzzletron/README.md
- ModelOpt 剪枝指南(mtp.prune 统一 API)— https://nvidia.github.io/Model-Optimizer/guides/3_pruning.html
- Megatron-Bridge 剪枝文档(Minitron 在 M-Bridge 的调用)— https://docs.nvidia.com/nemo/megatron-bridge/latest/modelopt/pruning.html
- Minitron 模型集合(HuggingFace)— https://huggingface.co/collections/nvidia/minitron
- HyQuant — https://github.com/jerrysfls/HyQuant
- SpQR — https://github.com/Vahe1994/SpQR
- QuIP# — https://github.com/Cornell-RelaxML/quip-sharp
- AQLM — https://github.com/Vahe1994/AQLM
- KVQuant — https://github.com/SqueezeAILab/KVQuant
- KVTuner — https://github.com/cmd2001/KVTuner
- bitsandbytes — https://github.com/TimDettmers/bitsandbytes
- AWD-LSTM 基线(论文实验用) — https://github.com/salesforce/awd-lstm-lm

## 工程文档

- vLLM Quantization — https://docs.vllm.ai/en/latest/features/quantization/
- vLLM quantized KV cache — https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- ICLR 2024 SpQR poster(发表状态) — https://iclr.cc/virtual/2024/poster/18686

### FP8/FP4 格式(2026-10 补)

- DeepSeek-V3 Technical Report — https://arxiv.org/abs/2412.19437
- OCP Microscaling Formats (MX) Specification — https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- NVIDIA Transformer Engine — https://docs.nvidia.com/deeplearning/transformer-engine/
- DeepGEMM(细粒度 FP8 GEMM 实现)— https://github.com/deepseek-ai/DeepGEMM
- MS-AMP(微软 FP8 训练框架,optimizer state 下放)— https://github.com/Azure/MS-AMP
