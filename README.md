# obrain - 个人大脑

> 技术知识库根入口(原 `INDEX.md`,2026-09 并入 README)。定位:存档为主,平时不常看 →「扫 30 秒能找到,打开一篇能读懂」。
> 导航维护:2026-11。共 10 个专题入口、106 篇知识条目;不含索引、来源页、工具文档与生成物。维护日期不代表所有技术版本已重新核验。

## 领域列表(10 个专题入口)

| 领域入口 | 查什么 | 篇数 | 导航更新 |
|---|---|---:|---|
| [Attention](attention/index.md) | 注意力算法、FlashAttention 内核与 FlexAttention 接口 | 7 | 2026-11 |
| [模型架构](model-architecture/index.md) | LLM 组件、模型谱系、DeepSeek 结构与混合线性注意力 | 10 | 2026-11 |
| [CUTLASS](cutlass/index.md) | GEMM/CuTe 机制、内核构建与性能调优 | 10 | 2026-11 |
| [GPU 随机数生成](gpu-rng/index.md) | Philox/MTGP、cuRAND 与数值模拟的随机流 | 1 | 2026-11 |
| [量化与压缩](quantization/index.md) | 权重/KV 量化、FP8/FP4、剪枝蒸馏、MoE 校准与验收 | 12 | 2026-11 |
| [vLLM](vllm/index.md) | 引擎机制、Runner、部署选型与推理生态 | 24 | 2026-11 |
| [计算机视觉](computer-vision/index.md) | ViT、检测分割、视觉表示与 ONNX/TensorRT 部署 | 7 | 2026-11 |
| [DeepSeek 基建](deepseek-infra/index.md) | 算子、通信、并行、存储、双平台适配与 DeepInfer | 22 | 2026-11 |
| [DeepSeek 视觉](deepseek-vision/index.md) | VL/VL2、OCR/OCR-2、Janus 与 DreamCraft3D | 4 | 2026-11 |
| [DeepSeek 模型](deepseek-models/index.md) | 数学/证明/代码模型、R1/ESFT 训练与配套工具 | 9 | 2026-11 |

目录是存储归属,不要求专题互斥。一条知识有一个主要维护入口,可以从多个领域导航;通用机制与产品实现用互链区分视角。

## 按问题找

| 我想查 | 从这里开始 |
|---|---|
| KV 显存不够,有哪些不同杠杆? | [头共享](model-architecture/concepts/attention-heads-mha-mqa-gqa.md) / [分页管理](vllm/concepts/paged-attention.md) / [KV 量化](quantization/concepts/kv-cache-quantization.md) / [流式逐出](attention/algorithm/attention-sinks-streaming.md) |
| 注意力怎么写、怎么选后端? | [FA 四代实现](attention/implementation/flash-attention-implementation.md) / [FlexAttention 接口](attention/frameworks/flexattention.md) / [引擎后端选型](vllm/practices/attention-backend-selection.md) |
| 没有量化 kernel,如何先验收精度? | [fake-quant 验收](quantization/concepts/vllm-fakequant-eval-export.md) → [引擎量化集成](vllm/concepts/quantization-gptq-awq-fp8.md) |
| 推理服务怎么上线、测容量? | [部署清单](vllm/practices/production-serving.md) / [GuideLLM](vllm/ecosystem/guidellm-bench.md) / [PD 分离](vllm/practices/pd-disaggregation.md) |
| 投机解码与草稿训练怎么衔接? | [解码机制](vllm/concepts/speculative-decoding.md) → [DeepSpec](deepseek-models/frameworks/deepspec.md) / [Speculators](vllm/ecosystem/speculators.md) |
| DeepSeek 改结构还是改训练? | [结构演进](model-architecture/concepts/deepseek-model-evolution.md) / [R1 训练](deepseek-models/concepts/deepseek-r1-reasoning.md) / [ESFT 机制](deepseek-models/concepts/deepseek-esft.md) / [ESFT 脚本](deepseek-infra/concepts/esft.md) |
| 视觉模型如何部署,OCR 两代有何不同? | [ONNX/TensorRT](computer-vision/frameworks/onnx-tensorrt-deployment.md) / [OCR 与 OCR-2](deepseek-vision/concepts/deepseek-ocr-ocr2.md) |
| GPU 随机流能否用于安全用途? | [GPU RNG](gpu-rng/concepts/gpu-random-number-generation.md):Philox/Threefry 面向数值模拟,不用于密码学或安全 token |

## 跨域速查

- **架构→内核→引擎**:[model-architecture](model-architecture/index.md) 管网络结构,[attention](attention/index.md) 管算法/实现/接口,[vLLM](vllm/index.md) 管服务;[CUTLASS](cutlass/index.md) 提供造内核的底座。FA2 的 CuTe 原语、FA3 的 collective 抽象和 FA4 的 CuTe DSL 见 [FA 主篇](attention/implementation/flash-attention-implementation.md)。
- **注意力成本的不同解法**:[精确省显存谱系](attention/algorithm/exact-attention-memory-lineage.md) 区分 online softmax、Rabe & Staats 与 FA 的优化目标;[MEGABYTE](attention/algorithm/megabyte-multiscale.md) 做架构分解,[SpAtten](attention/algorithm/attention-variants-laser-spatten.md) 做专用硬件稀疏。StreamingLLM 逐出策略与 PagedAttention 分页管理正交。
- **混合线性注意力**:[Mamba/RWKV/Gated DeltaNet 与 hybrid](model-architecture/concepts/hybrid-linear-attention.md) 的递归层状态在固定宽度下不随序列增长;含 full-attention 的混合模型仍需随长度增长的 KV,不能概括为“消灭序列状态”。
- **量化双视角**:[quantization](quantization/index.md) 管算法、误差与校准,[vLLM 量化](vllm/concepts/quantization-gptq-awq-fp8.md) 管注册/分派/内核集成。[FP8/FP4](quantization/concepts/fp8-fp4-formats.md) 硬件格式与 QuIP#/AQLM 算法压缩分开;训练 V3 recipe 见 [训练并行](deepseek-infra/concepts/training-parallelism-v3.md),推理低精度 KV 见 [FlashMLA](deepseek-infra/concepts/flashmla.md)。
- **硬件真实效率**:[CUTLASS](cutlass/index.md) 的 tiling/Tensor Core 是理解 FLOPs≠延迟的底层,[TRT-ViT](computer-vision/concepts/trt-vit-inference.md) 是视觉架构侧实例。
- **DeepSeek 结构↔领域能力**:[网络结构四篇](model-architecture/concepts/deepseek-model-evolution.md) 讲“怎么变”,[领域模型](deepseek-models/index.md) 讲数据、训练与评测。R1 的 [V3-Base 结构视角](model-architecture/concepts/deepseek-arch-v32-r1.md) 与 [RL 训练视角](deepseek-models/concepts/deepseek-r1-reasoning.md) 分开维护;ESFT 分机制与操作入口。
- **引擎↔产品取舍**:[vLLM](vllm/index.md) 的 concepts/practices/ecosystem 分引擎、实践与周边工具;[DeepInfer](deepseek-infra/concepts/deepinfer.md) 记录自家模型生产服务的定制取舍,机制对照见该篇 §14。Runner 默认行为须绑定源码快照与实际配置,不从笔记日期推断。

## 阅读建议

- **临时查证** → 先用问题表,或搜关键词(`PagedAttention`、`split-K`、`TiledMMA`、`sink token`);FlashAttention/FA/flash-attn 等别名也可搜索。
- **回顾某专题** → 打开领域 `index.md`,按导航跳。
- **完整学习** → 按领域索引阅读顺序;attention 为 algorithm/implementation/frameworks 三层,其余主要为 concepts,工具上手见 frameworks,生态见 vllm/ecosystem。各篇文末『溯源』回原始资料。

