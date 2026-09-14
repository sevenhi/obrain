# LLM 推理引擎栈

> 领域组索引。本组围绕「LLM 推理性能技术栈」,覆盖三个紧密相关且互相集成的项目。

## 一句话关系认知

**vLLM** 是面向应用的 LLM 推理引擎(消费 PagedAttention + 优化内核);
**FlashAttention** 是 vLLM 背后最核心的优化注意力内核之一(IO-aware tiling);
**CUTLASS** 是 FA/vLLM 等底层依赖的高性能 GEMM 模板库与脚手架(CuTe DSL)。

即:**CUTLASS(造内核的库)→ FlashAttention(注意力内核)→ vLLM(整机推理引擎)**,自底向上三层。

## 领域索引

| 主题 | 文件夹 | canonical 全文 | wiki pageId |
|---|---|---|---|
| [vLLM](vllm/index.md) | `llm-inference/vllm/` | `tracker/vllm-tracker/vllm-推理引擎与FlashAttention-完整技术文档.md` | 566290559 |
| [CUTLASS](cutlass/index.md) | `llm-inference/cutlass/` | `docs/CUTLASS-完整技术文档.md` | 566290114 |
| [FlashAttention](flash-attention/index.md) | `llm-inference/flash-attention/` | `tracker/flash-attention-tracker/flash-attention-annual-report.md` | 561077705 |

> 注:vLLM canonical 文档《vLLM 推理引擎与 FlashAttention 完整技术文档》同时覆盖 FA(第四主题),跨领域内容集中在此,未在 FA 页重复。

## 共同主题速查(跨领域)

- **显存/内存**:vLLM 的 PagedAttention/KV cache 是 host-vs-device lifetime 心智模型(见 `../gpu-memory-lifecycle/`)的最佳实例;FA/CUTLASS 都在 stream 上异步执行。
- **底层依赖链**:vLLM 集成 FA;FA1 基于 CUTLASS 3.x 构建。升级某个时注意链路兼容。

## 论文笔记

| 主题                             | 文件                                                | 一句话                                      |
| ------------------------------ | ------------------------------------------------- | ---------------------------------------- |
| HyQuant — LLM Attention 混合精度量化 | `papers/hyquant-hybrid-precision-quantization.md` | 竖线+局部窗口留全精度,其余 KV 压低比特(arXiv 2608.27875) |

## 更新维护(各 tracker 各自跑,本组只做索引)

三个项目各自有 `*-tracker/` 目录(定时检查 GitHub 提交 → 生成报告 → `sync_wiki.py` 同步 wiki),详见各领域 README 的「更新与同步」节。
