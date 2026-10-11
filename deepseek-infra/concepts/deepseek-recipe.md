---
title: deepseek-recipe — DeepSeek V4/V4.1 多格式 API 与 prompt 编解码库
type: concept
tags: [deepseek-recipe, api, prompt, chat-completions, responses, v4, v4.1, rust, python, serving]
date: 2026-10
source:
  - "code: deepseek-ai/deepseek-recipe | https://github.com/deepseek-ai/deepseek-recipe | 2026-10-11 | ↑primary"
  - "code: deepseek-recipe README.zh | https://github.com/deepseek-ai/deepseek-recipe/blob/main/README.zh.md | 2026-10-11"
---

# deepseek-recipe — DeepSeek V4/V4.1 多格式 API 与 prompt 编解码库

> **TL;DR**:deepseek-recipe 是一套 **Rust 库 + Python bindings**,把 Messages / Chat Completions / Responses 三种 API 请求格式统一转成 Conversation,再编码成 **DeepSeek V4 / V4.1** 的 prompt(或 token IDs);并把模型输出解析回对应格式的响应(含思考、工具调用、JSON、stop sequence)。**不含**模型推理、工具执行、HTTP 传输——由外部推理后端提供。定位是「推理后端 ↔ 多格式 API 服务」之间的**协议/词表胶水层**。

## 1. 是什么

- 把「各家 API 形状」和「DeepSeek 模型吃什么样的 prompt」解耦的**编解码库**。
- 语言栈:**Rust 为核心**(`deepseek-recipe` / `-core` / `-encoding` / `-image`),提供 `deepseek-recipe-python` bindings;另有 `server-py` / `server-rs` 示例。
- 服务对象是 **DeepSeek V4 与 V4.1** 的对话编码(与 [`flashmla`](flashmla.md)「只服务 V4.1」的口径对齐;V4 系架构见 [`open-model-landscape-2025-2026`](../../model-architecture/concepts/open-model-landscape-2025-2026.md))。

## 2. 支持范围

| 能力 | 支持情况 |
|---|---|
| 请求/响应格式 | Messages、**Chat Completions**、**Responses**;流式与完整响应 |
| 输入模态 | 文本、图片(base64 或 URL)、思考、客户端工具调用 |
| Prompt 编码 | DeepSeek **V4 / V4.1** 对话 → prompt 或 token IDs |
| 生成参数 | thinking 模式、reasoning effort、`temperature`、`top_p`、输出 token 上限 |
| 输出解析 | 思考、工具调用、JSON object、stop sequence |
| 图片 | OpenCV 实现的 V4.1 预处理 |
| 工具定义 | Function tools;Responses 另支持工具命名空间与 `apply_patch` 自定义工具 |

**明确不支持**(见 README):`logprobs`、文档/音视频输入、`file_id` 文件检索、服务端工具执行(如 `web_search`)、JSON Schema/regex 约束、`n>1` 多补全、Responses 的 `previous_response_id` 会话存储、加密思考内容等。

## 3. 代码结构

```
deepseek-recipe/
├── deepseek-recipe/              # 顶层 crate
├── deepseek-recipe-core/         # Conversation / 核心类型
├── deepseek-recipe-encoding/     # prompt / token IDs 编码
├── deepseek-recipe-image/        # V4.1 图片预处理(OpenCV)
├── deepseek-recipe-python/       # Python bindings
├── server-py/  server-rs/        # 示例服务
├── encoding-decoding-demo/
└── docs/                         # streaming / tokenizer 等
```

## 4. 在栈里的位置

```
 API 客户端(Chat Completions / Responses / Messages)
        │
        ▼
 deepseek-recipe   ← 本条目:格式归一 + prompt 编码 + 响应解析
        │
        ▼
 推理后端(外部,如 vLLM / 自研引擎)→ DeepSeek V4/V4.1
        │
        ▼
 工具执行 / HTTP 传输(外部)
```

- 与 [`vllm`](../../vllm/index.md) 的 OpenAI 兼容 API 是**互补**:vLLM 自带 API server;deepseek-recipe 面向「自己组装后端、但要兼容多格式 API」的场景,且**面向 V4/V4.1 的对话模板**。
- 与 [`deepseek-harness`](../../deepseek-models/frameworks/deepseek-harness.md) 无直接依赖:harness 是 agent 框架,recipe 是协议编解码。

## 5. 溯源

- 仓库:https://github.com/deepseek-ai/deepseek-recipe
- 中文 README:https://github.com/deepseek-ai/deepseek-recipe/blob/main/README.zh.md
- 库内互链:[deepseek-infra/index.md](../index.md)、[flashmla.md](flashmla.md)(V4.1 服务对象)、[open-model-landscape-2025-2026](../../model-architecture/concepts/open-model-landscape-2025-2026.md)(V4/V4.1 架构)
