---
title: deepseek-recipe — DeepSeek V4/V4.1 多格式 API 与 prompt 编解码库
type: concept
tags: [deepseek-recipe, api, prompt, chat-completions, responses, v4, v4.1, rust, python, serving]
date: 2026-10
source:
  - "code: deepseek-ai/deepseek-recipe (v0.2.0) | https://github.com/deepseek-ai/deepseek-recipe | 2026-10-20 | ↑primary"
  - "code: deepseek-recipe README.zh | https://github.com/deepseek-ai/deepseek-recipe/blob/main/README.zh.md | 2026-10-20"
  - "code: deepseek-recipe CHANGELOG (0.2.0) | https://github.com/deepseek-ai/deepseek-recipe/blob/main/CHANGELOG.md | 2026-10-20"
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
| 图片 | OpenCV 实现的 V4.1 预处理;V4 不接受图片,转换替换为 `[Unsupported Image]` |
| 工具定义 | Function tools;Responses 另支持工具命名空间与 `apply_patch` 自定义工具 |

**明确不支持**(见 README 0.2.0):文档内容、音视频输入、`file_id` 文件检索、服务端工具执行(如 `web_search`)、工具 `strict` 强制、`n>1` 多补全、Responses `apply_patch` 之外的自定义工具、Responses 的 `previous_response_id` 会话存储、加密思考内容。**`logprobs`/`top_logprobs`、JSON Schema/regex 约束已在 0.2.0 支持**(见下节),不再是「不支持」。

## 3. 代码结构

```
deepseek-recipe/
├── deepseek-recipe/              # 顶层 crate
├── deepseek-recipe-core/         # Conversation / 核心类型
├── deepseek-recipe-encoding/     # prompt / token IDs 编码
├── deepseek-recipe-image/        # V4.1 图片预处理(OpenCV)
├── deepseek-recipe-macros/    # v0.2.0 新增:proc-macro(`prompt!` 宏)、untagged 序列化辅助
├── deepseek-recipe-python/       # Python bindings
├── server-py/  server-rs/        # 示例服务
├── encoding-decoding-demo/
└── docs/                         # streaming / tokenizer 等
```

## 3.5 v0.2.0 增量(2026-10-20 发布,破坏性 API 变更)

0.1 → 0.2.0 改了**所有 crate 与 Python 包的公共 API**(0.1 之前的版本号;**1.0 之前 minor 可含不兼容改动**)。主要新增:

- **模型版本解析(ModelVersion)**:按请求里的 `model` 名解析 V4 或 V4.1,默认接受 `DeepSeek-V4-Pro` 与 `DeepSeek-V4.1-Flash`;其他名字(如 `deepseek-flash`)用 `ConversionOptions::with_model_alias` / `with_model_versions_by_name` 注册。`ConversationRequest::model_version` 返回结果。
- **按模型版本的采样参数解析**:`temperature` / `top_p` 不再透传,由转换层按 ModelVersion 解析——thinking 模式下 `temperature` 恒为 1.0;`top_p` 依 `ModelVersion::resolve_top_p`(thinking 模式 V4.1 抬到 ≥0.95,V4 用工具 0.95/无工具 1.0;非 thinking 模式 V4.1 用 1.0、V4 用请求值)。
- **Log probabilities**:Chat Completions 的 `logprobs`/`top_logprobs` 与 Responses 的 `top_logprobs` 转换进 `InferenceOptions`;调用方在后端 `InferenceChunk::Tokens` 提供值,adapter 输出(库本身不算概率)。
- **Reasoning token 计数**:`CompletionUsage::reasoning_tokens`,thinking 模式下 stream processor 附加 tokenizer 后对 reasoning 内容计数。
- **结构化输出**:新增 `ResponseFormat::JsonSchema` 与 `ResponseFormat::Regex`。转换**校验** Chat Completions `response_format` 或 Responses `text.format` 的 `json_schema`(0.1 是"接受但忽略"),渲染时把 schema 写进 prompt;`regex` 以 `ResponseFormat::Regex` 透传。是否由后端约束解码执行由调用方决定。
- **显式特殊 token**:新增 `deepseek-recipe-macros` crate 提供 `prompt!` 宏,`deepseek-recipe-core` 提供 `Prompt`;`RenderedPrompt::special_ranges` 记录显式特殊 token 的字节区间,`PromptTokenizer` 只把该区间编码为特殊 token——除 assistant/reasoning 内容的 grounding token 外,消息文本里的特殊 token 拼写按纯文本编码。
- **max_tokens 默认值**:`ConversationRequest::resolve_max_tokens` 按上下文长度校验 prompt 与 `max_tokens`,给未指定者取默认——非思考模式 8192、思考模式 65536、思考模式且 `max` effort 131072。

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
- CHANGELOG(0.2.0 增量):https://github.com/deepseek-ai/deepseek-recipe/blob/main/CHANGELOG.md
- 库内互链:[deepseek-infra/index.md](../index.md)、[flashmla.md](flashmla.md)(V4.1 服务对象)、[open-model-landscape-2025-2026](../../model-architecture/concepts/open-model-landscape-2025-2026.md)(V4/V4.1 架构)
