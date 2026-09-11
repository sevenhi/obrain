# vLLM 服务层与周边特性实现 — 分析笔记（总览）

> **任务**：分析 vLLM 服务层与周边特性实现（kanban t_be52d847）
> **代码版本**：commit `b74ac0bc61d8b8f74848664b1a3b4e878f047861`（`git describe`: `v0.28.1rc0-40-gb74ac0bc6`，main 分支，2026-08-28）
> **仓库路径**：`/home/cambricon/zhanbin/vllm`（官方 vllm-project/vllm）
> **说明**：本任务覆盖 5 大主题，每个主题的源码级分析分别成文于 `docs/vllm-*-notes.md`，本文为总览索引。

## 主题覆盖与验收对照

| # | 主题 | 笔记文件 | 关键实现细节数 |
|---|---|---|---|
| 1 | API server 层与 OpenAI 兼容接口 | `docs/vllm-api-server-openai-compat-notes.md` | ≥8 组（CLI 拓扑 / FastAPI 构建 / chat.callchain / 协议校验 / AsyncLLM 对接） |
| 2 | 模型加载与注册机制 | `docs/vllm-model-registry-notes.md` | ≥5 组（10 分类字典 / 懒加载 / 多架构复用 / _ModelInfo 子进程探测） |
| 3 | 量化支持（GPTQ/AWQ/FP8） | `docs/vllm-quantization-gptq-awq-fp8-notes.md` | ≥9 组（注册工厂 / auto_gptq / auto_awq / fp8 / MoE 后端） |
| 4 | 分布式执行（TP/PP、Ray、通信初始化） | `docs/vllm-distributed-ray-notes.md` | ≥5 组（Executor 抽象 / Ray actor / Rank 重排 / 通信组初始化） |
| 5 | 采样参数与惩罚机制 | `docs/vllm-sampling-penalties-notes.md` | ≥8 组（V1 采样管线 / Metadata / 三类惩罚算子 / top-k/p / 参数映射） |

验收标准：覆盖 5 主题、每主题 ≥3 个带源码引用的实现细节 —— **已满足**。

## 关键版本差异（区别于网上旧资料）

本版本（v0.28.1rc0-40-gb74ac0bc6）与经典结构差异较大，写报告 / 引用时务必注意：

- **API 层**：`vllm/entrypoints/openai/api_server.py` 已退化为 59 行 deprecation shim，转发到 `vllm.entrypoints.launchers`；每个协议端点拆成独立子包（`api_router.py` + `serving.py` + `protocol.py`）。
- **模型注册**：注册表从旧版 `model_registry.py` / `Models` 枚举 / `resolve_obj_cls`，演进为本版本的 `model_executor/models/registry.py` 分类字典集合 + `_VLLM_MODELS` 合并字典 + `_ModelRegistry.resolve_model_cls`。
- **量化**：`gptq.py / awq.py / gptq_marlin.py / awq_marlin.py` 合并为 `auto_gptq.py / auto_awq.py`；在线量化（fp8_per_tensor 等短名）实现在 `online/` 子目录。
- **分布式**：executor 从旧版 `vllm/executor/` 迁移到 `vllm/v1/executor/`（v1 引擎下）。
- **采样**：惩罚调用链位于 `vllm/v1/sample/ops/penalties.py`；新增 `vllm/v1/sample/logits_processor/` 按 argmax 不变性分组的框架。

## 各主题笔记链接

1. **API server 与 OpenAI 兼容接口** → `docs/vllm-api-server-openai-compat-notes.md`
2. **模型加载与注册机制** → `docs/vllm-model-registry-notes.md`
3. **量化支持（GPTQ/AWQ/FP8）** → `docs/vllm-quantization-gptq-awq-fp8-notes.md`
4. **分布式执行与 Ray 集成** → `docs/vllm-distributed-ray-notes.md`
5. **采样参数与惩罚机制** → `docs/vllm-sampling-penalties-notes.md`
