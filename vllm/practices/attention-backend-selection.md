---
title: vLLM V1 注意力后端选型:可插拔 backend 的机制、枚举与分派
type: comparison
tags: [vllm, v1, attention-backend, flash-attn, flashinfer, triton, mla, flashmla, cutlass-mla, rocm, xpu, backend-selection]
source:
  - "code: vllm/v1/attention/backends/registry.py (main) | https://github.com/vllm-project/vllm ↑primary"
  - "code: vllm/v1/attention/backend.py (main) | https://github.com/vllm-project/vllm"
  - "code: vllm/v1/attention/selector.py (main) | https://github.com/vllm-project/vllm"
  - "code: vllm/platforms/cuda.py (main) | https://github.com/vllm-project/vllm"
  - "web: vLLM 官方文档 V1 Guide | https://docs.vllm.ai/en/latest/usage/v1_guide/ | 阅读日期 2026-10-09"
date: 2026-10
---

# vLLM V1 注意力后端选型:可插拔 backend 的机制、枚举与分派

## TL;DR

**vLLM V1 的注意力内核是「可插拔 backend」**:同一套调度/执行骨架下,具体 attention 前向由不同 backend 实现,启动时按硬件 + 配置自动选一个。**分派链路(无 `ATTN_BACKEND` 环境变量)**:`get_attn_backend()` → `current_platform.get_attn_backend_cls()` → `get_valid_backends()`(按 priority 排序逐个 `validate_configuration()` 过滤)→ 选**最小 priority** 的合法 backend;显式指定走 **`--attention-backend` CLI**(等价 `AttentionConfig.backend`)。

**backend 枚举没有 TRT-LLM**。真实 main 枚举是:`FLASH_ATTN` / `FLASH_ATTN_DIFFKV` / `TRITON_ATTN` / `TRITON_ATTN_DIFFKV` / `FLASHINFER` / `FLEX_ATTENTION` / ROCM 系列 / **MLA 族(FLASHMLA / FLASHINFER_MLA / CUTLASS_MLA / TRITON_MLA / TOKENSPEED_MLA / FLASH_ATTN_MLA 及各自 *_SPARSE 变体)** / MiniMax M3 稀疏族(CUTLASS_MSA / TRITON_MSA)/ XPU / CPU / TURBOQUANT / HPC_ATTN / CUSTOM 等。**TRT-LLM 是独立推理引擎,不是 vLLM 的 attention backend**(它只作为 FlashInfer 内部一个可选内核路径出现,见 §4)。

本篇是「注意力内核 backend」层;引擎级选型(vLLM vs SGLang vs TRT-LLM)见 [inference-engine-selection.md](inference-engine-selection.md)。

---

## 1. 机制:可插拔的三件套抽象

V1 的 attention 抽象(backend.py)由三个类构成:

- **`AttentionBackend`(ABC,backend.py:61)**:backend 的元信息 + 能力声明。关键声明:`supported_dtypes` / `supported_kv_cache_dtypes`(默认 fp16/bf16)、`get_supported_head_sizes()`、`get_impl_cls()`(前向实现)、`get_builder_cls()`(metadata 构造器)。抽象方法:`get_name()` / `get_impl_cls()` / `get_builder_cls()`。
- **`CommonAttentionMetadata`(backend.py:389)**:所有 backend 共享的注意力元数据(序列长、block table、slot 等)。
- **`AttentionMetadataBuilder`(backend.py:593)**:从调度输出构造各 backend 所需 metadata 的构造器。

**注册机制(registry.py)**:`AttentionBackendEnum`(registry.py:34)枚举所有 backend,值为默认类路径;模块级函数 `register_backend()`(registry.py:243)通过 `_ATTN_OVERRIDES` 字典(registry.py:239)把某个 backend 名**覆盖/绑定**到自定义实现;`CUSTOM`(registry.py:130,value=None)是第三方/custom backend 的占位枚举,使用前须先 register。**枚举本身没有 `.register()` 方法**。**这是 V1 的新机制,不是 V0 的 `registerAttentionBackend`/`BackendMetadata`**(V0 概念已弃用)。

---

## 2. 默认分派链路(无 ATTN_BACKEND 环境变量)

```
get_attn_backend(head_size, dtype, kv_cache_dtype, use_mla, ...)   # selector.py:105
  └─ 组装 AttentionSelectorConfig(head_size/dtype/kv_cache_dtype/use_mla/use_sparse/...)
  └─ _cached_get_attn_backend(backend=attention_config.backend, ...)  # selector.py:204
       └─ current_platform.get_attn_backend_cls(backend, config, num_heads)  # cuda.py:438
            ├─ 若显式指定 backend → validate_configuration(),失败抛 ValueError
            └─ 否则 get_valid_backends()(cuda.py:375):
                 ├─ _get_backend_priorities(use_mla, device_capability, ...)  # cuda.py:83
                 │    按平台/是否 MLA/量化 KV 排出一个优先级列表
                 ├─ 逐个 backend_class.validate_configuration(device_capability, **config)
                 │    返回非法原因列表(空 = 合法);ImportError/OSError 视为不可用
                 └─ 返回 (合法候选列表, 非法原因字典)
            └─ min(candidates, key=priority)  → 选最小 priority(即最高优先级)
```

**要点**:

1. **优先级由 `_get_backend_priorities` 决定**(cuda.py:83),不是全局固定一张表——它按 `use_mla` / `device_capability.major`(10=Blackwell、12=Rubin SM120、9=Hopper)/ `kv_cache_dtype`(量化 KV)/ `num_heads` 分叉出不同顺序。
2. **量化 KV cache 会影响优先级**:例如 SM100 MLA + FP8 KV 时,`FLASHINFER_MLA_SPARSE` 排在 `FLASHMLA_SPARSE` 前(FlashInfer 对 FP8 KV 更好);BF16 KV 且 head 数 ≤16 时也优先 FlashInfer(FlashMLA 有 padding)。
3. **显式指定无效会硬报错**:选定的 backend `validate_configuration` 不过 → 抛 `ValueError`,不会静默回退。
4. **`--block-size` 会排挤高优先级 backend**:若用户指定了 block size 导致某些 backend 因「block_size not supported」被排除,vLLM 会打 warning 提示可去掉 `--block-size` 让自动选。

---

## 3. backend 枚举(严格按 main registry.py)

非 MLA(通用 attention):

| backend | 类路径(节选) | 说明 |
|---|---|---|
| `FLASH_ATTN` | `flash_attn.FlashAttentionBackend` | FA2/FA3/FA4(`--flash-attn-version 2/3/4` 可选代次) |
| `FLASH_ATTN_DIFFKV` | `flash_attn_diffkv.FlashAttentionDiffKVBackend` | 差异化 KV 缓存场景 |
| `TRITON_ATTN` | `triton_attn.TritonAttentionBackend` | Triton 实现 |
| `TRITON_ATTN_DIFFKV` | `triton_attn_diffkv.TritonAttentionDiffKVBackend` | 差异化 KV 缓存场景(Triton) |
| `FLASHINFER` | `flashinfer.FlashInferBackend` | FlashInfer 内核 |
| `FLEX_ATTENTION` | `flex_attention.FlexAttentionBackend` | PyTorch flex attention |
| `ROCM_ATTN` / `ROCM_AITER_FA` / `ROCM_AITER_UNIFIED_ATTN` | rocm_*.py | AMD ROCm 系列 |
| `XPU_MLA_SPARSE` | `mla.xpu_mla_sparse` | Intel XPU |
| `TORCH_SDPA` | — | 仅 ViT 用 tag |
| `CPU_ATTN` / `NO_ATTENTION` / `TURBOQUANT` / `HPC_ATTN` | 各自模块 | 各平台/量化专用 |
| `CUSTOM` | value=None | 第三方/custom backend 占位,须 `register_backend()` 先注册 |

MLA 族(DeepSeek MLA / sparse MLA,`vllm/v1/attention/backends/mla/` 子包):

| backend | 类路径 |
|---|---|
| `FLASHMLA` | `mla.flashmla.FlashMLABackend` |
| `FLASHMLA_SPARSE` | `mla.flashmla_sparse.FlashMLASparseBackend` |
| `FLASHINFER_MLA` | `mla.flashinfer_mla.FlashInferMLABackend` |
| `FLASHINFER_MLA_SPARSE`(含 `_SM120` 变体) | `mla.flashinfer_mla_sparse.*` |
| `CUTLASS_MLA` | `mla.cutlass_mla.CutlassMLABackend` |
| `TRITON_MLA` | `mla.triton_mla.TritonMLABackend` |
| `TOKENSPEED_MLA` | `mla.tokenspeed_mla.TokenspeedMLABackend` |
| `FLASH_ATTN_MLA` / `FLASH_ATTN_MLA_SPARSE` | `mla.flashattn_mla.*` |
| `ROCM_AITER_MLA` / `ROCM_AITER_TRITON_MLA` / `ROCM_AITER_MLA_SPARSE` | `mla.rocm_aiter_mla.*` |
| `AMX_MLA` / `CPU_MLA` | `mla.amx_mla` / `mla.cpu_mla` |
| `FLASHMLA_SPARSE_DSV4` / `FLASHINFER_MLA_SPARSE_DSV4` / `ROCM_FLASHMLA_SPARSE_DSV4` | `vllm.models.deepseek_v4*` 模型驱动 backend(V4 专用) |
| `MINIMAX_M3_SPARSE` / `CUTLASS_MSA` / `TRITON_MSA` | `vllm.models.minimax_m3.*` | MiniMax M3 稀疏 MLA |

**关键结论**:

- **没有 TRT-LLM backend**。整个枚举里唯一带 TRT-LLM 字样的是 FlashInfer 内部一条**内核路径** `use_trtllm_attention`(`AttentionConfig` 字段,注释「use the TRTLLM attention backend *in flashinfer*」),以及 FlashInfer MLA sparse 类名 `FlashInferMLASparseTRTLLMBackend`——都在 FlashInfer 内,不是独立 vLLM backend。
- **FlashMLA 是 NVIDIA + 昇腾双平台 backend**,不是昇腾专属(NVIDIA 侧 sparse decode 410 TFLOPS、昇腾侧 sparse prefill 410 TFlops,见 [deepseek-infra FlashMLA](../../deepseek-infra/concepts/flashmla.md))。
- V4/V4.1 的稀疏 MLA 走**模型驱动 backend**(`vllm.models.deepseek_v4*`),与通用 MLA backend 分名注册,避免 V4.1 模型解析到 V4.0 后端类。

---

## 4. 选型对比表

| backend | 平台 | 启动条件(依硬件/版本/量化) | 特征 |
|---|---|---|---|
| FLASH_ATTN | NVIDIA(Ampere+)、部分其他 | 默认高优先级;head_size 匹配;`--flash-attn-version` 可选 2/3/4 | 通用 dense attention 主力;FA3 仅 Hopper、FA4 仅 Blackwell(见 attention/implementation) |
| FLASHINFER | NVIDIA | SM100 上非 MLA 时优先级高于 FLASH_ATTN | 稀疏/长上下文/Paged 内核强 |
| TRITON_ATTN | 各平台(Triton) | 无专用内核时兜底 | 可移植、易读,性能中 |
| MLA 族(FLASHMLA/FLASHINFER_MLA/CUTLASS_MLA/TRITON_MLA/TOKENSPEED_MLA) | NVIDIA + 昇腾(FlashMLA 双平台) | `use_mla=True`(DeepSeek V2/V3/V4 系);量化 KV 或 head 数改变优先级 | 服务 MLA 低秩压缩;FP8 KV 优先 FlashInfer MLA,BF16 低 head 也偏 FlashInfer,否则偏 FlashMLA |
| ROCM 系列 | AMD | ROCm + aiter | AMD 专用内核 |
| XPU | Intel | XPU | |
| FLEX_ATTENTION | PyTorch | 兜底/可组合 mask | 灵活性最高,性能一般 |

**对 FP8 / 长上下文 / 稀疏注意力的支持差异**:

- **FP8 KV**:dense 走各 backend 的 FP8 内核;MLA 场景 FP8 KV 时 `FLASHINFER_MLA_SPARSE` 优先级高于 `FLASHMLA_SPARSE`(registry 注释引 issue #35807 基准)。
- **长上下文**:FlashInfer 与 FlashMLA 各有稀疏/长序优化;FLASH_ATTN 靠 SplitKV(`--flash-attn-max-num-splits-for-cuda-graph`)。
- **稀疏注意力**:DeepSeek V4 稀疏 MLA 有专门 `FLASHMLA_SPARSE_DSV4*` / `FLASHINFER_MLA_SPARSE_DSV4*` 模型驱动 backend;MiniMax M3 有 `CUTLASS_MSA` / `TRITON_MSA`;与 DeepSelect 出 indices 的 DSA 链路对接(见 [deepseek-infra](../../deepseek-infra/index.md))。

---

## 5. 实践要点(checklist + 坑)

1. **别用 `ATTN_BACKEND` 环境变量**——它不存在。显式选后端用 `--attention-backend`(如 `--attention-backend FLASH_ATTN`);或 `AttentionConfig.backend_per_kind` 按 KV-cache-group 分后端。
2. **「自动」已足够好,别乱指定**:默认 `_get_backend_priorities` + `validate_configuration` 会按平台/MLA/量化 KV/head 数自动选;显式指定错了直接 ValueError。调试时看启动日志 `Using X attention backend out of potential backends: [...]`。
3. **TRT-LLM 定位别写错**:引擎层选型才是 TRT-LLM(vs vLLM/SGLang,见 [inference-engine-selection.md](inference-engine-selection.md));在本层它最多是 FlashInfer 内的一条内核路径。
4. **FlashMLA 别当昇腾专属**:NVIDIA 侧也用它(尤其 MLA dense/sparse);昇腾侧经 `FLASH_MLA_BUILD_TARGET_PLATFORM=ASCEND` 编译(见 [ascend-deployment-checklist.md](../../deepseek-infra/practices/ascend-deployment-checklist.md))。
5. **量化 KV 会换后端**:FP8 KV 下 MLA 优先级翻转(FlashInfer 提前);调量化方案时重查生效 backend,别沿用 BF16 时的结论。
6. **`--block-size` 会静默降级**:指定 block size 可能把高优先级 backend 排挤掉(只打 warning),除非有硬约束否则别设。
7. **读源码认准 main**:枚举/优先级随版本变,本文口径是 2026-10 main;库内旧 v1-architecture §4 曾误写 TRTLLM backend(已更正,见 [v1-architecture.md](../concepts/v1-architecture.md))。

---

## 6. 溯源

- **registry.py**(main):`AttentionBackendEnum` 全枚举(行 34-130)、模块级 `register_backend()`(243)+ `_ATTN_OVERRIDES`(239)、`CUSTOM` 占位(130)—— https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py
- **backend.py**(main):`AttentionBackend`(61)、`CommonAttentionMetadata`(389)、`AttentionMetadataBuilder`(593)—— https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backend.py
- **selector.py**(main):`get_attn_backend`(105)、`_cached_get_attn_backend`(204)—— https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/selector.py
- **cuda.py**(main):`_get_backend_priorities`(83)、`get_valid_backends`(375)、`get_attn_backend_cls`(438)—— https://github.com/vllm-project/vllm/blob/main/vllm/platforms/cuda.py
- **config/attention.py**(main):`AttentionConfig.backend` / `backend_per_kind` / `use_trtllm_attention` / `mla_prefill_backend`—— https://github.com/vllm-project/vllm/blob/main/vllm/config/attention.py
- **库内交叉**:[v1-architecture.md](../concepts/v1-architecture.md)(前向链路里 backend 的位置)、[inference-engine-selection.md](inference-engine-selection.md)(引擎层选型)、[../../deepseek-infra/concepts/flashmla.md](../../deepseek-infra/concepts/flashmla.md)(FlashMLA 双平台与 DSA)、[../../attention/implementation/flash-attention-implementation.md](../../attention/implementation/flash-attention-implementation.md)(FA 代次机制)。
