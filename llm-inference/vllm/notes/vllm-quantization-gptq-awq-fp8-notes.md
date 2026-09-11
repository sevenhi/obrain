# vLLM 量化实现解析：GPTQ / AWQ / FP8 的注册、分派与内核集成

> **代码版本**：commit `b74ac0bc61d8b8f74848664b1a3b4e878f047861`（`git describe`: `v0.28.1rc0-40-gb74ac0bc6`，main 分支，2026-08-28）
>
> **注意**：本版本与旧版 vLLM 结构差异较大——`gptq.py / awq.py / gptq_marlin.py / awq_marlin.py` 已合并为 `auto_gptq.py / auto_awq.py`，Marlin 变体不再是独立 Config；LinearMethod 不再各自实现 repack/apply 内核细节，统一委托 `vllm/model_executor/kernels/linear` 的 `choose_mp_linear_kernel()` / `MPLinearLayerConfig`。下文全部以实际代码为准。

## 0. 目录结构与总览

量化相关代码集中在 `vllm/model_executor/layers/quantization/`（本版本共 33 个一级条目）：

| 文件/目录 | 对应量化方法 |
|---|---|
| `auto_gptq.py` | GPTQ（`gptq`/`gptq_marlin`/`auto_gptq` 均映射到此） |
| `auto_awq.py` | AWQ（`awq`/`awq_marlin`/`auto_awq` 均映射到此） |
| `awq_triton.py` | AWQ Triton kernel 后端 |
| `fp8.py` | FP8（checkpoint 离线 + MoE + KV cache） |
| `online/` | 在线量化（`fp8_per_tensor` 等短名的实际实现在此） |
| `compressed_tensors/` | compressed-tensors |
| `base_config.py` | `QuantizationConfig` 基类 |
| `__init__.py` | **注册表/工厂**：`get_quantization_config()` |
| `kv_cache.py` | KV cache 量化方法 |
| `mxfp4.py`, `moe_wna16.py`, `modelopt.py`, `quark/`, `torchao.py`, `inc/`, `humming.py` 等 | 其他方法 |

## 1. 注册与发现机制：方法名 → Config 类

### 1.1 方法名白名单

`vllm/model_executor/layers/quantization/__init__.py:12-46` 用 `Literal` 类型列出所有量化方法名（`QuantizationMethods`），并生成了 `QUANTIZATION_METHODS` 列表（`__init__.py:47`）。在线量化短名（`fp8_per_tensor` 等）也作为字符串列在此处以避免循环导入（`__init__.py:24-46`）。

### 1.2 核心工厂：get_quantization_config()

`vllm/model_executor/layers/quantization/__init__.py:108-180` 的 `get_quantization_config(quantization)` 是核心映射函数：

- **懒加载**：内部 `from vllm.config.quantization import _ONLINE_SHORTHANDS`（避免过早触发 `torch.compile`），并在函数体内按需 import 各 Config 类（`__init__.py:112-137`）。
- **映射表** `method_to_config`（`__init__.py:139-168`）：
  - `"awq" / "awq_marlin" / "auto_awq"` → `AutoAWQConfig`
  - `"auto_gptq" / "gptq" / "gptq_marlin"` → `AutoGPTQConfig`
  - `"fp8"` → `Fp8Config`
  - `"mxfp8"`（checkpoint 标签）→ `ModelOptMxFp8Config`
  - `"online"` 与各在线短名 → `OnlineQuantizationConfig`（`__init__.py:174-175` 用 `setdefault` 注册，保证既有的 checkpoint 方法名不被覆盖）
- 自定义注册：`register_quantization_config(quantization)` 装饰器（`__init__.py:58-105`）把用户自定义 Config 类写入 `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG`，并在 `__init__.py:178` 合并进映射表。

### 1.3 用户侧 config.json 的 quantization_config 如何生效

- 入口在 `vllm/config/model.py:1245 _verify_quantization()`：
  - 从 HF config 取 `quant_cfg = self.model_arch_config.quantization_config`（`model.py:1251`）；当 `quant_cfg is not None` 时读 `quant_method = quant_cfg["quant_method"]`（`model.py:1254`）。
  - **override 探测**（`model.py:1258-1290`）：维护 `overrides` 顺序列表，逐一对每种方法调用 `me_quant.get_quantization_config(name).override_quantization_method(quant_cfg, self.quantization, hf_config=...)`；若返回非 `None` 即认定 checkpoint 属于该方法的 override 变体并 `break`（`model.py:1291-1312`）。这就是为何 `gptq` 排在列表前面的原因（对 GPTQ 尤重要）。
  - 校验：`self.quantization`（CLI 传入）与 config.json 中 `quant_method` 不一致时报错；调用 `current_platform.verify_quantization()`（`model.py:1316-1332`）。
- CLI `--quantization` 与 `--quantization-config` 的合并逻辑在 `vllm/config/quantization.py:158-194 resolve_quantization_config()`：`quantization` 短名经 `_ONLINE_SHORTHANDS`（`quantization.py:116-148`）展开为完整 `QuantizationConfigArgs`；两者同给时 `quantization_config` 中显式字段优先。
- `vllm/config/quantization.py:26-36 QUANT_KEY_NAMES` 提供用户可读的量化粒度名（`fp8_per_tensor_static`、`fp8_per_block_dynamic` 等）→ 内部 `QuantKey` 常量的映射。

> 本版本已把旧版"`vllm/model_executor/layers/quantization/__init__.py` 直接维护大字典 + `vllm/config/model.py` 单独解析"的链路统一到 `get_quantization_config()` + `_verify_quantization()` 两条清晰入口，上文定位即权威路径。

## 2. GPTQ 实现（auto_gptq.py）

### 细节 1 — Config 与量化参数
`AutoGPTQConfig`（`auto_gptq.py:97`）：
- `TYPE_MAP`（`auto_gptq.py:101-104`）把 `(num_bits, is_sym)` 映射到 `scalar_types.uint4b8 / uint8b128`，只支持 4bit/8bit 对称量化。
- `pack_factor = 32 // weight_bits`（`auto_gptq.py:151`），int32 打包。
- `desc_act`（激活重排/权重量化顺序）与 `group_size`（`auto_gptq.py:152-153`）；当 `desc_act and group_size == -1` 时降级为 `desc_act=False`（`auto_gptq.py:118-121`，说明单组时 act_order 无意义）。
- 支持 GPTQModel 的 `dynamic` 属性对逐模块覆盖量化位宽/组大小（正则前缀 `+:` 正匹配、`-:` 跳过）（`auto_gptq.py:123-146`）。

### 细节 2 — override 与注册
`override_quantization_method`（`auto_gptq.py:219`）在 `_verify_quantization()` 中被调用，用于识别 checkpoint 属于 GPTQ 家族；`get_quant_method()` 为 Linear 层返回 `GPTQLinearMethod` 或 Marlin 变体（`auto_gptq.py:261-269` 设置 `marlin_input_dtype` / `input_dtype`）。

## 3. AWQ 实现（auto_awq.py）

### 细节 1 — 分组权重的打包参数
`BaseAWQLinearMethod.create_weights`（`auto_awq.py:840-898`）：
- 用 **`PackedvLLMParameter`** 定义 `qweight`（`auto_awq.py:857`）与 `qzeros`（`auto_awq.py:872`），`packed_dim=1, packed_factor=pack_factor`，数据类型 `torch.int32`——即 4bit 权重/零点按列打包进 int32。
- `scales` 用 `GroupQuantScaleParameter`（`auto_awq.py:885`），形状 `(num_groups, output_size_per_partition)`。
- 形状校验：`input_size_per_partition % group_size == 0`、`output_size_per_partition % pack_factor == 0`（`auto_awq.py:841-854`，张量并行切分过大时会触发）。

### 细节 2 — Triton 路径的 token 数启发式
`AutoAWQLinearMethod.apply`（`auto_awq.py:913`）：
- `FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256`（`auto_awq.py:927`）。
- 当 numel ≥ 256（或开启 batch-invariant 模式）时，用 `ops.awq_dequantize()` 反量化后走 `torch.matmul`（大批量下反量化开销摊薄、cuBLAS 更快）；否则走专用 `ops.awq_gemm()` Triton kernel（`awq_triton.py`）。

### 细节 3 — Config 与 override
`AutoAWQConfig`（`auto_awq.py:171`）同样实现 `override_quantization_method`（`auto_awq.py:260`），`get_quant_method()` 为 Linear 返回 `AutoAWQLinearMethod`（或经由 `choose_mp_linear_kernel()` 选中的 Marlin/AWQ Marlin kernel 后端）。

## 4. FP8 实现（fp8.py）

### 细节 1 — 配置解析与方法分派
- `Fp8Config.from_config()`（`fp8.py:155-172`）：`"fp8" in quant_method` → `is_checkpoint_fp8_serialized`；读 `activation_scheme`（static/dynamic）、`weight_block_size`（块量化，`fp8.py:160`）、`store_dtype`、`ignored_layers`/`modules_to_not_convert`（`fp8.py:159-165`）。
- `get_quant_method()`（`fp8.py:174-221`）按层类型分派：
  - `LinearBase` → 序列化 checkpoint 走 `Fp8LinearMethod`，否则走在线量化 `Fp8PerTensorOnlineLinearMethod`（`online/fp8.py`）（`fp8.py:185-196`）；被 `ignored_layers` 命中则返回 `UnquantizedLinearMethod`。
  - `RoutedExperts`（MoE）→ `Fp8MoEMethod`；当 `store_dtype=="mxfp4"` 时转 `Mxfp4MoEMethod`（`fp8.py:205-218`）。
  - `Attention` → `Fp8KVCacheMethod`（KV cache 量化）（`fp8.py:219-220`）。
- **`get_config_filenames()` 返回空列表**（`fp8.py:147-148`）——FP8 参数全部从 config.json 的 `quantization_config` 读取，无独立 `quantize_config.json`。

### 细节 2 — 量化粒度 key 与 kernel 初始化
`Fp8LinearMethod.init`（`fp8.py:251-290`）构造 weight/activation 的 `QuantKey`：
- 块量化（DeepSeek 风格 128×128）→ `kFp8Static128BlockSym` + `kFp8Dynamic128Sym`，且 `assert not act_q_static`（块量化强制动态激活）；**scale 参数名特意用 `weight_scale_inv`**（`fp8.py:348-349`，注释标明"intentional for deepseekv3"）。
- per-tensor → `kFp8StaticTensorSym`；**动态激活**在 `cutlass_fp8_supported()` 时升级为 per-token（`kFp8DynamicTokenSym`），否则退回 per-tensor（`fp8.py:283-290`）。
- `create_weights()` 末尾调用 `init_fp8_linear_kernel()`（`fp8.py:357`）选 kernel；若选中 `MarlinFP8ScaledMMLinearKernel` 则 `use_marlin=True`（`fp8.py:366`）——即无 FP8 硬件的 GPU 上用 Marlin 做 weight-only FP8。
- `process_weights_after_loading()`（`fp8.py:368`）对 per-tensor 融合层（如 QKV）：Marlin 路径先把 weight 转成 `(K, N)`（`fp8.py:372`）；非 Marlin 路径用 `process_fp8_weight_tensor_strategy` 取 max scale 重量化为单一 scale（`torch._scaled_mm` 要求）。

### 细节 3 — MoE 后端选择与硬件适配
- `Fp8MoEMethod`（`fp8.py:462`）在 `__init__`（`fp8.py:475`）中 `select_fp8_moe_backend()`（Triton/CUTLASS/DeepGEMM 等，见 `vllm/model_executor/layers/fused_moe/oracle/fp8.py`）。
- `process_weights_after_loading()`（`fp8.py:687`）：MI300x/MI325x 上 `normalize_e4m3fn_to_e4m3fnuz()` 转换 FP8 格式；w13 的 w1/w3 双 scale 用 `process_fp8_weight_tensor_strategy_moe` 取 max 重量化；最后 `convert_to_fp8_moe_kernel_format()`（`fp8.py:658`） + `make_fp8_moe_kernel()`（`fp8.py:679`）组装运行时 kernel，`apply()` 转发到 `self.moe_kernel`（`fp8.py:797`）。
- `Fp8LinearMethod.apply()`（`fp8.py:416`）在 `VLLM_BATCH_INVARIANT` 模式下非 block-quant 路径退化为 BF16 反量化 + `F.linear`。

## 5. 与旧版结构的关键差异（写报告时注意）

1. `gptq.py / awq.py / gptq_marlin.py / awq_marlin.py` → 合并为 `auto_gptq.py / auto_awq.py`，Marlin 变体不再是独立 Config。
2. LinearMethod 不再各自实现 repack/apply kernel 细节，统一委托 `vllm/model_executor/kernels/linear` 的 `choose_mp_linear_kernel()` + `MPLinearLayerConfig`。
3. MoE 量化统一走 `fused_moe/oracle/` 的 backend 选择（`select_wna16_moe_backend` / `select_fp8_moe_backend`）与 modular kernel（`make_*_moe_kernel`）。
4. 在线量化（`online/` 子包）与离线 checkpoint 量化在同一个 Config 的 `get_quant_method` 里分派（如 `Fp8Config`）。
5. `fp8.py` 中 `Fp8Config.get_config_filenames() == []`——FP8 参数全部从 config.json 的 `quantization_config` 读取，无独立 `quantize_config.json`。
6. 用户侧接入点统一到 `vllm/config/model.py:1245 _verify_quantization()`（override 探测）与 `vllm/config/quantization.py:158 resolve_quantization_config()`（CLI 短名展开与合并）。
