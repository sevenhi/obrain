---
title: vLLM 量化:GPTQ / AWQ / FP8 的注册、分派与内核集成
type: concept
tags: [vllm, quantization, gptq, awq, fp8, moe, kv-cache, model-executor]
source:
  - "code: vLLM V1 源码 (commit b74ac0bc6, v0.28.1rc0-40) | vllm/model_executor/layers/quantization/ 一手实读 (经 notes 提炼)"
  - "web: vLLM docs — Quantization | https://docs.vllm.ai/en/latest/features/quantization/"
date: 2026-09
---

# vLLM 量化:从 checkpoint 到 kernel 的完整分派链

> 本文由 `notes/vllm-quantization-gptq-awq-fp8-notes.md` 深度提炼而成(**notes 已删,以源码 commit b74ac0bc6 为准**)。源码基准 commit `b74ac0bc61d8b8f74848664b1a3b4e878f047861`(v0.28.1rc0-40,2026-08-28)。全文 path:line 均为实际代码位置。

## TL;DR

vLLM 的量化是"**一条分派链 + 三个决策点**":用户/checkpoint 声明量化方法 → **注册表 `get_quantization_config()`** 映射到 Config 类 → **Config 按层类型分派 `get_quant_method()`** 到对应的 LinearMethod/MoE/KV-cache 实现 → **LinearMethod 选 kernel 后端**(int4/int8 走 Marlin/Triton,FP8 走 CUTLASS/DeepGEMM)。

**看懂这条链,量化从"黑盒参数"变成可预测的选型**。本版(v0.28)三大结构变化:① `gptq.py/awq.py/gptq_marlin.py/awq_marlin.py` **合并为 `auto_gptq.py/auto_awq.py`**(Marlin 不再是独立 Config);② LinearMethod 不再各自实现 repack/apply,**统一委托 `choose_mp_linear_kernel()` + `MPLinearLayerConfig`**;③ 在线量化与离线 checkpoint 量化在**同一个 Config 的 `get_quant_method` 里分派**。

> **先记住一个反转直觉**:vLLM 的量化从不"在 Config 里写死内核",而是**分层解耦**——Config 管"该对这个 layer 用什么量化"(语义),kernel 选择器管"用什么内核实现"(机制),两层靠 `QuantKey`/`MPLinearLayerConfig` 对接。改内核不进 Config,改语义不进 kernel。

---

## 1. 分派链全景(一图看懂)

```
用户侧 config.json / --quantization
   │
   ▼
① resolve_quantization_config()      vllm/config/quantization.py:158   CLI 短名→完整 Args,含 override
   ▼
② _verify_quantization()             vllm/config/model.py:1245         读 checkpoint 的 quant_method
   │    └─ override_quantization_method()  逐方法探测,判定 checkpoint 属于哪个家族
   ▼
③ get_quantization_config()          vllm/model_executor/layers/quantization/__init__.py:108
   │    └─ method_to_config 映射: "gptq"→AutoGPTQConfig, "awq"→AutoAWQConfig, "fp8"→Fp8Config ...
   ▼
④ Config.get_quant_method(layer)     按层类型 ∈ {LinearBase, RoutedExperts, Attention}
   │    ├─ Linear  → Fp8LinearMethod / GPTQLinearMethod / AutoAWQLinearMethod / UnquantizedLinearMethod
   │    ├─ MoE     → Fp8MoEMethod / Mxfp4MoEMethod / WNA16MoE
   │    └─ Attn    → Fp8KVCacheMethod (KV cache 量化)
   ▼
⑤ LinearMethod.init() → choose_mp_linear_kernel() + MPLinearLayerConfig
   │    ├─ int4/int8: Marlin / Triton (AWQ 大 batch 退化 FP16 matmul)
   │    └─ FP8:      CUTLASS / MarlinFP8 / Triton / DeepGEMM (MoE)
   ▼
⑥ apply() → 反量化(或直接 FP8 内核) → matmul
```

**三个决策点**(普通用户至少要知道存在):注册表映射(**③**)、按层分派(**④**)、kernel 选型(**⑤**)。改一个量化方法从"口头说不清"变成"定位到这三个点之一"。

---

## 2. 注册表:方法名 → Config 类的懒加载工厂

### 2.1 方法名白名单与核心工厂

`get_quantization_config()`(`__init__.py:108-180`)是核心映射,懒加载避免过早触发 `torch.compile`:

```python
# vllm/model_executor/layers/quantization/__init__.py (示意,非逐字)
def get_quantization_config(quantization):
    # ★ 懒加载:函数体内按需 import,避免 import 时触发 torch.compile
    from vllm.config.quantization import _ONLINE_SHORTHANDS
    method_to_config = {
        "awq": AutoAWQConfig, "awq_marlin": AutoAWQConfig, "auto_awq": AutoAWQConfig,
        "gptq": AutoGPTQConfig, "gptq_marlin": AutoGPTQConfig, "auto_gptq": AutoGPTQConfig,
        "fp8": Fp8Config,
        "mxfp8": ModelOptMxFp8Config,
        # "online" 与各在线短名 → OnlineQuantizationConfig (setdefault,不覆盖既有 checkpoint 方法)
    }
    # 自定义注册:register_quantization_config() 装饰器注入 _CUSTOMIZED_METHOD_TO_QUANT_CONFIG
    return method_to_config.get(quantization)  # 简化
```

**注意竞态语义**:`__init__.py:174-175` 用 `setdefault` 注册在线短名,**保证既有的 checkpoint 方法名不被覆盖**——这是"在线量化(`online/`)与离线统一进同一 Config"的关键。

### 2.2 用户侧 config.json 怎么生效

- 入口 `vllm/config/model.py:1245 _verify_quantization()`:从 HF config 取 `quant_cfg = arch_config.quantization_config`,`quant_cfg["quant_method"]` 读方法名(`model.py:1251-1254`)。
- **override 探测**(`model.py:1258-1312`):对每种方法依次调 `get_quantization_config(name).override_quantization_method(...)`,返回非 None 即认定属该方法家族并 break。**这就是 `gptq` 排列表前面的原因**——越靠前越先判定。
- CLI `--quantization` 与 `--quantization-config` 合并(`vllm/config/quantization.py:158-194`):短名经 `_ONLINE_SHORTHANDS`(`:116-148`)展开;两者同给时 `quantization_config` 显式字段优先。
- **`get_config_filenames()` 返回空列表**(`fp8.py:147-148`):FP8 **没有独立 `quantize_config.json`**,全部参数从 config.json 的 `quantization_config` 读。这是排查"FP8 参数去哪了"时的关键事实。

---

## 3. 按层分派:一个 Config 管三类层

`get_quant_method()`(`fp8.py:174-221` 是范例)按层类型三路分派,**同一 Config 处理 Linear/MoE/Attention**:

| 层类型 | 方法 | 关键点 |
|---|---|---|
| `LinearBase` | 序列化 checkpoint → `Fp8LinearMethod`;在线 → `Fp8PerTensorOnlineLinearMethod`(`online/fp8.py`);`ignored_layers` 命中 → `UnquantizedLinearMethod` | **被忽略层**是分派链第一个"出口" |
| `RoutedExperts` (MoE) | `Fp8MoEMethod`;`store_dtype=="mxfp4"` → `Mxfp4MoEMethod` | MoE 有独立 backend 选择 |
| `Attention` | `Fp8KVCacheMethod` | KV cache 量化,与先验 kernel 无关 |

**这是"在线 vs 离线"一目了然的点**:`Fp8Config.get_quant_method` 里,同一个 `Fp8Config` 既分派离线(序列化 checkpoint)也分派在线(online 子包)——**不是两个独立的 Config**。曾以为"在线量化要单独配/单独 Config"的认知是错的。

---

## 4. int4/int8:GPTQ 与 AWQ

### 4.1 GPTQ — 注册 + 打包 + 降级

`AutoGPTQConfig`(`auto_gptq.py:97`):
- **TYPE_MAP**(`:101-104`):`(num_bits, is_sym)` → `uint4b8 / uint8b128`,**只支持 4bit/8bit 对称量化**。
- **pack_factor = 32 // weight_bits**(`:151`),int32 打包 → 4bit 时 8 个权重一个 int32。
- **desc_act 降级**:`desc_act and group_size == -1` 时降为 `desc_act=False`(`:118-121`)——**单组时 act_order(激活重排)无意义**。这是配置里最容易漏的隐性约束。
- 逐模块覆盖:支持 GPTQModel 的 `dynamic` 属性(`:123-146`),正则 `+:` 正匹配 / `-:` 跳过,可对特定模块单独调位宽/组大小。

### 4.2 AWQ — 打包参数与 Triton 启发式

`BaseAWQLinearMethod.create_weights`(`auto_awq.py:840-898`):
- **`PackedvLLMParameter`** 定义 `qweight`/`qzeros`(`:857/:872`),`packed_dim=1, packed_factor=pack_factor`,类型 `torch.int32`——**4bit 权重/零点按列打包进 int32**。
- `scales` 用 `GroupQuantScaleParameter`(`:885`),形状 `(num_groups, output_size_per_partition)`。
- 形状校验:`input_size % group_size == 0`、`output_size % pack_factor == 0`(`:841-854`)——**张量并行切分过大时触发**,排查 TP+AWQ 报错看这里。

**Triton 路径的 token 数启发式**(`AutoAWQLinearMethod.apply`, `auto_awq.py:913`):

```python
FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256   # auto_awq.py:927
# numel >= 256: 反量化后走 torch.matmul (cuBLAS,大批量反量化开销摊薄)
# numel <  256: 走专用 ops.awq_gemm() Triton kernel (awq_triton.py,小批量专用)
```

**这是 AWQ 最容易忽略的性能语义**:不是"AWQ 永远走 Triton kernel"——**大批量时反量化 + cuBLAS 反而更快**,启发式自动切。看到 AWQ 有时快有时慢,想的是这条折线,不是玄学。

### 4.3 Marlin:为什么 int4/int8 常落在它

本版 Marlin 不再是独立 Config,而是 **kernel 后端**:`get_quant_method()` 对 Linear 返回 `AutoGPTQLinearMethod`/`AutoAWQLinearMethod`,其内部再经 `choose_mp_linear_kernel()` 选中 **Marlin/AWQ-Marlin kernel**(见 §5)。`marlin_input_dtype`/`use_marlin` 在 `fp8.py:259-260`(FP8 无硬件时的 Marlin 兜底路径)设置,GPTQ/AWQ 的 `marlin_input_dtype` 归 `auto_gptq.py` 的 `get_marlin_input_dtype` 逻辑。

---

## 5. kernel 选型:MPLinearLayerConfig 统一网关

**本版本最大的架构抽象**(旧版每个 LinearMethod 自己 repack/apply 内核):

```python
# 语义: LinearMethod 只声明"用哪个 QuantKey"(fp8_static_128block / fp8_dynamic / gptq_marlin ...)
# 机制: choose_mp_linear_kernel(config) 选内核,产出 MPLinearLayerConfig 描述 kernel 需求
# 对接: LinearMethod 拿 MPLinearLayerConfig 去 repack 权重、选算子
```

### 5.1 FP8 的 QuantKey 与硬件适配(最值得精读)

`Fp8LinearMethod.init`(`fp8.py:251-290`)构造 weight/activation 的 `QuantKey`:
- **块量化**(DeepSeek:权重 128×128;激活动态 per-token-group)→ `kFp8Static128BlockSym` + `kFp8Dynamic128Sym`,且 **`assert not act_q_static`(块量化强制动态激活)**;scale 参数名**特意用 `weight_scale_inv`**(`:348-349`,注释明确 "intentional for deepseekv3")。
- **per-tensor** → `kFp8StaticTensorSym`;**动态激活**在 `cutlass_fp8_supported()` 时**升级 per-token**(`kFp8DynamicTokenSym`),否则退回 per-tensor(`:283-290`)。
- **无 FP8 硬件的 GPU** → `init_fp8_linear_kernel()` 选 `MarlinFP8ScaledMMLinearKernel` → `use_marlin=True`(`:366`)——**FP8 weight-only 降级到 Marlin**。
- `process_weights_after_loading()`(`:368`):per-tensor 融合层(如 QKV)Marlin 路径先转 `(K,N)`(`:372`);非 Marlin 用 `process_fp8_weight_tensor_strategy` 取 max scale 重化为单一 scale(`torch._scaled_mm` 要求)。

**MoE 后端选择**(`Fp8MoEMethod.__init__`, `fp8.py:475`):`select_fp8_moe_backend()`(Triton/CUTLASS/DeepGEMM,见 `vllm/model_executor/layers/fused_moe/oracle/fp8.py`)。`process_weights_after_loading()`(`:687`)中 **MI300x/MI325x 上 `normalize_e4m3fn_to_e4m3fnuz()` 转 FP8 格式**——AMD 平台独有,换平台排查 FP8 精度先查这条。

### 5.2 决策表(怎么选内核,memory-resident)

| 场景 | 内核落点 | 决策依据 |
|---|---|---|
| int4/int8,对称 | GPTQ → Marlin 系 | `choose_mp_linear_kernel()`,(默认 kernel,最优吞吐) |
| FP8,有 Blackwell/Hopper FP8 硬件 | CUTLASS FP8 / cutlass_fp8_supported | `cutlass_fp8_supported()` 为 True |
| FP8,无 FP8 硬件 | MarlinFP8 (weight-only) | `init_fp8_linear_kernel()` 兜底 |
| AWQ,大 batch | 反量化 + cuBLAS | `numel() >= 256` 启发式 |
| AWQ,小 batch | `awq_gemm()` Triton | 同上,未达阈值 |
| MoE FP8 | Triton / CUTLASS / DeepGEMM / WNA16 | `select_fp8_moe_backend()` |

---

## 6. 实践要点(checklist)

1. **量化报错先归因三层**:① Config 层(方法名映射/override)→ ② 分派层(get_quant_method 走错分支)→ ③ kernel 层(选型/硬件)。`Fp8Config.get_quant_method` 是 FP8 排查的主力入口。
2. **`quantize_config.json` 不存在是正常**:FP8 的 `get_config_filenames()==[]`,参数从 config.json 的 `quantization_config` 读——别再找不存在的独立量化配置文件。
3. **`--quantization` 与 config.json 冲突时**:override 探测先跑,`gptq` 排最前;CLI 与 checkpoint 的 `quant_method` 不一致会报错(`model.py:1316-1332` `verify_quantization`)。
4. **AWQ 性能不是一条直线**:大批量(numel≥256)走反量化+cuBLAS,小批量走 Triton kernel——调吞吐发现 AWQ 瓶颈时,先看 batch 是否跨过了这条启发式阈值。
5. **TP+AWQ/反量化形状校验**:`input_size%group_size`、`output_size%pack_factor` 不整除会炸——TP 并行切分过大时优先查这两条。
6. **AMD 平台 FP8**:MI300x/MI325x 会做 `e4m3fn → e4m3fnuz` 格式转换(`normalize_e4m3fn_to_e4m3fnuz`),换平台后 FP8 精度异常先查这条。
7. **看代码认链**:量化不是"配好参数就自动最优",把问题定位到 注册→分派→kernel 三段中的一段,再去对应文件(`quantization/__init__.py` / `auto_gptq.py` / `auto_awq.py` / `fp8.py` / `kernels/linear/`)。

---

## 相关条目

- [PTQ:GPTQ / AWQ / SmoothQuant](../../quantization/concepts/ptq-gptq-awq-smoothquant.md)——算法侧视角(为什么这么量化、误差怎么控制);本篇是引擎侧(量化怎么落地到注册/分派/kernel),两者互补。
- [KV Cache 量化](../../quantization/concepts/kv-cache-quantization.md)——本篇 §3 表中 `Fp8KVCacheMethod` 一行的 KV cache 量化展开。

## 7. 溯源

- 源码一手实读: `vllm/model_executor/layers/quantization/`(`auto_gptq.py` / `auto_awq.py` / `fp8.py` / `online/` / `__init__.py`),基准 commit `b74ac0bc6`
- 注册/配置链路: `vllm/config/model.py:1245`(`_verify_quantization`) / `vllm/config/quantization.py:158`(`resolve_quantization_config`)
- 版本差异基线: 本文第 1 节的"三大结构变化"对照自 notes `vllm-quantization-gptq-awq-fp8-notes.md`(已删)
