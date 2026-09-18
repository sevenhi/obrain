---
title: vLLM ModelRegistry:架构名到实现类的懒加载映射
type: concept
tags: [vllm, model-registry, architecture, lazy-loading, model-executor, transformers]
source:
  - "code: vLLM V1 源码 (commit b74ac0bc6, v0.28.1rc0-40) | vllm/model_executor/models/registry.py 一手实读 (经 notes 提炼)"
  - "web: vLLM docs — Basic Model(新增模型贡献指南) | https://docs.vllm.ai/en/latest/contributing/model/basic/"
date: 2026-09
---

# ModelRegistry:架构名如何变成模型实现类

> 本文由 `notes/vllm-model-registry-notes.md` 深度提炼而成(**notes 已删,以源码 commit b74ac0bc6 为准**)。源码基准 commit `b74ac0bc6`(v0.28.1rc0-40,2026-08-28)。path:line 均实际验证。

## TL;DR

vLLM 的模型注册表本质是一张 **"架构名字符串 → (模块相对名, 类名)"的扁平映射**,按任务类型分成 **10 个分类字典**再合并为 `_VLLM_MODELS`,运行时包装成 `_LazyRegisteredModel` 实现**零 import 成本的懒加载**。架构解析优先级:**Transformers 后端(显式或兜底)→ 注册表逐架构(含后缀归一化)→ 报错(区分加载失败/已废弃/插件外置/完全不支持四类)**。

**看懂它,就能回答两个高频问题**:① "我怎么加一个新模型?"——只需在分类字典加一行 + 建实现文件(见 §4);② "报错 'Unrecognized architecture' 时到底哪错了?"——取决于架构是未注册、已废弃、被插件接管还是真不支持(见 §3)。

> **先记住一个反转直觉**:vLLM 的**模型注册和模型加载是解耦的**。注册表只存字符串,import `registry.py` **不加载任何模型类**;真正 import 类靠 `load_model_cls` 懒加载。同时**能力探测(`_ModelInfo`)也和加载解耦**——走磁盘缓存 + 子进程,避免主进程初始化 CUDA。这两层解耦是 registry 文件膨胀到 1500+ 行仍保持启动开销可控的关键。

---

## 1. 注册表数据结构:10 个分类字典 + 懒加载

### 1.1 不是 enum,是 10 个字典

"支持的模型列表"不是单一枚举,而是 **10 个按任务类型分类的字典**(`registry.py`):

| 字典 | 行号 | 用途 |
|---|---|---|
| `_TEXT_GENERATION_MODELS` | 72 | 文本生成(decoder-only LM) |
| `_EMBEDDING_MODELS` | 213 | 嵌入/池化模型 |
| `_LATE_INTERACTION_MODELS` | 266 | ColBERT 类晚交互检索 |
| `_REWARD_MODELS` | 282 | 奖励模型 |
| `_TOKEN_CLASSIFICATION_MODELS` | 288 | token 分类 |
| `_SEQUENCE_CLASSIFICATION_MODELS` | 309 | 序列分类 |
| `_MULTIMODAL_MODELS` | 338 | 多模态 |
| `_SPECULATIVE_DECODING_MODELS` | 596 | 投机解码 draft(EAGLE/MTP) |
| `_TRANSFORMERS_SUPPORTED_MODELS` | 672 | 显式声明走 Transformers 后端 |
| `_TRANSFORMERS_BACKEND_MODELS` | 699 | Transformers 通用后端类自身 |

每个条目 value 是 **`(模块相对名, 类名)` 二元组**,不是类引用:

```python
# registry.py:146
"LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
```

### 1.2 合并 + 全局单例

10 个字典在 `registry.py:734-745` 合并为 `_VLLM_MODELS`,再在 `:1478-1486` 构建全局单例:

```python
ModelRegistry = _ModelRegistry({
    model_arch: _LazyRegisteredModel(
        module_name=_resolve_module_name(mod_relname),
        class_name=cls_name,
    )
    for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
})
```

**两个关键设计**:

**(a) 多架构复用**——一个实现类挂多个架构名,即多个 key 指向相同二元组。例:`llama.py` 的 `LlamaForCausalLM` 同时服务(registry.py:88,134,146,149):

```python
"LlamaForCausalLM":   ("llama", "LlamaForCausalLM"),   # 本名
"LLaMAForCausalLM":   ("llama", "LlamaForCausalLM"),   # decapoda-research 旧拼写
"InternLM3ForCausalLM":("llama", "LlamaForCausalLM"),  # 结构兼容,复用
"TeleChat3ForCausalLM":("llama", "LlamaForCausalLM"),
```

`_EMBEDDING_MODELS` 里还有一段**字典推导**(`:229-234`)把 `_TEXT_GENERATION_MODELS` 中映射到 `LlamaForCausalLM` 的架构批量复制进嵌入表——同一批 Llama 系架构在 `--task embed` 下也能解析。

**(b) 全限定模块路径**——`_resolve_module_name`(`:1469-1475`):模块名以 `vllm.` 开头则全限定直接用(如 `"vllm.models.deepseek_v32"`,硬件隔离布局);否则拼 `vllm.model_executor.models.` 前缀指默认平铺目录。

---

## 2. 懒加载与 _ModelInfo 子进程探测

### 2.1 零 import 成本的懒加载

`ModelRegistry` 的 value 全是 `_LazyRegisteredModel`(`:913-1049`),只存 `module_name`/`class_name` 字符串,**import registry 时不加载任何模型类**。取类才 `importlib.import_module` + `getattr`(`load_model_cls`, `:1047-1049`)。

### 2.2 能力探测:磁盘缓存 + 子进程

vLLM 加载权重前需知道模型能力(是否池化/多模态/支持 PP),由 `inspect_model_cls`(`:998-1045`)提供 `_ModelInfo` frozen dataclass(`:817-877`)。**问题**:import 模型类可能初始化 CUDA(子进程 fork 后不能重初始化)。两级优化:

1. **磁盘缓存**:`_ModelInfo` 存 JSON `$VLLM_CACHE_ROOT/modelinfos/<module>-<class>.json`,键为模块源文件内容 hash(`_get_modelinfo_module_hash`, `:930-947`;若模块是 `__init__.py` 递归纳入整包)。hash 未变直接用缓存。
2. **子进程探测**:缓存 miss 时 `_run_in_subprocess`(`:1491-1518`)用 cloudpickle 把闭包发给 `python -m vllm.model_executor.models.registry` 子进程执行(`_run`, `:1521-1532`),**主进程零 CUDA 副作用**,结果写回缓存。

`_try_load_model_cls`/`_try_inspect_model_cls`(`:1052-1076`)套 `lru_cache(128)`,加载前调 `current_platform.verify_model_arch(model_arch)` 让硬件平台拦截不支持的架构。

---

## 3. 解析流程:从 architectures 到实现类

入口 `vllm/model_executor/model_loader/utils.py:203 _get_model_architecture`:从 `model_config.hf_config.architectures`(按序优先)取架构名,`registry.resolve_model_cls(architectures, ...)` 返回 `(实现类, 命中架构名)`。

`_ModelRegistry.resolve_model_cls`(`:1326-1378`)优先级:

1. **Transformers 后端**(显式 `model_impl=="transformers"`,或 `"auto"` 且全部架构不在注册表):`_try_resolve_transformers`(`:1178-1246`)——查 `_TRANSFORMERS_BACKEND_MODELS`,再 `getattr(transformers, architecture, None)` 找 Transformers 原生实现(`trust_remote_code` 下支持 config `auto_map` 动态模块),`is_backend_compatible()` 通过后返回通用后端类(如 `TransformersForCausalLM`)。**未注册的架构只要 transformers 有 backend 兼容实现,就能零注册跑起来**。
2. **主路径**:逐个遍历 `architectures`,先 `_normalize_arch`(`:1248-1272`)做**架构名归一化**——若原名不在注册表,用 `try_match_architecture_defaults`(`vllm/config/model.py:2209-2226`,基于 `_SUFFIX_TO_DEFAULTS` 后缀表 `config/model.py:2185-2202`,如 `ForCausalLM→generate`、`EmbeddingModel→pooling`)识别 runner/convert 语义后缀,再把后缀替换成其它候选逐一试注册表(例 `XxxForSequenceClassification → XxxForCausalLM`)。**一个 LM 实现类无需额外注册即可服务 `--task classify`/`--task embed`**。
3. **找不到**:`_raise_for_unsupported`(`:1133-1164`)四级错误——架构在表里但加载失败(看日志)、在 `_PREVIOUSLY_SUPPORTED_MODELS`(`:753-807`,废弃表,附最后支持版本如 `"MiniMaxM1ForCausalLM": "0.23.0"`)、在 `_OOT_SUPPORTED_MODELS`(`:809-814`,移到插件的架构,附插件 URL 如 bart-plugin)、否则 `ValueError` 附**完整 `Supported architectures: {...}` 列表**(用户看到的报错全文)。

**chokepoint**:`ModelConfig.registry` property(`vllm/config/model.py:1027-1030`)是所有 inspect/resolve 的唯一入口,顺带执行 `_maybe_register_model_class_overrides`(`:1032-1055`)注入用户 `model_class_overrides`(开发调试),外部插件也走 `ModelRegistry.register_model`(`:1087-1131`)扩展(`"模块:类"` 字符串或类对象,重复注册覆盖)。

---

## 4. 注册之后:权重加载链(补全"注册+加载"闭环)

注册表只负责"架构名 → 类",权重注入由 `model_loader` 完成另一段分派:`get_model_loader()`(`vllm/model_executor/model_loader/__init__.py:119`)据 `load_config.load_format` 选 loader——`DefaultModelLoader`(`default_loader.py:43`,从 HF Hub/Safetensors 流式加载、按层名匹配,默认)或 `ShardedStateLoader`/`TensorizerLoader`/`RunaiStreamerLoader` 等变体;`get_model()`(`model_loader/__init__.py:127`)实例化 `nn.Module` 后调 `loader.load_weights()` 注入权重。TP/PP 切分无需用户干预——`ColumnParallelLinear`/`RowParallelLinear` 在层内自动完成。即:**架构解析(本文 §3)→ 类实例化 → loader 分派注入权重**,三段接力构成"注册+加载"的完整范围。

## 5. 新增一个模型要改哪里(可复现操作)

以新增原生支持架构为例,**只改两处**:

1. **实现类**:在 `vllm/model_executor/models/` 新建/复用实现文件,如 `llama.py` 定义 `LlamaForCausalLM`(`llama.py:446`)。该目录与注册表模块名直接对应(`("llama","LlamaForCausalLM")` → `.../models/llama.py::LlamaForCausalLM`),当前共 281 个实现文件;需硬件隔离布局的放 `vllm/models/` 并写全限定模块名。
2. **注册表条目**:在 `registry.py` 对应分类字典加一行 `"架构名": ("模块相对名", "类名")`。文件头注释(`registry.py:3-6`)要求**同步在 `tests/models/registry.py` 补该架构的示例 HF 模型**供 CI。

完成后,解析、能力探测(`_ModelInfo` 自动从类 mixin 读出)、Transformers 回退、错误提示支持列表**全部自动生效,无需其它接线**。

---

## 6. 实践要点(checklist)

1. **加新模型只动两处**:实现文件 + 注册表一行(记得补 `tests/models/registry.py` 示例)。别试图改解析逻辑。
2. **"Unrecognized architecture" 是四类之一**:先分清——加载失败(看日志)/ 已废弃(查 `_PREVIOUSLY_SUPPORTED_MODELS` 的最后版本)/ 插件外置(查 `_OOT_SUPPORTED_MODELS` 的插件 URL)/ 真不支持(看 `Supported architectures` 列表)。不同类别的解法完全不同。
3. **多架构复用靠"多 key 同 value"**:想复用某个兼容架构,只需在注册表多映射一行,不用复制实现类。
4. **任务级复用靠后缀归一化**:调 `--task classify`/`--task embed` 时,LM 类能复用是因为 `_normalize_arch` 把后缀从 `...ForSequenceClassification` 归一化到 `...ForCausalLM` —— **别给每个任务写一套 LM 实现**。
5. **未注册架构 ≠ 不能跑**:只要 transformers 里有 backend 兼容(`is_backend_compatible()`),`model_impl="transformers"` 或 `"auto"` 兜底能零注册跑——但这会失去 vLLM 原生 kernel 优化,做服务前先确认后端兼容与性能。
6. **能力探测离线**:`_ModelInfo` 走磁盘缓存 + 子进程,不污染主进程 CUDA 状态;改模型实现后若能力信息不更新,删 `$VLLM_CACHE_ROOT/modelinfos/` 下对应缓存(内容 hash 一般会自动失效)。
7. **看代码认路径**:要改注册行为找 `registry.py`(数据/懒加载/解析),要改配置链找 `vllm/config/model.py` 的 `ModelConfig.registry` property——两者是"数据"与"入口"的分工。

---

## 7. 溯源

- 源码一手实读: `vllm/model_executor/models/registry.py`(分类字典 / `_VLLM_MODELS` / `_ModelRegistry.resolve_model_cls` / `_ModelInfo` / 子进程探测)+ `vllm/model_executor/model_loader/`(`get_model_loader` / `get_model`,§4 加载链),基准 commit `b74ac0bc6`
- 配置链入口: `vllm/config/model.py:1027`(`ModelConfig.registry`) / `:1245`(`_verify_quantization` 无关,但 arch 解析同文件 `:2209`)
- 版本差异基线: 本文对照自 notes `vllm-model-registry-notes.md`(已删);早期 `Models` 枚举 / `_MODELS` 字典 / `resolve_obj_cls` 在本版演进为分类字典 / `_VLLM_MODELS` / `_ModelRegistry.resolve_model_cls`
