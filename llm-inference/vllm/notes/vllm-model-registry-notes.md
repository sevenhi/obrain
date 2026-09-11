# vLLM ModelRegistry：架构名到实现类的映射机制

> **代码版本**：commit `b74ac0bc61d8b8f74848664b1a3b4e878f047861`（`git describe`: `v0.28.1rc0-40-gb74ac0bc6`，main 分支，2026-08-28）
>
> 注意：本版本的注册表文件是 `vllm/model_executor/models/registry.py`（而非旧版的 `vllm/model_executor/model_registry.py`）；早期版本的 `Models` 枚举 / `_MODELS` 字典 / `resolve_obj_cls` 函数在本版本中分别演进为分类字典集合 / `_VLLM_MODELS` 合并字典 / `_ModelRegistry.resolve_model_cls` 方法，下文以实际代码为准。

## 1. 注册表数据结构：分类字典 + 懒加载

"支持的模型列表"在源码中不是单一枚举，而是 **10 个按任务类型分类的字典**，位于 `vllm/model_executor/models/registry.py`：

| 字典 | 行号 | 用途 |
|---|---|---|
| `_TEXT_GENERATION_MODELS` | registry.py:72 | 文本生成（decoder-only LM） |
| `_EMBEDDING_MODELS` | registry.py:213 | 嵌入/池化模型 |
| `_LATE_INTERACTION_MODELS` | registry.py:266 | ColBERT 类晚交互检索 |
| `_REWARD_MODELS` | registry.py:282 | 奖励模型 |
| `_TOKEN_CLASSIFICATION_MODELS` | registry.py:288 | token 分类 |
| `_SEQUENCE_CLASSIFICATION_MODELS` | registry.py:309 | 序列分类 |
| `_MULTIMODAL_MODELS` | registry.py:338 | 多模态 |
| `_SPECULATIVE_DECODING_MODELS` | registry.py:596 | 投机解码 draft 模型（EAGLE/MTP） |
| `_TRANSFORMERS_SUPPORTED_MODELS` | registry.py:672 | 显式声明走 Transformers 后端的架构 |
| `_TRANSFORMERS_BACKEND_MODELS` | registry.py:699 | Transformers 通用后端类自身 |

每个条目的 value 不是类引用，而是 **`(模块相对名, 类名)` 二元组**，例如 registry.py:146：

```python
"LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
```

10 个字典在 registry.py:734-745 合并为总表 `_VLLM_MODELS`，随后在 registry.py:1478-1486 构建全局单例：

```python
ModelRegistry = _ModelRegistry({
    model_arch: _LazyRegisteredModel(
        module_name=_resolve_module_name(mod_relname),
        class_name=cls_name,
    )
    for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
})
```

两个关键设计：

**（a）多架构复用** —— 一个实现类挂多个架构名，就是注册表里多个 key 指向相同二元组。例如 `llama.py` 中的 `LlamaForCausalLM` 同时服务（registry.py:88, 134, 146, 149）：

```python
"LlamaForCausalLM":  ("llama", "LlamaForCausalLM"),   # 本名
"LLaMAForCausalLM":  ("llama", "LlamaForCausalLM"),   # decapoda-research 旧拼写
"InternLM3ForCausalLM": ("llama", "LlamaForCausalLM"),  # 结构兼容，直接复用
"CwmForCausalLM":    ("llama", "LlamaForCausalLM"),
"TeleChat3ForCausalLM": ("llama", "LlamaForCausalLM"),
```

`_EMBEDDING_MODELS` 里更有一段字典推导（registry.py:229-234），把 `_TEXT_GENERATION_MODELS` 中所有映射到 `LlamaForCausalLM` 的架构批量复制进嵌入表——即同一批 Llama 系架构名在 `--task embed` 下也能解析，HF 纯 backbone 名（`LlamaModel`、`MistralModel`，registry.py:228/235）同样映射到生成式实现类做嵌入。

**（b）全限定模块路径** —— `_resolve_module_name`（registry.py:1469-1475）规定：模块名以 `vllm.` 开头则视为全限定路径直接使用（如 `"vllm.models.deepseek_v32"`，registry.py:94-97，硬件隔离布局）；否则拼接 `vllm.model_executor.models.` 前缀指向默认平铺目录。

## 2. 懒加载与 _ModelInfo 子进程探测

`ModelRegistry` 的 value 全部是 `_LazyRegisteredModel`（registry.py:913-1049）——只存 `module_name`/`class_name` 字符串，import registry 时不加载任何模型类。真正取类时才 `importlib.import_module(self.module_name)` + `getattr(mod, self.class_name)`（`load_model_cls`，registry.py:1047-1049）。

vLLM 在加载权重前还需要知道模型的能力（是否池化、是否多模态、是否支持 PP 等），这由 `inspect_model_cls`（registry.py:998-1045）提供，返回 `_ModelInfo` frozen dataclass（registry.py:817-877）——通过 `interfaces.py` / `interfaces_base.py` 中的一族探测函数（`is_pooling_model`、`supports_multimodal`、`supports_pp` 等）从实现类上挂的 mixin/属性静态读出。为避免 import 模型类时初始化 CUDA（子进程 fork 后不能重初始化 CUDA），探测走两级优化：

1. **磁盘缓存**：`_ModelInfo` 以 JSON 存在 `$VLLM_CACHE_ROOT/modelinfos/<module>-<class>.json`，键为模块源文件内容 hash（`_get_modelinfo_module_hash`，registry.py:930-947；若模块是 `__init__.py` 还会递归纳入整个包内容）。hash 未变直接用缓存。
2. **子进程探测**：缓存 miss 时 `_run_in_subprocess`（registry.py:1491-1518）用 cloudpickle 把闭包发给 `python -m vllm.model_executor.models.registry` 子进程执行（`_run` 入口，registry.py:1521-1532），主进程零 CUDA 副作用，结果再写回缓存。

`_try_load_model_cls` / `_try_inspect_model_cls`（registry.py:1052-1076）套了 `lru_cache(128)`，且加载前会先调 `current_platform.verify_model_arch(model_arch)` 让硬件平台拦截不支持的架构。

## 3. 解析流程：从 hf_config.architectures 到实现类

入口在 `vllm/model_executor/model_loader/utils.py:203` 的 `_get_model_architecture`：从 `model_config.hf_config.architectures`（HF config.json 的 `architectures` 列表，按序优先）取出架构名，调用 `model_config.registry.resolve_model_cls(architectures, model_config=model_config)`，返回 `(实现类, 实际命中的架构名)`。

`_ModelRegistry.resolve_model_cls`（registry.py:1326-1378）按以下优先级解析：

1. **`model_impl == "transformers"`**（显式要求 Transformers 后端）或 **`model_impl == "auto"` 且全部架构不在注册表**：调 `_try_resolve_transformers`（registry.py:1178-1246）——先查 `_TRANSFORMERS_BACKEND_MODELS`；再从 `getattr(transformers, architecture, None)` 找 Transformers 原生实现（支持 `trust_remote_code` 下 config `auto_map` 的动态模块），`is_backend_compatible()` 通过后返回 `model_config._get_transformers_backend_cls()` 选出的通用后端类（如 `TransformersForCausalLM`）。这意味着**未注册的架构只要 transformers 里有 backend 兼容实现，就能零注册跑起来**。
2. **主路径**：逐个遍历 `architectures`，先 `_normalize_arch`（registry.py:1248-1272）做**架构名归一化**——若原名不在注册表，用 `try_match_architecture_defaults`（`vllm/config/model.py:2209-2226`，基于 `_SUFFIX_TO_DEFAULTS` 后缀表，config/model.py:2185-2202，如 `ForCausalLM`→generate/none、`EmbeddingModel`→pooling/embed）识别出 runner/convert 语义后缀，再把该后缀替换成其它候选后缀逐一试注册表（例如 `XxxForSequenceClassification` → `XxxForCausalLM`）。这让一个 LM 实现类无需额外注册即可服务 `--task classify`/`--task embed`。命中后 `_try_load_model_cls` 真正 import 类。
3. **找不到**：`_raise_for_unsupported`（registry.py:1133-1164）给出三级错误信息——架构在表里但加载失败（提示看日志）；在 `_PREVIOUSLY_SUPPORTED_MODELS`（registry.py:753-807，废弃表，附最后支持版本，如 `"MiniMaxM1ForCausalLM": "0.23.0"`）或 `_OOT_SUPPORTED_MODELS`（registry.py:809-814，移到插件的架构，附插件 URL 如 bart-plugin）；否则抛 `ValueError` 并附**完整 `Supported architectures: {...}` 列表**，即用户看到的"支持的模型列表"报错全文。

补充：`ModelConfig.registry` property（`vllm/config/model.py:1027-1030`）是所有 inspect/resolve 的唯一 chokepoint，会顺带执行 `_maybe_register_model_class_overrides`（config/model.py:1032-1055）把用户配置的 `model_class_overrides`（开发调试用途）经 `ModelRegistry.register_model`（registry.py:1087-1131）注入——该方法支持 `"模块:类"` 字符串或类对象两种形式，重复注册时覆盖旧条目。外部插件也走同一入口扩展注册表。

## 4. 新增一个模型要改哪里

以新增一个原生支持的架构为例，改动集中在两处：

1. **实现类**：在 `vllm/model_executor/models/` 下新建（或复用）实现文件，如 `models/llama.py` 定义 `LlamaForCausalLM`（llama.py:446）。该目录与注册表模块名直接对应（`("llama", "LlamaForCausalLM")` → `vllm/model_executor/models/llama.py::LlamaForCausalLM`），目前共 281 个实现文件；需要硬件隔离布局的模型则放 `vllm/models/` 下并在注册表里写全限定模块名。
2. **注册表条目**：在 `registry.py` 对应任务类型的分类字典（如 `_TEXT_GENERATION_MODELS`）加一行 `"架构名": ("模块相对名", "类名")`。文件头注释（registry.py:3-6）明确要求**同步在 `tests/models/registry.py` 补上该架构的示例 HF 模型**，供 CI 验证。

完成这两步后，解析、能力探测（`_ModelInfo` 自动从类上的 mixin 读出）、Transformers 后端回退、错误提示中的支持列表全部自动生效，无需其它接线。

## 小结

- 注册表本质是"架构名字符串 → (模块名, 类名)"的扁平映射，按任务类型分成 10 个源字典再合并为 `_VLLM_MODELS`，运行时包装成 `_LazyRegisteredModel` 实现零 import 成本的懒加载。
- 多架构复用靠"多 key 同 value"表达；任务级复用（generate 类实现跑 embed/classify）靠 `_normalize_arch` 的后缀归一化，不必重复注册。
- 解析优先级：Transformers 后端（显式或兜底）→ 注册表逐架构（含归一化）→ 报错，报错信息区分"加载失败 / 已废弃 / 插件外置 / 完全不支持"四类，并附全量支持列表。
- 能力探测与类加载解耦：`_ModelInfo` 经磁盘缓存 + 子进程探测获得，避免主进程初始化 CUDA，这是 registry 文件膨胀到 1500+ 行仍保持启动开销可控的关键。
