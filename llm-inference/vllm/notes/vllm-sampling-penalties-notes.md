# vLLM 采样参数与惩罚机制实现研究笔记

> **代码版本**：commit `b74ac0bc61d8b8f74848664b1a3b4e878f047861`（`git describe`: `v0.28.1rc0-40-gb74ac0bc6`，main 分支，2026-08-28）
>
> 注意：本版本的采样相关结构较旧版有较大演进——惩罚调用链位于 `vllm/v1/sample/ops/penalties.py`，新增 `vllm/v1/sample/logits_processor/` 框架（`interface.py`/`builtin.py`/`state.py`）把 logits 处理器按"是否 argmax 不变"分组，采样元数据放 `vllm/v1/sample/metadata.py`。下文以实际代码为准。

## 1. V1 采样管线总览 —— Sampler

文件：`vllm/v1/sample/sampler.py`，类 `Sampler(nn.Module)`（sampler.py:21）。其 docstring（21-60 行）给出的标准处理顺序，与 `forward` / `sample` / `apply_logits_processors` 实现一致：

1. **logprobs 预留**：若请求了 logprobs，先在原始 logits 上计算。`raw_logprobs` 模式调用 `compute_logprobs`（sampler.py:305，即 `log_softmax(dim=-1, dtype=float32)`）；`raw_logits` 模式直接 clone。关键差异（sampler.py:81-84，`NOTE(woosuk)`）：**V1 用惩罚与温度缩放之前的原始 logits 计算 top-k logprobs**，而 V0 用的是采样时的处理后 logits。
2. **转 float32**（sampler.py:97）。
3. **allowed token ids 白名单**：`allowed_token_ids_mask` 2D bool 掩码，越界位 `masked_fill_(-inf)`（sampler.py:393-394）。
4. **bad words 排除**：`apply_bad_words`（sampler.py:397-398，实现在 `vllm/v1/sample/ops/bad_words.py`）。
5. **非 argmax 不变的 logits processor**（会影响贪心解码的）：min tokens、logit bias，经 `sampling_metadata.logitsprocs.non_argmax_invariant` 循环应用（sampler.py:401-402）。
6. **三类惩罚**：repetition → frequency → presence，入口为静态方法 `Sampler.apply_penalties`（sampler.py:421-438）。
7. **采样**（`sample`，sampler.py:244-303）：
   - 若非全随机批，先算贪心候选 `greedy_sample`（sampler.py:241，`argmax(dim=-1)`）；全贪心批直接返回。
   - `apply_temperature`（sampler.py:228-238）：原地 `logits.div_(temp.unsqueeze(1))`；非全随机批中温度低于 `_SAMPLING_EPS = 1e-5`（sampler.py:18）的行置为 1.0 防除零。
   - **argmax 不变 processor（默认 min_p）在温度之后应用**（sampler.py:283-284）——这就是 min_p 不影响贪心结果的原因。
   - top-k/top-p 交给 `TopKTopPSampler`（sampler.py:287-292）。
   - 混合批逐行选择：`torch.where(temperature < 1e-5, greedy_sampled, random_sampled)`（sampler.py:297-302），即温度≈0 的行走贪心、其余走随机。
8. **收集 logprobs**：`gather_logprobs`（sampler.py:310-358）用 `torch.topk` + `gather` 取 top-N 与采样 token 的 logprob，用 `batched_count_greater_than` 计算 rank；并对 batch 维做 `torch._dynamo.decorators.mark_unbacked`（sampler.py:347-348）避免 batch 从 1→≥2 时重编译。

其它已核对细节：

- **推测解码与惩罚的交互**：`predict_bonus_token` 为真且存在惩罚/bad_words 时，`_combine_outputs_with_spec_tokens`（sampler.py:360-371）把草稿 token 并入 `output_token_ids` 再施加惩罚（sampler.py:384-390），保证惩罚能看到草稿 token。
- **thinking budget**：惩罚之后由 `thinking_budget_state_holder.apply_to_logits` 追加处理（sampler.py:406-418），状态更新只用已提交输出。

## 2. 采样元数据 —— SamplingMetadata

文件：`vllm/v1/sample/metadata.py`，dataclass `SamplingMetadata`（14-55 行）。这是请求级 `SamplingParams` 落到 GPU worker 侧的张量化形态，字段直接反映引擎如何组织采样参数：

- `temperature` / `top_p` / `top_k`：**逐请求 batch 张量**（`torch.Tensor | None`），而非标量——支持同一批内混合参数。
- `all_greedy` / `all_random`：整批标记，用于走捷径。
- `generators: dict[int, torch.Generator]`：行号 → 请求级种子 RNG（对应 `SamplingParams.seed`）。
- `no_penalties: bool` + `prompt_token_ids: torch.Tensor | None` + `frequency_penalties` / `presence_penalties` / `repetition_penalties: torch.Tensor`：惩罚三元组为逐请求张量；`prompt_token_ids` 一并传入惩罚计算，说明**重复惩罚的统计范围覆盖"提示 + 已生成"全部 token**。
- `output_token_ids: list[list[int]]`：已生成 token 的 CPU 侧 Python 嵌套列表（每步由 engine 侧维护）。
- `allowed_token_ids_mask`（max_batch × vocab 的 bool 掩码）、`bad_words_token_ids: dict[req_index → list[list[int]]]`。
- `logitsprocs: LogitsProcessors`：按 argmax 不变性分两组（实现在 `vllm/v1/sample/logits_processor/` 的 `builtin.py` / `interface.py` / `state.py`）。

## 3. 惩罚机制调用链与逐 token 算式

### 3.1 三层调用链

**第一层**：`Sampler.apply_penalties`（`vllm/v1/sample/sampler.py:421-438`）——`no_penalties` 为真直接返回原 logits；否则把 `prompt_token_ids`、三个惩罚张量、`output_token_ids` 传给 `apply_all_penalties`。

**第二层**：`apply_all_penalties`（`vllm/v1/sample/ops/penalties.py:10-38`）：

- `_convert_to_tensors`（penalties.py:41-56）把 Python 嵌套列表转成 int64 张量，用 `make_tensor_with_pad(pad=vocab_size, pin_memory=PIN_MEMORY)` 打 pad（词表内不存在 id=vocab_size 的 token，天然安全），再 `non_blocking` 上传 GPU。
- penalties.py:29：`output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)`——async 调度下无惩罚的行会含 -1 占位 token id，替换成合法的 pad 值以免后续 scatter 越界（注释 24-28 行；代码自注 "currently quite inefficient and will be reworked"）。

**第三层（算式所在）**：`from vllm.model_executor.layers.utils import apply_penalties`（penalties.py:6），定义于 `vllm/model_executor/layers/utils.py:51`：

```python
def apply_penalties(logits, prompt_tokens_tensor, output_tokens_tensor,
                    presence_penalties, frequency_penalties, repetition_penalties):
    _, prompt_mask   = get_token_bin_counts_and_mask(prompt_tokens_tensor, ...)
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(output_tokens_tensor, ...)
    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)
    # 遵循 OpenAI API 定义
    logits -= frequency_penalties.unsqueeze(1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(1) * output_mask
    return logits
```

### 3.2 分桶计数统计

`get_token_bin_counts_and_mask`（utils.py:34-48）：对每条序列建 `vocab_size+1` 长的 long 计数桶（`+1` 留给 pad 值 vocab_size），用 `bin_counts.scatter_add_(1, tokens, ones)` 统计各 token 出现次数，再 `bin_counts[:, :vocab_size]` 截掉 pad 列得到 `bin_counts`，并导出 bool 掩码 `mask = bin_counts > 0`：

- `output_bin_counts`：**出现频次**（frequency penalty 用）。
- `output_mask` / `prompt_mask`：**是否出现过**（presence / repetition penalty 用）。

### 3.3 repetition penalty —— 乘性修正（自定义算子）

`apply_repetition_penalties`（`vllm/_custom_ops.py:336`）在 CUDA 且连续张量时走 `_C.apply_repetition_penalties_` 内核（`csrc/libtorch_stable/sampler.cu:16` `apply_repetition_penalties_kernel`），否则走 PyTorch 优雅降级 `apply_repetition_penalties_torch`（`_custom_ops.py:309`），两者数学一致：

```python
penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)
scaling   = torch.where(logits > 0, 1.0 / penalties, penalties)
logits *= scaling
```

即 **logit 为正则除以 penalty，为负则乘以 penalty**；token 只要出现在 prompt 或 output 中就施加（掩码并集），未出现则乘 1.0 无操作。这是抑制重复的经典乘性修正。

### 3.4 presence / frequency penalty —— 加性修正（OpenAI API 语义）

utils.py:87-88，注释明确"We follow the definition in OpenAI API"（引用 `https://platform.openai.com/docs/api-reference/parameter-details`）：

```python
logits -= frequency_penalties * output_bin_counts   # 按出现次数线性扣除
logits -= presence_penalties     * output_mask       # 出现即一次性扣除（次数无关）
```

- **frequency penalty** 与 token 出现**次数**成正比，打蔁高频重复。
- **presence penalty** 只看是否出现（bool 掩码），对已出现 token 统一扣除固定量。
- 两者都是**加性**（减 logits），与 repetition 的乘性不同。

## 4. top-k / top-p 截断与重采样算法

文件：`vllm/v1/sample/ops/topk_topp_sampler.py`。

### 4.1 truncation：apply_top_k_top_p_pytorch（367-408 行）

1. 对 vocab 维升序排序：`logits.sort(dim=-1, descending=False)`。
2. **top-k**：`size(1) - k` 即升序视图下第 k 大元素的索引，gather 出该值作阈值，`logits_sort < 阈值` 的位置填 -inf（并列值保留，实际保留 ≥ k 个）。
3. **top-p**：对排序后 logits 做 softmax → 原地 `cumsum`（`torch.cumsum(..., out=probs_sort)`，复用内存）；`probs_sum <= 1 - p` 的位置填 -inf，且 `top_p_mask[:, -1] = False` 保证至少保留 1 个。数学上等价于标准 nucleus 语义：保留概率质量 ≥ p 的最小高概率后缀集合。
4. `logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)` 原地恢复原始词表顺序。

路由 `apply_top_k_top_p`（349-364 行）：CPU 且有 Triton → triton kernel；GPU 上 batch ≥ 8 且有 Triton → `apply_top_k_top_p_triton`；小 batch 走上述 PyTorch sort 实现。

**仅 top-k 的免排序路径 `apply_top_k_only`（411-431 行）**：不排全词表，用 `logits.topk(max_top_k)` + `gather(k-1)` 取每行第 k 大值做阈值；`k == vocab_size` 的"无 top-k"行掩码填 -inf 使其不被改动。注释明确代价：`k.max()` 引发一次 GPU→CPU 同步，损害 async 调度性能。

### 4.2 随机采样：指数竞赛 / Gumbel-argmax（450-472 行）

`random_sample` + `sample_with_exponential_noise`（441-447 行）：**不用 `torch.multinomial`**（会触发 CPU-GPU 同步），而是取 `q ~ Exp(1)`，返回 `argmax(probs / q)`，统计上等价于按 probs 采样。带请求级种子的行逐行 `q[i].exponential_(generator=...)` 覆写（`TODO(woosuk)`：逐行处理较慢，待优化）；`use_fp64_gumbel` 时用 float64 噪声走倒数乘法路径。

`flashinfer_sample`（475 行起）：基于**拒绝采样**避免排序，作者注明"统计等价但不保证与 native 输出逐位一致"。

### 4.3 多后端分发（TopKTopPSampler.init，85-129 行）

CUDA→FlashInfer（受 `VLLM_USE_FLASHINFER_SAMPLER` 与算力门控，`flashinfer_sampler_supported()`，28-74 行）；CPU 的 POWERPC/RISCV 回退 native（PowerPC 上 argmax 在 `torch.compile` 下有错，PR #26987）；XPU、ROCm/aiter 各有专属 forward。`forward_cuda`（155-182 行）在三种情况回退 native：无 k 且无 p、存在逐请求 generator（FlashInfer 0.2.3+ 不支持）、或启用 fp64 gumbel；调用前强制 `contiguous()`（flex_attn/triton_attn fp32 切片可能产生非连续张量）。

## 5. OpenAI 兼容层参数映射（to_sampling_params）

文件：`vllm/entrypoints/openai/chat_completion/protocol.py`，`ChatCompletionRequest.to_sampling_params`（661-721+ 行）。请求字段到 `SamplingParams` 的转换：

- 每个参数采用"**请求字段 → 服务器默认 → 协议内建默认**"三级回退：字段为 `None` 时取 `default_sampling_params`（server 配置），再取 `self._DEFAULT_SAMPLING_PARAMS`（protocol 内建，如 `"repetition_penalty": 1.0`，629 行）。
- `repetition_penalty`（protocol.py:667-671）是 **vLLM 扩展参数**，非 OpenAI 原生——OpenAI 规范只有 presence/frequency，此处补充支持。
- `stop_token_ids` 合并 server 默认（如 gpt-oss 的 `</call>`）与请求指定，用 `dict.fromkeys` 去重保序（689-699 行）。
- `prompt_logprobs`：请求未显式指定且 `echo=True` 时继承 `top_logprobs`（701-703 行）。
- 最终统一走 `SamplingParams.from_optional(...)`（712-721+ 行），把可选字段归一成具体值（`presence_penalty=0.0`/`frequency_penalty=0.0`/`repetition_penalty=1.0`/`temperature=1.0`/`top_p=1.0`，sampling_params.py:447-456）。

## 6. SamplingParams 字段定义与校验

文件：`vllm/sampling_params.py`（1249 行），类 `SamplingParams`（215 行，dataclass）。

### 6.1 关键字段（240-267 行，含默认值）

- `presence_penalty: float = 0.0`、`frequency_penalty: float = 0.0`、`repetition_penalty: float = 1.0`
- `temperature: float = 1.0`、`top_p: float = 1.0`、`top_k: int = 0`、`min_p: float = 0.0`
- `seed: int | None = None`、`logit_bias: dict[int, float] | None = None`

### 6.2 from_optional 的 logit_bias 处理（418-443 行）

- 支持 `dict[str, float]` 形式（token 为字符串时用 `get_tokenizer_group().get_token_id` 转换）。
- 所有 bias 值经 `min(100.0, max(-100.0, bias))` **clamp 到 [-100, 100]**（435 行）。
- 无法转换的 key 抛 `ValueError` 并提示用 `get_tokenizer_group` 转换。

### 6.3 __post_init__ 校验（483-545+ 行）

- **temperature**：`0 < temperature < _MAX_TEMP` 时告警并 max 到 `_MAX_TEMP`（防 NaN/Inf，484-492 行）；`temperature < _SAMPLING_EPS` 视为贪心，强制 `top_p=1.0, top_k=0, min_p=0`（529-533 行）。
- **seed**：`-1` 归一为 `None`（494-495 行）。
- **stop / stop_token_ids / bad_words**：`None`→`[]`，字符串→单元素列表；`stop_token_ids`/`bad_words` 用 `dict.fromkeys` 去重保序（501-514 行）。
- **logprobs / prompt_logprobs**：布尔 True 归一为 1（516-520 行）。
- **stop 缓冲**：`include_stop_str_in_output=False` 时设 `output_text_buffer_length = max(len(s)) - 1`（524-525 行），供 stop string 延迟求值。

### 6.4 _verify_args（545-560+ 行）

- `presence_penalty` / `frequency_penalty` 必须在 **[-2, 2]**（547-553 行）。
- `repetition_penalty` 必须有限（`math.isfinite`）且 `> 0`（555-560 行）。

## 小结

- **采样管线顺序**（`Sampler`）：原始 logits 算 logprobs → float32 → allowed token 白名单 → bad words → 非 argmax 不变 processor（min tokens/logit bias）→ 惩罚（repetition→frequency→presence）→ 温度缩放 → argmax 不变 processor（min_p）→ top-k/top-p → 混合批逐行贪心/随机。
- **惩罚分两类数学**（`layers/utils.py:51`）：repetition 为**乘性**（logit 正除负乘，`_custom_ops.py:309`）；presence/frequency 为**加性**（`logits -= penalty * counts` / `logits -= penalty * mask`，遵循 OpenAI API 语义）。计数用 `bincount` + scatter_add，统计范围覆盖 prompt+output。
- **采样避免 CPU-GPU 同步**（`topk_topp_sampler.py`）：随机采样用指数竞赛（`argmax(probs/q), q~Exp(1)`）而非 `multinomial`；仅 top-k 时才有一次 `k.max()` 同步代价；FlashInfer 走拒绝采样。
- **logits 处理器按 argmax 不变性分组**（`logits_processor/`）：影响贪心的（min tokens/logit bias）在温度前、不影响的（min_p）在温度后。
- **参数映射三级回退** + `SamplingParams.from_optional` 归一 + `__post_init__/_verify_args` 校验（温度/惩罚范围/clamp logit_bias），构成从 OpenAI API 到 GPU kernel 的完整闭环。
