---
title: vLLM 采样与惩罚机制:从 OpenAI 参数到 GPU kernel 的闭环
type: concept
tags: [vllm, sampling, penalties, repetition-penalty, top-k, top-p, openai-api, sampler]
source:
  - "code: vLLM V1 源码 (commit b74ac0bc6, v0.28.1rc0-40) | vllm/v1/sample/ + vllm/sampling_params.py + entrypoints/openai/chat_completion/protocol.py 一手实读 (经 notes 提炼)"
  - "web: vLLM docs — SamplingParams | https://docs.vllm.ai/en/latest/api/vllm/sampling_params/"
  - "web: OpenAI API Reference — Create chat completion (penalty 参数语义) | https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/"
date: 2026-09
---

# vLLM 采样与惩罚机制:从 OpenAI 参数到 GPU kernel 的闭环

> 本文由 `notes/vllm-sampling-penalties-notes.md` 深度提炼而成(**notes 已删,以源码 commit b74ac0bc6 为准**)。源码基准 commit `b74ac0bc6`(v0.28.1rc0-40,2026-08-28)。path:line 均实际验证。

## TL;DR

V1 的采样是一条**单趟 GPU 管线**(`vllm/v1/sample/sampler.py` 的 `Sampler`):原始 logits 算 logprobs → float32 → 白名单/bad words → 非 argmax 不变 processor(min tokens/logit bias)→ **三类惩罚(repetition→frequency→presence)** → 温度 → argmax 不变 processor(min_p)→ top-k/top-p → 混合批逐行贪心/随机。三类惩罚是**两类数学**:repetition 是**乘性**(logit 正除负乘),presence/frequency 是**加性**(遵循 OpenAI API 语义)。整条管线的两个工程主题:**避免 CPU-GPU 同步**(指数竞赛替代 multinomial)与**按 argmax 不变性给 processor 排序**(min_p 在温度后,不影响贪心)。入口侧,OpenAI 请求参数经 `to_sampling_params` 三级回退 + `SamplingParams` 校验归一,才进入这条管线。

**看懂它,就能回答三个高频问题**:① "调 repetition/presence/frequency penalty 到底改了什么?"——乘性 vs 加性、按次数 vs 按出现,§2;② "为什么采样会卡 GPU?"——大概率是 `k.max()` 或逐行 generator 这类同步点,§3;③ "请求参数没写时默认值哪来的?"——三级回退链,§4。

> **心法**:采样层的设计约束是"**每步 decode 都要走一遍,任何一次 CPU-GPU 同步都会把异步调度的重叠打回原形**"。penalty 的分桶计数、指数竞赛、免排序 top-k,全是这条约束的推论。

---

## 1. Sampler 管线:顺序本身就是机制

`sampler.py:21` 的 `Sampler(nn.Module)`,标准处理顺序(docstring 与 forward 实现一致):

```
1. logprobs 预留(有请求时,在【原始 logits】上算)      sampler.py:81-84
2. 转 float32
3. allowed_token_ids 白名单(2D bool 掩码,-inf 填充)
4. bad words 排除(ops/bad_words.py)
5. 非 argmax 不变 processor:min tokens / logit bias     sampler.py:401-402
6. 三类惩罚:repetition → frequency → presence          apply_penalties, sampler.py:421-438
7. 温度缩放(logits.div_(temp);<1e-5 的行防除零置 1.0)
8. argmax 不变 processor(默认 min_p)                   sampler.py:283-284
9. top-k/top-p(TopKTopPSampler)
10. 混合批逐行选择:where(temp < 1e-5, greedy, random)  sampler.py:297-302
```

**两个"为什么"藏在顺序里**:

- **logprobs 用原始 logits 算**(sampler.py:81-84 `NOTE(woosuk)`):V1 在惩罚与温度缩放**之前**的 logits 上取 top-k logprobs,V0 用的是采样时的处理后 logits。语义差异:V1 返回的是"模型本来怎么想",不是"惩罚后还剩多少"。
- **processor 按 argmax 不变性分组**(`vllm/v1/sample/logits_processor/` 的 interface/builtin/state,元数据挂 `SamplingMetadata.logitsprocs`):影响贪心结果的(min tokens、logit bias)必须在温度**前**应用;不影响的(min_p)放温度**后**——这就是 min_p 不改变贪心输出的原因。分组框架是本版本新增(相对旧版的 `logits_processors` 列表逐个应用)。

**与推测解码的交互**:`predict_bonus_token` 为真且有惩罚/bad_words 时,`_combine_outputs_with_spec_tokens`(sampler.py:360-371)先把草稿 token 并入 `output_token_ids` 再施惩罚(sampler.py:384-390)——惩罚的统计范围必须看到草稿 token,否则草稿重复不受罚。

---

## 2. 惩罚三算子:两类数学,一层分桶计数

### 2.1 调用链(三层)

```
Sampler.apply_penalties        sampler.py:421-438   no_penalties → 直接返回
  └─ apply_all_penalties       v1/sample/ops/penalties.py:10-38
       ├─ _convert_to_tensors  penalties.py:41-56    嵌套列表 → pad 后 int64 张量 → non_blocking 上传
       └─ apply_penalties      model_executor/layers/utils.py:51   ← 算式所在
```

pad 用 `make_tensor_with_pad(pad=vocab_size)`:词表内不存在 id=vocab_size,天然安全。penalties.py:29 把 async 调度下无惩罚行的 `-1` 占位 `masked_fill_` 成 vocab_size,防 scatter 越界(代码自注 "currently quite inefficient and will be reworked")。

### 2.2 分桶计数:一次 scatter 统计出"次数"和"出现过"

`get_token_bin_counts_and_mask`(utils.py:34-48)对每条序列建 `vocab_size+1` 的计数桶,`scatter_add_` 累加,截掉 pad 列后同时导出两个视图:

- `output_bin_counts`:**出现次数**(frequency 用);
- `output_mask` / `prompt_mask`:**是否出现**(presence / repetition 用)。

**统计范围覆盖"prompt + 已生成"全部 token**(`prompt_token_ids` 一并传入 SamplingMetadata)——repetition penalty 连你提示词里的重复都算,这是与直觉最容易偏差的点。

### 2.3 乘性 vs 加性(核心算式)

```python
# utils.py:51 apply_penalties(节选)
apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)
# 遵循 OpenAI API 定义
logits -= frequency_penalties.unsqueeze(1) * output_bin_counts   # 按次数线性扣除
logits -= presence_penalties.unsqueeze(1) * output_mask          # 出现即一次性扣除
```

**repetition(乘性,自定义算子)**:`_C.apply_repetition_penalties_` CUDA 内核(`csrc/libtorch_stable/sampler.cu:16`),无 CUDA 时 PyTorch 降级实现(`_custom_ops.py:309`)数学一致:

```python
penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)
scaling   = torch.where(logits > 0, 1.0 / penalties, penalties)
logits *= scaling          # 正 logit 除以 penalty,负 logit 乘 penalty
```

token 只要出现在 prompt **或** output(掩码并集)就施加;未出现乘 1.0 无操作。正logit压、负logit更负——双向抑制,所以参数语义是"1.0 = 不惩罚,>1.0 抑制重复"。

**presence/frequency(加性,OpenAI 语义)**:utils.py:87-88 注释明确 "We follow the definition in OpenAI API"。frequency 与出现**次数**成正比(打高频重复);presence 只看 bool(出现即扣固定量,不管几次)。两者参数域 [-2, 2],负值 = 鼓励重复。

| 惩罚 | 数学 | 依据 | 抑制对象 | OpenAI 原生? |
|---|---|---|---|---|
| repetition_penalty(默认 1.0) | **乘性**(正÷负×) | prompt∪output 出现即施加 | 一切已见 token(含提示) | ❌ vLLM 扩展 |
| frequency_penalty(默认 0.0) | 加性,按**次数** | output_bin_counts | 高频重复 | ✅ |
| presence_penalty(默认 0.0) | 加性,按**出现** | output_mask | 已出现 token(次数无关) | ✅ |

---

## 3. top-k / top-p 与采样:把同步从算法里挤出去

文件 `vllm/v1/sample/ops/topk_topp_sampler.py`。

### 3.1 截断(apply_top_k_top_p_pytorch,367-408 行)

升序排序 → **top-k**:gather 升序视图第 k 大值作阈值,`< 阈值` 填 -inf(并列保留,实际 ≥ k 个);**top-p**:softmax 后原地 `cumsum`(复用内存),`probs_sum <= 1-p` 填 -inf 且末位必保 1 个——数学等价标准 nucleus 语义(概率质量 ≥ p 的最小高概率后缀);最后 `scatter_` 恢复词表顺序。

**免排序捷径(仅 top-k 时,apply_top_k_only,411-431 行)**:不排全词表,`logits.topk(max_top_k)` + gather(k-1) 取每行第 k 大值做阈值。**代价**:取 `k.max()` 引发一次 GPU→CPU 同步,损害 async 调度性能——纯 top-k 场景的已知取舍。

路由:CPU+Triton → triton kernel;GPU 且 batch≥8+Triton → `apply_top_k_top_p_triton`;否则 PyTorch sort。

### 3.2 随机采样:指数竞赛,不用 multinomial

`sample_with_exponential_noise`(441-472 行):取 `q ~ Exp(1)`,返回 `argmax(probs / q)`——统计上等价按 probs 采样,却**避开 `torch.multinomial` 的 CPU-GPU 同步**。这是"Gumbel-argmax"一族的做法:加指数噪声取 argmax,天然适配 batch 化。

带请求级种子(`SamplingParams.seed`)的行逐行 `q[i].exponential_(generator=...)` 覆写(TODO(woosuk):逐行慢,待优化);`use_fp64_gumbel` 走 float64 噪声路径。`flashinfer_sample`(:475)基于**拒绝采样**免排序,作者注明"统计等价但不保证与 native 输出逐位一致"——换后端别期待逐位可复现。

### 3.3 后端分发(TopKTopPSampler.init,85-129 行)

CUDA → FlashInfer(受 `VLLM_USE_FLASHINFER_SAMPLER` 与算力门控 `flashinfer_sampler_supported()`);CPU 的 POWERPC/RISCV 回退 native(PowerPC argmax 在 `torch.compile` 下有错,PR #26987);XPU、ROCm/aiter 各有专属 forward。`forward_cuda`(155-182)三种情况回退 native:无 k 无 p / 存在逐请求 generator(FlashInfer 0.2.3+ 不支持)/ fp64 gumbel;调用前强制 `contiguous()`(flex_attn/triton_attn 的 fp32 切片可能非连续)。

---

## 4. 参数入口:to_sampling_params 三级回退 + SamplingParams 校验

请求参数到引擎参数的完整归一链(OpenAI 兼容层的字段拆解见 [api-server-openai-compat.md](api-server-openai-compat.md) §6):

**三级回退**(`chat_completion/protocol.py:661` `ChatCompletionRequest.to_sampling_params`):每个参数为 `None` 时依次取 ① 请求字段 → ② 服务器默认 `default_sampling_params`(模型 `generation_config.json`)→ ③ 协议内建 `_DEFAULT_SAMPLING_PARAMS`(如 `"repetition_penalty": 1.0`,protocol.py:629)。另有:`stop_token_ids` 与服务端默认(如 gpt-oss 的 `</call>`)`dict.fromkeys` 去重合并;`echo=True` 时继承 `prompt_logprobs=top_logprobs`;最终统一 `SamplingParams.from_optional(...)`(sampling_params.py:447-456)把可选项归一成具体值。

**校验**(`vllm/sampling_params.py:215`,`__post_init__` 483-545 + `_verify_args` 545-560):

- `temperature` 越界告警并 clamp 到 `_MAX_TEMP`;`< _SAMPLING_EPS(1e-5)` 视为贪心,强制 `top_p=1.0 / top_k=0 / min_p=0`;
- `seed=-1` 归一 `None`;stop/bad_words 去重保序;logprobs 布尔归一为 1;
- `logit_bias` 支持 `dict[str, float]`(字符串 token 经 tokenizer 转换),值 clamp 到 **[-100, 100]**(sampling_params.py:435);
- presence/frequency 必须 **[-2, 2]**;repetition 必须有限且 **> 0**。

> 实践含义:同一批请求可混不同参数——`SamplingMetadata`(metadata.py:14-55)把 temperature/top_p/top_k/三惩罚全张量化成**逐请求 batch 张量**,`all_greedy`/`all_random` 整批标记走捷径,`generators: dict[行号→torch.Generator]` 承载请求级种子。

---

## 5. 实践要点(checklist)

1. **调惩罚先分清两类数学**:想"少重复用过的词"用 repetition(乘性,1.0 起);想"按 OpenAI 语义控频率/存在感"用 frequency/presence(加性,[-2,2])。repetition 连 prompt 里的重复都算,长系统提示下比想象中激进。
2. **temperature=0 的真实行为**:走 `<1e-5 视为贪心` 路径,整行强制贪心并清空 top_p/top_k/min_p——不要再叠 top_k=1 之类的冗余参数。
3. **要可复现就设 seed**:请求级种子经逐行 generator 覆写噪声;但换采样后端(FlashInfer 拒绝采样)不保证逐位一致,复现实验固定 `VLLM_USE_FLASHINFER_SAMPLER`。
4. **性能排查看同步点**:纯 top-k 有一条 `k.max()` GPU→CPU 同步;带 seed 的行逐行采样较慢(源码 TODO 自认)——吞吐敏感的场景别乱开。
5. **min_p 与贪心共存**:min_p 在温度后应用、argmax 不变——贪心请求不受它影响,这是 logits_processor 分组框架的保证,不是巧合。
6. **参数没写≠引擎默认**:OpenAI 请求字段缺省时先吃模型 `generation_config.json` 再吃协议内建默认——排查"为什么行为和文档不一样"先看模型自带的 generation_config。
7. **V0/V1 资料差异**:旧资料里惩罚在 `LogitsProcessor` 列表逐个应用;V1 是 `vllm/v1/sample/` 的 Sampler + argmax 不变性分组框架,读代码认 `vllm/v1/sample/`。

---

## 6. 溯源

- **V1 源码一手实读**: `vllm/v1/sample/sampler.py`、`vllm/v1/sample/ops/`(penalties.py / topk_topp_sampler.py)、`vllm/v1/sample/logits_processor/`、`vllm/v1/sample/metadata.py`、`vllm/model_executor/layers/utils.py:51`、`vllm/sampling_params.py`、`entrypoints/openai/chat_completion/protocol.py:661`,基准 commit `b74ac0bc6`(本文由库内 notes 深度提炼,notes 已删,以源码 commit 为准)
- **vLLM SamplingParams 官方文档**(字段/默认值口径): https://docs.vllm.ai/en/latest/api/vllm/sampling_params/
- **OpenAI API Reference — Create chat completion**(presence/frequency penalty 官方语义): https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/
- 版本差异基线: 源码注释明确 presence/frequency "We follow the definition in OpenAI API"(utils.py:87-88);repetition_penalty 为 vLLM 扩展(OpenAI 规范无此参数)
