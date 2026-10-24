---
title: 投机解码:为什么一次验证多个候选 token 能加速
type: concept
tags: [vllm, speculative-decoding, draft-model, eagle, mtp, medusa, ngram, acceptance-rate, latency, kv-cache, dspark, dflash, adaptive-verification]
source:
  - "pdf: Fast Inference from Transformers via Speculative Decoding (Leviathan et al., ICML 2023) | arXiv:2211.17192 ↑primary"
  - "pdf: Accelerating LLM Decoding with Speculative Sampling (Chen et al., DeepMind, 2023) | arXiv:2302.01318"
  - "pdf: EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | arXiv:2401.15077"
  - "pdf: EAGLE-2: Faster Inference with Dynamic Draft Trees | arXiv:2406.16858"
  - "pdf: EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test | arXiv:2503.01840"
  - "pdf: Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads | arXiv:2401.10774"
  - "pdf: Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding | arXiv:2401.07851"
  - "code: vllm/model_executor/models/registry.py `_SPECULATIVE_DECODING_MODELS`(main, 行 596) | https://github.com/vllm-project/vllm"
  - "web: vLLM 官方文档 speculative_decoding/ 子页 | https://docs.vllm.ai/en/latest/features/spec_decode.html | 阅读日期 2026-10-09"
date: 2026-10
---

# 投机解码:为什么一次验证多个候选 token 能加速

## TL;DR

**投机解码(speculative decoding)的核心洞察是「难任务里藏着容易的子任务」**:用一个更小/更快的「草稿模型」(draft model)先**串行**猜出 γ 个候选 token,再由大模型(目标模型)对这批候选**一次前向并行验证**,接受多少用多少、拒绝处回退重新采样。因为大模型 decode 是**访存受限**(memory-bound)、单 token 前向时算力大量空闲,「一次并行验证 γ 个」几乎不额外花时间,却一次产出多个 token → **不改变输出分布、不重训模型**,端到端 2~3× 加速(Leviathan 2022 T5-XXL 实测 2~3×)。

**关键公式(Leviathan 2022 原式)**:设草稿接受率 α、草稿长度 γ,则**每步期望产出 token 数 = (1−α^(γ+1))/(1−α)**(含那个必有的 bonus token)。墙钟加速还要除以 draft + verify 的算力/带宽开销——接受率才是第一驱动,草稿越长边际收益递减,而高并发下草稿成本会吃掉收益。

---

## 1. 机制:为什么「猜」能加速

### 1.1 问题:自回归 decode 是串行 + 访存受限

自回归生成每出一个 token 都要跑一遍完整前向,而前向的访存量(读权重 + 读 KV cache)远大于单 token 的计算量 → GPU 算力大量闲置,**decode 阶段卡在「搬数据」不是「算」**(这也是 vLLM 连续批处理/PagedAttention 的同一背景,见 [continuous-batching.md](continuous-batching.md)、[paged-attention.md](paged-attention.md))。

### 1.2 方案:draft → 并行 verify

下图是这套机制的主流程:草稿串行猜 γ 个 → 目标模型一次前向并行验证 → 接受 k 个(外加一个 bonus token),被拒处回退重采样:

<div class="archify-figure">
<iframe src="../../../diagrams/vm-spec-draft-verify.workflow.html"
        style="width:100%;height:660px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="投机解码:draft 串行猜 + 目标模型并行验证"></iframe>
</div>

```
标准自回归:   每步 1 token,串行,访存受限,算力闲置
投机解码:     草稿模型串行猜 γ 个 → 目标模型一次前向并行验证 γ 个
              └─ 验证是 memory-bound,γ 个 token 的前向 ≈ 1 个 token 的前向
              └─ 接受 k 个(0≤k≤γ) → 一步产出 k+1 个 token(含 bonus)
```

**为什么不改输出分布**:验证用**拒绝采样(rejection sampling)**。目标模型对这 γ 个候选并行算各自的条件概率,与草稿概率比较,按规范采样决定接受/拒绝——接受的一定是目标模型自己会产出的 token(Leviathan 2022 定理:接受率即无偏;Chen et al. 2023 的 `speculative sampling` 是同一算法的并行化形式)。所以**投机解码不近似目标模型,输出与标准采样严格同分布**。

### 1.3 期望 token 数与加速上限

设每位置草稿被接受的概率为 α(i.i.d. 简化),草稿长 γ:

```
E[每步 token 数] = (1 − α^(γ+1)) / (1 − α)
```

这是几何级数的部分和 Σ_{i=0}^γ α^i,含那个「至少会产出」的 bonus token。数值表(见 §5)里 γ=3/α=0.8 时 ≈ 2.95。

**墙钟加速 ≠ 期望 token 数**,还要扣草稿成本:

下图把时间轴摊开对比(示意量级,非实测):基线每 token 20ms 串行;投机下 4 个草稿各 2ms 叠成 8ms,验证 4 个候选仍只占 22ms——这就是「多验几个几乎不花时间」的来源:

<div class="archify-figure">
<iframe src="../../../diagrams/vm-spec-timeline.waterfall.html"
        style="width:100%;height:600px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="投机解码的耗时结构对比"></iframe>
</div>

```
加速比 = E[每步 token] × T_single / (T_draft(γ) + T_verify(γ))
```

- `T_draft(γ)`:草稿模型串行出 γ 个 token 的时间(草稿越小越快);
- `T_verify(γ)`:目标模型并行验证 γ 个候选的时间(≈ 一次前向,memory-bound 下 ≈ `T_single`)。

所以**接受率 α 是第一驱动**,草稿质量差 → α 低 → 大量候选被拒、白算;草稿长度 γ 是第二旋钮,但边际收益随 γ 递减(α<1 时 α^γ 指数衰减),且 γ 越大验证的峰值显存/算力越高。**目标模型的 KV cache 复用**是实现的隐性前提:draft 与 verify 共享同一 KV cache,验证时只需算新增候选位置的增量 KV。

---

## 2. 主流方案:草稿从哪来(五条路线)

下图是 vLLM 官方速选表的决策路径:先问目标模型有没有原生草稿头,再逐层退到 EAGLE → 独立小模型 → n-gram;右侧标出已被禁用的 MLP Speculator:

<div class="archify-figure">
<iframe src="../../../diagrams/vm-spec-five-routes.workflow.html"
        style="width:100%;height:680px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="五条草稿路线的选择路径"></iframe>
</div>

| 路线 | 代表 | 草稿来源 | 特点 / 代价 |
|---|---|---|---|
| **独立草稿模型** | Leviathan 2022 / Chen 2023 | 另一个更小的同 vocab 模型 | 最通用、零训练;要额外显存放草稿;vocab 需对齐(或走 TLI,见 §4) |
| **自投机 / 无外部模型** | Medusa、n-gram / prompt-lookup、suffix | 目标模型自身 | 免外部模型;Medusa 加多个「提前解码头」并行出 γ 个;n-gram 直接查 prompt 里出现过的 n-gram,免训练但增益有限 |
| **EAGLE 族** | EAGLE → EAGLE-2 → EAGLE-3 / PEagle | 从目标模型**最后一层特征**外推下一 token | 特征级 draft 比 token 级快得多,α 高;单层/轻量头,训练成本低;EAGLE-2 用动态草稿树,EAGLE-3 训时即测 |
| **MTP(多头预测)** | DeepSeek-V3/V4、Gemma 4、Qwen3.5 等 | 目标模型**原生多 token 预测头** | 目标模型自带草稿头,无需外部模型、共享隐藏层;只在模型原生支持时可用 |
| **DSpark / DFlash** | DeepSeek V4 系列、Qwen3 系 | 目标是草稿的专用变体(带 confidence head / 单层 flash) | 2025-26 新增的模型原生投机路径;DSpark 带置信度头,支撑 adaptive verification(§3) |

**路线选择的心法**(据 vLLM 官方方法速选表):模型原生支持 MTP → 首选 MTP;要通用高增益 → EAGLE/EAGLE-3;有可用 MLP speculator → MLP;只想轻量零训练 → n-gram/suffix。**高 QPS(吞吐敏感)场景增益普遍缩水**,因为验证阶段的「多 token × 大 batch」把 memory-bound 推成 compute-bound,草稿成了与真实 token 抢算力的负担(见 §3 动态/自适应两节正是解这个)。

---

## 3. vLLM 落地:注册表 + 后端支持状态

### 3.1 `_SPECULATIVE_DECODING_MODELS`(registry.py 当前 main,行 596)

vLLM 把投机草稿模型登记在 `vllm/model_executor/models/registry.py` 的 `_SPECULATIVE_DECODING_MODELS` 字典(架构名 → (模块, 类)),**当前 main 行号 ~596**。真实条目(2026-10 抓取)覆盖:

- **EAGLE 族**:`EagleLlamaForCausalLM` / `EagleCohereForCausalLM` / `EagleMistralForCausalLM` / `EagleLlama4ForCausalLM`、`Eagle3{...}ForCausalLM`(Llama/Qwen3/MiniMax/DeepSeek 等)、`PEagleDraftModel`(= Eagle3 Llama);
- **MTP 族**:`DeepSeekMTPModel`、`DeepseekV32MTPModel`、`DeepSeekV4MTPModel`、`Gemma4MTPModel`、`Qwen3_5MTP`、`MiniMaxM3MTP`、`Glm4MoeMTPModel`、`ErnieMTPModel`、`ExaoneMoeMTP` 等一长串模型原生 MTP;
- **DSpark / DFlash**:`DSparkDraftModel`(deepseek_v4)、`Qwen3DSparkModel`、`DFlashDraftModel` / `DFlash2DraftModel`(qwen3_dflash)、`Gemma4DSparkModel`、`K3DSparkModel` 等;
- **Medusa**:`MedusaModel`;
- **n-gram / suffix**:走独立 proposer(`ngram_proposer.py` / `suffix_decoding.py`),不进这个字典。

### 3.2 关键支持状态(如实标注)

- ✅ **EAGLE / EAGLE-3 / MTP / Medusa / n-gram / suffix / draft model / DSpark / DFlash** 均可用,各有独立文档页。
- ⚠️ **MLP Speculator 在 V1 已被注释禁用**:`_SPECULATIVE_DECODING_MODELS` 末尾(行 667-669)原条目 `MLPSpeculatorPreTrainedModel` 被注释,注释原文 `# Temporarily disabled. # TODO(woosuk): Re-enable this once the MLP Speculator is supported in V1.` —— 别当成 V1 可用。
- **adaptive verification 目前只支持 DSpark**(带 confidence head),且需 full cudagraph、不支持 LoRA/PP(§3.4)。
- **dynamic SD 实测覆盖 Eagle/Eagle-3/DFlash**,其他方法「可能不 out of the box」。

### 3.3 配置(可复现)

```python
# draft model(独立草稿模型)
llm = LLM(model="Qwen/Qwen3-8B",
          speculative_config={"model": "Qwen/Qwen3-0.6B",
                              "num_speculative_tokens": 5,
                              "method": "draft_model"})

# EAGLE(特征级 draft)
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=4,
          speculative_config={"model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
                              "draft_tensor_parallel_size": 1,
                              "num_speculative_tokens": 2,
                              "method": "eagle"})

# n-gram(零训练)
speculative_config={"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}
```

命令行等价:`vllm serve ... --speculative-config '{"model": "...", "num_speculative_tokens": 5, "method": "draft_model"}'`。

### 3.4 2025-26 新增的三个机制(何时不值得用 / 怎么救)

下图对比两个「救场」机制各自的干预点:dynamic SD 按并发档位改 γ,adaptive verification 给每个槽位打生存概率再按预算录取——两者不冲突,但后者实测只覆盖 DSpark:

<div class="archify-figure">
<iframe src="../../../diagrams/vm-spec-dynamic-adaptive.workflow.html"
        style="width:100%;height:660px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="dynamic SD 与 adaptive verification 的干预点"></iframe>
</div>

1. **Dynamic Speculative Decoding(动态投机)**:`num_speculative_tokens_per_batch_size` 按并发区间动态调 γ。为什么需要:验证阶段有效 batch = BS×K,一旦越过临界点 SD 反而拖慢 decode(TPOT)。适用:同部署波动并发、RL rollout 尾部长尾请求。
2. **Adaptive Verification(自适应验证)**:不再静态定 γ,而是每步对每个 (request, position) 草稿槽位按「生存概率 = 各位置置信度连乘」打分,按全局预算录取——高置信请求保留整块、低置信请求提前截断。预算来自启动时 profiling 的成本模型。目前 DSpark-only,且**强制 full cudagraph、不支持 LoRA/流水线并行**。
3. **Per-Request Acceptance Metrics**:`--per-request-spec-decode-metrics summary|detailed` 在响应 `metrics.speculative_decoding` 里回每请求的 `mean_acceptance_length` / `draft_acceptance_rate` / `acceptance_histogram` 等,用于实测调参(见 §5 表怎么来的)。

### 3.5 验收口径:为什么不能从日志抓「Mean acceptance length」

**结论先行**:vLLM 周期打印的 `SpecDecoding metrics: Mean acceptance length: X.XX` **只覆盖最近一个 logging interval,不是整场均值**,不能当验收基准;验收应从 `/metrics` 的累计计数器自己算。

#### 机制:日志行是「打完即清零」的窗口值

`vllm/v1/spec_decode/metrics.py` 里两条聚合路径的行为完全不同:

- `SpecDecodingLogging` 按 **logging interval** 聚合:`observe()` 把每个 scheduler step 的 `num_drafts / num_draft_tokens / num_accepted_tokens / num_accepted_tokens_per_pos` 追加进 list;`log()` 对 list 求和后算 `mean_acceptance_length = 1 + num_accepted_tokens / num_drafts`,**打印完最后一步是 `self.reset()` 把 list 全部清空**(且 `num_drafts` 为空时直接 return,连日志都不打)。所以每行日志只代表「上次打印以来」的窗口——用小样本(如 8 条 smoke prompt)去 grep 这一行,抽到的可能是抖动很大的一段,或恰好没有请求的空窗。
- `SpecDecodingProm` 用 `prometheus_client.Counter` 注册**只增不减**的累计计数器,`counter_specs` 为 `vllm:spec_decode_num_drafts` / `vllm:spec_decode_num_draft_tokens` / `vllm:spec_decode_num_accepted_tokens`(Counter 导出时加 `_total` 后缀,`/metrics` 里实际名字见下),永不清零。

`SpecDecodingProm` 的 docstring 给了两个 PromQL 公式,**注意分母不同**:

```
drafts   = vllm:spec_decode_num_drafts_total           # 发生了多少次投机
accepted = vllm:spec_decode_num_accepted_tokens_total  # 累计接受了多少草稿 token
接受率(α)   = accepted / vllm:spec_decode_num_draft_tokens_total  # 分母是草稿 token 数
平均接受长度 = 1 + accepted / drafts                              # 分母是投机次数,+1 惯例含 bonus token(同 §1.3)
```

可复现取数:压测结束后抓一次 `/metrics`,取上面三个计数器按公式即得**开服至今的整场均值**;要区间口径就用 PromQL `rate(...[$interval])` 相除(同一公式)。ModelOpt 的 smoke 脚本正是这样改的(PR #2930 修的第 4 个问题:「acceptance length 只覆盖最后几秒」)。

#### 三种取数方式对比

| 方式 | 覆盖范围 | 适用场景 |
|---|---|---|
| 日志行 `Mean acceptance length` | 仅最近一个 logging interval(`log()` 末尾 `reset()` 清零) | 人肉观察趋势/抖动;**不做验收** |
| `/metrics` 累计计数器 | 进程启动至今(Counter 只增不减) | 验收门禁、整场均值——自己算 `1 + accepted/drafts` |
| per-request 响应字段 | 单请求 | 逐请求诊断/调参,见 §3.4 第 3 条 `--per-request-spec-decode-metrics`(此处不重复) |

#### 三条可复用的门禁规则(从 PR #2930 的 smoke 脚本硬化提炼)

1. **设了阈值就必须评估它**:设了最低接受长度阈值,但 `/metrics` 里没有任何 draft 计数(说明根本没走投机路径,如 draft 没加载成功)时,**必须判 FAIL**,而不是「没解析到指标就跳过检查」——从未被评估的阈值不得通过。原脚本的 bug 正是「从不投机的 drafter 反而 PASS」。
2. **上一次运行的结果不能被当成本次的**:启动前先 `rm -f` 结果 JSON;结果文件在 `trap ... EXIT` 里写,且**参数校验放在 trap 设好之后**——这样即使配置非法(`NUM_SPEC_TOKENS` / `SERVER_TIMEOUT` 非正整数、`MIN_ACCEPTANCE_LENGTH` 非有限数),留下的也是本次的 `invalid_config` 失败记录,而不是上次的成功记录。
3. **stage 字段是排障主键**:结果 JSON 带 `stage ∈ {invalid_config, starting, draft_missing, server_died, server_timeout, serving, completed}`,下游 gate 一眼区分「不可服务」与「可服务但低于阈值」;进入 `serving` 阶段先预置 `reason="exited while sending test prompts"`,中途崩了也有原因可查。

两条脚本级细节一并记住:除法用 python 做、分母为 0 时输出 `?`(`ratio()` 帮助函数),别用 `bc`——部分镜像没有它,且旧写法在 `bc` 失败时退化成 `echo 0`,把「测量失败」静默变成「0 tok/s」;直接拼进 JSON 的数值参数用 `^[1-9][0-9]*$` 校验(`08` 带前导零不是规范 JSON number,会破坏配置)。

**边界**:本节门禁/脚本实践来自 ModelOpt 的实验 launcher(`tools/launcher/common/specdec/vllm_smoke_test.sh`,PR #2930,2026-10-23),**不是 vLLM 官方 API**;vLLM 侧的机制依据是 `vllm/v1/spec_decode/metrics.py`(核对 HEAD `b74ac0bc61d8`,2026-08-28)。

---

## 4. 实践要点(checklist + 坑)

1. **先判「是否访存受限」再上**:低 QPS、小 batch(单请求/少并发)时 SD 收益最大;高并发把验证推成 compute-bound,草稿抢算力 → 收益转负。用 dynamic SD 或直接关掉。
2. **接受率决定一切,先实测**:上生产前用 acceptance metrics(或 offline 脚本)量 α;α<0.5 时基本不值得。
3. **γ 别贪大**:α<1 时边际收益指数衰减,大 γ 抬高峰值显存与验证算力,还放大拒绝的浪费。从 γ=1~2 起调。
4. **vocab 对齐是硬约束**:默认草稿/目标要共享 vocab;跨家族草稿需 `use_heterogeneous_vocab: true` 走 TLI(Token-Level Intersection),目前仅支持 greedy 草稿采样。
5. **显存账**:draft model 或 MTP 头要额外显存;目标模型权重 + KV cache 之外再算一份草稿。显存紧张时优先 n-gram/suffix(零额外模型)或 EAGLE(单层头)。
6. **MLP Speculator 别用**:V1 已注释禁用(§3.2),文档虽有 mlp.md 页但 registry 里是关的。
7. **采样方式交互**:非 greedy 采样(temperature/top-p)会降低 α、影响拒绝采样开销;`draft_sample_method`(greedy vs probabilistic)直接影响接受率,按模型调。

---

## 5. 数据对比:接受率 × 草稿长度 → 期望产出

`E[每步 token] = (1−α^(γ+1))/(1−α)`(含 bonus token;数值为公式计算,非实测基准):

| α \ γ | γ=3 | γ=5 | γ=7 |
|---|---|---|---|
| 0.5 | 1.875 | 1.969 | 1.992 |
| 0.6 | 2.176 | 2.383 | 2.458 |
| 0.7 | 2.533 | 2.941 | 3.141 |
| 0.8 | 2.952 | 3.689 | 4.161 |

读法:α 从 0.6→0.8,γ=5 时收益从 2.38→3.69;而 γ 从 5→7 在 α=0.8 时只从 3.69→4.16(边际递减)。**「提接受率」比「拉长草稿」划算**——这也是 EAGLE(特征级、α 高)比独立小模型(α 低)有效的原因。注意这是**期望 token 上限**,墙钟加速还要扣 §1.3 的草稿+验证开销;T5-XXL 上 Leviathan 2022 实测 2~3×,中文场景常低于此。

---

## 6. 溯源

- **Leviathan et al.**, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023 — https://arxiv.org/abs/2211.17192 (期望 token 公式、拒绝采样、T5-XXL 2~3×)
- **Chen et al.**, *Accelerating Large Language Model Decoding with Speculative Sampling* — https://arxiv.org/abs/2302.01318 (并行化 speculative sampling;vLLM 文档引此 URL 为定义出处)
- **EAGLE** — https://arxiv.org/abs/2401.15077 ;**EAGLE-2** — https://arxiv.org/abs/2406.16858 ;**EAGLE-3** — https://arxiv.org/abs/2503.01840
- **Medusa** — https://arxiv.org/abs/2401.10774
- **Survey**(Heming Xia et al.) — https://arxiv.org/abs/2401.07851
- **vLLM 官方文档 speculative_decoding/**(README 方法速选表、draft_model / eagle / mtp / n_gram / dynamic_speculative_decoding / adaptive_verification / acceptance_metrics / speculators 各子页):https://docs.vllm.ai/en/latest/features/spec_decode.html
- **源码一手实读**:`vllm/model_executor/models/registry.py`(main, `_SPECULATIVE_DECODING_MODELS` 行 596;MLP Speculator 行 667-669 注释禁用)—— https://github.com/vllm-project/vllm
- **验收口径**:NVIDIA/Model-Optimizer PR #2930 *launcher: harden the vLLM smoke test and newest-checkpoint selection*(2026-10-23)— https://github.com/NVIDIA/Model-Optimizer/pull/2930
- **源码一手实读(§3.5)**:`vllm/v1/spec_decode/metrics.py`(`SpecDecodingLogging.log()` 末尾 `self.reset()`;`SpecDecodingProm` docstring 的接受率与 mean-acceptance-length 两个 PromQL 公式、`counter_specs` 指标名)—— 核对 HEAD `b74ac0bc61d8b8f74848664b1a3b4e878f047861`(2026-08-28),只读克隆 `~/zhanbin/repos/vllm-project/vllm`
- **库内交叉**:[continuous-batching.md](continuous-batching.md)(decode 访存受限背景)、[paged-attention.md](paged-attention.md)(KV cache 分页,验证阶段增量 KV 的底座)、[v1-architecture.md](v1-architecture.md)(batch queue 异步调度与 SD 的交互)、[attention-backend-selection.md](../practices/attention-backend-selection.md)(adaptive verification 要求 backend 容忍 device 决定的 query 长度)。
