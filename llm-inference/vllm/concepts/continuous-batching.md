---
title: 连续批处理与统一 Token 预算调度:vLLM 的吞吐引擎
type: concept
tags: [vllm, continuous-batching, scheduler, decode, prefill, chunked-prefill, throughput]
source:
  - "pdf: PagedAttention 论文 (SOSP 2023) | arXiv:2309.06180 — Figure 3 吞吐对比 (Orca 等 baseline, 2-4x)"
  - "web: vLLM Blog — Easy, Fast, and Cheap LLM Serving with PagedAttention | https://vllm-project.github.io/2023/06/20/vllm.html"
  - "code: vLLM V1 源码(commit b74ac0bc6) 一手实读 (经 notes 提炼,notes 已删,以源码 commit 为准)"
date: 2026-09
---

# 连续批处理(Continuous Batching)与统一 Token 预算调度

## TL;DR

**连续批处理 = 迭代级(iteration-level)调度**:不是"一批一起算完再换下一批",而是**每一步都重新调度整个 batch**——完成的请求立刻移出、等待中的新请求按预算补进来、运行中请求再续 1 个 token。配合 V1 的**统一 token 预算模型**,decode + prefill 在同一 step 混批(且 prefill 被 chunk),尽量(至多)把 GPU 每一轮填到 `max_num_batched_tokens`(默认 2048;低负载时填不满)。

> 吞吐数字别挂错账:相对真正静态批处理(HF/transformers)FasterTransformer 的连续批处理本身就快 **14-24×**;SOSP 论文的 **2-4×** 是 **vLLM(含 PagedAttention + 连续批处理)整机** vs **Orca**(本身已是迭代级调度)的同延迟吞吐——而这 2-4× 的优势主要来自 **PagedAttention 的内存管理**,不是调度粒度本身(对 Orca 而言连续批处理已不是增量)。两个数字、两种归因,别混。

> **心法**:传统批处理让 GPU 在 decode 阶段(每请求每次只产 1 token)饿肚子;连续批处理把"喂满 GPU"变成每一轮的主动动作。调度器从"分派一次"变成"每一次都重排"——吞吐的钥匙在调度频率,不在单次批次大小。

---

## 1. 传统静态批处理(或 waiting-batch)为什么浪费

**静态批处理**:把一组请求积累起来一次性跑完,全部生成结束,再接纳下一组。

```
传统:  ┌──── batch 1 ────┐  ┌──── batch 2 ────┐
       │ prefill → decode │  │ prefill → decode │
       └── GPU 有空转 ────┘  └── 等新请求攒够 ──┘
```

两个致命浪费:
1. **等待浪费**:请求要攒够一批才开跑(延迟 + 吞吐双输);
2. **decode 浪费**:batch 里请求步调不一致,短的先结束,GPU 越跑越空;decode 每请求每步只产 1 token,算术强度极低,GPU 在"算一点、等很多"。

**ORCA 论文(2022)提出 iteration-level scheduling**:每次迭代都把新到达的请求加进正在跑的 batch、把完成的移出——这就是"连续批处理"的雏形。vLLM 是第一个把"连续批处理 + 分页 KV cache"做成**工业可用引擎**的(整机 vs Orca 2-4×,增量主要来自分页内存管理,见 TL;DR 口径)。

---

## 2. V1 的统一 token 预算模型(Scheduler 的灵魂)

V1 调度器**不区分"prefill 阶段"和"decode 阶段"**(这是和网上大量 V0 资料的本质区别)。源码注释(`scheduler.py:502-511`)原话:

> 每个 request 只有 `num_computed_tokens` 和 `num_tokens_with_spec`。每一步调度器尽力让每个 request 的 `num_computed_tokens` 追上 `num_tokens_with_spec`。

一切问题都化约为一个追赶目标:

```
更新:  num_computed_tokens → num_tokens_with_spec
       (已计算 token 数 → 应计算 token 数)
```

这个统一抽象**通吃**:
- 普通 decode:`num_tokens_with_spec = num_computed_tokens + 1`(每步续 1)
- prefill:`num_tokens_with_spec` 一次跳很多(整段 prompt 要计算的 token 数)
- **chunked prefill**:一次只给它一部分 token 预算
- **prefix caching**:命中的前缀直接算已 computed(跳过)
- **speculative decoding**:`num_tokens_with_spec` 含 speculative token 的 lookahead

> 为什么 V1 更强(与 V0 的准确区别):**V0 和 V1 都有 running/waiting 双队列**;区别在调度语义——**V0 里 waiting 队列只做 prefill、running 只做 decode(两条代码路径 + swap/recompute 两种抢占),V1 里两队列仍存在于每步 schedule(),但调度逻辑统一为 `num_computed_tokens` 追赶模型,prefill/decode 不再区分阶段**。很多优化(prefix caching/spec decoding 的 token 落差、chunked prefill)在 V1 的\"每步重排、谁差多少补多少\"下一个模型通吃。

---

## 3. 双队列 + 每步重排(Scheduler 如何工作)

Scheduler 维护两个队列,V1 调度顺序(scheduler.py:548 起,先 RUNNING 后 WAITING):

```
schedule() 每步:
1. 先调度 RUNNING(running 队列,FCFS 高优先)
   → 每个请求续 num_new_tokens = num_tokens_with_spec − num_computed_tokens(通常 1)
   → 受 token_budget(max_num_scheduled_tokens) + input_budget(max_num_batched_tokens) 双重约束
2. 再调度 WAITING(waiting 队列)
   → 空余预算 + 未达 max_num_running_reqs(默认 128)才取队首
   → 先做 prefix cache 查询(命中跳过),再按 chunked prefill 切块塞进剩余预算
```

**关键数据结构**:
- `SchedulerOutput`:每步产物(`scheduled_new_reqs` / `scheduled_running_reqs` / `req_to_new_blocks` / `preempted_reqs`)
- `Request`:携带 `num_computed_tokens`、`num_tokens_with_spec`、`block_ids`、`status`(WAITING/RUNNING/FINISHED...)
- 预算:`max_num_batched_tokens` 默认 2048、`max_num_seqs` 默认 128、`enable_chunked_prefill` 默认 True

---

## 4. Chunked Prefill:让 prefill 不再"独占一步"

**为什么需要**:不需要 chunk 时,一个长 prefill 会吃掉整步预算,decode 请求被挤到后面,延迟飙升、GPU 利用率被一次大矩阵乘主导。

**chunked prefill(默认开)**：prompt 超过当前 step 剩余 `token_budget` 时,只调度一部分 token,其余留到后续 step:

```
step 1: [decode A][decode B][prefill C 前 512 tok]
step 2: [decode A][decode B][prefill C 中 512 tok]   ← prefill 被切成小块缝进 decode 空隙
step 3: [decode A][decode B][prefill C 尾 200 tok]
```

- **收益**：decode 和 prefill 同 batch 混合,GPU 每步都填满;
- **代价**：单个 prefill 的延迟略增(拆开做),这正是默认开启、但 `long_prefill_token_threshold` 可对超长 prefill 限流的原因。

> 取舍心法:chunked prefill 是在 **吞吐(填满 GPU) vs prefill 延迟(一次做完)** 之间做权衡。服务场景默认要吞吐,所以翻成"拆开缝进空隙"。

---

## 5. Continuous Batching 的 step 循环(EngineCore)

V1 的连续批处理由 EngineCore 的 step 循环驱动(`core.py:597` `step()`):

```
EngineCore.step()                        # core.py:597(行号以 597 为准,v1-architecture 篇亦同)
  if not scheduler.has_requests(): return
  scheduler_output = scheduler.schedule()          # 重排整个 batch
  future = executor.execute_model(non_block)       # 异步下发到 worker
  grammar = scheduler.get_grammar_bitmask()        # 与 GPU 计算重叠
  model_output = future.result()
  scheduler.update_from_output(output)             # 更新 num_computed_tokens
```

- **每步全量重排**：不固定 batch 成员,完成/中止移出、新请求按预算进、运行中续 token;
- **异步调度**：用 batch queue 缓存多个已调度 batch,GPU 跑 batch N 时 CPU 已调度 batch N+1(CPU/GPU 重叠);
- **请求生命周期**：`add_request → waiting → (schedule 选中) running(分配 block) → 每步续 token → FINISHED/ABORTED → free`。

---

## 6. 与 PagedAttention 的协同(PagedAttention 是地基)

连续批处理能"每步任意增减请求",**前提是 KV cache 能按 token 数精确分配/释放**——这正是 PagedAttention 的分页块池给的:

- 调度器 `allocate_slots` 成功 → 请求加入;返回 None(块不足)→ 触发**抢占**(默认抢 running 队尾)。**V1 只有 recompute 一种模式**:释放被抢占请求的 KV cache 块(进 LRU 前缀缓存)、`num_computed_tokens=0` 退回 waiting 重排——重排时靠 **prefix cache 命中**减少重算,不是"保留部分进度"(那是 V0 的 counted 概念,别照搬);
- 完成后 `free` 块 → 池里腾出 → 下一请求可用。

> 没有分页,连续批处理每接纳/移除一个请求都要搬移大段连续显存(或预留浪费),调度器根本不敢每步重排。两个招牌特性是**一体两面**。

---

## 7. 实践要点(checklist)

1. **吞吐瓶颈看调度频率,不是 batch 大小**:服务器频繁新请求时,谁能每步重排谁就赢——这解释了 vLLM 相对静态 batching 的倍增。
2. **decode 阶段算术强度极低**:它是 memory-bound;连续批处理的核心价值就是**不让 decode 阶段 GPU 空转**(用混进来的 prefill 填满)。
3. **chunked prefill 默认开但要懂取舍**:要吞吐混批 → 开;要单请求 prefill 低延迟(如交互式) → 考虑关或调 `long_prefill_token_threshold`。
4. **max_num_batched_tokens 是"每步喂多少"的旋钮**:示例默认 2048 但源码注释明确说它\"mainly for convenience when testing\"——**实际部署会被 EngineArgs 按模型/长度自动调整,别拿 2048 当线上真值**,查启动日志里的生效值再调。
5. **注意 V0/V1 资料差异**:网上大量调度文章讲的是旧版(V0 里 waiting 只 prefill、running 只 decode 两条路径);当前主线 `vllm/v1/` 是统一 token 预算模型(以 `vllm/v1/` 源码为准,不是老教程)。
6. **抢占(块不足)是连续批处理的兜底**:`allocate_slots` 失败触发抢占(默认抢 running 队尾),释放块进 LRU、被抢占请求重排靠 prefix cache 减重算——调参时看 **KV cache 余量比看 batch 数字更本质**(可观测指标:`vllm:num_preemptions_total` 看抢占次数、scheduler 指标看每步 batch 填充,别凭感觉)。

---

## 8. 溯源

- **PagedAttention 论文**(iteration-level scheduling、吞吐对比 Figure 3): Kwon et al., SOSP 2023, arXiv:2309.06180
- **ORCA**(首次提出 iteration-level batching, vLLM 的吞吐 baseline 参照): arXiv:2207.03718
- **vLLM 博客**(吞吐测量、与 baseline 对比): https://vllm-project.github.io/2023/06/20/vllm.html
- **V1 源码一手实读**: `vllm/v1/core/sched/scheduler.py`(调度器/统一预算/chunked prefill)与 `vllm/v1/engine/core.py`(step 循环),基准 commit `b74ac0bc6`(原 notes `vllm-core-architecture-notes.md` §1/§3 **已删,以源码 commit b74ac0bc6 为准**)
