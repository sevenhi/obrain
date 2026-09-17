---
title: vLLM V1 引擎架构:调度执行解耦的多进程心智模型
type: concept
tags: [vllm, v1, architecture, scheduler, worker, async, multiprocess, engine]
source:
  - "code: vLLM V1 源码(commit b74ac0bc6) notes 一手实读 | notes/vllm-core-architecture-notes.md §0 §4 §5"
  - "web: vLLM 官方 Architecture Overview | https://docs.vllm.ai/en/latest/design/arch_overview/"
  - "web: vLLM V1 Guide | https://docs.vllm.ai/en/latest/usage/v1_guide/"
  - "doc: notes/vllm-core-architecture-notes.md (库内源码笔记)"
date: 2026-09
---

# vLLM V1 引擎架构:调度执行解耦的多进程心智模型

## TL;DR

**V1 把「调度(决策)」和「执行(算)」解耦**:一个 `EngineCore` 进程专职做调度决策、一个或多个 `Worker`(每 GPU 一个)专职跑模型前向。**注意:Worker 是否独立进程取决于 Executor(见 §5)——单 GPU 走 UniProcExecutor 时 Worker 与 EngineCore 同进程持有,跨进程保证的核心是 前端(AsyncLLM)↔ EngineCore**。调度在 GPU 上一步还跑着时,CPU 已把下一步排好队(**异步调度**,CPU/GPU 重叠)。整套结构从上到下三层:API 层(AsyncLLM)→ 调度核心(EngineCore)→ 执行层(Worker/GPUModelRunner)。理解 V1,只需要记住一句话:**调度搬进独立进程 + 每步全量重排 + 异步下发**——剩下的都是这条主线的展开。

> **心法**:vLLM 的核心不是"一个跑模型的库",而是一对**异步的生产者-消费者**——调度器(CPU,生产计划)和执行器(GPU,消费计划)之间靠"指令队列"(SchedulerOutput)交换,而不是"函数调用"。CPU 永远在给 GPU 准备下一步,GPU 永远在消化 CPU 排好的批。

---

## 1. 三层结构(从上到下)

```
┌────────────────────────────────────────────────────────────┐
│  AsyncLLM (API 层)       vllm/v1/engine/async_llm.py        │
│  └─ asyncio 接口,请求队列 / 流式输出                        │
│         │  (ZMQ socket / 消息传递)                           │
│         ▼                                                    │
│  EngineCore (调度核心)   vllm/v1/engine/core.py             │
│   ├─ Scheduler            vllm/v1/core/sched/scheduler.py   │
│   ├─ KVCacheManager       vllm/v1/core/kv_cache_manager.py  │
│   └─ ModelExecutor(抽象)  vllm/v1/executor/abstract.py      │
│         │  (RPC / 共享内存张量)                              │
│         ▼                                                    │
│  Worker (每 GPU 一个;同进程或独立进程视 Executor)  vllm/v1/worker/gpu_worker.py      │
│   ├─ GPUModelRunner       vllm/v1/worker/gpu_model_runner.py│
│   ├─ AttentionBackend     vllm/v1/attention/backends/*      │
│   └─ 模型权重 nn.Module    vllm/model_executor/             │
└────────────────────────────────────────────────────────────┘
```

**关键设计决策**:调度(CPU)与执行(GPU)解耦。

- **EngineCore 在独立进程内做调度决策**(不在 API 进程,也不在 Worker 进程)——调度不会阻塞在 API 进程的事件循环里;
- **Worker 只负责**按 `SchedulerOutput` 执行模型前向,不关心"为什么排这个批"(单卡 UniProc 时与 EngineCore 同进程,多卡各自独立进程);
- 中间用 **ZMQ socket(API↔EngineCore)** 和 **RPC / 共享内存张量(EngineCore↔Worker)** 通信。

---

## 2. 为什么要把调度搬进独立进程

对照旧版 / 朴素实现:

| 方案 | 问题 |
|---|---|
| 调度写在 API 进程 | 一个长请求的 tokenize/流式逻辑会阻塞调度,大并发下 CPU 忙不过来 |
| 调度写在 GPU 执行进程 | 每 GPU 一个调度器,跨卡状态(state: block table / 预算)要同步,复杂且不一致 |
| **V1:独立 EngineCore 进程** | 调度状态一份、与 GPU 计算天然重叠,Worker 无脑执行 |

核心收益是**异步调度**(async scheduling)。注意区分两个 step 路径——**不启用异步时单步 `step()`**(core.py:597,排的是**本步** batch N,`future.result()` 等的就是这个 batch;行号以 597 为准,注释引用较多):

```
EngineCore.step()                            # core.py:597 (每步:schedule 本步 batch N → 下发 → 取回结果)
  scheduler.schedule()
  executor.execute_model(non_block)
  model_output = future.result()             # 等的就是刚下发的 batch N 自己
```

**启用异步时走 `step_with_batch_queue()`**(core.py:638)——用 batch queue 缓存多个已调度 batch,**GPU 跑 batch N 时 CPU 已在调度 batch N+1**:
```
  scheduler.schedule()                        # 排好 batch N+1
  ...                                         # CPU 继续调度 / 处理 grammar
  model_output = future.result()              # 等 batch N 的 GPU 结果
```

> ⚠ batch queue **并非普适机制**:仅当 `max_concurrent_batches > 1`(PP>1,或显式异步调度时为 2)才存在(core.py:210-216);`async_scheduling` 默认由 `VllmConfig` 自动裁决(多种特性组合下被强制关,可 `--async-scheduling` 显式开)。调试时别默认它开着。

**SchedulerOutput 字段(调度器→执行层的\"指令\",理解三层如何协作的关键)**:
```
SchedulerOutput:
  scheduled_new_reqs       # 新加入的请求
  scheduled_cached_reqs    # 命中 prefix cache 的请求
  scheduled_running_reqs   # 继续 decode 的请求
  num_scheduled_tokens     # 本步喂给 GPU 的 token 总量(≈ max_num_batched_tokens 内)
  req_to_new_blocks        # 本步为新请求分配的 block 映射
  preempted_reqs           # 被抢占的请求(触发 recompute)
  finished_req_ids / aborted_req_ids
```
Worker 完全按这个结构执行,不关心\"为什么这么排\"——这就是\"指令队列而不是函数调用\"的具体形态。

**端到端请求生命周期(三层如何协作的完整路径)**:
```
AsyncLLM.add_request()  →  ZMQ  →  EngineCore 入 waiting 队列
  → Scheduler.schedule()   →  SchedulerOutput(选中 → running,分配 block)
  → Worker.execute_model() →  GPUModelRunner 前向 + 采样 → ModelRunnerOutput
  → EngineCore.update_from_output()(推进 num_computed_tokens)
  → ZMQ  →  AsyncLLM 流式返回该 token
```
每个请求 token 走完这一圈;连续批处理让它**不止一次**(每步续 1 token,循环若干轮直到 FINISHED)。

> **输出回路**:EngineCore 只产 logits/采样结果,结果经 ZMQ 回 **AsyncLLM(前端进程)**,由 OutputProcessor / detokenize 后**流式返回**(`vllm/v1/engine/output_processor.py`、`core_client.py`)——decode 与流式发送留在 API 层,EngineCore 不碰。

---

## 3. 初始化链路(EngineCore 起来时做了什么)

`EngineCore.__init__` 的 5 步(notes §0):

1. `register_all_kvcache_specs` + `get_kv_cache_specs()`:收集模型每层对 KV cache 的需求(block_size、head 数等);
2. `determine_available_memory()`:跑 dummy batch 测峰值显存,算出可用于 KV cache 的余量;
3. `get_kv_cache_configs()`:据此决定 block 数量与布局(`kv_cache_layout`);
4. `initialize_from_config()` + `compile_or_warm_up_model()`:Worker 上分配 KV cache 张量、编译/捕获 CUDA graph;
5. 构造 `Scheduler`(注入 `KVCacheConfig` 与 `block_size`)。

> 注意 **KV cache 大小是"profiling 出来的"**,不是写死的:启动时先探一次显存余量,把多的都给 KV cache。这跟 PagedAttention 篇的 gpu_memory_utilization 联动。

**KV cache 余量怎么算(可复现)**:`determine_available_memory()` 在 Worker 所在 GPU 上跑一个 dummy batch(profile_run,dummy 权重/输入),用 `torch.cuda.max_memory_allocated()` 的差值量出**模型 + 激活的峰值显存**;再从 `total_memory × gpu_memory_utilization`(默认 0.92)里减去它,剩下的**全部给 KV cache**。块数再除单块大小:

```
kv_cache_bytes = total_memory × gpu_memory_utilization − (weights + activations 峰值)
num_blocks     = kv_cache_bytes ÷ (num_layers × num_kv_heads × block_size × head_dim × 2 × dtype_bytes)
```

(单块大小 = 每层 K+V × head × block_size × head_dim × dtype;NumLayers/kv_heads 来自各层 KVCacheSpec,`get_kv_cache_specs()` 收集。paged-attention 篇的 checklist 那条就是这里的落地。)

---

## 4. 执行层:Worker 与 GPUModelRunner

**Worker(每 GPU 一个,同进程或独立进程视 Executor,UniProc 时同进程)**:
- `load_model()`:加载权重(nn.Module);
- `execute_model()`:处理 pipeline parallel 的 inter-stage 张量传递,再调 model_runner;
- `determine_available_memory()`:测峰值显存(配合 EngineCore 的 KV 余量计算)。

**GPUModelRunner(执行的编舞)**:
- `_prepare_inputs()`:把 `SchedulerOutput` 转成模型输入张量(`input_ids`、`positions`、`block_table`、`slot_mapping`、`cu_seqlens`);
- `_build_attention_metadata()`:构造各 attention backend 所需 metadata;
- `_model_forward()`:`torch.compile` 编译后的前向,支持 piece-wise CUDA graph——**在 attention 边界切图 + 按输入 shape 分桶**,不是任意变长直接捕获(变长 batch 也能命中桶);
- `_sample()`:采样。

**前向链路**:
```
GPUModelRunner.execute_model()
  → _update_states(scheduler_output)   # 更新请求槽位/block table
  → _prepare_inputs()                  # 组装张量
  → _model_forward()                   # nn.Module.forward
      → 各 transformer 层
      → Attention: FlashAttentionImpl.forward()   # 以 FA 后端为例;AttentionBackend 可插拔
          → do_kv_cache_update()  # 写回 KV cache
          → flash_attn_varlen_func # 分页注意力(默认后端依硬件/版本,亦可用 FlashInfer/TRTLLM/FlashMLA/Triton)
  → _sample()                         # 采样
  → ModelRunnerOutput
```

---

## 5. 执行器:UniProc / Multiproc / Ray 三选一

`Executor.get_class()` 据 `ParallelConfig` 选实现(notes §4.1/§5):

| Executor | 场景 | 机制 |
|---|---|---|
| `UniProcExecutor` | 单 GPU | 直接持有 Worker |
| `MultiprocExecutor` | 单节点多 GPU | 子进程 + **共享内存张量** |
| `RayDistributedExecutor`(默认)/ `RayExecutorV2`(需 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` 开启) | 跨节点 | Ray actor + NCCL;由 `distributed_executor_backend == "ray"` 分派 |

**分布式三并行(ParallelConfig 决定)**:
- **TP(张量并行)**:权重按 head 切分到各 GPU,`ColumnParallelLinear`/`RowParallelLinear` 前向自动 all-reduce / all-gather;
- **PP(流水线并行)**:模型按层分 stage,stage 间 `isend/irecv_tensor_dict` 传中间张量,Worker 据自身 PP rank 决定收/发;
- **DP(数据并行)**:多组独立 EngineCore 各处理请求子集(`DPEngineCoreProc`)。

**通信后端**:`pynccl`(NCCL)/ `custom_all_reduce`(自定义,低延迟 P2P)/ `shm_broadcast`(共享内存广播,同机)/ `ray_communicator`(跨节点),由 `GroupCoordinator` 统一封装成 `all_reduce/all_gather/reduce_scatter`。

---

## 6. 与上游主题的咬合

- **调度器 ↔ KV cache**:Scheduler 通过 KVCacheManager(唯一入口)操作分页块——调度决策(KV cache 装得下吗)直接决定能否接纳请求([paged-attention.md](paged-attention.md#5-v1-里的落地blockpool--引用计数--内容哈希));
- **执行层 ↔ 连续批处理**:Worker 不自己排批,只消费 SchedulerOutput;每步全量重排发生在 EngineCore,Worker 无状态地跑([continuous-batching.md](continuous-batching.md));
- **异步 ↔ 吞吐**:独立进程让"CPU 调度"与"GPU 算"能并行,配合异步下发把调度延迟藏进计算(本章 §2)。

---

## 7. 实践要点(checklist)

1. **先认 V1 再谈架构**:网上教程多是 V0(engine/llm_engine.py 两阶段调度),当前主线是 V1 统一 token 预算 + 独立 EngineCore——读源码先看 `vllm/v1/`。
2. **定位问题看层**:请求进不来 → EngineCore/调度器(预算/KV 块);跑得慢 → Worker/GPUModelRunner(kernel/attention backend);卡在 API → AsyncLLM(流式缓冲)。
3. **KV cache 是 profiling 出来的**:启动时 `determine_available_memory` 探余量 → 多的给 KV cache;想给 KV 更多空间调 `gpu_memory_utilization`。
4. **多 GPU 用 Executor 三选一**:单卡 UniProc、单节点多卡 Multiproc、跨节点 Ray;TP/PP/DP 各自对应权重切分/层分片/请求分片。
5. **调试吞吐**:先用 `EngineCore.step()` 的节奏想问题(每步:schedule → 异步下发 → 更新),异步调度是否生效看 batch queue 是否在 GPU 忙时排队。
6. **P/D 分离另说**:prefill/decode 跨节点拆需要 KVCacheConnector(vllm/distributed/kv_transfer/),调度器查远端命中、异步加载 KV——是 V1 之上的追加架构,不是默认。

---

## 8. 溯源

- **V1 源码一手实读**: notes/vllm-core-architecture-notes.md §0(架构总览/初始化链路)、§4(加载与推理路径)、§5(分布式)
- **官方 Architecture Overview**: https://docs.vllm.ai/en/latest/design/arch_overview/
- **V1 Guide**: https://docs.vllm.ai/en/latest/usage/v1_guide/
