# vLLM 核心技术架构分析笔记

> **代码版本**：commit `b74ac0bc6`（main 分支，`v0.28.1rc0-40-gb74ac0bc6`）
> **分析范围**：整体架构、Scheduler、PagedAttention / KV Cache、连续批处理、模型加载与推理路径、分布式推理
> **仓库根**：`/home/cambricon/zhanbin/repos/vllm`
> 文中 `path:line` 引用均为实际源码位置，基于一手实读。本笔记聚焦本任务要求的 6 个重点，与同目录下覆盖 API server / 模型注册 / 量化 / 采样 / 分布式细节的分篇互补，不重复其内容。

---

## 0. 架构总览

vLLM 当前生产路径是 **V1 架构**（`vllm/v1/`），旧的 `vllm/engine/llm_engine.py` 仅作兼容保留。V1 采用「引擎进程 + 多个 Worker 进程」的多进程架构，核心抽象分三层：

```
┌─────────────────────────────────────────────────────────────┐
│  AsyncLLM (API 层)       vllm/v1/engine/async_llm.py:72     │
│  └─ asyncio 接口，请求队列 / 流式输出                       │
│         │  (ZMQ socket / 消息传递)                            │
│         ▼                                                      │
│  EngineCore (调度核心)   vllm/v1/engine/core.py:105          │
│   ├─ Scheduler          vllm/v1/core/sched/scheduler.py:73   │
│   ├─ KVCacheManager     vllm/v1/core/kv_cache_manager.py:118 │
│   └─ ModelExecutor (抽象) vllm/v1/executor/abstract.py:38    │
│         │  (RPC / 共享内存张量)                               │
│         ▼                                                      │
│  Worker (每 GPU 一个进程) vllm/v1/worker/gpu_worker.py:143   │
│   ├─ GPUModelRunner     vllm/v1/worker/gpu_model_runner.py:501│
│   ├─ AttentionBackend   vllm/v1/attention/backends/*.py      │
│   └─ 模型权重 nn.Module  vllm/model_executor/                │
└─────────────────────────────────────────────────────────────┘
```

**关键设计决策**：调度（CPU）与执行（GPU）解耦。`EngineCore` 在独立进程内做调度决策，`Worker` 进程只负责按 `SchedulerOutput` 执行模型前向。V1 引入了 **异步调度（async scheduling）**：调度器在上一批还在 GPU 上跑时就为下一步排好队，实现 CPU/GPU 重叠（`vllm/v1/core/sched/async_scheduler.py`）。

初始化链路（`EngineCore.__init__` `core.py:108-205`）：
1. `register_all_kvcache_specs` + `get_kv_cache_specs()` 收集模型每层对 KV cache 的需求；
2. `determine_available_memory()` profiling 峰值显存，算出可用于 KV cache 的余量；
3. `get_kv_cache_configs()` 据此决定 block 数量与布局（`kv_cache_layout`）；
4. `initialize_from_config()` + `compile_or_warm_up_model()` 在 Worker 上分配 KV cache 张量并编译/捕获 CUDA graph；
5. 构造 `Scheduler`（注入 `KVCacheConfig` 与 `block_size`）。

---

## 1. 调度器 Scheduler

**文件**：`vllm/v1/core/sched/scheduler.py:73` — `class Scheduler`

### 1.1 核心职责

调度器不区分"prefill 阶段"和"decode 阶段"。源码注释（`scheduler.py:502-511`）明确指出：

> 每个 request 只有 `num_computed_tokens` 和 `num_tokens_with_spec`。每一步调度器尽力让每个 request 的 `num_computed_tokens` 追上 `num_tokens_with_spec`。这一抽象足够通用，覆盖了 chunked prefill、prefix caching、speculative decoding 以及未来的 jump decoding。

这种**统一 token 预算模型**是 V1 区别于旧版（分 running/waiting 两阶段）的核心。

### 1.2 双队列与调度顺序

Scheduler 维护两个队列：
- `self.running`：已分配 KV cache、正在生成的请求（FCFS，优先级高）；
- `self.waiting` / `self.skipped_waiting`：新到达、等待首次调度的请求。

`schedule()`（`scheduler.py:500`）的处理顺序：
1. **先调度 RUNNING**（`scheduler.py:548-760`）：遍历 `running`，为每个请求计算 `num_new_tokens = num_tokens_with_spec - num_computed_tokens`，受 `token_budget`（=`max_num_scheduled_tokens`）和 `input_budget`（=`max_num_batched_tokens`）双重约束。RUNNING 请求通常只需 1 个新 token（decode），保证低延迟。
2. **再调度 WAITING**（`scheduler.py:771-1080`）：若有空余预算且未达 `max_num_running_reqs`（默认 128，`scheduler.py:43`），从 waiting 取队首请求。先做 prefix cache 查询（`_get_local_prefix_cache_hit` `scheduler.py:467`），命中则跳过这些 token 的计算；再按 `chunked prefill` 把长 prompt 切块塞进剩余预算。

### 1.3 chunked prefill

默认开启（`config/scheduler.py:74` `enable_chunked_prefill: bool = True`）。当 prompt 超过当前 step 剩余 `token_budget` 时，只调度一部分 token，其余留到后续 step。这使 prefill 和 decode 可在同一 batch 混合，提升吞吐——代价是 prefill 延迟略增。`long_prefill_token_threshold`（`scheduler.py:587`）可对超长 prefill 限流单步 token 数。

### 1.4 抢占（preemption）

当 `allocate_slots` 返回 `None`（KV cache 块不足）时触发抢占（`scheduler.py:654-716`）：
- **PRIORITY 策略**：抢占优先级最低的运行中请求（`scheduler.py:667-700`）；
- **默认策略**：抢占队尾（`running.pop()`，`scheduler.py:702`）；
- 被抢占请求经 `_preempt_request()`（`scheduler.py:1370`）处理：**V1 只有 recompute 一种模式**——释放其 KV cache 块（进 LRU 前缀缓存）、`num_computed_tokens = 0`、退回 waiting 重排（重排靠 prefix cache 命中减重算；V0 的 counted/swap 不再存在，2026-09 实读源码确认）。

### 1.5 关键数据结构

- `SchedulerOutput`（`vllm/v1/core/sched/output.py`）：每步调度的产物，含 `scheduled_new_reqs` / `scheduled_running_reqs` / `req_to_new_blocks` / `num_scheduled_tokens` / `preempted_reqs`，传给 ModelExecutor。
- `Request`（`vllm/v1/request.py`）：携带 `num_computed_tokens`、`num_tokens_with_spec`、`block_ids`、`spec_token_ids`、`status`（WAITING/RUNNING/FINISHED...）。
- `token_budget` / `input_budget`：`max_num_batched_tokens` 默认 2048（`scheduler.py:42`）。

---

## 2. KV Cache 管理与 PagedAttention

### 2.1 分页 KV Cache（核心创新）

vLLM 将 KV cache 划分为固定大小的 **block**（默认 16 token，`config/cache.py:79` `DEFAULT_BLOCK_SIZE = 16`），每个序列的逻辑 KV 存储通过**block table**（块表）映射到物理块。这使 KV cache 可非连续分配，避免传统"每序列连续分配"导致的显存碎片——这就是 PagedAttention 的核心思想。

物理 KV cache 张量形状（GPU 上，FlashAttention 路径）：
```
[num_blocks, num_kv_heads, block_size, 2 * head_size]   # flash_attn.py:966
```
即 K 和 V 拼在最后一维（`2 * head_size`），按 (block, head, block内位置) 索引。

### 2.2 BlockPool（块池）

**文件**：`vllm/v1/core/block_pool.py:143` — `class BlockPool`

管理物理块的生命周期：
- **free list**：空闲块池，`get_new_blocks(n)` 分配（`block_pool.py:647`）；
- **cached hash map**：`BlockHashToBlockMap`（`block_pool.py:33`）按 block 内容哈希索引，支持 prefix caching 命中查找；
- **引用计数**：每个 `KVCacheBlock` 带 `ref_cnt`，序列释放时递减，归零则回收到 free list（`block_pool.py:719` `free_blocks`）；
- **LRU 淘汰**：`_maybe_evict_cached_block()`（`block_pool.py:679`）在空闲块不足时按 LRU 淘汰已缓存的块。

### 2.3 KVCacheManager

**文件**：`vllm/v1/core/kv_cache_manager.py:118` — `class KVCacheManager`

封装对 BlockPool 的高层操作，是 Scheduler 操作 KV cache 的唯一入口：
- `get_computed_blocks()`（`kv_cache_manager.py:232`）：prefix cache 查询，返回命中的块列表与命中 token 数；
- `allocate_slots()`（`kv_cache_manager.py:347`）：为请求分配新块。源码 393-414 行用图示标注了块的布局分区：

```
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
```
  分别是已计算、新命中、外部(connector)命中、待计算、speculative lookahead。分配分三阶段（注释 `kv_cache_manager.py:431-438`）：释放冗余块 → 处理 prefix token → 为待计算 token 分配新块。返回 `None` 表示空间不足（触发抢占）。

- `free()`（`kv_cache_manager.py:570`）、`evict_blocks()`（`kv_cache_manager.py:622`）、`reset_prefix_cache()`（`kv_cache_manager.py:630`）。

### 2.4 Prefix Caching（自动前缀缓存）

默认开启（`config/cache.py:138` `enable_prefix_caching: bool = True`）。原理：
1. 每个满 block 由 `Request` 算出**内容哈希**（基于 token ids + 前驱块哈希，构成哈希链）；
2. `BlockPool.cache_full_blocks()`（`block_pool.py:225`）把满块登记进 `cached_block_hash_to_block`；
3. 新请求到达时 `_get_local_prefix_cache_hit()`（`scheduler.py:467`）查哈希链，命中即复用已有物理块（ref_cnt++），跳过这些 token 的 prefill 计算——system prompt / 共享前缀场景显著省算力。

**copy-on-write**：被多个序列引用的块在被写入时先复制（CoW），避免污染。块布局注释中"ref_cnt not increased yet"区即指待 CoW 的块。

### 2.5 KV Cache 写入 kernel

每步前向后，新计算的 K/V 需写回分页 cache。关键 op（`vllm/_custom_ops.py`）：
- `reshape_and_cache_flash()`（`_custom_ops.py:2631`）→ `torch.ops._C_cache_ops.reshape_and_cache_flash`：FlashAttention 路径用，按 `slot_mapping` 散列写入 `key_cache`/`value_cache`；
- `reshape_and_cache()`（`_custom_ops.py:2609`）：通用路径；
- `swap_blocks()`（`_custom_ops.py:2903`）：CPU↔GPU 间的块交换（用于 offloading）。

CUDA 实现见 `csrc/cache.h`（声明）与 `csrc/` 下对应 `.cu`。`slot_mapping` 是把 (block_id, slot_in_block) 展平成一维索引，kernel 据此 `atomic` 写入。

### 2.6 Attention 计算如何使用分页 cache

V1 的 FlashAttention 后端（`vllm/v1/attention/backends/flash_attn.py`）不使用旧版 `paged_attention_v1/v2` 自定义 CUDA kernel，而是调用 **flash_attn_varlen_func**（`flash_attn.py:1162`），它原生支持"变长序列 + block table"形式的 paged KV：
- `FlashAttentionImpl.forward()`（`flash_attn.py:947`）先将 cache 拆出 K/V（`flash_attn.py:1014` `kv_cache.transpose(1,2).split(...)`）；
- `do_kv_cache_update()`（`flash_attn.py:1219`）调用 `reshape_and_cache_flash` 写入新 K/V；
- 再以 block table + cu_seqlens 调 `flash_attn_varlen_func` 做带分页的 attention。

即：**PagedAttention 在 V1 中的体现 = 分页物理存储（BlockPool）+ block table 索引 + flash_attn_varlen 的变长分页注意力**。旧版手写 `paged_attention` kernel 仍保留（`csrc/attention/`）但非主路径。

---

## 3. 连续批处理（Continuous Batching）

### 3.1 流程

V1 的连续批处理由 Scheduler + EngineCore 的 step 循环驱动（`core.py:597` `step()`）：

```
EngineCore.step():                          # core.py:597
  if not scheduler.has_requests(): return    # 606
  scheduler_output = scheduler.schedule()    # 608  ← 核心：重排整个 batch
  future = executor.execute_model(non_block) # 609  ← 异步下发到 worker
  grammar = scheduler.get_grammar_bitmask()  # 610  ← 与 GPU 计算重叠
  model_output = future.result()             # 615
  scheduler.update_from_output(output)       # 622  ← 更新 num_computed_tokens
```

每一步：
1. **重新调度整个 batch**：不固定 batch 成员——已完成/被中止的请求移出，waiting 中新请求按预算进入，运行中请求继续追加 token；
2. **执行**：把 `SchedulerOutput` 发给所有 Worker，各 Worker 用 `GPUModelRunner` 组装输入张量并跑模型；
3. **采样**：`GPUModelRunner._sample()`（`gpu_model_runner.py:3699`）；
4. **状态更新**：`update_from_output()`（`scheduler.py:1767`）推进 `num_computed_tokens`，标记 FINISHED/ABORTED。

### 3.2 为何能提升吞吐

传统静态批处理必须等一批全部生成完才接纳新请求，decode 阶段 GPU 利用率低。连续批处理允许：
- **decode 与 prefill 混批**：同一 step 里既有只产 1 token 的 decode 请求，也有切块的 prefill 请求（chunked prefill），填满 `max_num_batched_tokens`；
- **即时进出**：短请求完成后立刻让位给 waiting 队列，无需等齐。

### 3.3 异步调度（async scheduling）

`core.py:638` `step_with_batch_queue()` 与 `vllm/v1/core/sched/async_scheduler.py` 实现更激进的重叠：用一个 batch queue 缓存多个已调度的 batch，在 GPU 跑 batch N 时 CPU 已在调度 batch N+1。`async_scheduling` 配置项控制。

### 3.4 请求生命周期

```
add_request (async_llm.py:283)
  → tokenize → Request 对象 → EngineCore.add_request (core.py:452)
  → scheduler.add_request (scheduler.py:2354) → 进 waiting 队列
schedule() 首次选中 → waiting → running（分配 block）
每 step：running 中生成 token → FINISHED/ABORTED → _free_request (scheduler.py:2445)
```

---

## 4. 模型加载与推理路径

### 4.1 ModelExecutor（执行器层）

**抽象**：`vllm/v1/executor/abstract.py:38` — `class Executor`，定义 `execute_model`、`determine_available_memory`、`get_kv_cache_specs`、`initialize_from_config` 等接口。具体实现：
- `UniProcExecutor`（`uniproc_executor.py:51`）：单 GPU，直接持有 Worker；
- `MultiprocExecutor`（`multiproc_executor.py`）：多 GPU 通过子进程 + 共享内存张量；
- `RayExecutor`（`ray_executor.py`）/ `RayExecutorV2`：跨节点用 Ray。

`Executor.get_class()`（`abstract.py:49`）据配置选实现。

### 4.2 Worker 与 ModelRunner

`vllm/v1/worker/gpu_worker.py:143` — `class Worker`，每 GPU 一个，持有模型权重与 KV cache 张量：
- `load_model()`（`gpu_worker.py:455`）→ `GPUModelRunner.load_model`；
- `execute_model()`（`gpu_worker.py:1067`）：处理 pipeline parallel 的 inter-stage 张量传递（`gpu_worker.py:1112-1124` `irecv_tensor_dict`），再调用 `model_runner.execute_model()`；
- `determine_available_memory()`（`gpu_worker.py:480`）：跑 dummy batch 测峰值显存。

`vllm/v1/worker/gpu_model_runner.py:501` — `class GPUModelRunner`，执行的核心：
- `_prepare_inputs()`（`gpu_model_runner.py:1961`）：把 `SchedulerOutput` 转成模型输入张量（`input_ids`、`positions`、`block_table`、`slot_mapping`、`cu_seqlens`）；
- `_build_attention_metadata()`（`gpu_model_runner.py:2297`）：构造各 attention backend 所需 metadata；
- `_model_forward()`（`gpu_model_runner.py:3896`）：`torch.compile` 编译后的前向，支持 piece-wise CUDA graph；
- `_sample()`（`gpu_model_runner.py:3699`）：采样。

### 4.3 模型注册与加载

- **架构映射**：`vllm/model_executor/models/` 下每个模型文件用 `@support_*` 装饰器注册到 `ModelRegistry`；`get_model_architecture()`（`model_executor/model_loader/__init__.py` 附近）据 `model_config.architecture` 选类。
- **权重加载器**：`get_model_loader()`（`model_loader/__init__.py:119`）据 `load_config.load_format` 选 loader：
  - `DefaultModelLoader`（`default_loader.py:43`）：从 HF Hub/Safetensors 流式加载权重，按层名匹配；
  - `ShardedStateLoader`、`TensorizerLoader`、`RunaiStreamerLoader` 等变体。
- **模型实例化**：`get_model()`（`model_loader/__init__.py:127`）→ 实例化 `nn.Module` → `loader.load_weights()` 注入权重。TP/PP 切分在层内通过 `ColumnParallelLinear`/`RowParallelLinear` 等自动完成。

### 4.4 推理前向链路

```
GPUModelRunner.execute_model()
  → _update_states(scheduler_output)   # gpu_model_runner.py:1188 更新请求槽位/block table
  → _prepare_inputs()                  # 1961 组装张量
  → _model_forward()                   # 3896 nn.Module.forward
      → 各 transformer 层
      → Attention 层: FlashAttentionImpl.forward()  # flash_attn.py:947
          → do_kv_cache_update()  # 写回 KV cache
          → flash_attn_varlen_func # 分页注意力
  → _sample()                          # 3699
  → ModelRunnerOutput
```

---

## 5. 分布式推理

### 5.1 三种并行

`vllm/distributed/parallel_state.py` 管理并行组，`init_model_parallel_group()`（`parallel_state.py:1315`）据 `ParallelConfig`（`config/parallel.py`）建立 TP/PP/DP 通信组：

- **张量并行（TP）**：`tensor_model_parallel_size`，权重按 head 切分到各 GPU，`ColumnParallelLinear`/`RowParallelLinear` 在前向自动 all-reduce / all-gather。通信组构造见 `parallel_state.py:1837`。
- **流水线并行（PP）**：`pipeline_model_parallel_size`，模型按层分 stage，stage 间用 `PipelineParallelGroup` 传中间张量（`gpu_worker.py:1112` `irecv_tensor_dict` / `isend_tensor_dict`）。Worker 据自身 PP rank 决定接收还是发送（`gpu_worker.py:1083-1124`）。
- **数据并行（DP）**：`data_parallel_size`，多组独立 EngineCore 各自处理请求子集，`DPEngineCoreProc`（`core.py:2004`）管理 DP 协调。

### 5.2 通信后端

`vllm/distributed/device_communicators/` 提供多种实现：
- `pynccl.py` + `cuda_wrapper.py`：基于 NCCL 的 GPU 集合通信（all_reduce/all_gather/reduce_scatter）；
- `custom_all_reduce.py`：自定义 all-reduce（绕过 NCCL，低延迟，多 GPU P2P）；
- `shm_broadcast.py`：共享内存广播（同机多进程高效传张量）；
- `ray_communicator.py`：Ray 跨节点。

`GroupCoordinator`（`parallel_state.py:380`）统一封装，提供 `all_reduce()`/`all_gather()`/`reduce_scatter()`（`parallel_state.py:662-745`）。

### 5.3 Ray 集成

跨节点部署用 Ray：`vllm/v1/executor/ray_executor.py`（V1）/ `ray_executor_v2.py` 在每个 GPU 节点启动一个 Ray actor 承载 Worker。`vllm/ray/` 含 Ray 相关工具。Executor 通过 Ray 远程调用 Worker 的 `execute_model`，用 NCCL 做集合通信。

### 5.4 KV Cache 跨节点传输（P/D 分离）

`vllm/distributed/kv_transfer/` 支持 Prefill-Decode 分离架构：prefill 节点算出 KV cache 后传给 decode 节点。Scheduler 侧 `connector`（`scheduler.py:845-902`）查询远端缓存命中、异步加载 KV（`load_kv_async`），`num_external_computed_tokens` 记录远端命中量。

---

## 6. 关键数据结构速查

| 结构 | 文件:行 | 作用 |
|------|---------|------|
| `VllmConfig` | `config/__init__.py` | 聚合所有配置子项的顶层配置 |
| `SchedulerConfig` | `config/scheduler.py:29` | max_num_batched_tokens=2048, max_num_seqs=128, enable_chunked_prefill=True |
| `CacheConfig` | `config/cache.py` | block_size=16, enable_prefix_caching=True, gpu_memory_utilization=0.92 |
| `ParallelConfig` | `config/parallel.py` | tp/pp/dp size |
| `Request` | `vllm/v1/request.py` | 单请求状态机 |
| `SchedulerOutput` | `vllm/v1/core/sched/output.py` | 单步调度结果 |
| `KVCacheManager` | `vllm/v1/core/kv_cache_manager.py:118` | 块分配/释放/前缀查询 |
| `BlockPool` | `vllm/v1/core/block_pool.py:143` | 物理块池 + LRU + 哈希索引 |
| `KVCacheBlock` | `vllm/v1/core/kv_cache_utils.py` | 单块元数据（id, ref_cnt, hash） |
| `KVCacheSpec` | `vllm/v1/kv_cache_interface.py:145` | 每层对 KV cache 的需求描述（block_size, head 数等） |
| `EngineCore` | `vllm/v1/engine/core.py:105` | 调度核心进程 |
| `AsyncLLM` | `vllm/v1/engine/async_llm.py:72` | 异步 API 层 |
| `Worker` | `vllm/v1/worker/gpu_worker.py:143` | 每 GPU 执行进程 |
| `GPUModelRunner` | `vllm/v1/worker/gpu_model_runner.py:501` | 模型前向 + 输入组装 |
| `FlashAttentionImpl` | `vllm/v1/attention/backends/flash_attn.py:833` | FA 注意力实现 |
| `Executor` | `vllm/v1/executor/abstract.py:38` | 执行器抽象 |

---

## 7. 可补充到 wiki 的技术要点清单

以下要点建议补入 wiki vLLM 页面，每条均有源码支撑：

1. **V1 架构是当前主路径**：`vllm/v1/` 取代旧 `engine/`；多进程（EngineCore + Worker）+ 异步调度。
2. **统一 token 预算调度**：不再分 prefill/decode 阶段，一切以 `num_computed_tokens` 追赶 `num_tokens_with_spec`（`scheduler.py:502-511`）。
3. **chunked prefill 默认开启**：prefill 与 decode 同 batch 混合，`max_num_batched_tokens=2048`（`scheduler.py:42,74`）。
4. **分页 KV cache**：block_size 默认 16（`cache.py:79`），物理张量 `[num_blocks, num_kv_heads, block_size, 2*head_size]`（`flash_attn.py:966`）。
5. **PagedAttention V1 实现**：BlockPool 分页存储 + block table 索引 + `flash_attn_varlen_func` 变长分页注意力；非旧版 `paged_attention` 自定义 kernel。
6. **自动 prefix caching**：默认开启（`cache.py:138`），基于 block 内容哈希链，命中复用物理块，CoW 处理写共享。
7. **KV cache 写回**：`reshape_and_cache_flash` 按 `slot_mapping` 散列写入分页张量（`flash_attn.py:1244`，`_custom_ops.py:2631`）。
8. **抢占机制**：块不足时抢占低优先级/队尾请求，recompute 退回 waiting（`scheduler.py:654-716`）。
9. **分布式三并行**：TP（权重切分+all_reduce）、PP（stage 间 tensor 传递，`gpu_worker.py:1112`）、DP（独立 EngineCore，`core.py:2004`）。
10. **P/D 分离**：KVCacheConnector 跨节点传 KV cache，scheduler 查远端命中（`scheduler.py:845-902`）。
11. **执行器多后端**：UniProc / Multiproc / Ray，按配置选择（`abstract.py:49`）。
12. **CUDA graph 编译**：`compile_or_warm_up_model()`（`core.py:345`）捕获计算图，piece-wise graph 在 `GPUModelRunner` 中支持变长 batch。

---

## 附：与既有分篇的边界

本笔记聚焦调度器 / KV cache / PagedAttention / 连续批处理 / 模型加载推理路径 / 分布式框架六大主题。同目录已有分篇覆盖的细分内容不再重复：
- API server 层与 OpenAI 兼容 → `vllm-api-server-openai-compat-notes.md`
- 模型注册机制 → `vllm-model-registry-notes.md`
- 量化（GPTQ/AWQ/FP8）→ `vllm-quantization-gptq-awq-fp8-notes.md`
- Ray 分布式执行细节 → `vllm-distributed-ray-notes.md`
- 采样与惩罚 → `vllm-sampling-penalties-notes.md`
