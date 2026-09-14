# vLLM 分布式执行与 Ray 集成研究笔记

> 所阅读代码版本：vLLM v0.28.1rc0-40-gb74ac0bc6（commit `b74ac0bc61d8b8f74848664b1a3b4e878f047861`，main 分支）
> 仓库：vllm-project/vllm（github.com/vllm-project/vllm）
> 说明：文中所有 path:line 引用均来自实际读取源码验证；未深入的部分在文末「待补项」明确列出，未做推测性填充。

## 1. 架构总览：Executor 抽象与后端选择

文件：`vllm/v1/executor/abstract.py`（v1 引擎下 executor 已从旧版 `vllm/executor/` 迁移到 `vllm/v1/executor/`）

- 核心抽象类 `Executor`（`abstract.py:38`，ABC），两个关键类属性：
  - `uses_ray: bool = False` —— 是否使用 Ray 编排
  - `supports_pp: bool = False` —— 是否支持 pipeline parallel
- 后端分派入口 `Executor.get_class(vllm_config)`（`abstract.py:49`），按 `parallel_config.distributed_executor_backend` 选择：
  - `"ray"` → `RayDistributedExecutor`（`vllm/v1/executor/ray_executor.py`）；若 `envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND` 开启则改用 `RayExecutorV2`（`ray_executor_v2.py`）
  - `"mp"` → `MultiprocExecutor`（`multiproc_executor.py`）
  - `"uni"` → `UniProcExecutor`（`uniproc_executor.py`）
  - `"external_launcher"` → `ExecutorWithExternalLauncher`
  - 也支持传入自定义类或任意可解析的 qualname 字符串（`resolve_obj_by_qualname`），必须为 `Executor` 子类，否则 `TypeError`
- `Executor.init`（`abstract.py:96` 起）持有完整 `VllmConfig` 各子配置，并调用子类实现的 `_init_executor()`

## 2. Ray 执行器：Actor 创建与 Rank 重排（核心实现细节）

文件：`vllm/v1/executor/ray_executor.py`，类 `RayDistributedExecutor(Executor)`（`ray_executor.py:64`，`uses_ray=True`、`supports_pp=True`）

### 2.1 Worker Actor 的创建

`_init_workers_ray(placement_group)`（`ray_executor.py:156` 起）：

1. **Bundle 选择**：遍历 `placement_group.bundle_specs`，按 `current_platform.ray_device_key` 过滤出含 GPU 资源的 bundle，取前 `world_size` 个作为 `bundle_indices`；用户可用环境变量 `VLLM_RAY_BUNDLE_INDICES` 显式指定（带长度与去重断言）。
2. **Actor 创建**：对每个 rank 执行
   `ray.remote(num_cpus=0, num_gpus=envs.VLLM_RAY_PER_WORKER_GPUS, scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=..., placement_group_capture_child_tasks=True, placement_group_bundle_index=bundle_id))(RayWorkerWrapper).remote(rpc_rank=rank)`
   - 非 `"GPU"` 资源键的平台（如 TPU）改走 `resources={current_platform.ray_device_key: num_gpus}`。
   - worker 类是 `RayWorkerWrapper(WorkerWrapperBase)`（`vllm/v1/executor/ray_utils.py:56`），每个 actor 对应一个 vLLM worker 进程。
3. **关键设计——创建顺序 ≠ 逻辑 rank**：Ray actor 实际落位顺序不确定，因此引入 `RayWorkerMetaData`（`ray_executor.py:51`，字段 worker / created_rank / adjusted_rank / ip）。拿到各 worker IP 后按 `sort_by_driver_then_worker_ip`（`ray_executor.py:247`）重排：① 与 driver 同节点的 worker 优先；② worker 数少的节点优先；③ IP 小者优先。这样 TP 组内 worker 倾向于落在同一节点、driver 节点排最前，优化通信局部性。重排后调用 `collective_rpc("adjust_rank", args=(rerank_mapping,))` 把 created_rank → adjusted_rank 映射下发到所有 worker。

### 2.2 节点/GPU 拓扑收集与环境传播

- 通过 `ray.get(worker.get_node_and_physical_gpu_ids.remote())` 收集每个 worker 所在节点与物理 GPU ID；物理 ID 经 `current_platform.device_control_id_to_physical_device_id()` 转换后按整数排序（注释明确引用 issue #5590：GPU 数 >9 时字符串排序会出错）。
- 一致性校验：`n_nodes != n_ips` 时抛 `RuntimeError`（要求每节点 IP 唯一，提示检查 `VLLM_HOST_IP`）。
- 环境变量传播：`get_env_vars_to_copy(exclude_vars=WORKER_SPECIFIC_ENV_VARS, additional_vars=current_platform.additional_env_vars, destination="workers")`（来自 `vllm/ray/ray_env.py:56`），再经 `collective_rpc("update_environment_variables", ...)` 批量下发。

### 2.3 通信组初始化调用链（引擎启动 → worker 建组）

单节点场景下 driver IP 直接退化为 `127.0.0.1` 回环地址（注释解释：节点可能有多个网卡 IP，回环永远可用）。随后：

1. `distributed_init_method = get_distributed_init_method(driver_ip, get_open_port())` —— 生成 `tcp://ip:port` 形式的 rendezvous 地址（`vllm/utils/network_utils.py:131`）。
2. 按 rank 逐 worker 组装 kwargs：`vllm_config / assigned_physical_gpu_ids / local_rank / rank / distributed_init_method / is_driver_worker=(rank % tensor_parallel_size == 0)` —— 每个 PP 组的第一个 rank（`rank % TP == 0`）即该组的 driver worker。
3. 固定初始化序列：
   `collective_rpc("init_worker", (all_kwargs,))` → `collective_rpc("init_device")` → `collective_rpc("load_model")`（若 `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH` 开启则改为 `elastic_ep_execute("load_model")`）→ `_update_block_size`。
4. **PP×TP 二维编址**：`pp_tp_workers[pp_rank][tp_rank]`，`rank = pp_rank * tensor_parallel_size + tp_rank`（rank 布局是 TP 连续、PP 分块，例如 PP=2/TP=4 时 `[[0,1,2,3],[4,5,6,7]]`）。

worker 侧最终会进入 `vllm/distributed/parallel_state.py` 的 `init_distributed_environment`（`parallel_state.py:1586`）建立 `torch.distributed` 进程组，再由 `initialize_model_parallel`（`parallel_state.py:1751`）划分 TP/PP 组（这两个函数体本次未读完，见文末待办）。

### 2.4 其他已验证细节

- **异步执行模型**：`execute_model`（`ray_executor.py` 约 400 行起）在 `uses_sampler` 且有调度 token 时暂存 scheduler_output，返回预置的 `COMPLETED_NONE_FUTURE`（`ray_executor.py:46`），真正前向延迟到 `sample_tokens()` 触发 —— 支持非阻塞式两段式执行。
- **TPU/XPU 特化**：`_init_executor` 中对 TPU/XPU 平台强制 `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=shm`，绕开 NCCL 通道。
- **旧版 Ray 执行器的已知限制**：`ec_transfer_config` 非 None 且 `world_size > 1` 时直接 `NotImplementedError`，提示改用 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1` 或 mp 执行器（EC connector 元数据只能取单 worker 输出）。
- **弹性重配置**：`reinitialize_distributed`（接收 `ReconfigureDistributedRequest`）下发给所有 worker；当 `new_data_parallel_rank == ReconfigureRankType.SHUTDOWN_CURRENT_RANK` 时整体 `shutdown()`。
- **Ray 集群初始化**：`initialize_ray_cluster`（`vllm/v1/executor/ray_utils.py:538`）负责 Ray 集群初始化与 placement group 获取；`ray_utils.py` 顶部对 `import ray` 做失败保护（`ray is not None` 判断），Ray 缺失时相关类型置 None。

## 3. 并行状态管理（parallel_state.py）

文件：`vllm/distributed/parallel_state.py`（2359 行），已确认的关键符号：

| 符号 | 位置 |
|------|------|
| `init_distributed_environment` | `parallel_state.py:1586` |
| `initialize_model_parallel` | `parallel_state.py:1751` |
| `get_tensor_model_parallel_world_size` | `parallel_state.py:2068` |
| `get_tensor_model_parallel_rank` | `parallel_state.py:2073` |

配合 `vllm/distributed/utils.py`（733 行，含辅助工具）。`vllm/distributed/` 目录还包含：`communication_op.py`（通信算子封装）、`stateless_coordinator.py`（无状态协调）、`ec_transfer/`、`kv_transfer/`、`weight_transfer/`（权重传输）、`elastic_ep/`、`eplb/` 等子模块。

## 4. 通信后端概览

目录：`vllm/distributed/device_communicators/`，实际存在以下后端（文件级确认，函数级细节本次未展开）：

- `pynccl.py` + `pynccl_wrapper.py` + `pynccl_allocator.py` —— PyNccl 集合通信（对 NCCL 的 Python ctypes 封装及分配器）
- `custom_all_reduce.py` —— 自定义 allreduce（小消息优化）
- `cuda_communicator.py` / `base_device_communicator.py` / `cpu_communicator.py` / `xpu_communicator.py` —— 按平台分派的设备通信器基类与实现
- `symm_mem.py`（对称内存）、`shm_broadcast.py` + `shm_object_storage.py`（共享内存广播）、`ray_communicator.py`（Ray 通道）、`flashinfer_all_reduce.py`、`quick_all_reduce.py`、`aiter_custom_all_reduce.py`、`all2all.py`、`all_reduce_utils.py`、`mnnvl_compat.py`

## 5. 执行器对比（单机 vs Ray）

基于 `abstract.py` 的分发逻辑与文件结构：

| | UniProcExecutor | MultiprocExecutor | RayDistributedExecutor |
|---|---|---|---|
| 进程模型 | 单进程单 GPU | 本机多进程（1127 行） | 跨节点 Ray actor |
| uses_ray | False | False | True |
| supports_pp | False | False | True |
| 部署范围 | 单卡 | 单节点多卡 | 多节点 |

Ray 路径独有的开销/能力：placement group 调度、bundle 级 GPU 绑定、rank 重排、driver→worker 环境变量同步、TCP rendezvous 地址下发；而 mp 路径这些都在本机内完成（其 `multiproc_executor.py` 的具体 rank/GPU 分配逻辑本次未读）。

## 6. 待补项（供后续补充，本次未验证）

1. `parallel_state.py:1586` `init_distributed_environment` 函数体 —— 进程组后端选择、store 初始化、与 pynccl 的衔接。
2. `parallel_state.py:1751` `initialize_model_parallel` 函数体 —— TP/PP 进程组的 new_group 划分、rank 计算细节。
3. `vllm/v1/executor/ray_utils.py` 中 `RayWorkerWrapper` 的 `init_worker` / `adjust_rank` 实现与 `initialize_ray_cluster` 的 placement group 创建。
4. `ColumnParallelLinear` / `RowParallelLinear`（应在 `vllm/model_executor/layers/` 下）的 TP 权重切分与前向 all-reduce 位置。
5. `pynccl.py` / `custom_all_reduce.py` 内部实现与 `cuda_communicator.py` 的选择逻辑。
6. `ray_executor_v2.py`（V2 Ray 执行器）与旧版的差异。
