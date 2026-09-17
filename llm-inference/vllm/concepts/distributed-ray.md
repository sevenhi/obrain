---
title: vLLM 分布式执行与 Ray 集成:从 Executor 到通信组
type: concept
tags: [vllm, distributed, ray, executor, tp, pp, parallel-state, ncl]
source:
  - "code: vLLM V1 源码 (commit b74ac0bc6, v0.28.1rc0-40) | vllm/v1/executor/ + vllm/distributed/ 一手实读 (经 notes 提炼)"
  - "web: vLLM docs — Parallelism and Scaling | https://docs.vllm.ai/en/latest/serving/parallelism_scaling/"
date: 2026-09
---

# vLLM 分布式执行:Executor 抽象、Ray 集成与通信组初始化

> 本文由 `notes/vllm-distributed-ray-notes.md` 深度提炼而成(**notes 已删,以源码 commit b74ac0bc6 为准**)。源码基准 commit `b74ac0bc6`(v0.28.1rc0-40,2026-08-28)。path:line 均实际验证;notes 中标注"未深入"的(parallel_state 函数体、ColumnParallel 切分)在本文以"边界"明示,不推测。

## TL;DR

vLLM 的分布式执行是"**一层抽象 + 三条后端**":顶层 **`Executor` 抽象**(`vllm/v1/executor/abstract.py`)按 `distributed_executor_backend` 分派到 **UniProcExecutor(单卡)/ MultiprocExecutor(单节点多卡)/ RayDistributedExecutor(跨节点)**。跨节点走 Ray 时,真正的复杂度在**三件事**:① 用 placement group 为每个 rank 创建 Worker actor;② **重排 rank 以优化通信局部性**(driver 同节点优先、同节点 worker 数少者优先);③ 用 TCP rendezvous 地址 + 固定初始化序列建通信组。

**记住一句话**:vLLM 不自己发明分布式语义,它**复用 torch.distributed + NCCL,只是负责"把每个 GPU 的 worker 编排起来、按拓扑重排 rank、下发环境变量和通信地址"**。瓶颈通常在编排,不在通信本身。

> **先记住一个反转直觉**:Worker actor 的**创建顺序 ≠ 逻辑 rank**。Ray 把 actor 落到哪些 GPU 是不确定的,vLLM 必须拿到各 worker 的 IP 后**显式重排**(adjust_rank)才能保证 TP 组内通信局部性。看到"rank 重排"别以为是 bug,这是刻意的设计。

---

## 1. Executor 抽象与后端选择

`Executor`(`abstract.py:38`,ABC)两个关键类属性决定后端能力:

| 属性 | 含义 |
|---|---|
| `uses_ray: bool` | 是否用 Ray 编排 |
| `supports_pp: bool` | 是否支持 pipeline parallel |

**后端分派入口 `Executor.get_class(vllm_config)`**(`abstract.py:49`),按 `parallel_config.distributed_executor_backend`:

```python
# 示意(真实沿 abstract.py:49 逻辑)
if backend == "ray":          RayDistributedExecutor   # ray_executor.py; VLLM_USE_RAY_V2_EXECUTOR_BACKEND → RayExecutorV2
elif backend == "mp":         MultiprocExecutor        # multiproc_executor.py (本机多进程)
elif backend == "uni":        UniProcExecutor          # uniproc_executor.py (单进程单 GPU)
elif backend == "external_launcher": ExecutorWithExternalLauncher
else:                         支持自定义 qualname 字符串 → resolve_obj_by_qualname()
```

**Ray 路径独有的开销/能力**:placement group 调度、bundle 级 GPU 绑定、rank 重排、driver→worker 环境变量同步、TCP rendezvous 地址下发。**mp 路径这些都在本机内完成**——明白这一点,部署时选后端的判断依据就清楚了:单节点多卡用 mp(无 Ray 开销),跨节点才用 Ray。

---

## 2. Ray 执行器:Worker 创建与 Rank 重排(核心)

### 2.1 Worker actor 的创建

`_init_workers_ray(placement_group)`(`ray_executor.py:156` 起):

1. **Bundle 选择**:遍历 `placement_group.bundle_specs`,按 `current_platform.ray_device_key` 过滤出含 GPU 的 bundle,取前 `world_size` 个作 `bundle_indices`;用户可用 `VLLM_RAY_BUNDLE_INDICES` 显式指定(带长度/去重断言)。
2. **Actor 创建**:每 rank 执行 `ray.remote(num_cpus=0, num_gpus=..., scheduling_strategy=PlacementGroupSchedulingStrategy(...))(RayWorkerWrapper).remote(rpc_rank=rank)`。非 `"GPU"` 键平台(TPU)改走 `resources={ray_device_key: num_gpus}`。**worker 类是 `RayWorkerWrapper`(`ray_utils.py:56`),每个 actor 一个 vLLM worker 进程**。
3. **关键:创建顺序 ≠ 逻辑 rank**。

### 2.2 Rank 重排:为什么必须做

拿到各 worker IP 后,`sort_by_driver_then_worker_ip`(`ray_executor.py:247`)重排,排序键:
1. **与 driver 同节点的 worker 优先**;
2. **worker 数少的节点优先**;
3. **IP 小者优先**。

重排后 `collective_rpc("adjust_rank", args=(rerank_mapping,))` 把 `created_rank → adjusted_rank` 映射下发所有 worker。**目的**:让 TP 组内 worker 倾向落在同一节点、driver 节点排最前。通信局部性直接影响 TP 的 all-reduce 延迟。

**物理 GPU 排序还有一个隐藏坑**(`ray_executor.py` 注释引用 issue #5590):物理 ID 转成整数排序,否则 **GPU 数 > 9 时字符串排序会出错**(`"10" < "9"`)。

### 2.3 节点拓扑收集与环境传播

- 经 `ray.get(worker.get_node_and_physical_gpu_ids.remote())` 收集节点与物理 GPU;`n_nodes != n_ips` 抛 `RuntimeError`(要求每节点 IP 唯一,提示查 `VLLM_HOST_IP`)。
- 环境变量传播:`get_env_vars_to_copy(exclude_vars=WORKER_SPECIFIC_ENV_VARS, ...)`(`vllm/ray/ray_env.py:56`),再 `collective_rpc("update_environment_variables", ...)` 批量下发。

---

## 3. 通信组初始化调用链

**单节点**:driver IP 退化 `127.0.0.1`(注释:节点多网卡时回环永远可用)。随后:

1. `distributed_init_method = get_distributed_init_method(driver_ip, get_open_port())` → `tcp://ip:port` rendezvous 地址(`vllm/utils/network_utils.py:131`)。
2. 逐 worker 组装 kwargs:`vllm_config / assigned_physical_gpu_ids / local_rank / rank / distributed_init_method / is_driver_worker=(rank % tensor_parallel_size == 0)`——**每个 PP 组的第一个 rank(`rank % TP == 0`)即该组 driver worker**。
3. **固定初始化序列**:
   `collective_rpc("init_worker")` → `collective_rpc("init_device")` → `collective_rpc("load_model")`(弹性 EP 则 `elastic_ep_execute("load_model")`)→ `_update_block_size`。
4. **PP×TP 二维编址**:`pp_tp_workers[pp_rank][tp_rank]`,`rank = pp_rank * tensor_parallel_size + tp_rank`(**TP 连续、PP 分块**,例 PP=2/TP=4 → `[[0,1,2,3],[4,5,6,7]]`)。

worker 侧进入 `vllm/distributed/parallel_state.py` 的 `init_distributed_environment`(`:1586`)建 `torch.distributed` 进程组,再 `initialize_model_parallel`(`:1751`)划 TP/PP 组。

### 已验证的边界(诚实标注)

本文**不虚构** `init_distributed_environment`/`initialize_model_parallel` 的函数体细节(notes 未读完),只确认它们 在 位置(:1586 / :1751)并承担"建进程组 / 划 TP/PP"职责。要写得更深须重读这两个函数——不在本文范围内。

---

## 4. 通信后端概览(文件级确认)

`vllm/distributed/device_communicators/`:

| 后端 | 作用 |
|---|---|
| `pynccl.py` + wrapper + allocator | PyNccl 集合通信(NCCL 的 Python ctypes 封装 + 分配器) |
| `custom_all_reduce.py` | 自定义 allreduce(小消息优化) |
| `shm_broadcast.py` / `shm_object_storage.py` | 共享内存广播(同机) |
| `ray_communicator.py` | Ray 通道 |
| `cuda_communicator.py` / `base_device_communicator.py` / `cpu_communicator.py` / `xpu_communicator.py` | 按平台分派 |
| `symm_mem.py` / `all2all.py` / `all_reduce_utils.py` / `flashinfer_all_reduce.py` / `quick_all_reduce.py` / `aiter_custom_all_reduce.py` | 各专用通道 |

`vllm/distributed/` 还含 `communication_op.py`(通信算子封装)、`stateless_coordinator.py`、`ec_transfer/`、`kv_transfer/`、`weight_transfer/`、`elastic_ep/`、`eplb/`。

> **KV 传输(`kv_transfer/`)不是默认**:跨节点 prefill/decode 拆分需要 KVCacheConnector(见 [v1-architecture.md](v1-architecture.md) §7),这是 V1 之上的**追加架构**,不是分布式默认。看到 `kv_transfer/` 别以为它是标配路径。

---

## 5. 执行器对比(选型速查)

| | UniProcExecutor | MultiprocExecutor | RayDistributedExecutor |
|---|---|---|---|
| 进程模型 | 单进程单 GPU | 本机多进程(1127 行) | 跨节点 Ray actor |
| uses_ray | False | False | True |
| supports_pp | False | False | True |
| 部署范围 | 单卡 | 单节点多卡 | 多节点 |
| 编排开销 | 无 | 低 | placement group + rank 重排 |

**选型判断**:单卡 UniProc;单节点多卡优先 Multiproc(免 Ray 开销);跨节点才 Ray。**PP 只有在 Ray(或 mp)下才支持**(`supports_pp=True`);UniProc 单进程不支持 PP。

---

## 6. 实践要点(checklist)

1. **选后端看部署范围,不看偏好**:单节点多卡用 `mp`(无 Ray 编排开销),跨节点才 `ray`。不要默认上 Ray 让单机也背 placement group 成本。
2. **多节点报通信问题先查两个环境变量**:`VLLM_HOST_IP`(每节点 IP 唯一性校验 `n_nodes != n_ips`)、`VLLM_RAY_BUNDLE_INDICES`(GPU bundle 显式指定)。
3. **worker 落位是"创建后再重排"**:看到日志里 created_rank ≠ 你想的 rank 是正常,`adjust_rank` 是设计不是 bug。
4. **GPU > 9 的排序坑**:物理 ID 排序按整数(引用 issue #5590),字符串排序会在 10+ GPU 时出错——升级/排查拓扑时留意。
5. **PP 的 driver worker 判定**:`rank % TP == 0` 是每个 PP 组的 driver;PP×TP 编址 TP 连续、PP 分块,定位跨卡问题先算清 rank 布局。
6. **TPU/XPU 特化**:`_init_executor` 对 TPU/XPU 强制 `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=shm`,绕开 NCCL 通道——非 NVIDIA 平台部署记得查这条。
7. **旧 Ray executor 的 EC 限制**:`ec_transfer_config` 非 None 且 world_size>1 时 `NotImplementedError`,提示改用 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1` 或 mp——做弹性/上下文扩展(EC)时这是硬边界。
8. **跨进程通信语义归 torch.distributed**,vLLM 只负责编排——想在通信层动手(自定义 allreduce)看 `device_communicators/`,想在编排层动手看 `executor/`。

---

## 7. 溯源

- 源码一手实读: `vllm/v1/executor/`(`abstract.py` / `ray_executor.py` / `ray_utils.py`) + `vllm/distributed/`(`parallel_state.py` / `device_communicators/`)
- 版本差异基线: 本文对照自 notes `vllm-distributed-ray-notes.md`(已删);其"待补项"(parallel_state 函数体、ColumnParallel 切分)作为边界在 §3 明示,待后续深读
