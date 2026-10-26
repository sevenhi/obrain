---
title: Model Runner V2(MRV2):vLLM 执行层的重设计——持久 batch 解耦、async-first、StagedWriteTensor 与显式 CUDA Graph
type: concept
tags: [vllm, model-runner, mrv2, persistent-batch, async-scheduling, cuda-graph, uva, triton, sampling, gumbel, speculative-decoding, kv-cache]
source:
  - "code: vLLM 源码一手实读 (main @2d40c0609, 2026-10-22 抓取) | https://github.com/vllm-project/vllm/tree/2d40c0609 | vllm/config/vllm.py(use_v2_model_runner 判定)、vllm/v1/worker/gpu_worker.py、vllm/v1/worker/gpu/{model_runner.py, states.py, buffer_utils.py, cudagraph_utils.py, input_batch.py, sample/gumbel.py, sample/logprob.py}"
  - "web: vLLM Model Runner V2 Design Document | https://docs.vllm.ai/en/latest/design/model_runner_v2/ | 2026-10-23 实抓(与仓库 docs/design/model_runner_v2.md 逐字一致,206 行)"
  - "code: vLLM 文档源文件 docs/design/model_runner_v2.md | https://github.com/vllm-project/vllm/blob/main/docs/design/model_runner_v2.md | 首次提交 PR https://github.com/vllm-project/vllm/pull/35819 (2026-03-02)"
  - "web: vLLM 官方文档首页 | https://docs.vllm.ai/en/latest/ | 2026-10-23 实测 200"
date: 2026-10
---

# Model Runner V2(MRV2):vLLM 执行层的重设计

> **TL;DR**:MRV2 不是"再快一点"的调优,而是把 V1 执行层里**耦合掉的东西拆开**——① 持久 batch 状态与每步输入张量解耦(请求拿到**永久行号**,preemption 当结束处理),于是不再需要 `CachedRequestState` 这份备份;② 明确以 **async-first** 为前提(执行 = 一条没有 CPU 同步点的 CUDA stream),用"状态不 pin + 每次 pin 一份临时副本"从根上消掉 V1 的 async barrier;③ 大状态(block table / `num_computed_tokens` / `all_token_ids`)改走 **`StagedWriteTensor` + UVA 环形缓冲**,只搬 diff、一次 kernel 落盘;④ 采样器 Triton 化(Gumbel-max 单遍、top-k logprob 只算选中的 token);⑤ CUDA Graph 有显式 `CudaGraphManager`,能把投机解码的多步 draft **融进一张图**。**默认已开**(2026-10 的 main:`use_v2_model_runner` 除非命中不支持特性/无 Triton/特定 ROCm 架构,否则为 True),所以"新代码跑在 MRV2 上"是默认事实,旧的 `gpu_model_runner.py` 只在回退路径里。

> ⚠ 官方文档原文仍写着 *"not yet feature-complete, not rigorously tested, and still has open design decisions"*——但代码里它已经是**默认路径**。正确姿势:**以源码为准判断"当前跑哪个 runner",以设计文档理解"为什么这么设计"**,两者都要看。

---

## 0. 为什么需要重写:V1 的三笔技术债

官方设计文档开篇列了三类"事后看是设计失误"的东西,都是**加特技加出来的**:

| 债 | V1 的做法 | 后果 |
|---|---|---|
| 持久 batch 与输入张量耦合 | 持久状态张量**直接**当模型/采样器输入 | 布局与顺序被硬约束;请求 join/finish 时常要"整张量重排"而非插入/删一行 |
| 需要 `CachedRequestState` | 因为持久张量的行可能被覆盖,只能再留一份请求状态备份 | 双份真相,记账复杂度翻倍,异步下更难对 |
| async 是后补的 | 为异步调度加**异步 barrier** 包住临界区 | ① 保护的缓冲区容易漏(漏=竞态 bug);② 所有 CPU 工作被迫塞进 barrier 内;③ 同步点本身吃掉 overlap |
| `dummy_run` 承担一切 | 显存 profile + `torch.compile` + CUDA Graph 捕获 + warmup + EP+DP 空跑 | `dummy_run` 与 `execute_model` 行为漂移 → 一类"profile 时正常、上线才崩"的 bug |

对读代码的人,这解释了一件事:**V1 的 `gpu_model_runner.py` 为什么动不得**——它的复杂度来自耦合,不是来自功能。

---

## 1. Persistent Batch:每请求一条永久行

> 下图把原文那两行 ASCII 展开成完整生命周期:行怎么占、每步怎么 gather、finish 与抢占为什么都退化成「标记空闲」。


<div class="archify-figure">
<iframe src="../../../diagrams/vm-persistent-batch.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="Persistent Batch 生命周期"></iframe>
</div>


### 1.1 机制

MRV2 把"持久状态"和"本步输入"彻底分家:

1. 预分配固定 `max_num_reqs` 行的状态张量(官方:多数平台默认 **1024**;源码里这个默认来自 `ArgUtils.get_batch_defaults`,≥70 GiB 显存且非 A100 的卡上 `max_num_seqs` 默认就是 1024,而 `SchedulerConfig.DEFAULT_MAX_NUM_SEQS=128` 只是 class 级兜底);
2. 每个活跃请求**在整个生命周期内固定占用一行**(直到 finish 或被抢占);
3. **把 preemption 当 completion**:恢复时按"全新请求"重新写入状态,而不是想办法把旧行搬回来。

于是每步只剩两件事:**按本步顺序 gather 出输入张量**(顺序通常由 attention backend 决定),执行,回写。`CachedRequestState` 没有必要存在了。

```text
持久状态(常驻 GPU / UVA)                本步输入(临时)
idx: 0   1   2   3  ... 1023            query_start_loc  [0, 5, 9, ...]
     │   │   │   │                        seq_lens         [..  ..  ..]
     └───┴───┴───┴──  gather(idx_mapping) ──► input_ids / positions / block_tables
                                               (顺序 = 本步调度顺序,可任意)
```

### 1.2 为什么"永久行"是关键的工程取舍

- **代价**:行数上限 = `max_num_seqs` 预分配,空转也占着显存(`all_token_ids` 是 `(max_num_reqs, max_model_len) int32`,源码注释直说"可能是几个 GB");
- **收益**:插入/删除变成"标记一行为空闲",**没有任何跨行搬移**;preemption 的语义被简化成"结束 + 重新开始",省掉一类"部分状态还在、部分已清"的中间态;
- **配套**:正因为 `all_token_ids` 太大,MRV2 给它用 **UVA 而不是 GPU 显存**(`StagedWriteTensor(..., uva_instead_of_gpu=True)`)——GPU kernel 直接读 CPU 侧内存,不复制一份到显存。

---

## 2. Async-First,以及"不用 barrier 也能防竞态"

> 两种写法并列:V1 靠同步点保护同一块内存,MRV2 让读写落在两块内存上——后者从结构上就不可能漏。


<div class="archify-figure">
<iframe src="../../../diagrams/vm-async-snapshot.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="async barrier vs 快照"></iframe>
</div>


### 2.1 问题形状

异步调度的本质:GPU 跑第 `N` 步的同时,CPU 准备第 `N+1` 步。危险在于**同一块内存 CPU 在写、GPU 异步拷贝在读**:

```python
# V1 的隐患写法(文档原文示例)
self.states = torch.zeros(max_num_reqs, dtype=torch.int32, device="cpu", pin_memory=True)

def execute_step(...):
    self.states[req_idx] = new_req.data                  # CPU 写
    states = self.states.to("cuda", non_blocking=True)   # GPU 异步读同一块
```

### 2.2 MRV2 的解法:换缓冲,而不是加锁

```python
# 状态不 pin,写的是自己的私有内存
self.states = torch.zeros(max_num_reqs, dtype=torch.int32, device="cpu", pin_memory=False)

def execute_step(...):
    self.states[req_idx] = new_req.data
    tmp_states = self.states.pin_memory()      # 拷出一份 pinned 快照
    states = tmp_states.to("cuda", non_blocking=True)   # GPU 读快照
```

CPU 只写 `self.states`,GPU 只读 `tmp_states`,**不需要任何显式同步**。代价是每步多一次 CPU 内拷贝(量小),换来的是"永远不会漏掉某块缓冲没被保护"——这比"记得给每块缓冲加 barrier"可靠一个量级。

> **可复用的判断**:CPU↔GPU 竞态用"**双缓冲/快照**"消,而不是用"**同步点**"消;只要数据量小于一次同步的代价,双缓冲总是更优且更难写错。

### 2.3 这条设计直接改了 `max_concurrent_batches`

源码 `VllmConfig.max_concurrent_batches`:

```python
if self.scheduler_config.async_scheduling:
    if self.use_v2_model_runner:
        return pp_size + 1          # MRV2:PP 每级一条 + 一条在飞
    if pp_size <= 1:
        return 2                    # V1:只能 2 条(注释:V1 不完整支持 async + PP)
return pp_size
```

这个数决定 engine 的 `batch_queue` 深度(`vllm/v1/engine/core.py` 用它建 `deque(maxlen=...)`),也决定 UVA 环形缓冲要几层(见 §3.2)——**同一份配置同时约束"能排几个 batch"和"要准备几份缓冲",两边必须对齐**。

---

## 3. StagedWriteTensor + UVA:大状态只搬 diff

### 3.1 机制:五步代替"整张量重拷"

> 数据流图:`stage_write` 只攒 diff,`apply_write` 一次 kernel 落盘,`cu_lens` 的前缀和让一个下标切出任意段。


<div class="archify-figure">
<iframe src="../../../diagrams/vm-staged-write.dataflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="StagedWriteTensor 五步"></iframe>
</div>


block table 这类张量每步**只有几行变化**,整份 H2D 拷贝是纯浪费。`StagedWriteTensor`(`vllm/v1/worker/gpu/buffer_utils.py`)的做法:

1. 基张量常驻 GPU;
2. CPU 侧累积 diff:`stage_write(index, start, x)` / `stage_write_elem(index, x)`,内部只记四个 list:`_staged_write_indices / _starts / _contents / _cu_lens`;
3. `apply_write()` 把 indices/starts/cu_lens 装进 **UVA 环形缓冲**、把 contents 走一次 `async_tensor_h2d`;
4. 起**一个** kernel(`_apply_write_kernel`,`BLOCK_SIZE=1024`)按 `(row, start, cu_len)` 把内容写进 GPU 基张量;
5. 清空 staged 列表。

```python
state = StagedWriteTensor(size=(1024, 1000), dtype=torch.int32, device="cuda")
state.stage_write(index=2, start=3, x=[3, 1, 2])
state.stage_write(index=0, start=1, x=[-1, -2, -5])
state.apply_write()      # 1 次 H2D(数据)+ 3 次小拷贝(元数据)+ 1 个 kernel
```

`cu_lens` 是**前缀和**(累积长度)而不是每段长度——这样 kernel 用一个下标就能切出"第 i 段写哪些元素",避免每段一个 kernel。

`FusedStagedWriter.apply` 更进一步:把**多个** `StagedWriteTensor` 的 staged 写合并成一次 kernel(带 `group_id` 区分目标张量),进一步减少 launch 数——启动开销在 decode 步只有几百 token 时是真实成本。

### 3.2 UVA 环形缓冲:并发深度必须 ≥ 在飞步数

```python
_DEFAULT_MAX_CONCURRENCY = 2   # 注释:必须 >= engine batch_queue_size
def set_default_max_concurrency(n): _DEFAULT_MAX_CONCURRENCY = max(2, n)

class UvaBufferPool:
    def copy_to_uva(self, x):
        self._curr = (self._curr + 1) % self.max_concurrency   # round-robin
        buf = self._uva_bufs[self._curr]
        buf.cpu[:len(x)] = x          # CPU→CPU 写进 pinned+UVA 视图
        return buf.uva[:len(x)]       # 返回 GPU 可直接读的地址
```

**为什么要环形**:第 `N` 步的元数据在 GPU 还在读的时候,第 `N+1` 步就要写下一份;单缓冲会覆盖正在读的内容。所以"并发层数"不是性能旋钮而是**正确性参数**,MRV2 在 worker 初始化时按 `max_concurrent_batches` 反推并 `set_default_max_concurrency()`。

`UvaBackedTensor` 是同一族但语义不同:它是"CPU 是真相、GPU 有视图"的单值张量(`prompt_len` / `prefill_len` / `num_computed_tokens_np` 这类标量的 per-request 镜像),改值后 `copy_to_uva(n)` 刷一次。

| 结构 | 真相在哪 | 适合 | 源码里的实例 |
|---|---|---|---|
| `UvaBackedTensor` | CPU | 小、每步可能整排刷新 | `prompt_len`、`prefill_len` |
| `StagedWriteTensor` | GPU(UVA 可选) | 大、每步只改几行几列 | `all_token_ids`(UVA)、`total_len`、`num_computed_tokens`、`logit_bias.allowed_token_ids`、`logprob.token_ids` |

---

## 4. GPU-native 输入准备与 Triton 采样器

### 4.1 输入元数据在 GPU 上算

`input_ids` / `positions` / `query_start_loc` / `seq_lens` 改成 Triton kernel 生成。两个动机:

1. **异步正确性**:投机解码时某些值只有 GPU 知道(draft 的结果),CPU 想准备就得同步等待——那就破坏了 async-first;
2. **省 CPU**:Python 构造大张量是纯开销,GPU 侧算几乎免费。

### 4.2 Gumbel-max 采样:一个 kernel 顶替 softmax + multinomial

> kernel 内三步:按 (seed,pos) 定位随机流 → 造 Gumbel 噪声 → 取 argmax;右侧是原文点出的三个细节。


<div class="archify-figure">
<iframe src="../../../diagrams/vm-gumbel-sampling.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="Gumbel-max 采样"></iframe>
</div>


`sample/gumbel.py` 的核心:

```python
@triton.jit
def gumbel_noised_argmax(logits, keys, mask, seed, pos, temp, USE_FP64):
    if temp != 0.0:
        gumbel_seed = tl.randint(seed, pos)
        u = tl_rand32(gumbel_seed, keys, includes_zero=False)
        gumbel_noise = -tl.log(-tldevice.log1p(-u))    # log1p:u→0 时 fp32 仍可分辨
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))
    return tl.max(logits, axis=0, return_indices=True)
```

三个细节值得抄:

- **数学等价**:`argmax(logits + Gumbel(0,1))` 与按 `softmax(logits)` 采样同分布 —— 于是**不需要物化 softmax、不需要 multinomial、不需要 CPU-GPU 同步**;
- **`keys` 索引噪声、`(seed, pos)` 定位流**:同一 token 在同一请求的同一步拿到同一份噪声,所以 **draft 与验证阶段能对同一随机流达成一致**(投机解码接受判定的前提);
- **`log1p` 而非 `log(1-u)`**:`u→0` 时 Gumbel 尾部是决定胜负的地方,`log(1-u)` 在 fp32 下会退化成 0 → 噪声 = inf;`log1p(-u)` 保住精度。fp64 可选但"H100/Ada/Blackwell 上吞吐是 1/32–1/64,而经验上与 fp32 无法区分",所以默认 fp32。

### 4.3 top-k / prompt logprob 的两种省显存

- **top-k**:V1 先算**全词表** logprob 再取 top-k;MRV2 先用 `torch.topk` 拿 top-k token id,再用 `_topk_log_softmax_kernel` **只对选中的 id** 算 log-softmax(`tl.load(row_ptr + topk_ids)`)。峰值显存从 `V` 降到 `k`;
- **prompt logprob**:支持在**单个 prompt 内部**再切 chunk,长 prompt 不再一次性冲出一个大缓冲;
- **投机解码**:不再把 per-request 采样状态**展开**成 per-logit 形状,而是 kernel 内用 `idx_mapping` 间接寻址(`expanded_idx_mapping`)——这是"复杂采样参数 + logits processor 与投机解码共存"能实现的原因。

---

## 5. 显式 CUDA Graph:一个 manager、一个描述符键

### 5.1 `CudaGraphManager`(`vllm/v1/worker/gpu/cudagraph_utils.py`)

> 分派流程图:查表 → 逐个兼容性判定 → 命中重放 / 显式回退。`cg_mode=NONE` 是正常路径,不是异常。


<div class="archify-figure">
<iframe src="../../../diagrams/vm-cudagraph-dispatch.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="CudaGraphManager.dispatch"></iframe>
</div>


```python
self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
self._candidates: dict[tuple[int, int], list[BatchExecutionDescriptor]] = {}

def dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None):
    key = (num_tokens, self._resolve_effective_loras(num_active_loras))
    if self._graphs_captured and num_tokens > 0 and key in self._candidates:
        for desc in self._candidates[key]:
            if _is_compatible(desc, num_reqs, num_tokens, uniform_token_count, ...):
                return desc
    return BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, ...)   # 明确回退,不是静默
```

设计要点:**图的生命周期、捕获条件、分派规则全在一个类里**;`dispatch` 找不到匹配就返回 `cg_mode=NONE` 的显式描述符(而不是"以为命中了"),没命中可以从返回的 desc 直接看出来。LoRA 场景预先算好 `num_active_loras → captured case` 的映射表,把每次调用的 bisect 换成一次 dict 查表。

`ModelCudaGraphManager` 在其上加模型相关状态(捕获 `hidden_states` / `aux_hidden_states` / `intermediate_tensors`),`profile_cudagraph_memory()` 则用"只捕获最大的 N 个描述符 + 按内存增量外推"的方式**在不捕获全部图的前提下**估出显存——这正是把 `dummy_run` 的职责拆出来的结果。

### 5.2 融多步 draft 解码进一张图(最值得学的部分)

自回归投机解码在一个调度步内要跑 γ 个**有依赖**的 draft step。V1 的等价做法:每步一张图、replay γ 次。MRV2 的融合路径:

- **post-prefill 的所有 draft step 捕获进一张全图**;
- attention metadata **在循环前建一次**;步间变化的张量保持**地址稳定**、原地更新;
- 需要派生态的 backend 必须实现 `AttentionMetadataBuilder.update_draft_decode_metadata()`——**这个 hook 在捕获期运行,只有它发出的 GPU 操作会被记录进图,Python 主体在 replay 时不再执行**;因此实现必须只用 capture-safe 操作、并把所有要重放的状态放持久存储;
- 门禁:每个 draft attention group 都要声明 `supports_draft_decode_metadata_update`,否则**回退**成"每步重建 metadata";position 不变的 draft 模型不需要这个更新。

> **这条是本篇最可迁移的经验**:CUDA Graph 的边界不在"能不能捕获",而在"**捕获期执行一次的 Python 逻辑,replay 时不会重跑**"。凡是 metadata 里有"每次都要根据新输入重算"的部分,都必须显式改成"图内算子"或"per-replay 原地更新"。

---

## 6. 版本与开关:v0.28 基线 → v0.32 现状

| 项 | 库内旧基线(2026-09) | 当前(2026-10-22 一手核实) |
|---|---|---|
| 源码快照 | `b74ac0bc6` = `v0.28.1rc0-40` | main `2d40c0609` = `v0.32.1rc0-353`;PyPI 最新发布 **v0.32.0** |
| Model Runner | `vllm/v1/worker/gpu_model_runner.py`(V1) | **默认 `vllm/v1/worker/gpu/model_runner.py`(MRV2)**;V1 仅回退路径 |
| 开关 | 无 | `VLLM_USE_V2_MODEL_RUNNER`(bool,`None` = 走自动判定) |

### 6.1 自动判定逻辑(`VllmConfig.use_v2_model_runner`,源码 `config/vllm.py`)

```text
env 显式给了 → 用它
否则:
  ROCm 且 architecture ∈ ROCM_DEFAULT_MRV1_ARCHITECTURES → False(性能原因,日志会提示)
  没有 Triton(HAS_TRITON=False)                        → False
  命中任一"尚不支持的特性"                              → False(日志列出具体哪几项)
  否则                                                  → True
```

### 6.2 会把你踢回 V1 的特性清单(实测源码,部署前必查)

| 类别 | 触发条件 |
|---|---|
| 编译 | `CompilationMode.STOCK_TORCH_COMPILE`(没用 vLLM 的 compile 后端) |
| 并行 | 开 sequence parallelism 且 TP>1;`external_launcher` + PP>1;**dual batch overlap(enable_dbo)**;**elastic EP** |
| 投机解码 | method ∈ {`ngram`, `ngram_gpu`};或 method 不在 {`eagle`, `eagle3`, `mtp`, `dflash`, `dspark`, `extract_hidden_states`};EAGLE 的 `parallel_drafting`(dflash/dspark 除外);EAGLE3 + PP |
| 自定义 | 配了 `logits_processors`(含通过 entry point `vllm.logits_processors` 注册的插件) |
| KV | `kv_sharing_fast_prefill` |
| 平台 | 无 Triton;特定 ROCm 模型架构 |

**实践含义**:这些条件**任意一条命中就整机回退 V1**,不是"降级某个功能"。所以调优前先看启动日志里是 `Using V2 Model Runner` 还是 `Model Runner V2 does not yet support ...; using the V1 model runner instead`——**日志直接告诉你在哪条跑**,别猜。

---

## 7. 实践 checklist

**上线前**

1. **先确认跑的是哪个 runner**:grep 启动日志 `Using V2 Model Runner` / `does not yet support`;这决定了后面 `gpu_model_runner.py` 还是 `gpu/model_runner.py` 里的行号对得上。
2. **do not trust "文档说不完整就以为没开"**:官方设计文档那句"not feature-complete"是 2026-03 的措辞,而代码里它已是默认;同理**反向**——别看默认开了就假设自己的特性被支持,查 §6.2 那张表。
3. **`max_num_seqs` 的显存账要按 MRV2 重算**:`all_token_ids` 是 `(max_num_seqs, max_model_len)` int32,虽走 UVA 不占显存,但**占 pinned/UVA 内存**,几个 GB 级;`--max-model-len` 与 `--max-num-seqs` 一起翻倍时先算这笔账。
4. **并发深度与 UVA 层数必须自洽**:`max_concurrent_batches`(async+MRV2 = `pp_size+1`)决定 `batch_queue` 深度,也决定 UVA 环形缓冲最小层数;自己改并发相关配置后,若出现"元数据串味"(block table 偶发错行),先查这里。

**采样与 logprob**

5. **温度 0 走纯 argmax**(kernel 内 `if temp == 0.0` 直接跳过噪声),不要用极小温度代替 0——浮点上仍有随机性且更慢。
6. **要 bitwise 复现就固定 `(seed, pos)` 语义**:Gumbel 噪声由 seed+位置决定,**同一请求重放同一步**能拿到同一噪声;跨请求/跨 batch 组合变化不保证,别拿"两次跑同样输入结果应一致"当断言写进测试。
7. **fp64 Gumbel 只在你已知需要时开**:H100/Ada/Blackwell 上吞吐掉 1/32–1/64,官方判断是经验上不可区分。

**CUDA Graph**

8. **融合 draft 解码前先查 backend 是否实现了 `update_draft_decode_metadata()` 与 `supports_draft_decode_metadata_update`**:没实现就得回退"每步重建 metadata",否则图内状态不更新 = 结果错但不报错。
9. **捕获期逻辑只跑一次**:任何"根据本步输入重算"的 Python 逻辑,要么改成图内算子,要么做成稳定地址 + 原地更新。
10. **`dispatch` 返回 `cg_mode=NONE` 是正常回退**,不是 bug;怀疑图没命中时打印返回的 `BatchExecutionDescriptor`,比读代码猜快。

**读源码**

11. **顺序**:`config/vllm.py::use_v2_model_runner`(是否启用/是否回退)→ `v1/worker/gpu_worker.py:395-440`(构造哪个 runner)→ `gpu/model_runner.py`(执行主循环)→ `states.py` + `buffer_utils.py`(状态怎么存/怎么写)→ `sample/`(`gumbel.py`/`logprob.py`)→ `cudagraph_utils.py`(图)。
12. **`gpu/model_runner.py` 头部有作者写的纪律**:这个文件被所有模型共享(文本/多模态、生成/embedding),"be paranoid about adding lines",特性逻辑必须在别处(`model_states/`、`mm/`、`spec_decode/` 等子目录)。想改它之前先读那段注释——这是本库其它 vLLM 篇里"59 行 shim"同一种设计哲学。

---

## 相关条目

- [V1 引擎架构](v1-architecture.md)——EngineCore/Worker/GPUModelRunner 三层与调度执行解耦;本篇是它执行层部分的"下一代"
- [连续批处理与统一 Token 预算](continuous-batching.md)——调度侧;MRV2 改的是**执行侧**(拿到 batch 之后怎么把张量喂给模型)
- [投机解码](speculative-decoding.md)——本篇 §5.2 的"多步 draft 融一张图"是它的执行层落地
- [PagedAttention 与分页 KV Cache](paged-attention.md)——block table 正是 `StagedWriteTensor` 的主要用户
- [生产部署与观测清单](../practices/production-serving.md)、[注意力后端选型](../practices/attention-backend-selection.md)

## 溯源

- **设计文档(主源)**:*Model Runner V2 Design Document*,vLLM 官方文档 https://docs.vllm.ai/en/latest/design/model_runner_v2/ ;仓库内同源文件 `docs/design/model_runner_v2.md`(206 行,首次提交 #35819,2026-03-02)。2026-10-23 实测网页与仓库文件**逐字一致**,故本篇章节结构完全按该文档的 9 节组织。
- **源码(判定"当前实际行为"的唯一依据,2026-10-22 main `2d40c0609`)**:
  - `vllm/config/vllm.py`:`use_v2_model_runner`(环境变量覆盖 + 自动判定)、`_get_v2_model_runner_unsupported_features`(回退特性清单)、`max_concurrent_batches`
  - `vllm/v1/worker/gpu_worker.py`:V1/MRV2 构造分派、`_num_workspace_lanes`
  - `vllm/v1/worker/gpu/model_runner.py`(2112 行,头部有共享-文件的纪律注释)、`states.py`(RequestState)、`buffer_utils.py`(`UvaBuffer`/`UvaBufferPool`/`UvaBackedTensor`/`StagedWriteTensor`/`FusedStagedWriter`)、`input_batch.py`(InputBatch)
  - `vllm/v1/worker/gpu/cudagraph_utils.py`(`CudaGraphManager`/`ModelCudaGraphManager`/`dispatch`/`profile_cudagraph_memory`)、`sample/gumbel.py`、`sample/logprob.py`、`sample/sampler.py`
  - `vllm/config/scheduler.py`(`DEFAULT_MAX_NUM_SEQS = 128`)、`vllm/engine/arg_utils.py::get_batch_defaults`(按显存给出的 1024/256 默认)
- **版本事实**:PyPI `vllm` 最新发布 **0.32.0**(上传 2026-10-18,2026-10-23 实查);main 描述为 `v0.32.1rc0-353-g2d40c0609`(2026-10-22);GitHub release 页 v0.32.0 的 published_at 为 2026-10-18。
- **交叉**:库内旧基线 `b74ac0bc6`(v0.28.1rc0-40)的 V1 结论仍然成立,但**只适用于回退路径**;凡涉及 `gpu_model_runner.py` 的 path:line,在 MRV2 默认开启的部署上需重新定位。
