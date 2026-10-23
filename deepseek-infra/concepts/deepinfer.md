---
title: DeepInfer — DeepSeek 生产推理框架(V4.1-Flash,EPD 分离 + 零开销调度)
type: concept
tags: [deepinfer, deepseek, inference, serving, v4.1, dflash, yoco, swa, eplb, wide-ep, zero-overhead-scheduling, ascend, nvidia]
date: 2026-10
source:
  - "code: deepseek-ai/DeepInfer (main, HEAD 9bd9cf243f, 2026-10-21;此前 774ce244b8/2026-10-20) | https://github.com/deepseek-ai/DeepInfer | 2026-10-21 | ↑primary"
  - "code: DeepInfer scripts/build_vendor.sh + Makefile + requirements-nvidia.txt(vendored pin 与构建链) | https://github.com/deepseek-ai/DeepInfer/blob/main/scripts/build_vendor.sh | 2026-10-21"
  - "code: DeepInfer PR #157(Bump DeepEP to f6aa38fe / drop SSH->HTTPS rewrite) | https://github.com/deepseek-ai/DeepInfer/pull/157 | 2026-10-21"
  - "code: deepseek-ai/DeepGEMM(上游 main 5081f8513a,含 #478;DeepInfer pin 仅到 1fc9c66975) | https://github.com/deepseek-ai/DeepGEMM | 2026-10-21"
  - "code: DeepInfer doc/inference_deployment_qa.md(官方部署 Q&A) | https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md | 2026-10-20"
  - "code: DeepInfer examples/README.md(benchmark 指标口径) | https://github.com/deepseek-ai/DeepInfer/blob/main/examples/README.md | 2026-10-20"
  - "code: DeepSeek-V4.1-Flash-1017(默认 checkpoint) | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 | 2026-10-20"
---

# DeepInfer — DeepSeek 生产推理框架

## TL;DR

**DeepInfer 是 DeepSeek 自家大规模生产推理框架的首次开源**(2026-10-20),服务 DeepSeek-V4.1-Flash,同时支持 NVIDIA SM100 GPU 与昇腾 950 NPU。README Acknowledgment 明写它**最初主要基于 vLLM 构建**,另受 SGLang / TensorRT-LLM 启发——但它不是通用 serving 框架的 fork 维护版,而是围绕 DeepSeek 生产条件(EPD 分离、Wide-EP、低精度 KV、多厂商硬件)深度改造的**单模型引擎**:开源版只支持 V4.1-Flash 一个模型,且 KV cache 存储客户端与跨 DP 负载均衡器**不在开源范围内**(README「Required for a complete inference service (not included)」)。

独有机制三件事:
1. **零开销调度(ZOS)**:decode 步 N 的 post-process 推迟到步 N+1 的 forward 之后执行,CPU 调度与 GPU 计算全程重叠;代价是 CPU 侧 seqlens 滞后一步,必须在 GPU 上用上一步的采样结果修正(`_correct_seqlens_qk`)。
2. **DSpark confidence-scheduled varlen 投机解码**:DFlash 草稿头出 5 个 draft token,Confidence Head 预测每个 token 的接受率,按边际收益在 GPU 上动态分配验证长度——实测比定长 5 少验证约 30% token,60 并发生产点吞吐 +4–7%。
3. **SWA bounded replay + YoCo 截断**:长上下文命中缓存后只重算末尾 128 token、query 可见域收窄,重算成本变成与上下文长度无关的常数;decoder 层 prefill 计算近乎减半。

官方口径(单源,出自部署 Q&A):单集群日服务 **155 万亿 token**,谷时定价下 **$38/GPU-hour** 整体收益。

---

## 1. 定位与边界:开源了什么,没开源什么

**开源的**:离线推理引擎全栈——C++ scheduler、Python engine/worker、模型 runtime(V4.1 的 YoCo/DSA/mHC/Engram 模块)、EPLB、投机解码、结构化输出、benchmark 工具链。支持特性(README):优化内核与 sampler runtime、Attention DP / balanced CP、MoE EP + EPLB、DSpark 与 confidence-scheduled verification、SWA bounded replay、异步调度、batch-invariant 推理、确定性采样、结构化输出。

**没开源的**(README 列在「required for a complete inference service, not included」;Q&A 措辞是 *not yet fully open-sourced*,并承诺「Inference code for future DeepSeek models, along with subsequent performance work, will continue to be open-sourced」——所以这条边界是**本 release 的边界,不是永久边界,可能前移**):
- **KV cache 存储系统**的客户端逻辑(kvcache-store、Entrypoint 的命中匹配与内存目录);
- **VE/Prefill/Decode 分离编排**(引擎侧功能大部分已实现——`EngineType` 有 PROMPT/COMPLETION 专用引擎——但生产编排层未放出);
- **跨 DP 副本的 prefill/decode 负载均衡器**。

Q&A 解释:KV cache 管理与负载均衡与内部基础设施深度耦合,设计细节见官方[知乎博客](https://zhuanlan.zhihu.com/p/2095560295930508953)。同日配套开源了 [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)(Rust 实现的 token 编解码/API 协议转换组件,与官方 API 行为对齐)。

**单模型边界**:只支持 DeepSeek-V4.1-Flash,没有 vLLM 那样的通用模型注册表;默认 checkpoint `deepseek-ai/DeepSeek-V4.1-Flash-1017` 首次运行自动下载到 HF cache,可用 `DEEPINFER_HF_MODEL_DIR` 指向本地副本。模型架构侧(V4.1 的 YoCo/CED/DSA)见 [../../model-architecture/concepts/open-model-landscape-2025-2026.md](../../model-architecture/concepts/open-model-landscape-2025-2026.md)。

---

## 2. 部署形态:EPD 分离 + Wide-EP(以下部署数字均出自官方部署 Q&A,单源、未经第三方复核)

> 下图把原文的 ASCII 拓扑重画成可交互版本:三段异构算力(VE / Prefill 小 EP / Decode Wide-EP)与两类 KV 存储的分工,以及「哪个组件没开源」。

<div class="archify-figure">
<iframe src="../../../diagrams/di-epd-topology.architecture.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="DeepInfer 部署形态:EPD 分离 + Wide-EP"></iframe>
</div>


```
        请求(API 协议)
             │
      ┌──────▼──────┐   deepseek-recipe:协议→token(分段并行编码,防特殊 token 注入)
      │ Entrypoint  │   缓存命中匹配(hash-chain 前缀匹配,只算 metadata)※未开源
      └──┬───┬───┬──┘
         │   │   │        ┌────────────────────────────┐
   ┌─────▼┐ ┌▼─────┐      │  KVCache Storage(3FS 集群)│
   │  VE  │ │Prefill│◄────►│  分页 block KV:10+2 EC     │
   │单 GPU│ │EP2–8 │ KV   │  ~100 PiB,TTL 3d/7d        │
   └──────┘ │DP2–8 │      │  GC worker 按 LRU 回收      │
            └──┬───┘      └────────────▲───────────────┘
               │ P→D 传输(block KV + SWA state 同路径)
            ┌──▼────┐   host 内存:SWA KV(单副本,400G VLAN 跨机拉取)
            │Decode │
            │EP64–128│  Wide-EP:每 GPU 只放少量 expert
            │DP64–128│
            └───────┘
```

- **并行选择**:Prefill 用 **EP2–8 / DP2–8**,原则是「HBM 装得下的最小 EP/DP」;Decode 用 **EP64–128 / DP64–128(Wide-EP)**,原则是「现有 GPU 能撑的最大 EP/DP」;VE 单 GPU 即可、推荐便宜卡;**生产不用 TP/CP**(代码支持 CP,benchmark 有 `--cp`,但 Q&A 明写 "TP and CP are not used for now")。attention 沿用 V3/R1 系统的 DP(#DP = #EP)。
- **Wide-EP 为什么值得**:decode 是访存题,EP 越大每 GPU 常驻的 expert 权重越少,per-step MoE 访存越低;配合 EPLB 把热点 expert 复制摊平,TPS 有效提升。Q&A 给了 EP8 vs EP64 的 decode pareto 图(`doc/images/wide_ep_decode_pareto.png`,图内数字无法从文本提取,此处只记定性结论)。
- **最小部署单元**:1P+1D 可以跑,但 decode 推荐 EP64 起,所以**最小高效单元 = 64 卡 prefill + 64 卡 decode**;P:D GPU 配比约 **1:1**。
- **跨数据中心**:V4.1 KV 压缩率高,P/D 跨 DC 技术可行,预算**每 8 GPU ~400 Gbps** 互联;DeepSeek 实际部署在相邻的不同数据中心。
- **存储配比**:每台 KV 存储服务器推荐 20 × 30.73 TB SSD、≥40 GB/s 读 + ≥5 GB/s 写、容量够存 24 小时;在线 GPU 每卡约 **14.4 TB 逻辑容量**(物理需求更大)。

---

## 3. 引擎实现分层:C++ Scheduler + Python Engine + 多进程 Worker

> 下面的分层图对应原文那段调用栈:决策为什么在 C++、Python 侧持有什么、以及每 GPU 一个 worker 进程的生死绑定关系。

<div class="archify-figure">
<iframe src="../../../diagrams/di-engine-layers.architecture.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="引擎实现分层"></iframe>
</div>


```
Entrypoint(生产侧未开源;引擎基类留虚函数 process_output / pe_do_save_kvcache / send_processed_prompts)
   │
LLMEngine(deepinfer/engine/llm_engine.py:130)──────── step() 主编排
   │   scheduler.schedule()  ──► C++ Scheduler(csrc/engine/scheduler.hpp,pybind11)
   │   DeferredRelease(llm_engine.py:67)── 池化对象延后一轮释放
   ▼
Worker(deepinfer/worker/worker.py:51)── 每 GPU 一进程;slave worker 循环
   │   set_pdeathsig(SIGKILL) 后阻塞在 cp_broadcast_object(worker/base.py:130–133)
   ▼   ——父进程死,子进程被内核直接 SIGKILL,不留僵尸
ModelRunner(execute_main_model / execute_sample / post_process)
   ▼
CUDA Graph runner + TileLang/DeepGEMM/FlashMLA/DeepEP 内核
```

- **EngineType 三种**(`deepinfer/engine/engine_type.py` 与 `csrc/engine/scheduler.hpp:39–43` 要求保持一致):`NORMAL`(prompt+decode 混跑,单机/未分离部署)、`PROMPT`(只做 prefill,即 PE)、`COMPLETION`(只做 decode,即 CE)。COMPLETION 没有 prefill 通道、无法重算被抢占请求,启动时硬拦截:`TORCH_CHECK(engine_type != COMPLETION || stop_preempted_request)`(scheduler.hpp:101–104)。
- **C++/Python 边界怎么传**:C++ 侧持有 `Sequence` 对象池,Python 拿到的是 `ObjectPoolPyObject<SequenceMetadata>` 包装;`schedule()` 返回 `(seq_metadata_list, SchedulerOutputs)`,其中 `SchedulerOutputs` 的 `clone_states/clone_blocks/zero_states` 是对 C++ 内部 vector 的 **const 引用**,Python 必须在下一次 schedule 前消费完(C++ 侧会 clear 复用,scheduler.hpp:280–285 注释)。请求输出以 **uint64 指针数组**(`np.ndarray[np.uint64]`)跨边界传递,由 `RequestOutputPool` 池化。
- **DeferredRelease**(llm_engine.py:67–87):step 产出的 SequenceMetadata / RequestOutput / 临时 pinned 内存不立即释放,而是 collect 起来,在**下一步的 `pre_sampler_sync`**(llm_engine.py:830,sample kernel 启动后、同步前)统一 release——释放动作与 GPU 计算 overlap,且保证调用方在下一 step 前还能读。这就是 `step()` 文档里「RequestOutput 会在下一次 step() 自动释放,要留存必须 pool.clone」的原因。
- **step() 主流程**(llm_engine.py:860):schedule(C++ 投票/组批)→ clone/zero states(YoCo 抢占重算相关)→ SO 簿记校验 → decode 步的 KV 落盘 copy → 多模态 fetch → `_execute_model_async_step()`(llm_engine.py:961)广播到所有 worker 执行。

---

## 4. NORMAL 引擎:prefill/decode 同型投票(ALL 点 + ANY 点 + 两轮 all-reduce MIN)

> 这张流程图把 ALL 点 / ANY 点的两种 MIN 极性、以及「第二轮确认」为什么必需画在一起——后者是原文指出死循环的那个坑。

<div class="archify-figure">
<iframe src="../../../diagrams/di-normal-voting.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="NORMAL 引擎同型投票"></iframe>
</div>


NORMAL 引擎混跑 prompt 和 decode,但 attention DP 要求**全组每一步同型**(要么都 prompt、要么都 decode),靠 DP 组内投票统一。源码(`csrc/engine/scheduler.hpp:295–343`,节选):

```cpp
// 两类尝试点:
//   ALL 点(每 prefill_attempt_interval 个 decode step):全员都有 prefill 活才做,
//     防止一个 rank 干活、其余 rank 陪跑空 prompt step。
//   ANY 点(每 prefill_any_attempt_interval 个 decode step):一个 rank 有活就一起做,
//     给孤立 rank 兜底。两类点重合时按 ANY 算。
bool is_any_point = (decode_step != 0) && (decode_step % prefill_any_attempt_interval == 0);
bool is_all_point = (decode_step % prefill_attempt_interval == 0);
if (!is_any_point && !is_all_point) return _schedule_decode(acquire);
bool want = (get_gpu_cache_usage() <= GPU_CACHE_PREFILL_WATERMARK) && _has_prefill_waiting();
if (is_any_point)
    do_prefill = (_dp_allreduce(want ? 0 : 1, c10d::ReduceOp::MIN) == 0); // ANY:任一 rank 想做即触发
else
    do_prefill = (_dp_allreduce(want ? 1 : 0, c10d::ReduceOp::MIN) == 1); // ALL:全员想做才触发
...
// 第二轮投票:确认这轮真有 rank 排出了 prefill。want 只看了 waiting 非空,
// 但 _schedule_prefill 仍可能空手(slot 满、KV 块不够)。
bool any_scheduled = (_dp_allreduce(outputs ? 0 : 1, c10d::ReduceOp::MIN) == 0);
if (!any_scheduled) return _schedule_decode(acquire);
```

机制要点:
- **两类投票点、同一个 MIN all-reduce 原语**:ANY 点把「想做」编码成 0,MIN==0 即任一 rank 想做;ALL 点把「想做」编码成 1,MIN==1 即全员想做。默认间隔:ALL 点 `DEEPINFER_PREFILL_ATTEMPT_INTERVAL=70`、ANY 点 `DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL=100`(environ.py:174–178)——孤立 rank 的 prefill 最多被饿 100 个 decode step。
- **GPU cache watermark**:`want` 还要求 GPU KV cache 占用 ≤ `GPU_CACHE_PREFILL_WATERMARK = 0.9`(scheduler.hpp:80)——cache 快满时 prefill 会挤爆 decode,先让 decode 消化。
- **第二轮投票为什么必需**:第一轮 `want` 只检查 waiting 队列非空,但 `_schedule_prefill` 可能因 slot 满/KV 块不够而空手返回。若全组都空手还硬做空 prompt step,prompt step 会把 `decode_step` 清零 → 下一步又落在 ALL 点 → 又投票又空手——「running 满 + waiting 非空」时**死循环**。第二轮 MIN 确认「确有 rank 排出了活」,全组空手就转 decode;部分 rank 有活时,没活的 rank 陪跑空 prompt step 保持步型一致(scheduler.hpp:326–341 注释写明了这条因果)。
- `decode_step` 各 rank 一致,所以投票次数与语义全组天然对齐,不会卡死在 allreduce。

---

## 5. 零开销调度 ZOS(核心机制)

### 5.1 气泡从哪来

> 同一时间轴上的对照:标注为「同步调度基线」的两条 span 就是 GPU 空转的气泡,其余 CPU span 是 ZOS 把 post-process 整体推迟一步后挪进 GPU 计算窗口的那部分工作。

<div class="archify-figure">
<iframe src="../../../diagrams/di-zos-overlap.waterfall.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="零开销调度 ZOS 的 CPU/GPU 重叠"></iframe>
</div>


同步调度下 decode 步 N 的时间线是 `schedule(N) → forward(N) → sample(N) → [GPU-CPU 同步] → post-process(N)`,其中 schedule 和 post-process 是纯 CPU 逻辑,且 post-process 前必须同步拿到 sample 结果 → **GPU 在 post-process(N) 和 schedule(N+1) 期间空转**。ZOS 的目标是消掉这段空闲:GPU 跑 `forward(N) → sample(N)` 的同时,CPU 并发跑 `schedule(N+1) → launch_forward(N+1) → post-process(N−1) → launch_sample(N+1)`(Q&A 原文口径,并注明沿用 SGLang / NanoFlow 的思路)。

### 5.2 「默认开启」的真实条件(以源码为准)

README 只写 "Enabled by default"。真实逻辑在 `EngineArgs.__post_init__`(deepinfer/engine/arg_utils.py:48–55):

```python
os.environ.setdefault(
    "DEEPINFER_ZERO_OVERHEAD_SCHEDULING",
    "1" if self.engine_type.default_zero_overhead_scheduling else "0",
)
environ.rearm("DEEPINFER_ZERO_OVERHEAD_SCHEDULING")
```

而 `default_zero_overhead_scheduling`(engine_type.py:15–17)= `engine_type in (NORMAL, COMPLETION)`。即:**NORMAL 与 COMPLETION(decode)引擎默认开,PROMPT(prefill 专用)引擎默认关**;`setdefault` 语义意味着显式设过环境变量则以环境变量为准。

### 5.3 Worker 侧四阶段与「post-process 推迟一步」

`execute_model_step`(deepinfer/worker/worker.py:89–189)每步固定四阶段:

```python
# 1. forward curr step(driver worker 先做 seqlens 修正,见 5.4)
correction = self._correct_seqlens_qk(seq_metadata_list, is_prompt, total_q)
main_model_output = self.model_runner.execute_main_model(..., metadata_zos=metadata_zos, ...)
# 2. post process prev step —— 与第 1 步已 launch 的 GPU forward 重叠
request_outputs = self._post_process_step(self.prev_step_state)
# 3. sample curr step(先用 D2H 回来的 seqlens_q 修正 CPU 侧元数据)
if correction.d2h_evt is not None: correction.d2h_evt.synchronize()
scheduler_outputs.update_metadata_for_sampling_zos(...)
sampler_output = self.model_runner.execute_sample(...)
# 4. prompt 步或 non-ZOS 当场 post process;ZOS decode 步留给下一步
if is_prompt or not envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING:      # worker.py:183
    request_outputs = np.concatenate([request_outputs, self._post_process_step(curr_step_state)])
    self.prev_step_state = None
else:
    self.prev_step_state = curr_step_state
```

`prev_step_state`(`WorkerStepState`,deepinfer/worker/zos_status.py:10–20)持有上一步的 seq_metadata / scheduler_outputs / sampler_output / sample_end_evt;`_post_process_step` 先 `sample_end_evt.synchronize()` 再做 CPU 侧 token 处理与 engine 回调。**prompt 步永远当场处理**(prefill 没有可重叠的连续 decode 流)。收尾时若没有在跑的请求,`flush_inflight_reqs()`(worker.py:274–283,引擎侧 llm_engine.py:977)把还挂在 `prev_step_state` 里的最后一步冲刷掉——offline 入口正是「还有 unfinished 就 step(),否则 flush」(entrypoints/llm.py:178),不然最后一步的输出会丢。

### 5.4 滞后一步的 seqlens:必须在 GPU 上修正

> 修正链路的数据流:remap 与滞后长度从 C++ 到 GPU,两个 TileLang kernel 就地改长度,只有采样要用的 q 长度 D2H 回读。

<div class="archify-figure">
<iframe src="../../../diagrams/di-zos-seqlens-fix.dataflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="滞后 seqlens 的 GPU 侧修正"></iframe>
</div>


ZOS 的根本矛盾:`schedule(N+1)` 运行时,**步 N 的采样结果还在 GPU 上没回 CPU**,所以 CPU 侧算出的 seqlens_q/k 是「假设上一步什么都没产出」的滞后值。修正链路(`_correct_seqlens_qk`,worker.py:191–245):

1. C++ `prepare_zos_inputs`(csrc/engine/zos.cpp:142)对比当前 batch 与上一 batch,生成 pinned 的 `remap`(remap[i]=j 表示当前第 i 条对应上一批第 j 条,新请求为 −1)与初版 seqlens_q/k;无交集直接返回,不修正。
2. H2D 后两个 TileLang kernel 在 GPU 上就地修正:
   - `compute_seqlens_q_by_confidence_rates`(仅 `DEEPINFER_ENABLE_MTP_VARLEN=1`):按上一步 confidence 率把 total_q 预算重新分给各请求(见 §6);
   - `compute_seqlens_k_zos`(nextn>0):`seqlens_k[i] += seqlens_q[i]; seqlens_k[i] -= nextn − prev_num_accepted[remap[i]]`——**仅当 `remap[i] = j ≠ −1`**(即该请求上一步也在跑)才减;上一步投机了 nextn 个 token、实际只接受了 num_accepted 个,K 长度要按差值回退。新请求(`remap=−1`)和 placeholder 行不做回退(`compute_seqlens_k_zos.py:22–23` 的 docstring 与 `:45–52` 的参考实现都带这个 mask)。
3. varlen 模式下修正后的 seqlens_q **D2H 回写 pinned 副本**并记 event;第 3 阶段 sample 前 CPU 同步该 event,用真实长度更新采样元数据(worker.py:147–152)。
4. 上一步的 `sampled_token_ids / predict_token_ids / num_accepted_tokens` 以 GPU tensor 形式装进 `MetadataZOS`(zos_status.py:23–40),下一步 forward 的输入 gather 直接从 GPU 读——**新 token 全程不经过 CPU 往返**,这是 ZOS 能省掉同步的前提。

**placeholder 从哪来(与 CUDA graph 分桶咬合)**:修正链路的输入不是「本步真实请求数」,而是 `n_seq_with_padding`。decode 且开 CUDA graph 时(`worker.py:203–207`):

```python
cuda_graph_batch_size, _ = self.model_runner.cuda_graph_runner.select(total_q)  # 按 total_q 选最近的 graph 桶
n_seq_with_padding = n_seq + cuda_graph_batch_size - total_q                    # 差额用 placeholder 补齐
```

即 CPU 预算先定 `total_q`(§6 的 confidence 排序)→ 选 graph 桶 → 差额补 placeholder 行。`prepare_zos_inputs` 对 padding 段给 `remap=−1`、`seqlens_q=1`(zos.cpp:190–194),而修正 kernel 只在 `remap≠−1` 的行上回退 `seqlens_k`,所以 placeholder 行天然免修正。这也解释了为什么 `DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL`(默认 8)会间接影响 decode 效率:桶粒度越粗,placeholder 陪跑行越多。

### 5.5 位置余量:runtime_max_model_len

异步调度下,在途请求的 position 会超前于 CPU 已确认的位置。`deepinfer/config.py:11–16`:

```python
def _get_runtime_max_model_len(max_model_len: int) -> int:
    nextn = envs.DEEPINFER_USE_NEXTN_DECODE
    scheduler_overhead = nextn + (nextn + 1 if envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING else 0)
    draft_overhead = nextn
    return max_model_len + scheduler_overhead + draft_overhead
```

nextn=5、ZOS 开:额外 5+6+5=16 个位置。KV block 表按 runtime_max_model_len 分配,否则 ZOS 下最后几个位置会越界。

---

## 6. 投机解码:DFlash 草稿 + DSpark confidence-scheduled varlen 验证

> 下图是 DSpark 的预算决策链:草稿与置信度 → CPU 只用它定 total_q → 逐请求长度必须在 GPU 上重算,以及「哪一轮的 accept rate 能给谁用」这条时序纪律。

<div class="archify-figure">
<iframe src="../../../diagrams/di-dspark-varlen.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="DSpark 动态验证长度"></iframe>
</div>


- **草稿模型**:DFlash(`deepinfer/model_executor/models/mtp/dflash.py:82`),MTP 式草稿头,生产上一次生成 5 个 draft token(`DEEPINFER_USE_NEXTN_DECODE=5`,设 0 关闭);带可选 Markov embedding 的 **Confidence Head**(融合 kernel `confidence_head_fused`,tilelang_ops/dflash.py)预测每个 draft token 的接受率。
- **为什么定长 5 浪费**:各请求、各位置的接受概率差异大,低概率 token 也占验证算力。DSpark 按「预期接受率排序 + 硬件吞吐曲线」选**边际收益最大的前缀子集**做验证(Q&A)。
- **CPU 侧调度**(csrc/engine/mtp_varlen.hpp:150–260):维护 graph_size→baseline_ms 的 `_decode_perf_curve`(启动时实测、`DEEPINFER_DECODE_PERF_CACHE_DIR` 可落盘缓存,TTL 默认 3600s)+ 运行时斜率/采样耗时中位数估计器(窗口 9、最少 5 样本)。把全 batch 的 (accept_rate, seq) 排序后,**贪心逐个加入 draft token,每加一个算一次 `expected_tokens / time_ms`,取吞吐最大的 batch size 作为 total_q 预算**。关键的时序纪律(源码注释,`schedule_cnt` 与 `process_cnt` 两个计数器决定「accept_rates 是第几轮」):**算 total_q 预算永远用 N−2 轮的 accept rate**;**算每请求 seqlens_q:非 ZOS 用 N−1 轮,ZOS 下 CPU 只能先按 N−2 轮出初值(第 N−1 轮结果还在 GPU 上),真正的按请求分配在 GPU 上重算**(下条)。这条纪律的目的是「第 i 个 draft 是否被验证」不受它自己及之后的 confidence 影响——否则是用答案出题。
- **异步调度下的矛盾与解法**:`schedule(N+1)` 时步 N 的 confidence 还没回 CPU。解法是 CPU 只定 total_q 预算,**每请求长度在 GPU 上由 TileLang kernel 重算**(Q&A 原文:cumulative product、global sort for top-k、fused remap scatter),结果写回 pinned buffer——对应 `compute_seqlens_q_by_confidence_rates`:cumprod 得每个 draft 位置的累计接受率,全局稳定排序(按 −rate,同率按序号)取 top-(total_q−n_seq),scatter 回各请求的 seqlens_q;新请求(remap=−1)confidence 视为 0。
- **实测数据**(EP64/DP64,出自官方部署 Q&A,单源):动态长度把每步验证的 draft token 减约 30%(5.00 → 3.2–4.4),平均接受长度基本不变(3.29 vs 3.20);生产点(~60 并发/rank)吞吐 **+4–7%**。gain 列我按 (dynamic−fixed)/fixed 逐行重算过:16 并发 −1.5%、32 并发 +1.6%、48 并发 +3.6%、**60 并发 +5.4%**、72 并发 +6.7%、96 并发 +11.0%——与表中数字一致,低并发下动态验证反而略亏(调度收益盖不过碎片化)。

| conc(seq/rank) | 验证长度 fixed-5→dyn | 接受长度 fixed-5→dyn | 吞吐 tok/s/rank fixed-5→dyn | gain |
|---:|---|---|---|---:|
| 16 | 5 → 4.44 | 3.32 → 3.24 | 4891 → 4816 | −1.5% |
| 32 | 5 → 3.65 | 3.25 → 3.17 | 8964 → 9111 | +1.6% |
| 48 | 5 → 3.94 | 3.30 → 3.26 | 12032 → 12467 | +3.6% |
| 60 | 5 → 3.58 | 3.30 → 3.21 | 13730 → 14466 | +5.4% |
| 72 | 5 → 3.24 | 3.29 → 3.12 | 14962 → 15961 | +6.7% |
| 96 | 5 → 3.24 | 3.30 → 3.15 | 16817 → 18660 | +11.0% |

与 vLLM/SGLang 的对照见 [../../vllm/concepts/speculative-decoding.md](../../vllm/concepts/speculative-decoding.md):通用框架的投机解码通常是「定长 draft + 全量验证」,confidence 驱动的**变长预算分配**是 DeepInfer 的差异点。

---

## 7. 长上下文侧:YoCo 结构与 SWA bounded replay

### 7.1 YoCo producer/consumer 分层

V4.1 的层分两段:前 `n_prefill_layer` 层是 producer(全量 prefill),其后是 **consumer 层**(判定:`yoco_enabled = n_prefill_layer < num_hidden_layers`,consumer 层 = `[n_prefill_layer, n_layer)`,deepinfer/model_executor/models/v4_modules/yoco.py:14–21)。CED 架构下 decoder 层的 block KV 由最后一个 encoder 层 hidden 投影而来,所以 consumer 层 prefill 时只需算 SWA 部分。

**prefill 截断两模式**(`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE`,environ.py:188–194):
- `mask`:保留全部 query,只把 consumer 层的 SWA 可见域收窄成「原滑窗 ∩ 末尾 KEEP_TOKENS 矩形」;
- `compact`:真正把截断区的 query 从 consumer 层移除——compact 边界上 hidden/residual(含 mHC 的 HCLazyPostData 四条流)、topk_indices 等 per-token 数据都要 gather 成紧凑布局(`yoco_boundary_compact` / `compact_yoco_boundary`,yoco.py:55–123;CP 开启时还要跨 rank all_gather 后再选)。

硬约束:`KEEP_TOKENS ≥ window_size + DEEPINFER_MAX_BACKTRACKING_TOKENS`(model_runner.py:109–113 断言);backtracking 默认 = `max(0, nextn+1)` = 6(environ.py:132),V4.1 默认配置 window 128 → README 给的生产值是 **compact + KEEP_TOKENS=140**(128+6=134 ≤ 140)。

### 7.2 SWA bounded replay:重算成本变常数

> 命中判定与重算的分支:命中长度由 block KV 定,最后一块是否要重算由 SWA 定;缺失时只重算末尾 128 token。

<div class="archify-figure">
<iframe src="../../../diagrams/di-swa-bounded-replay.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="SWA bounded replay 命中判定"></iframe>
</div>


缓存命中后 SWA KV 可能缺失。精确重建要把末尾一长段重跑一遍(官方例子:20 层 decoder、每层末尾 128 token,合计 2,560 token-层);**bounded replay 只重算末尾 128 token,且把 query 的可见域收窄到 `max(s, i−127)..i`**(s 为 replay 起点)——得到一个近似 SWA KV,重算成本与上下文长度、层数都无关(Q&A)。两处用法:
- **Encoder 侧**:prefix cache 命中长度只由 block KV 决定;SWA 缺失时重算命中前缀的末尾 128 token,与非命中后缀一起 forward,replay 段只重建 SWA KV,block KV 原样复用、不重算不覆写。
- **Decoder 侧**:每次 prefill 只在 decoder 层重算 prompt 末尾 128 token,结果**仅供本次 decode、不进 prefix cache**——总 prefill 计算近乎减半(Q&A 口径)。

### 7.3 命中判定顺序与 L−134 上限

原则:**命中长度由 block KV 定,最后一块是否要重算由 SWA 定**。YoCo 截断要求下,长度 L 的输入最大命中长度 ≤ **L−134**(128 窗口 + 6 投机回退);先查 block KV 得 L1(128 的倍数),再查 SWA:若 `[L1−128, L1]` 区间存在 SWA 副本,命中长度 = L1,否则 = L1−128(128 即重算窗口)。这个 134 与 §8 的 StateBlock 134 token 槽长是同一个数。

---

## 8. KV cache 三级存储与 StateBlock

### 8.1 三级布局(Q&A 表格重排;数字均为官方单源)

> 三级存储到 swap-in 的数据通路,以及三段开销(匹配 / 取 KV / swap-in)各自藏在哪里。

<div class="archify-figure">
<iframe src="../../../diagrams/di-kv-tiers.dataflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="KV cache 三级存储"></iframe>
</div>


| 内容 | 介质 | 副本策略 | 管理者 |
|---|---|---|---|
| in-flight block KV + state KV | 推理 GPU 显存 | — | Prefill/Decode 实例 |
| SWA KV | 本机 host 内存 | 单副本,**不上 3FS**;跨机命中走 400G VLAN 拉取 | P/D 实例 + Entrypoint 内存目录(逻辑统一、按 key 物理分片) |
| 分页 block KV(main KV + indexer K) | 专用 3FS 存储集群 | 无应用层副本,靠 3FS 自身冗余(10+2 EC) | kvcache-store(未开源) |

- **为什么用专用存储集群而非分层本地存储**:任何 prefill 机器都能命中任何前缀 → 避免「prefix affinity 与负载均衡路由互相打架」;存储扩缩容与计算解耦;计算节点故障率比存储节点高一个数量级以上,KV 不受计算节点故障影响。
- **3FS 细节**:10+2 erasure coding,每组容忍 2 节点同时故障;一次读**并发请求组内全部 11 个节点、任意 10 个响应即可解码**(拿延迟换尾);GC Worker 每存储节点一个,LRU 回收过期缓存、把磁盘水位压在阈值下。总量 ~100 PiB;TTL:agent 流量 3 天、chatbot 流量 7 天——chatbot 长尾重(少量 KV 被读非常多次),3 天 TTL 对 chatbot 命中率的代价实测 0.5%–1%。文件系统设计背景见 [./3fs.md](./3fs.md)。
- **开销重叠三段式**:匹配(Entrypoint,hash-chain 只比 metadata,但命中长度须在路由前算出、藏不掉)→ 取 KV(独立进程异步 3FS→host 共享内存)→ swap-in(引擎进程内线程,**逐层 H2D 与 forward 重叠**)。swap-in 用 **GDR 而非 CUDA copy engine**:copy engine 对 KV block 的细粒度拷贝性能差,GDR 让 RDMA 网卡直接打满 PCIe 带宽、不占 SM。
- decode 期间**不做 KV offload**(decode 不受容量约束,也不用 Hi-Sparse),但 decode 产生的 KV 仍写盘以提高后续命中率。

### 8.2 StateBlock:一个序列一个定长槽

SWA KV 不进分页 block,住 **state cache**:每序列一个定长 `StateBlock` 槽,槽数 = `int(max_num_seqs × DEEPINFER_NUM_STATE_SLOTS_RATIO(1.2)) + 1`(cache_engine.py:201–202)。布局(cache_engine.py:221–306):

```
[stored region: SWA(需落盘,前 20 层)]
[dropped region: YOCO consumer SWA | compression state(ca_state/ica_state)]
```

- 分区原则:**要落盘的全放前面**——落盘时按长度截取前 `state_size_to_disk` 字节即可整个排除 dropped region;PE→CE 传输则两部分都传。
- dropped region 里,YoCo consumer SWA 靠 prefill 重算恢复(bounded replay),压缩边界累加器 `ca_state/ica_state` 的压缩窗口天然对齐 128 token 落盘粒度(`KVSTORE_GRAIN=128`,cache_engine.py:32),无跨界依赖,所以都可以丢。
- **落盘字节数验算**:Q&A 给 `state_size_to_disk` = 20 层 × 134 token × 528 B = **1,415,040 B**/序列。我逐项核过:528 B/token 即 `V41_FORMAT` 的 512B FP8 数据 + 16B scale(§11);134 = inference SWA 槽长 = window 128 + backtracking 6(`get_inference_swa_block_size`,cache_engine.py:88–93);20 = V4.1 的 stored(非 consumer)SWA 层数(Q&A 明写 "the stored layers are the first 20")。134×528=70,752 恰好 32B 对齐,20×70,752=1,415,040,与 Q&A 完全一致。
- **落盘位置纪律**:只在 128 的整倍数位置存。prefill 后:预留末尾 1 token、向下取整到 128 倍数;不足一个完整 128 单元则本轮不存;chunked prefill 中途不存、只有最后一个 chunk 存。decode 期在显存另开 replica ring,每跨 128 边界就 device 内拷一次当前窗口;请求结束时取 replica 里最近一次完整拷贝,拷贝位置最多超出边界 5 token,由槽 ring 比窗口多出的 6 个位置吸收(128+6=134 的另一处由来)。

### 8.3 分段加载计划

swap-in/重排走统一的 **BlockLoadPlan**(csrc/engine/input_metadata/block_load_plan.cpp:9–47):把任意「源 → 目的」的 KV 搬运拆成三种 segment——`append_paged_range`(按 block table 查分页源块)、`append_ring_range`(源是环形缓冲,`src_begin = pos % block_size`)、`append_zero_fill`(src_block_id=−1 的零填充段)——最终打成 `[n_seg, 4]` 的 pinned int32 表 `(src_block_id, src_begin, src_end, dst_offset)` 一次交给 kernel。分页 KV、SWA ring、缺失补零三种形态用同一套原语描述。

---

## 9. EPLB:冗余专家 + 贪心放置(机制开源,生产参数见 Q&A)

与训练侧的 EPLB(见 [./lplb-epb-load-balancing.md](./lplb-epb-load-balancing.md))同源思想,推理侧实现在 `deepinfer/model_executor/eplb/`:

- **逻辑/物理专家**:logical expert 是模型定义的、gate 从中选择的;physical expert 是 GPU 上实际放置的副本,数量 = num_logical + num_redundant(policy.py 模块注释)。
- **冗余数取整公式**(eplb/__init__.py:21–28):

```python
def resolve_num_extra_experts(num_routed_experts, min_num_extra_experts, ep_size):
    padding = -(num_routed_experts + min_num_extra_experts) % ep_size
    return min_num_extra_experts + padding   # 使 num_physical 能被 ep_size 整除
```

  下限来自 `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS`(benchmark 默认 32,examples/README.md 说明它只是下界,实际取整后的值才写盘/画图)。Q&A 的生产参数:prefill 每 GPU **4** 个冗余、100 步统计窗、每 1000 步重排;decode 每 GPU **1** 个冗余、1000 步窗、每 10000 步重排。EP64 下 Q&A 原文明写「this works out to **7 experts per GPU** after rounding」,即 448 = 7 × 64 个物理专家(不含 shared expert);routed 专家数 **384** 已由 V4.1-Flash HF config 核实(`n_routed_experts: 384`),代入公式 `resolve_num_extra_experts(384, 64, 64)` → padding = −448 % 64 = 0 → extra = 64 → 448/64 = 7,与原文自洽。实测 `moe_lb`(最慢 rank/均值):prefill **1.074**(EP4DP4)/ **1.045**(EP2DP2),decode **1.090**(EP64)。
- **两级贪心**(policy.py):
  1. `rebalance_experts_global`(policy.py:73–140):决定**谁被复制**——每轮给「加一个副本后,最大每副本负载(按历史 step 求和)最小」的 logical expert 复制;
  2. `_batch_greedy_indices`(policy.py:12–70):决定**副本放哪**——physical expert 按每副本历史总负载降序,逐个尝试放到各 rank,目标函数是 `min over placements of Σ_steps max_rank load`(源码注释原文),满 capacity 的 rank 屏蔽,同分取最小 rank ID。逐层独立求解。
- **权重 shuffle**:重排后专家权重要跨 GPU 搬迁;EP 组在 NVLink 域内且是 GPU 时走 `symm_mem` IPC 单机 shuffle(镜像 buffer 只需容纳一层、逐层刷新复用,eplb/__init__.py:195–209);**NPU 尚不支持 symm_mem**(源码 TODO)。首次 rearrange 前 `torch.cuda.empty_cache()` 防 OOM。
- **统计口径**:`DEEPINFER_MOE_LB_STAT_INTERVAL=128` 步做一次 EP all_gather,`moe_load_balance_ratio = (Σ mean_over_ranks, Σ max_over_ranks)`,相除即 moe_lb。这个 all_gather 有十几 ms 抖动,benchmark 窗口要剔除撞上它的区间(§13)。统计窗(`EXPERT_COUNT_WINDOW_SIZE`,默认 100)必须 ≤ 重排间隔(启动时校验)。

---

## 10. 负载均衡(未开源,仅记 Q&A 设计口径)

- **Prefill LB 四级分桶**:按总上下文长度分 <16K / 16–64K / 64–256K / ≥256K 四档,各档引擎数按上一轮算力配额需求占比动态重分配,每档有独立的 per-batch 计算配额上限。跨 DP 组按剩余配额拉活,**无 prefix affinity**(任何引擎都可能命中任何前缀,与专用存储集群的设计互为因果)。单 GPU 内 FIFO 出队,**chunk 大小不固定而是解出来的**:每 prefill step 给一个纳秒级时间预算(`compute_quota_limit`),用拟合的二次代价模型 `cost(q, c)`(q=本步 token 数,c=已算前缀长)反解最大的 q,受 32,768 token 预算封顶;预算本身再经 DP 组内 all-reduce MIN 收紧。
- **Decode LB 双维度**:同时均衡请求数(决定 per-step batch)与总 KV 长度(决定 attention 成本)——短请求发给低并发但 KV 总量大的 GPU,长请求发给 KV 总量小的 GPU(Q&A 指向 DualPath 的组内调度算法)。
- **VE LB**:机器粒度随机选,机内把图交给当前 patch 数最少的 GPU(patch 数 = (w/patch)×(h/patch) 直接算)。
- **整体效果**:达到完美均衡 offline benchmark 的 **~80%**。
- 弹性与容错:prefill 按非缓存 token 数扩缩、decode 按 running 请求数扩缩(限速、有上下界);配额利用率 >0.8 禁止缩容;decode CE 收到下线通知后有 1 分钟 grace period,把未完成请求连同 KV 一次性转发给其他 CE;大 EP 下单 GPU 故障即整实例不可用,重启时按当前健康 GPU 数重新定 EP size。

---

## 11. 低精度 KV 与 FLOPs 记账

### 11.1 KV cache 格式(字节数为我按 `cache_common.py` 定义逐项验算)

`KVCacheFormat`(deepinfer/model_executor/layers/tilelang_ops/cache_common.py:24–107)用「量化元素数 + 每 scale 覆盖元素数 + scale 格式」描述每 token 物理布局,main/SWA cache 是 token-interleaved 的 `[data | scales]`:

| 格式常量 | 用途 | data | scale | 字节/token(验算) |
|---|---|---|---|---|
| `V41_FORMAT`(:109) | V4.1 main/SWA KV(MXFP8) | 512×e4m3=512B | per-32 ue8m0 → 16×1B | 512+16 = **528** |
| `V41_FP4_FORMAT`(:119) | V4.1 FP4 main KV | 512×e2m1=256B | per-16 e4m3 → 32×1B | 256+32 = **288** |
| `FP4_INDEX_KV_FORMAT`(:129) | indexer KV(MXFP4) | 128×e2m1=64B | per-32 ue8m0 → 4×1B | 64+4 = **68** |
| `MXFP8_INDEX_KV_FORMAT`(:147) | V4.1 indexer KV | 128×e4m3=128B | per-32 ue8m0 → 4×1B | 128+4 = **132** |
| `FP8_FLOAT32_SF_INDEX_KV_FORMAT`(:138) | V4 indexer(NPU 不支持) | 128×e4m3=128B | per-128 fp32 → 1×4B | 128+4 = **132** |

细节:FP4 main KV 的 scale 是 per-16 e4m3——Q&A 描述为「像 NVFP4 但没有 global scale」;两个 132B 的 index 格式字节布局撞车,靠 `sf_format` 显式消歧(`get_kv_cache_format`)。MoE routed expert 权重也是 MXFP4(per-32 ue8m0)。KV 量化格式谱系见 [../../quantization/concepts/kv-cache-quantization.md](../../quantization/concepts/kv-cache-quantization.md)。

### 11.2 FP8-equivalent FLOPs 记账

调度器的代价模型不按真实 FLOPs 而按 **FP8 等效**记账(csrc/engine/flops_estimator.hpp:13–19):

```cpp
// GEMM coefficients include multiply-add (2 FLOPs) and the FP8-equivalent weight.
constexpr int64_t FP4 = 1;  constexpr int64_t FP8 = 2;
constexpr int64_t BF16 = 4; constexpr int64_t TF32 = 8;
constexpr int64_t FP32 = TF32; // Accounting reference, not actual kernel precision.
```

即 FP4 GEMM 记 1 份、FP8 记 2 份、BF16 记 4 份(乘加 2 FLOPs 已含在系数里)。两处值得注意的口径:MoE 层「FP8 激活 × FP4 权重」仍按 **FP8** 记(FP4 只是存储格式,不代表 FP4×FP4 计算,flops_estimator.hpp:84–89);mHC 的 fn 投影按 TF32、mix 收缩按 FP32 记(与 [./mhc-manifold-hyper-connections.md](./mhc-manifold-hyper-connections.md) 的实现一致)。这份记账直接喂给 prefill 调度的 `attn_tflops/total_tflops`(SchedulerOutputs 字段),是「32,768 token 预算 + 二次代价模型」的基础。

### 11.3 batch-invariant 与确定性采样(NVIDIA only)

- **batch-invariant**(`DEEPINFER_BATCH_INVARIANT=1`):不同 batch 组合下模型输出逐位一致。offline 示例在 NVIDIA 上默认开(examples/offline_inference.py:43)。
- **确定性采样**:三件套 = `DEEPINFER_BATCH_INVARIANT=1` + `DEEPINFER_ENABLE_MTP_VARLEN=0`(关掉变长投机——验证长度依赖运行时 confidence,天然不确定)+ `SamplingParams` 固定 `seq_seed`(README)。
- **与 EPLB 的因果**:DeepEP buffer 的 `allow_multiple_reduction = not (BATCH_INVARIANT and EXPERT_REARRANGEMENT_STEP_INTERVAL > 0)`(parallel_utils/userbuffer.py:79)——**开了 batch-invariant 又开 EPLB 动态重排时,必须关掉 multiple-reduction 优化**。重排会改变各 rank 持有的专家集合,从而改变 combine 阶段部分和的归约组合;multiple reduction 允许的归约拆分方式与「跨步逐位一致」冲突。源码只写了这个联动条件,更深的归约语义在 DeepEP 侧(见 [./deepep-nvidia-vs-ascend.md](./deepep-nvidia-vs-ascend.md))。
- 回归保障:`tests/models/test_batch_invariant.py` + PPL 黄金基线(`tests/models/offline_decode_ppl.json`,逐样本精确比对,平台/负载分键)。

---

## 12. 结构化输出:编译与推理解耦

流程(Q&A):SO 编译在**专用 SO 机器**上异步跑,请求进引擎前先取编译结果;运行时每个 decode step 推进自动机、生成合法 token mask 施加到采样。开源侧的编译器包装 `XGrammarCompiler`(deepinfer/structure_output/compiler.py):
- 支持 json(strict mode)/ regex / ebnf 三种 schema(`dispatch_and_compile`,:54);
- **缓存安全**:xgrammar ≤0.1.x 开缓存出过生产事故,`DEEPINFER_XGRAMMAR_DISABLE_CACHE` 留作 kill switch;缓存默认上限 10 GiB(`DEEPINFER_XGRAMMAR_CACHE_LIMIT_BYTES=10*1024³`,environ.py:170),设 ≤0 时启动即警告「无上限缓存随 distinct schema 无限增长,是 DoS 向量,见 **CVE-2025-32381**」(compiler.py:16–30)——公网服务必须设正的上限;
- 编译超时 `DEEPINFER_XGRAMMAR_TIMEOUT=20`s、builder 进程 `DEEPINFER_SO_BUILDER_CPUS=4`;
- 引擎侧对最后一个 prefill chunk 做 SO 簿记校验,matcher 缺失直接判死该请求(FINISHED_SO_FAILED)而不拖垮整批(llm_engine.py:910–928)。

---

## 13. benchmark 口径与坑(examples/README.md,数字口径极细,照抄前先对齐)

- **prefill throughput**(tokens/s/GPU):跳过 5 个 warmup chunk 后,取最后 **11 个样本 = 10 个稳态区间**(首样本只贡献起始时间戳、其 token 不计入);token 总数 ÷ 区间总时长 ÷ 设备数。供给不足两个采样点打 `N/A`。
- **decode avg_SPS**:跳过 50 warmup + 60 个带日志步后,取最后 **61 个样本 = 60 个区间**;撞上 MoE LB 统计 flush(EP all_gather,十几 ms 抖动)的区间**剔除后再平均**;同时打印净区间里的最慢单步 `max_step`,正常应接近 `1000/avg_SPS`。
- **decode throughput** = `avg_SPS × decode_num_requests × (1+nextn) / cp`,**满接受口径**——不管真实接受率,每请求每步记 1+nextn 个 token(默认 64 请求 × 6 = 384 token/步)。画 pareto 时 sweep 脚本按草稿接受率(如 0.75)另行折算。
- **profiler 三条纪律**:① profiler 退出后仍有 **~7–11% 残留减速**(bsz=1 实测),decode profiler 必须放在全部测量之后;② trace 内的 step 时间含 profiler 即时开销(bsz=1 最多抬高近 **50%**),只能看 kernel 构成、**不能拿来算 SPS**;③ sweep 一律 `PROFILE=0`。
- **日志与测量隔离**:测量段加倍——前半段开日志(prefill 每 5 步、decode 每 20 步,moe_lb 从这里读;benchmark 把 `DEEPINFER_MOE_LB_STAT_INTERVAL` 设为 10 让日志段有真值),后半段静默出数;组批/warmup/profile 窗全程 `DEEPINFER_LOGGING_INTERVAL_STEP=0`。
- **benchmark 自己会覆写四个 env**(`examples/benchmark.py:164–169`,复现 state 尺寸/显存时必看):`DEEPINFER_MAX_BACKTRACKING_TOKENS=1`(于是 SWA 槽长 = 128+1 = **129** 而非默认的 134)、`DEEPINFER_FORCE_LOAD_BALANCE=1`、`DEEPINFER_MOE_LB_STAT_INTERVAL=10`、`DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL=32`(默认 8)。**由此 benchmark 运行时的 `state_size_to_disk` = 20 × align32(129 × 528) = 20 × 68,128 = 1,362,560 B,不是 §8.2 按默认配置算的 1,415,040 B**——两个数字别混用。
- prefill/decode 两阶段解耦:`--prefill_batch_size`/`--non_cached_tokens` 只管 prefill,`--decode_num_requests` 只管 decode batch;唯一耦合是 decode 复用 prefill 产出的共享前缀 KV。README 的 pareto 图口径:prefill 4 张 SM100、decode 64 张,各在一个 scale-up domain 内。

---

## 14. 与 vLLM 的关系(机制对照)

依据:README Acknowledgment("originally built primarily on vLLM")+ 大量文件头 `# Derived from vLLM; modified by DeepSeek.`(config.py、worker.py、model_loader.py 等)。**不要**把它说成「vLLM 某个 commit 的 fork」——仓库没有给出上游基线版本,这个说法没有依据。

| 维度 | vLLM(见 [../../vllm/concepts/v1-architecture.md](../../vllm/concepts/v1-architecture.md)) | DeepInfer |
|---|---|---|
| 核心不变量 | paged KV + continuous batching | **相同**(分页 block KV、连续批处理、block manager 概念都在) |
| 同源类名 | CacheEngine / ModelRunner / Scheduler / LLMEngine | 同名类保留,实现大改 |
| Scheduler | Python | **C++**(pybind11),含 DP 投票、FLOPs 记账 |
| 模型支持 | 通用注册表、上百模型 | **只有 V4.1-Flash** |
| MoE 均衡 | 已有 EPLB(`vllm/distributed/eplb/` + `--enable-eplb`,算法注释明写移植自 DeepSeek EPLB) | **原生内置**,与自身 ZOS / YoCo / SWA state 管理、DeepEP-V2 userbuffer 直接耦合(如 batch-invariant 与重排的 multiple-reduction 联动,§11.3) |
| 异步调度 | v1 有 async output processing | **ZOS**:post-process 整体推迟一步 + GPU 侧 seqlens 修正 |
| 投机解码 | 通用 spec-decode 框架 | **DFlash + DSpark** confidence-scheduled varlen |
| KV 层次 | GPU(+可选 CPU offload) | GPU / host SWA state / **3FS 分页 block KV** 三级 + bounded replay |
| 长上下文 | 全量注意力为主 | **YoCo producer/consumer + SWA 截断/compact** |
| 特有模块 | — | Engram([./engram.md](./engram.md))、mHC 融合内核、DSA indexer(FlashMLA/DeepSelect,[./flashmla.md](./flashmla.md)、[./deepselect.md](./deepselect.md)) |
| 硬件 | NVIDIA/AMD/… 插件制 | **SM100 + 昇腾 950 双平台一等公民**(Platform 抽象) |

---

## 15. 双平台抽象与构建

- `PlatformGPU` / `PlatformNPU`(deepinfer/platform.py:55–89):tilelang 后端 `cuda`/`asc`,分布式后端 `nccl`/`hccl`;NPU 侧把 `ASCEND_RT_VISIBLE_DEVICES` 映射到 `CUDA_VISIBLE_DEVICES` 以复用代码路径;NCCL 的 max/min CTA 数由 `DEEPINFER_NCCL_NUM_SMS=32` 钉死。内核族(FlashMLA/DeepGEMM/DeepEP/TileKernels/DeepSelect + 自研 TileLang kernel)的双平台对照,本库已各有专篇([./deepgemm-nvidia-vs-ascend.md](./deepgemm-nvidia-vs-ascend.md)、[./deepep-nvidia-vs-ascend.md](./deepep-nvidia-vs-ascend.md)、[./tilelang.md](./tilelang.md))。Q&A 补充:内部版 FlashMLA 多一处手调 SASS,开源版 decode kernel 慢 5–10%,端到端差距 ≤5%;Mega MoE 在 scale-up domain 内不论 EP/batch 都快于非融合 MoE。
- **构建**:Linux + GCC ≥13 + Python 3.12(`pyproject.toml` `requires-python = ">=3.12"`,空环境)+ CUDA 13(NVIDIA)或 CANN 9.2(昇腾);`export DEEPINFER_ACCELERATOR=nvidia|ascend && ./build.sh`。`build.sh` 三步固定:① `pip install -r requirements-${ACCELERATOR}.txt`;② NVIDIA 侧 `pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3`(**钉死 NCCL**,芯片侧对 Gin 后端版本敏感);③ `scripts/build_vendor.sh` 构建 vendored 算子轮子 → `scripts/build.sh build` 编主仓 Python 扩展。NVIDIA 依赖链里 `torch==2.13.0+cu132` + `flash-attn-4==4.0.0b34` 来自 `download.pytorch.org/whl/cu132`,不是 PyPI(装错源会拉到 CPU 版)。`Makefile` 把这几步拆成 `build-nvidia` / `build-ascend` / `vendor-nvidia` / `vendor-ascend` / `clean` / `stubgen` / `format` 七个目标,调 vendor 不必重编主仓。

**vendored 算子的 pin 是「已知快照」,不是「当前 main」(2026-10-21 增量)**

`scripts/build_vendor.sh` 把四个算子作为 git submodule 拉进 `vendor/` 再各自 `python setup.py bdist_wheel`,所以**引擎能用哪版内核由 DeepInfer 仓的 submodule pin 决定,与你另行 pip install 的算子版本无关**。2026-10-21 的 PR #157「Bump DeepEP to f6aa38fe and drop the SSH->HTTPS rewrite」只动了两处:submodule pin 前移 + 删掉 `GIT_REWRITE=(-c url."https://github.com/".insteadOf="git@github.com:")` 这段改写在(`build_vendor.sh` 里对主仓和每个 vendor 包各调用一次)。删除的理由写在 commit message 里:**DeepEP 自己的 `.gitmodules` 已把 `third-party/deep_jit` 从 `git@github.com:` 改成 `https://`,不再需要外层替它改写 URL**。对使用者的实际影响:内网只开了 HTTPS 出口、没配 SSH key 的构建机以前要靠这层 rewrite 才能 `submodule update`,现在上游已修好,少数依赖 rewrite 掩盖其它私仓 SSH 地址的场景会直接暴露(报 `Permission denied (publickey)`)。

当前 pin 集(`9bd9cf243f`,2026-10-21)与上游 main 的对照——**两处落后**:

| vendor 包 | DeepInfer pin | 上游 main HEAD | 差 | 影响 |
|---|---|---|---|---|
| DeepEP | `f6aa38fe06` (10-21) | `f6aa38fe06` (10-21) | 同步 | #157 就是为对齐它 |
| DeepEP-Ascend | `3b25377d04` (09-29) | `3b25377d04` (09-29) | 同步 | — |
| DeepGEMM | `1fc9c66975` (10-14) | `5081f8513a` (10-20) | **落后 #478** | ⚠ mega_mhc CUDA graph 屏障 bug(见下) |
| DeepGEMM-Ascend | `3a35326271` (10-14) | `3a35326271` (10-14) | 同步 | — |
| DeepSelect | `bfa4507d93` (09-30) | `bfa4507d93` (09-30) | 同步 | — |
| FlashMLA | `2e5429fc56` (09-30) | `0f11e178da` (10-15) | **落后 #230** | ABI 稳定化(构建层),无内核变化 |

**这两条落后要分开看**。FlashMLA 落在 `2e5429fc56` 只丢 #230 的 TORCH_LIBRARY/abi3 稳定化(见 [./flashmla.md](./flashmla.md) §ABI),那是打包层收益,不影响数值。DeepGEMM 落在 `1fc9c66975` 则是**实打实的功能缺陷**:`#478` 修的正是 mega_mhc 在 CUDA graph 重放下的屏障代数错误,而 DeepInfer 的 decode 路径**同时满足触发条件**——

1. 走 mega_mhc:`v4_modules/hc.py:99` 的门控是 `PLATFORM.is_gpu and hc_mult == 4 and hc_attn_fn.shape[1] % (hc_mult * 16*64) == 0`,而 V4.1 的 `mhc_mult` 默认就是 4;命中后 `hc.py:270` 直接 `deep_gemm.mega_mhc(...)`;
2. 走 CUDA graph:decode 侧 `model_runner.py:327` 走 `cuda_graph_runner.replay(...)`。

于是 pin 里那版 `scheduler/mega_mhc.cuh` 用 `encode_state(ptx::get_grid_idx())` 给 split 屏障打代数,而屏障缓冲区是 `csrc/apis/mega_mhc.hpp:26–38` 里**按 CUDA stream 缓存的 static 张量**(`split_barriers_by_stream`,首次 `torch::zeros` 分配后一直复用)——graph 重放时 `%gridid` 恒定、缓冲区又是同一块,上一轮的终态 `encode_state(grid_idx, kNumSplits)` 恰好等于本轮要找的值,迟到的初始化 CTA 会让 waiter 提前放行读到**旧 split partial**。症状是偶发数值错误、只在 graph 重放下出现(即启动每次 gridid 都变,不复现)。机制推导与 epoch 修法详见 [./mhc-manifold-hyper-connections.md](./mhc-manifold-hyper-connections.md) §5。

> **操作结论**:用 DeepInfer 的 NVIDIA 路径跑 decode(CUDA graph 默认开)、且吃 mega_mhc(`hc_mult=4`)时,别信 pin 里的 DeepGEMM——在 `vendor/DeepGEMM` 手动 `git checkout 5081f8513af5`(或只 pick #478)后 `make vendor-nvidia` 重编,或运行时 `DEEPINFER_ENFORCE_EAGER=1` 回避 graph 重放(代价是掉 graph 的性能)。这类「pin 落后上游」的坑没法靠 `pip list` 看出来,查 `git -C vendor/DeepGEMM rev-parse HEAD` 才是真的版本。
- **运行**:`python examples/offline_inference.py`(默认 ckpt 自动下载;`DEEPINFER_HF_MODEL_DIR` 指本地副本,不覆盖显式 `--model_name`)。

---

## 16. 实践 checklist

**上线前**
- [ ] 明确你只拿到「引擎」:KV 存储客户端、EPD 编排、跨 DP 负载均衡器要自己补(README 明示不含);先从 `EngineType.NORMAL` 单机形态起步。
- [ ] 生产参数对齐 Q&A:prefill EPLB 4 冗余/100 窗/1000 重排,decode 1 冗余/1000 窗/10000 重排;`moe_lb` 目标参考 1.045–1.090。
- [ ] 开 SO 服务必设 `DEEPINFER_XGRAMMAR_CACHE_LIMIT_BYTES` 正上限(CVE-2025-32381 DoS)。
- [ ] 要逐位可复现:`DEEPINFER_BATCH_INVARIANT=1 DEEPINFER_ENABLE_MTP_VARLEN=0` + 固定 `seq_seed`,且意识到它与 EPLB 重排同开时会关掉 multiple-reduction(性能代价,§11.3)。
- [ ] YoCo 截断:`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` + `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`(V4.1 默认;必须 ≥ window+backtracking,否则启动断言失败)。

**关键 env 速查**(全表见 `deepinfer/environ.py`)

| env | 默认 | 作用 |
|---|---|---|
| `DEEPINFER_ACCELERATOR` | nvidia | 平台选择(构建+运行都要) |
| `DEEPINFER_USE_NEXTN_DECODE` | 5 | 最大投机长度;显式 0 关闭 |
| `DEEPINFER_ENABLE_MTP_VARLEN` | 0 | DSpark 变长验证 |
| `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` | 随 engine_type(NORMAL/COMPLETION 开) | ZOS |
| `DEEPINFER_PREFILL_ATTEMPT_INTERVAL` / `_ANY_` | 70 / 100 | NORMAL 引擎 ALL/ANY 投票点间隔 |
| `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` | 0(benchmark 设 32) | EPLB 冗余下限,按 ep_size 向上取整 |
| `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE` / `_REARRANGEMENT_STEP_INTERVAL` | 100 / 0(关) | EPLB 统计窗/重排间隔 |
| `DEEPINFER_NUM_STATE_SLOTS_RATIO` | 1.2 | state 槽数 = max_num_seqs×1.2+1 |
| `DEEPINFER_ENFORCE_EAGER` / `_DFLASH_ENFORCE_EAGER` | 0 | 关 CUDA graph(PROMPT 引擎本就不用 graph) |
| `gpu_memory_utilization`(EngineArgs) | 0.90 | KV+state 显存占比(benchmark 0.95、offline 示例 0.85) |

**常见坑**
- KV block 不够启动报错("No available memory for the cache blocks"):调高 `gpu_memory_utilization` 或调低 `max_model_len`(引擎报错信息原文如此);注意 `runtime_max_model_len` 比 `max_model_len` 多出位置余量——ZOS 开时 `nextn+(nextn+1)+nextn` = **3×nextn+1**(默认 nextn=5 时 16),ZOS 关时 **2×nextn**(默认 10);`max_num_blocks` 按 `runtime_max_model_len` 算(model_runner.py:115),所以按「2×nextn+1」估会少算位置、块数偏小。
- CUDA graph 捕获阶段显存吃紧:`DEEPINFER_ENFORCE_EAGER=1` 先跑通再开 graph;graph batch 粒度 `DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL=8`。
- NPU 调试:`ASCEND_LAUNCH_BLOCKING=1`(算子同步执行、栈可读)+ `ASCEND_SLOG_PRINT_TO_STDOUT=1`(examples/README.md)。
- 拿 benchmark 数字对比时:先确认口径(满接受 ×(1+nextn)、LB flush 剔除、profiler 残留 ~7–11%),否则 SPS 会系统性偏高或偏低(§13)。
- ZOS 下最后一步输出要靠 `flush_inflight_reqs` 收口;自己写 serving 循环时别只调 `step()`。
- **别把 `pip list` 里的算子版本当成引擎实际用的版本**(§15):vendored 算子由 `vendor/*` 的 submodule pin 决定,`git -C vendor/<pkg> rev-parse HEAD` 才是真版本。2026-10-21 实查 pin 落后上游两处,其中 DeepGEMM 停在 `1fc9c66975` 缺 `#478`,而 V4.1 decode 恰好满足 mega_mhc + CUDA graph 双条件 → 偶发读到旧 split partial。上生产前先对齐 DeepGEMM 到 `5081f8513a` 或开 `DEEPINFER_ENFORCE_EAGER=1`。

---

## 溯源

**一手链接**(均取自仓库 README / Q&A 原文):
- 仓库:https://github.com/deepseek-ai/DeepInfer(HEAD 9bd9cf243f,2026-10-21)
- PR #157(bump DeepEP pin + 删 SSH→HTTPS rewrite):https://github.com/deepseek-ai/DeepInfer/pull/157
- vendored pin 的原子仓:`vendor/` 下 DeepEP/DeepEP-Ascend/DeepGEMM/DeepGEMM-Ascend/DeepSelect/FlashMLA 六个 submodule,版本见 §15 对照表
- 官方部署 Q&A:https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md
- benchmark 口径:https://github.com/deepseek-ai/DeepInfer/blob/main/examples/README.md
- 默认 checkpoint:https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017
- V4.1 Flash 推理系统知乎博客(Q&A 的设计篇前置):https://zhuanlan.zhihu.com/p/2095560295930508953
- V3/R1 推理系统(DP=#EP 的先例):https://zhuanlan.zhihu.com/p/27181462601
- deepseek-recipe(配套开源):https://github.com/deepseek-ai/deepseek-recipe
- 此前开源的 TileLang 与内核库(微信公号):https://mp.weixin.qq.com/s/X41mKH4Ds-VXUAnK6M8Eww
- 致谢的上游:[vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)、[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

**本篇逐文件核对的源码**(仓库相对路径):`README.md`;`doc/inference_deployment_qa.md`;`examples/README.md`;`examples/_config.py`;`examples/offline_inference.py`;`deepinfer/environ.py`;`deepinfer/config.py`;`deepinfer/engine/arg_utils.py`;`deepinfer/engine/engine_type.py`;`deepinfer/engine/llm_engine.py`;`deepinfer/entrypoints/llm.py`;`csrc/engine/scheduler.hpp`;`csrc/engine/zos.cpp`;`csrc/engine/mtp_varlen.hpp`;`csrc/engine/flops_estimator.hpp`;`csrc/engine/input_metadata/block_load_plan.cpp`;`deepinfer/worker/worker.py`;`deepinfer/worker/base.py`;`deepinfer/worker/zos_status.py`;`deepinfer/worker/cache_engine.py`;`deepinfer/worker/model_runner.py`;`deepinfer/worker/decode_perf_cache.py`;`deepinfer/model_executor/eplb/__init__.py`;`deepinfer/model_executor/eplb/policy.py`;`deepinfer/model_executor/layers/tilelang_ops/cache_common.py`;`deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py`;`deepinfer/model_executor/layers/tilelang_ops/compute_seqlens_k_zos.py`;`deepinfer/model_executor/layers/tilelang_ops/dflash.py`;`deepinfer/model_executor/models/mtp/dflash.py`;`deepinfer/model_executor/models/v4_modules/yoco.py`;`deepinfer/model_executor/parallel_utils/userbuffer.py`;`deepinfer/structure_output/compiler.py`;`deepinfer/platform.py`;`build.sh`;`scripts/build_vendor.sh`;`scripts/build.sh`;`Makefile`;`requirements-nvidia.txt`;`requirements.txt`;`requirements-dev.txt`;`pyproject.toml`;`.gitmodules`;`vendor/`(submodule pin);`deepinfer/model_executor/models/v4_modules/hc.py`(mega_mhc 门控与调用);`deepinfer/worker/cuda_graph_runner.py`(**2026-10-21 增量追加**)。

**单源标注**:§2/§9/§10 的部署与性能数字(155T token/天、$38/GPU-h、moe_lb、TTL、带宽、存储配比、~80% 均衡、+4–7%)全部出自官方部署 Q&A,单源、未经第三方复核。

**库内交叉**:[./tilelang.md](./tilelang.md)(内核 DSL 与 asc 后端)、[./mhc-manifold-hyper-connections.md](./mhc-manifold-hyper-connections.md)(残差流机制与 FLOPs 记账对应)、[./engram.md](./engram.md)、[./flashmla.md](./flashmla.md)、[./deepselect.md](./deepselect.md)、[./deepep-nvidia-vs-ascend.md](./deepep-nvidia-vs-ascend.md)、[./3fs.md](./3fs.md)(存储底座)、[./lplb-epb-load-balancing.md](./lplb-epb-load-balancing.md)(训练侧 EPLB 对照)、[../../vllm/concepts/continuous-batching.md](../../vllm/concepts/continuous-batching.md)、[../../vllm/concepts/paged-attention.md](../../vllm/concepts/paged-attention.md)、[../../vllm/practices/pd-disaggregation.md](../../vllm/practices/pd-disaggregation.md)(通用框架的 P/D 分离对照)、[../../quantization/concepts/kv-cache-quantization.md](../../quantization/concepts/kv-cache-quantization.md)、[../../model-architecture/concepts/open-model-landscape-2025-2026.md](../../model-architecture/concepts/open-model-landscape-2025-2026.md)(V4.1-Flash 模型侧)。
