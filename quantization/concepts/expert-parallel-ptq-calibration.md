---
title: MoE 专家并行 PTQ 校准:DP=EP all-to-all 传输与 amax 覆盖面
type: concept
tags: [quantization, ptq, moe, expert-parallel, all-to-all, torchrun, modelopt, activation-amax, determinism]
source:
  - "code: NVIDIA/Model-Optimizer PR #2738 *[OMNIML-6059] [1/21] Add the DP=EP all-to-all expert transport*(2026-10-24 合并)| https://github.com/NVIDIA/Model-Optimizer/pull/2738 | ↑primary"
  - "code: 源码 modelopt/torch/utils/moe.py(commit 0638a27,a2a_moe / torch_a2a_experts_forward / register_torch_a2a)| https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/utils/moe.py"
  - "code: 单测 tests/unit/torch/utils/test_moe_a2a.py + tests/gpu/torch/utils/test_moe_a2a.py(EP=1 逐 bit 对照 grouped_mm)| https://github.com/NVIDIA/Model-Optimizer/blob/main/tests/unit/torch/utils/test_moe_a2a.py"
  - "web: Hugging Face transformers *Expert parallelism* 文档(EP 基础设施最低版本 5.6、parallel style 概念)| https://huggingface.co/docs/transformers/en/expert_parallelism | 阅读日期 2026-10-24"
date: 2026-10
---

# MoE 专家并行 PTQ 校准:DP=EP all-to-all 传输与 amax 覆盖面

> **TL;DR**:大规模 MoE 的 PTQ 校准,两条现成路都不可行——复现(replicate)要每个 rank 装下整个模型;仅 FSDP2 分片虽然省了权重存储,但每次前向都把**所有专家 all-gather 回每个 rank**,显存根本没省下来。ModelOpt PR #2738(2026-10-24 合并,EP PTQ 系列 **1/21** 的第一块)加入 DP=EP 传输:稠密层 + router 走数据并行、专家按 rank 分片、token 走 plain NCCL `all_to_all_single` 的 dispatch/combine。它同时解决**显存**和**精度**两件事,后者才是关键:**每个专家看到来自所有 rank 路由给它的 token,amax 是在全局 token 池上统计的,而不是一个 rank 的那一份**——amax 的覆盖面本身就是精度变量。为保证导出的 activation scale 逐位可复现,combine 必须定序求和(旧 `index_add_` 版在 Qwen3-30B-A3B 上 **27% 的 input scale 与单卡不同、最大偏 28%**;GB200 EP=1 对照 `grouped_mm` 从 223 个差异元素降到 **0**),缓冲尽早释放把单层传输峰值 RSS 从 **2122 MiB 压到 706 MiB**。注意:本 PR 只是传输层,端到端 `hf_ptq.py` 的 EP 入口在 [21/21],**尚未全部落地**。

## 1. 为什么复现和纯 FSDP2 都不行

PR 正文的原话:"Calibrating a large MoE has no good option today."两条现成路线各自的死因不同,要分开看:

| 路线 | 显存 | 前向开销 | 精度(amax 覆盖面) |
|---|---|---|---|
| 复现(replicate) | 每个 rank 都要整模型 | 无额外通信 | 每 rank 只看自己的 token 切片 → **专家 amax 只覆盖局部 token** |
| 仅 FSDP2 分片 | 权重存储省了 | 每次前向都要 **all-gather 所有专家**回每个 rank → 专家权重支配显存与时间 | 同上,每 rank 仍只看到自己的 token 切片 |
| **DP=EP(本篇)** | 专家权重按 rank 分片,不复现 | 一次 dispatch + 一次 combine 的 all-to-all | **每个专家看到来自所有 rank 路由给它的 token → amax 覆盖全局 token 池** |

FSDP2 那条要特别说清:PR 正文的措辞是 "gathers every expert on every rank for each forward"——分片了又在每次前向时聚回来,**这不是优化不到位,而是根本没解决显存问题**。而精度那一列是两条旧路共同的、更隐蔽的死因:PTQ 的激活量化靠校准时统计的 amax 定 scale,数据并行下每个 rank 只喂自己的 batch 切片,某个专家若只被本 rank 的 token 路由到,它的 amax 就只见过全局 token 池的一个切片——这是**系统性偏差**,不是噪声。DP=EP 让 token 去找专家(而不是专家来找 token),每个专家的前向天然汇聚了所有 rank 路由给它的 token,amax 统计口径自动变成全局。

DP=EP 的具体分工:**dense 层 + router 数据并行**(每 rank 跑自己的 batch),**routed experts 按 rank 分片**(EP),token 经 all-to-all dispatch 到拥有目标专家的 rank、算完再 combine 回来。专家矩阵乘留在 transformers 自己的 `_grouped_linear` 上——[3/21] 会在那里像单进程一样挂上 ModelOpt 的 per-expert quantizer。

## 2. 机制:四步流程 + 八个必须做对的细节

入口是 `a2a_moe(group, hidden_states, topk_idx, topk_weights, num_local_experts, expert_fn)`(moe.py:54)。四步主流程:

```
(tok, expert) 对展开 → 按 dest_rank = global_expert // num_local 路由
  → all_to_all 把 token(+ 本地 expert id)发过去
  → 本地 _grouped_linear 跑本地专家
  → all_to_all 发回 → 按路由权重加权并求每个 token 的 top_k 和
```

伪代码(两次 all-to-all 与"权重不上路"是重点):

```python
# 伪代码:一层 MoE 的专家阶段。每个 rank:dense+router 跑自己的 batch,专家只持有 num_local 份
# ---- dispatch ----
dest = global_expert_id // num_local_experts         # 每个 (token, expert) 对的 owner rank;-1 槽位不发
# 按 dest 稳定排序;bincount 得 in_counts;一次 all_to_all_single 换 out_counts(host 必须 wait);Q = sum(out_counts)
all_to_all(recv_hidden[Q, H], send_hidden)           # token 以模型 dtype 上路
all_to_all(recv_eid[Q],       local_eid_sorted)      # 本地 expert id 随行
# 注意:topk_weights 不上路 —— 留在 token 的属主 rank,保持原 dtype

# ---- 本地专家计算 ----
# 按 recv_eid 稳定排序分组 → expert_fn(recv_hidden[order], offsets) → transformers _grouped_linear

# ---- combine ----
all_to_all(send_back[P_sent, H], eo_recv)            # 专家结果不带权回来(与 token 同 dtype)
out_rows[pair_rows_sorted] = send_back               # 放回 (token, slot) 序;-1 槽位的行为零
out[t] = Σ_slot topk_weights[t, slot] * out_rows[t, slot]   # 本地权重、固定次序:view(n, top_k, H).sum(1)
```

以下八个细节每一条都能在 `moe.py` 里指到行,且大多对应 PR incident 表里的真实事故:

**① -1 槽位不发送。**router 的 top-k 输出可能有"未选中"的占位槽(expert id = -1),`pair_rows = (exp >= 0).nonzero()` 先把它们滤掉,只为真实 (token, expert) 对排传输。combine 时这些槽位的行补零,保证每个 token 的 top_k 结果在内存里相邻。

**② 计数交换后 host 必须 wait。**按 `order = torch.argsort(dest, stable=True)` 把待发对按目的 rank 稳定排序成批,`torch.bincount` 算出 `in_counts`,再用一次 `all_to_all_single`(`async_op=True`)换回 `out_counts`,`Q = sum(out_list)` 决定接收缓冲的行数。源码注释写明:host 读计数来给 dispatch 定尺寸,**读的时机若和 collective 的 stream 竞争,会用垃圾值定尺寸,下一次 all-to-all 直接崩**——incident 表里对应 Qwen3.5-MoE EP 校准的真实事故,所以 `_cnt_work.wait()` 不能省。

**③ 路由权重不上路。**token 以模型 dtype(hidden_states 的 dtype)传输,专家结果**不带权**回来;持有该 token 的 rank 用它本地的、**保持原 dtype 的** `topk_weights` 相乘。推论:router 返回 fp32 权重时(Mixtral、DeepSeek-V3 的 router 都是),加权乘积就是 fp32 的——旧版曾把权重舍入到 bf16,单测里"fp32 路由权重 + bf16 专家、EP=1 与 `grouped_mm` 逐 bit 相同"这条在旧版会挂。权重的梯度也来自这个本地乘积。

**④ combine 必须定序。**`out_rows[pair_rows_s]` 把结果放回 (token, slot) 序后,按 `weighted.view(t1 - t0, top_k, H).sum(1)` 求和,一次一段 token(切片上限 `_COMBINE_SLICE_ELEMENTS = 2**25` 元素,临时量有界)。**为什么不能用 `index_add_`**:CUDA 上它是原子加,求和次序随 token 分布变 → 不同 EP degree、不同次运行舍入不同 → **导出的 activation scale 会变、checkpoint 不可复现**。实测:Qwen3-30B-A3B top-8,旧版 **27% 的 input scale 与单卡不同、最大偏 28%**;GB200 上 fp32 权重 EP=1 对照 `grouped_mm`(4,194,304 个元素),旧 `index_add_` 版有 **223** 个不同(4 卡各 rank 217–232),定序版 **0**。

**⑤ 2^31 元素溢出要沿 hidden 切片,且 hop 数全 rank 对齐。**`all_to_all_single` 用 **int32 元素偏移**寻址扁平缓冲,`rows × hidden > 2**31 - 1`(`_A2A_MAX_HOP_ELEMENTS`)就溢出,**collective 会一直挂到 NCCL 超时**。处理:沿 hidden 维切片,`step = _A2A_MAX_HOP_ELEMENTS // max(most_rows, 1)`。关键在 `most_rows` 的来源:**每层一次 `all_reduce(MAX)` 对齐各 rank 的行数**——每个 rank 发的行数取决于自己的路由、各不相同,而所有 rank 的 hop 次数必须一致;用本地尺寸各自决定会 desync → 死锁。事故口径:Qwen3.5-MoE EP 校准 batch 8 × seq 4096 → rows × hidden = **2.68e9** 元素。切片宽度的注释还有一句值得记:按"最多行数"算步宽,而不是把 H 均分——均分会把每片向上取整,可能反而超限。

**⑥ 传输缓冲尽早释放。**每个 `[rows, H]` 缓冲在其阶段一结束就 `del`(`send_hidden` / `recv_hidden` / `eo` / `eo_recv` / `send_back`),源码注释:"留到最后的六个 [rows, H] 缓冲就是这一层的峰值内存"。实测(CPU、4096 token、hidden 4096、top-8、fp32 路由权重):**峰值 RSS 2122 MiB → 706 MiB**。

**⑦ 路由塌缩告警,判据必须是"任何 rank 发出的最大值"。**某 rank 收到的 token 数 Q 超过"任何 rank 发出量的 4 倍"(`_A2A_SKEW_WARN_RATIO = 4`)就 warn,且在 O(Q×H) 分配**之前**发——塌缩的 router 可能在那里直接 OOM,告警要赶在前面。一个 rank 最多收到 world_size 倍,所以这条只在 >4 卡时可能触发。判据的细节是事故换来的:不能用"本 rank 发出的量"比,否则**大小不一的 batch + 均匀路由会误报**(单测专门覆盖这条,旧版会挂);all-reduce(MAX) 顺带把 `most_sent` 也取回来了,一次 collective 拿两个最大值。

**⑧ 梯度要过得去。**AutoQuantize 靠梯度打分,所以构造计算图且输入 `requires_grad` 时,collective 换成 `torch.distributed.nn.functional.all_to_all_single`(反向就是反向 all-to-all);纯前向走普通 `dist.all_to_all_single`。切片路径同理分叉:需要梯度时用 `torch.cat` 拼接(in-place `copy_` 会破坏 backward,cat 的高峰值对梯度打分的少量 token 可接受);纯前向时**原地写回**输出切片,省一份 `[rows, H]` 峰值——校准里最大的缓冲就是它。边界情况:"某个 rank 一个 token 都没发"(路由槽全 -1 或干脆没 token)也要跑一次反向 all-to-all,否则对端在反向里等死;单测给 group 设 10 s 超时,退化会失败而不是挂住。

## 3. 与 transformers 的接法:`register_torch_a2a()`

传输层不是独立框架,而是**注册进 stock transformers 的 EP 基础设施**(parallel style 概念见 transformers *Expert parallelism* 文档),且用法是内部的——PR 明说 "Internal to the expert-parallel loader ([9/21]); nothing to call directly"。

- **注册内容**:`ALL_EXPERTS_FUNCTIONS.register("torch_a2a", torch_a2a_experts_forward)`,以及 `ALL_PARALLEL_STYLES` 里的两个 style:
  - `torch_a2a_router`(继承 `RouterParallel`):把 router 输出**原样透传**。必须发**全局、未被 mask 的 expert id**,因为 a2a dispatch 靠 `global_expert // num_local` 算目的 rank;基类的 `_prepare_output_fn` 会重映射成本地 id 并 mask 掉非本 rank 的——那样路由信息就没了。
  - `torch_a2a_experts`(继承 `TensorParallelLayer`):`_prepare_input_fn` 的**第 4 个返回值**注入 EP `process_group`(`device_mesh.get_group()`),并把 all-reduce 关掉(a2a 的 combine 已经做了归约)。`should_use_local_tensors` 返回 True:a2a 前向自己用 `_grouped_linear` 跑本地专家分片,而 `aten._grouped_mm` 拒绝 DTensor 参数配普通激活;`shard_tensor` 是 no-op(专家权重经 packed_colwise/rowwise + `grouped_gemm` 分片)。
- **必须 lazy 注册**:在 `load_and_prepare_ep` 里调,不在 import 时做——否则 `import modelopt.torch.utils.moe` 就带 transformers 副作用。幂等由 `_TORCH_A2A_REGISTERED` 保证。
- **与 megamoe 的关系**:这两个 style 功能上等价于 transformers **5.10.4** 为 DeepGEMM Mega-MoE 加的 `megamoe_router` / `megamoe_experts`,这里重写一遍是为了在 **< 5.10.4** 也能用;基类(`RouterParallel`、`TensorParallelLayer`)在 transformers **>= 5.6**(EP 基础设施的最低版本,`load_and_prepare_ep` 强制)都是原生的。PR 自述相关单测在 transformers **5.18 和 5.5** 上跑过——注意 5.5 低于它自己声明的 5.6 下限,这与"下限是 `load_and_prepare_ep` 强制"的说法有张力:实际可用版本以端到端 loader([9/21] 起)的检查为准,别把 5.5 当成受支持基线。
- **版本兼容坑(值得单列)**:
  - transformers **5.16.0** 只导出 `EpRouterParallel`(没有 `RouterParallel` 别名),**5.16.1 才恢复** → 源码用 try/except ImportError 兜底。
  - **>= 5.16 改名了 hook**:`_prepare_input_fn` / `_prepare_output_fn` **不会被调用**,必须同时实现 `transform_inputs_pre_forward` / `transform_output_post_forward` / `should_use_local_tensors` / `shard_tensor`,否则 EP process_group 注入不进去、前向直接 raise(`torch_a2a_experts_forward` 对 `process_group=None` 显式报错,提示用 `torch_a2a_experts` style)。
- **per-expert bias 必须随权重一起分片**:GPT-OSS 的 `gate_up_proj_bias` / `down_proj_bias` 按**本地 expert id** 索引;若分片计划只切了权重、bias 留整份,**每个非首 rank 都会加上别的专家的 bias**——静默的数值错误。`torch_a2a_experts_forward` 在任何 collective 之前检查 bias 行数 == 本 rank 专家数,不符直接 raise("出错好过静默错")。`expert_fn` 里用 `torch.repeat_interleave(torch.arange(num_local), counts)` 把 per-expert bias 铺成 per-row bias 喂给 `_grouped_linear`。
- **`self.num_experts` 是本 rank 的份额**:EP loader 通过 transformers 的 expert-parallel sharding(`grouped_gemm` / packed styles)沿 dim 0 切专家权重,并把 `num_experts` 设成 per-rank 数量。

## 4. 数据与事故汇总

| 项 | 旧版 | 本版 | 口径 |
|---|---|---|---|
| 单层传输峰值 RSS | **2122 MiB** | **706 MiB** | CPU、4096 token、hidden 4096、top-8、fp32 路由权重 |
| EP=1 vs `grouped_mm` 差异元素(共 4,194,304) | **223**(1 GPU);4 GPU 各 rank **217–232** | **0** / 每 rank **0** | GB200、fp32 路由权重;旧版 combine 用 `index_add_` |
| 导出 input scale 与单卡不同 | **27%**,最大偏 **28%** | —(定序 combine 修掉) | Qwen3-30B-A3B、top-8、bf16 |
| int32 溢出挂死 | rows × hidden = **2.68e9** > 2^31,collective 挂到 NCCL 超时 | hidden 维切片 + 每层一次 all-reduce(MAX) 对齐 hop 数 | Qwen3.5-MoE EP 校准、batch 8 × seq 4096 |
| 塌缩告警误报 | 大小不一 batch + 均匀路由**误报**(比的是本 rank 发出量) | 不告警(比"任何 rank 发出的最大值") | 单测覆盖,阈值临时调低到 2 rank 可触发 |

三条 incident(计数交换竞争、int32 溢出、`index_add_` 不定序)都发生在真实模型的 EP 校准中,不是构造用例——这是"正确性优先的传输层"里那些不显眼代码的存在理由。注意本 PR **不报吞吐数字**:传输是 plain NCCL,定位就不是追吞吐(见 §7 与 DeepEP 的分野)。

## 5. 21 个 PR 的分期与当前可用边界

PR 正文明确:这是给 `examples/hf_ptq` 加 EP PTQ(torchrun 下)的系列 **1/21**,按顺序合并:

| 分期 | 内容 |
|---|---|
| [1/21]–[3/21] | 积木:本 a2a 传输、**在数据并行校准下仍成立的 amax 同步**([2/21])、`grouped_mm` 下的 fused-experts 量化([3/21]) |
| [4/21]–[12/21] | 分布计划:每 run 决定一次的 `DistributionPlan`,`load_pretrained` 按它加载(复现 / FSDP2 分片 / 专家分片);**到 [12/21],凡是 transformers 自带可用 EP 计划的 MoE 都能 EP 加载并校准** |
| [13/21]–[17/21] | 模型 spec 与导出:transformers 没有可用计划的(如 **DeepSeek-V3**)由 spec 接手,路径处理不了的直接拒绝;[14/21]–[17/21] 不 gather 地写出分片 HF checkpoint |
| [18/21]–[20/21] | 靠 spec 接入的模型:加载时解压 compressed-tensors 专家、Kimi 系列 |
| [21/21] | 在 `hf_ptq.py` 里串起来(Changelog 条目也随它落) |

**当前边界的正确措辞**:传输层已合并(2026-10-24,commit `0638a27`),端到端 `hf_ptq.py` 的 EP 入口在 [21/21],**尚未全部落地**——不要写成"ModelOpt 现在支持 EP PTQ 了"。本 PR 的 Usage 一节也只有一句:内部给 EP loader([9/21])用,没有可直接调用的入口。系列的模型跑通验证(4 到 64 ranks)挂在 [21/21] 的口径下。

## 6. 实践 checklist

**判断要不要走这条路**

- [ ] 模型是大规模 MoE、要做**带激活量化**的 PTQ(amax 要全局覆盖面),且单卡/复现装不下 → 复现和纯 FSDP2 都不成立,EP 是正解;但先确认系列进度(端到端在 [21/21])。
- [ ] 只想权重 RTN/无激活统计 → 覆盖面问题不出现,别为此上 EP。

**接线时(transformers 侧)**

- [ ] transformers >= 5.6(EP 基础设施下限);**5.16.0 有 `EpRouterParallel` 别名坑**,5.16.1 修复;>= 5.16 必须实现新 hook(`transform_inputs_pre_forward` 等),否则 process_group 注入不进去。
- [ ] router style 必须发**全局、未 mask** 的 expert id(dispatch 靠 `global_expert // num_local` 定目的 rank)。
- [ ] per-expert bias(GPT-OSS 类)必须随权重按 dim 0 分片;源码会在任何 collective 前 raise——见到这个报错去修分片计划,不要绕过。
- [ ] 注册要 lazy(用时调 `register_torch_a2a()`),不要 import 时注册。

**自己写/改 all-to-all 传输时(通用教训)**

- [ ] combine 求和**永远定序**(`view(n, top_k, H).sum(1)`),不要 `index_add_`——CUDA 原子加的次序不确定性会渗进导出的 activation scale。
- [ ] `rows × hidden` 接近 **2^31** 就要沿 hidden 切片;切片宽度按"全 rank 最大行数"算(all-reduce MAX 对齐),**hop 数各 rank 必须一致**,否则死锁。
- [ ] host 读 collective 的结果(计数)前必须 wait,别和 stream 竞争。
- [ ] 路由权重不上路:让结果不带权回来、在属主 rank 用原 dtype 权重加权,fp32 router 就保住 fp32 乘积。
- [ ] 每个传输缓冲用完即 `del`;峰值就是留到最后的那几个 `[rows, H]`。
- [ ] 梯度路径:构图 + `requires_grad` 时用 autograd 版 collective;"零发送" rank 也要参与反向 all-to-all,否则对端等死。
- [ ] 塌缩告警的判据用"任何 rank 发出的最大值",不是本 rank 的;告警发在大分配之前。

## 7. 与库内其他篇的关系

- [大规模 MoE 的 GPTQ 工程化](moe-gptq-layerwise-calibration.md)——那篇讲**单卡/单进程**的逐层 GPTQ(逐专家 Hessian、逐层校准、4.9 TB 单卡 GB300),解决的是"显存 × 层数";本篇是**多卡的另一个维度**:"显存 × 卡数 + 校准 token 的覆盖面"。两者**正交且可叠加**:逐层校准里每一层的前向仍然可以是 DP=EP。那篇的教训"专家是 MoE 的绝大部分权重"在本篇换了个形态——专家分片了,但每个专家的 amax/Hessian 要见到全局 token。
- [deepseek-infra/concepts/deepep-nvidia-vs-ascend](../../deepseek-infra/concepts/deepep-nvidia-vs-ascend.md)——**别把两者写成同一个东西**:DeepEP 是**训练/推理**用的高性能 EP 通信库(IBGDA、NVSHMEM、NCCL Gin,通信藏进专家 GEMM 间隙);本篇的 `torch_a2a` 是**校准工具链里的正确性优先实现**——plain NCCL `all_to_all_single`,不追吞吐,追"逐 bit 可复现、任意 rank 数、跨节点能跑"。同一个 all-to-all 语义,两套完全不同的取舍。
- [computer-vision/frameworks/onnx-tensorrt-deployment](../../computer-vision/frameworks/onnx-tensorrt-deployment.md)——ModelOpt 在那篇里是 TensorRT 部署的前处理;本篇说明 ModelOpt 远不止 TRT 前处理,它的 PyTorch 侧校准工具链本身就是分布式系统工程。
- [PTQ:GPTQ / AWQ / SmoothQuant](ptq-gptq-awq-smoothquant.md)——激活量化 scale 从哪来(amax 统计)的算法侧背景;本篇是"统计量在多卡下怎么不失真"的工程侧。

## 溯源

- 机制、incident 表、内存/GB200/Qwen3-30B-A3B 数据、21 PR 分期、测试口径:ModelOpt PR #2738 正文(2026-10-24 合并,commit `0638a27`,作者 shengliangxu)。数字均为 PR 自述,**未独立复现**;GPU 结果注明测自该分支更早 revision,本 revision 的 GPU 测试只跑 CI。
- API 名、常量(`_A2A_MAX_HOP_ELEMENTS = 2**31 - 1` / `_COMBINE_SLICE_ELEMENTS = 2**25` / `_A2A_SKEW_WARN_RATIO = 4`)、代码行为(计数 wait、切片 cat vs copy_ 分叉、bias raise、lazy 注册、5.16 兼容):`modelopt/torch/utils/moe.py` 全文(362 行)逐条 grep 核对。
- transformers EP 基础设施(>= 5.6 下限、parallel style / `ExpertsInterface` 概念):Hugging Face transformers *Expert parallelism* 文档,阅读日期 2026-10-24。
- 本篇写作时端到端入口([21/21])未合并,凡"当前可用"表述均以 2026-10-24 仓库状态(`f29bb60` → `0638a27` 区间)为准。
