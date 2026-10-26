---
title: 大规模 MoE 的 GPTQ 工程化:逐专家 Hessian、逐层校准与内存预算
type: concept
tags: [quantization, gptq, moe, fused-experts, layerwise, hessian, nvfp4, imatrix, modelopt, memory, single-gpu]
source:
  - "code: NVIDIA/Model-Optimizer PR #2840 *Run GPTQ on HF fused MoE experts*(2026-10-22 合并:逐专家 Hessian 的机制、内存公式、bit-exact 测试口径)| https://github.com/NVIDIA/Model-Optimizer/pull/2840 | ↑primary"
  - "code: PR #2864 *[2/3] Speed up layerwise GPTQ on MoE models*(2026-10-23 合并:三处白等的消除、RTX PRO 6000 分阶段与 GB300×4 端到端实测、KL 对照)| https://github.com/NVIDIA/Model-Optimizer/pull/2864"
  - "code: PR #2877 *Launch the fused NVFP4 GPTQ kernel on its tensors' GPU*(2026-10-23 合并:device_map=\"auto\" 多卡下的 Triton 当前设备 bug)| https://github.com/NVIDIA/Model-Optimizer/pull/2877"
  - "code: PR #2945 *Fit layerwise calibration to GPU memory*(**open,未合并**:65% 比例阈值 → 实测 headroom 放置 + 逐层 batch 探测)| https://github.com/NVIDIA/Model-Optimizer/pull/2945"
  - "code: PR #2943 *Fix AutoQuant automatic batch-size probing*(2026-10-23 合并:探测必须复现打分时的真实负载)| https://github.com/NVIDIA/Model-Optimizer/pull/2943"
  - "web: ModelOpt 公告 *Quantizing a 4.9 TB Qwen3.8 Model on a Single GPU*(2026-09-28;逐层校准的 loop interchange、逐层导出/断点续跑、生产 recipe 与结果;2026-10-25 重抓正文,其 batch note 已被 #2864 改写为「默认 8192 token」口径,#2945 还将再改为实测探测)| https://nvidia.github.io/Model-Optimizer/announcements/20260928-single-gpu-ptq.html"
  - "web: ModelOpt model_calib API 文档(gptq / imatrix / local_hessian / layerwise_calibrate 的参数与断言)| https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_calib.html"
  - "code: 源码 calib_utils.py(update_hessian / GPTQHelper / _hessian_offload_device,commit f29bb60)| https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/utils/calib_utils.py"
  - "code: 源码 hf_ptq.py(逐层 batch 三段演进:硬编码 1 → `LAYERWISE_CALIB_TOKENS=8192`@#2864 → 实测探测 + 上限 32768@open #2945)| https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/hf_ptq.py"
  - "code: 源码 layerwise_calib.py(`_layer_batch_fit` / `get_max_layerwise_batch_size` 的探测算法,open #2945)| https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/utils/layerwise_calib.py"
  - "code: 源码 dataset_utils.py(`tokenizer(all_samples, padding=True, ...)` 一次性 pad 全部样本 —— 判定「batch 是否改变 pad 口径」的一手依据)| https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/utils/dataset_utils.py"
  - "code: recipe general/ptq/nvfp4_experts_only-kv_fp8_layerwise_export.yaml(layerwise.export_dir 即开关)| https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/general/ptq/nvfp4_experts_only-kv_fp8_layerwise_export.yaml"
date: 2026-10
---

# 大规模 MoE 的 GPTQ 工程化:逐专家 Hessian、逐层校准与内存预算

> **TL;DR**:GPTQ 的算法(逐列贪心 + 逆 Hessian 误差补偿)早在 [PTQ 篇](ptq-gptq-awq-smoothquant.md)讲完;真正卡住大规模 MoE 的是**工程三件事**:①HF fused MoE 的专家权重是一个 3-D `gate_up_proj` + 一组 per-expert quantizer,**旧实现按 `is_quantized_linear` 过滤,直接跳过专家**——而 MoE 里专家就是绝大部分权重;②每个专家的 Hessian 是 `in_features²` fp32,MoE 一层就是 **13–23 GiB**、整模型 **638–945 GiB**,迫使校准必须**逐层**(一层一个 Hessian 集合);③逐层一开,原来"反正一次前向跑完 92 层"的隐含假设全失效,**校准 batch=1、多余的一次 max 前向、Hessian 被赶到 CPU** 三处白等一起暴露,Qwen3.8-Flash-Next 的逐层 GPTQ 从 **~20 h 降到 44 min**(>10×)。再往上,逐层校准把"整个模型驻留"降成"一层驻留":4.892 TB 的 Qwen3.8-2.4T-A95B 在**单张 GB300** 上 3 h 04 m 量化完、峰值 141.7 GB,产物 1.444 TB(3.4×)。

## 1. 为什么"逐专家 Hessian"不是可选优化

> 下图回答「旧实现为什么跳过 MoE 专家、修复后怎么做到 bit-exact 等价 dense」:3-D 权重 + per-expert 量化器被 is_quantized_linear 挡掉,修复为逐专家建 GPTQHelper 并靠 input quantizer hook 分桶。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-moe-gptq-per-expert-hessian.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="逐专家 Hessian 的修复机制"></iframe>
</div>



GPTQ 的每一步都围绕 Hessian `H = 2·XᵀX`(输入激活的二阶矩,`in_features × in_features`)展开:量化第 q 列时按 `[H⁻¹]_qq` 判定该列多敏感,并把误差沿 `[H⁻¹]_:,q` 摊到还没量化的列上。**没有 H,GPTQ 就退化成 RTN。**

而 HF 新式 MoE 的权重布局是两个 3-D 参数:

```python
# 一个 HF transformers fused experts 的 MoE 层:专家权重是 3-D 参数,量化器是 per-expert 的 ModuleList
mlp.gate_up_proj          # [num_experts, 2*intermediate, hidden]
mlp.down_proj             # [num_experts, hidden, intermediate]
# 每个专家各自一个 weight_quantizer;旧 gptq() 按 is_quantized_linear(2-D + 单个 quantizer)过滤,这批全被跳过
```

旧版 `gptq()` 只收集 `weight.dim()==2 且恰有一个 weight_quantizer` 的模块(`is_quantized_linear`),这批 3-D 参数**完全落在 GPTQ 之外**——它们走 RTN,或者在内置 IQ1/IQ2 recipe 里由 imatrix 兜底。代价有量级感:PR #2840 的原话是**在 dense 模型上,单靠 imatrix 只能补上 plain encode 到 GPTQ 之间 53–63% 的差距**;而在 MoE 里,被跳过的这批正是"绝大部分权重"。

修复后的机制值得记一下,因为它决定了能否做到 **bit-exact 等价于 dense**:

- `gptq()` 为**每个专家 × 每个投影**各建一个 `GPTQHelper`(`GPTQHelper(weight_quantizer=q, weight=w)`),`update_weights()` 原地写回 3-D 参数里的那一片;
- Hessian 从**投影共享的 input quantizer 上的 forward hook** 取,并靠 `_current_expert_idx` 分桶——于是每个专家拿到的**只有路由到它的 token、且是输入量化之后**的激活,与 dense 路径的口径一致;
- 校准集里**一个 token 都没路由到**的专家,Hessian 为零 → 退回纯搜索(和 dense 层"校准集没覆盖"的 fallback 同一条路),不是报错也不是留 NaN;
- 原生 Transformers FP8 专家的权重是 8-bit 打包、校准拿到的是 dequant 副本,**原地更新会丢** → GPTQ 与 imatrix 在动任何校准状态之前就拒绝,要求先 `unpack_weight()`。

## 2. 内存公式:一层 MoE 的 Hessian 有多大(可自己算)

> 下图把 Hessian 大小的计算链画成数据流(dim → 每投影 → 每专家 → 每层 → 整模型),回答「一层 MoE 的 Hessian 为什么能到 13–23 GiB、整模型 638–945 GiB」。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-moe-gptq-memory-formula.dataflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="一层 MoE 的 Hessian 有多大"></iframe>
</div>



每个专家的每个投影贡献一个 `in_features² × 4 B` 的 fp32 Hessian。所以一层的专家 Hessian ≈ `num_experts × Σ_proj 4·in²`,整模型再乘层数:

| 模型(PR #2840 口径) | 每专家 | 每 MoE 层 | 整模型 |
|---|---|---|---|
| Qwen3.8-Flash-Next(512 专家,hidden 2560,intermediate 640,48 层) | `4·2560²`=**26.2 MB** + `4·640²`=1.6 MB ≈ 27.9 MB | **13.3 GiB** | **638 GiB** |
| GLM-5.3-Flash(288 专家,hidden 4096,intermediate 2048) | 67.1 MB + 16.8 MB ≈ 83.9 MB | **22.5 GiB** | **945 GiB**(42 个 MoE 层) |

复算校验:512×27.9 MB = 13.3 GiB ✓;288×83.9 MB = 22.5 GiB ✓;26.2 MB 也正好是 PR #2864 里"每次 hook 调用拷 26 MB 到主机"的那个矩阵。**结论**:638–945 GiB 的 Hessian 不可能和 BF16 权重一起常驻——这就是为什么 IQ 系列 recipe 默认开 `layerwise`,也是 `gptq()` 在分配前先报大小、超了就警告的原因。

## 3. 逐层校准:一次 loop interchange 换来"一层驻留"

> 下图把常规校准与逐层校准的循环结构画成对照,回答「一次 loop interchange 怎么把整模型驻留降成一层驻留、为什么边界激活才 ~2 GiB」。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-moe-gptq-layerwise-loop.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="逐层校准的 loop interchange"></iframe>
</div>



公告把机制说得很干净——不是新算法,是**把数据在外、深度在内的循环换成深度在外、数据在内**:

```python
# 常规校准:每个 batch 走全部 92 层,整模型必须驻留(或每 batch 从 CPU/盘重灌一次全权重)
for batch in calib_data:
    h = embed(batch)
    for layer in model.layers:
        h = layer(h)

# 逐层校准:边界处一次性算好全部激活,之后深度在外
acts = [embed(batch) for batch in calib_data]      # 只留边界激活
for layer in model.layers:
    for i, h in enumerate(acts):
        acts[i] = layer(h)                          # 同一时刻只有一层活着
    calibrate(layer); quantize(layer); export(layer); release(layer)
```

正确性来自一条依赖关系:第 i 层的校准统计**只依赖进入第 i 层的激活**,而第 i-1 层刚产出它。内层循环一结束,这一层"见过的全部数据"就都见完了,于是同一步里可以校准、量化、**导出自己的 shard**、然后释放——不再需要第二次全模型遍历来做导出。

### 3.1 边界代价可以被算出来(而不是"大概慢一点")

常规校准每个 batch 只有一个激活张量在飞;逐层校准要把**整个校准集**的激活按层存着。这笔开销与**层数无关**,只随样本数 × 序列长 × hidden 增长:

| 校准集 × 序列 × hidden | 边界激活(bf16) |
|---|---|
| 512 × 512 × 2560 | 1.25 GiB |
| 512 × 512 × 4096 | 2.00 GiB |
| 1024 × 512 × 4096 | 4.00 GiB |

对照单个 batch(16×512 tokens、hidden 4096):0.06 GiB。也就是**用 ~2 GiB 常驻激活,换掉 ~600–900 GiB 的常驻权重/中间态**——这笔交易在任何单卡上都成立。

### 3.2 逐层导出顺带解决断点续跑

`layerwise.export_dir` 一设就开启(没有 CLI flag;`hf_ptq.py` 会把它改写成 `--export_path`,因为目标目录是每次运行指定的)。每层:量化 → 写自己的 checkpoint shard → 在 `<export_path>.layerwise_resume` 的 manifest 里**提交**这个 shard,并把"进入下一层的激活"存下来。

- shard 既是**产物**又是**大部分续跑状态**——不存在"半精度临时副本堆在旁边",也不欠最后一趟全模型导出;
- 续跑就是**原命令再跑一次**:已提交的层跳过,从最后一个提交边界接着校准;shard 落了但没提交的层会重做;若一次提交都没有过,重跑会**直接报错而不是覆盖**那些 shard(要重来就删掉导出目录);
- 公告的实测:一次 Qwen3.8 运行撑过 CUDA OOM、主动 kill、会话拆除、以及抹掉 site-packages 的机器重建,**跨四个进程生命周期**完成,零重算零重导;在 8 层里杀在第 2 层之后重启,产物与不间断运行**逐张量相同**;
- 续跑目录收尾时 **748 KB**,旁边是 1.4 TB 的 shard。

### 3.3 能力边界(哪些校准算法现在还配不上逐层)

- `max` / `mse` / `local Hessian`(FP8 与 NVFP4)只改 `_amax`,配 `calib_mutates_weights: false`,续跑状态小,**这是已验证的组合**;
- **GPTQ 会改写权重**,默认 `calib_mutates_weights: true`(每层要快照整层权重以支持续跑);公告明确写"per-layer export 的形状是为 GPTQ 设计的,但**这一对还没验证**,验证 PR 在路上"——所以 GPTQ + 逐层导出目前按**未验证组合**对待;
- 直接拒绝的配置:**AWQ / SVDQuant**(需要全模型 pre-quant scale 步骤,可能到第一层导出才暴露)、**tie_word_embeddings 模型**(改用 `export_hf_checkpoint()`)、**多进程如 FSDP2**(每个 rank 会写同一批 shard)、以及 `hf_ptq.py` 额外拒绝的 AutoQuantize recipe、投机解码模型、`--qformat int8_smoothquant`;
- 逐层导出会把内存里的模型留在 export 形态,所以 `hf_ptq.py` 自动替你加 `--skip_generate`。

## 4. 三处"白等":逐层 GPTQ 慢的真正原因

> 下图把三处白等(没人用的前向 / Hessian 被赶到 CPU / batch=1)与各自修复一一对应,回答「权重更新只占 9 s、整层却要 25 min 的时间去了哪」。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-moe-gptq-three-waits.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="三处白等与修复"></iframe>
</div>



原始症状很反直觉:一个 Qwen 形状的层,**GPTQ 的权重更新只占 9 s**,而整层 GPTQ 要 ~25 min。时间全在**前向**上,PR #2864 拆出三个各自独立的原因:

### 4.1 一次没人用的前向

`gptq()` 先用 `max_calibrate` 带校准前向跑一遍(为了激活 amax),再跑自己的 Hessian 前向。但 **weight-only 时权重 amax 来自权重本身**(`weight_only_quantize`),那次前向纯粹只喂激活量化器。修复:`max_calibrate(..., skip_forward_without_activation_calib=True)`——顶层 `max` 与 `imatrix` 早就是这么干的,现在 GPTQ 也加进来,**weight-only recipe 从两次前向变一次**。

### 4.2 GPU 空着,Hessian 却被赶到 CPU

旧规则:`GPTQHelper` 只要**权重所在 GPU 的使用率 > 65%** 就把 Hessian 放到 CPU。模型切在 4 张 GB300 上时,当前层所在卡 79% 占用、另外两张空着——于是每个专家的 Hessian 都去了 CPU,每次 hook 调用都要**把整个矩阵(26 MB)拷回主机并在 CPU 上重算**。

临时修复(已合并):把 Hessian 放到**本进程已经持有显存的、最空且放得下该矩阵字节数**的那张卡上,只有都不行才退 CPU;候选范围仅限本进程用过的卡,且**分布式运行保持旧规则**(别的卡可能属于别的 rank)。顺带把 `update_hessian` 的增量平均**折进一个 `addmm_`**,不再"先缩放一遍、再相加一遍"地重写整个 H:

```python
# calib_utils.py(commit f29bb60)
if hessian.device == input_flat.device:
    hessian.addmm_(input_flat.t(), input_flat, beta=scale, alpha=2 / n_samples)
else:
    hessian.mul_(scale).add_((input_flat.t() @ input_flat).to(hessian.device), alpha=2 / n_samples)
```

效果可量化:只上第 1、3 条改动时,**Hessian 落 CPU 的层要 474 s,留在 GPU 的层只要 ~80 s**。

**这条"65% 比例阈值"本质是错的**(后续 open PR #2945 已指出):阈值是"使用率 >65% 就把 Hessian 挪走",即要求卡上始终空着 **35%**——284 GB 的 GB300 上等于**预留约 100 GB**给一次校准前向,远超需要;而且该读数把 PyTorch **缓存但未使用**的 block 计为已用;更糟的是**每个 Hessian 独立决策**,同一层可能一半在卡上一半在 CPU。GLM-5.3-Flash 在 4×GB300 上跑到一半,本进程用的每张卡都过了 65%,此后每个 MoE 层的 Hessian 全进 CPU,单层 **142 s vs 43 s**,42 个 MoE 层里 21 个中招、83 min 的校准多花 **34.5 min**——而当时每张卡还空着 **90–100 GB**。修法是改成按**实测 headroom** 判:预留 = **1.5× (该卡上 GPTQ 校准前向的峰值 − 分配前已用量)**,并设 10%(占该卡容量)的下限;前向需求在 `gptq()` 的 Hessian 前向周围测量(PyTorch 的 peak 计数器不复位,历史更高的峰值只会**高估**需求——所以误差方向总是"更保守地放 CPU",不会变成 OOM);某张卡上还没测过前向时(第一层、非逐层运行)沿用原来的 35%。

### 4.3 batch=1

`hf_ptq.py` 之前给逐层运行硬编码 `--batch_size 1`,理由是"自动探测要跑整模型前向,既违背逐层初衷又可能 OOM"。但逐层的 batch 里**只有一层的激活**,按 token 定额完全安全;而 batch 1 会让**每个样本都重跑一遍 MoE 的逐专家 Python 循环**(HF fused experts 每次前向都是对专家的循环),样本成本几乎与 batch 无关(实测 **250 ms/样本 @ batch1**)。

改成一个 token 预算:

```python
LAYERWISE_CALIB_TOKENS = 8192        # 16 x 512
def _layerwise_calib_batch_size(calib_seq, num_samples):
    return max(1, min(LAYERWISE_CALIB_TOKENS // calib_seq, num_samples))
```

显式传 `--batch_size` 仍然直通(要旧行为就传 `--batch_size 1`)。

公告侧的实测补充:**batch 1 → 8 把墙钟砍掉大约一半**,代价是激活 `input_scale` 有细微变化(权重、`weight_scale`、`weight_scale_2` 仍逐位相同)。所以 batch 是单次运行里**最大的性能杠杆**,同时也是校准协议的一部分——见 §5 对这条因果链的更正。

### 4.4 第四个坑:每个专家都打一行日志

每个专家的 GPTQ helper 都会打自己 Hessian 加权误差,而这**要两次 weight×Hessian matmul + 一次主机同步**——GLM-5.3-Flash 一层 576 次。修复:专家级 helper `log_mse=False`,改由 `gptq()` 每个 experts 模块打一行"更新了多少专家、多少专家没被校准覆盖";dense 层保留原日志。

### 4.5 实测:分阶段与端到端

单个 Qwen 形状的层(RTX PRO 6000,512 专家、hidden 2560、intermediate 640、top-10 路由,gate_up 用 IQ2_XXS、32×512 校准):

| | GPTQ 总计 | 前向 | 专家更新 |
|---|---|---|---|
| 改前,batch 1 | 29.6 s | 2 × 9.8 s | 9.2 s |
| 改后,batch 1 | 19.0 s | 1 × 8.9 s | 9.3 s |
| 改后,batch 16 | **11.2 s** | **1 × 1.0 s** | 9.3 s |

端到端(oci-jhb,4×GB300,Qwen3.8-Flash-Next,gate_up IQ2_XXS,1024×512 校准 token):

| | 单层 GPTQ | 逐层校准(48 层;GLM 45) | 整个作业 |
|---|---|---|---|
| 改前(batch 1) | 1206–1663 s(linear-attn 层),full-attn 层到 **1862 s** | ~20 h | ~21 h |
| 本 PR(batch 16) | **29–35 s**(首层 65 s,含预热) | **44 min** | **70 min** |
| GLM-5.3-Flash 改前 | 739–2131 s | ~25 h(推算) | — |
| GLM-5.3-Flash 本 PR | 37–44 s;最后 8 个 MoE 层 126–136 s | **55 min** | **79 min** |

注意最后一行:GLM 的尾部 8 层慢 3×,正是 4.2 里"每张卡都过 65%"的副作用——这条只有 open 的 #2945 才真正修掉。

### 4.6 质量有没有被速度换掉

同一实验对 BF16 的 KL(wikitext-2,100×512,只算后半、25 500 token,按 chunk 配对):

| gate_up IQ2_XXS | 平均 KL |
|---|---|
| max | 0.2971 |
| imatrix | 0.2863(−0.0108 ± 0.0023 vs max) |
| **GPTQ(本 PR,batch 16)** | **0.2492**(−0.0479 ± 0.0033 vs max;比 imatrix 再好 −0.0371 ± 0.0034) |

GLM-5.3-Flash(全部专家)上:GPTQ **0.6298**,imatrix 0.7211,max 0.7402。**方向很重要**:补上专家 GPTQ 不只是变快,是**质量也更好**,因为 imatrix 只是对 encode 搜索加权、并不做误差补偿。

batch 16 vs batch 1 的 KL 没有直接对照(那批 batch-1 运行反复被抢占、取消了),但用 phi-2(NVFP4 W4A4 逐层,128×512 cnn_dailymail;wikitext-2 上 40×512 配对)测过:GPTQ 的 batch16−batch1 = **−0.0058 ± 0.0032**,local-Hessian **+0.0058 ± 0.0052**,都不显著。

## 5. 一个必须知道的语义细节:改 batch 会不会改结果

官方公告的说法是:batch 1→8 时**权重、`weight_scale`、`weight_scale_2` 逐位相同**,但**激活的 `input_scale` 会小幅变化**,理由是"batching 会把每条序列 pad 到 batch 内最长的那条,而这些 pad 位置参与激活 amax 统计"。

**这条因果链在后续 open PR #2945 里被纠正了**,而且从源码能直接验:`dataset_utils.py` 里 `tokenizer(all_samples, padding=True, truncation=True, max_length=...)` 是**一次性把全部样本 pad 到同一长度**、再交给 DataLoader 切 batch 的——所以 **batch 大小并不改变哪些 token 是 pad**。#2864 的 changelog 与公告这段都被 #2945 一并改正。于是 `input_scale` 的小幅差异更可能来自规约顺序(浮点求和次序),而不是"pad 口径变了";这与 phi-2 实测(GPTQ batch16−batch1 KL = −0.0058 ± 0.0032、local-Hessian +0.0058 ± 0.0052,**都不显著**)和"权重逐位相同"自洽。

#2945 给出的替换措辞(可直接对照当前线上公告的差异):`input_scale` 变动是因为**不同的 batch 形状让激活的规约/取整路径不同**("a different batch shape rounds the activations differently"),并明确补一句 **"Padding does not change: `hf_ptq.py` pads every calibration sample to the same length whatever the batch size."**

### 5.1 逐层 batch 默认值的三段演进(版本敏感,必须钉死)

> 下图把同一行代码在三个版本里的三种语义画成树,回答「逐层 batch 默认值为什么是『默认值会咬人』的教科书案例」。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-moe-gptq-batch-evolution.tree.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="逐层 batch 默认值三段演进"></iframe>
</div>



同一行代码在三个版本里有三种语义,这是"默认值会咬人"的教科书案例:

| 版本 | 逐层 batch 默认 | 依据 |
|---|---|---|
| ≤ #2864 之前 | **1**(`hf_ptq.py` 硬编码) | 理由:自动探测要跑整模型前向,违背逐层初衷、可能 OOM |
| #2864 合并后(main、commit f29bb60) | **8192 token**(seq 512 → 16) | `LAYERWISE_CALIB_TOKENS = 8192`;`max(1, min(8192//calib_seq, num_samples))` |
| #2945(**open,未合并**) | **实测探测**:每种解码层各跑一次前向,取"装得进半数空闲显存"的最大样本数,**上限 32768 token**(seq 512 → 64) | `get_max_layerwise_batch_size()`;探测失败/装不下时回退小值 |

两个可自检的换算:`8192 / 512 = 16`、`32768 / 512 = 64`(公告给的就是这两个数)。公告原文还留了一句值得注意的自述——这次 2.4 T 生产运行是在默认值仍为 1 的版本上做的,"which is why it passed `--batch_size 8`",即**它的墙钟数字带着 batch 8 的加速,而不是默认值的成绩**。

#2945 的探测之所以不能直接用现成的 `get_max_batch_size()`,是因为后者要跑**整模型**前向——既高估一层所需的显存,又可能自己 OOM。它的替代方案值得抄,因为这是"给逐层流程做 batch 探测"的通用写法:

```python
# modelopt/torch/quantization/utils/layerwise_calib.py(PR #2945,open)
def _layer_batch_fit(model, layer, seq_len, num_samples, memory_fraction, device):
    # 1) 对每种解码层(按 module 类集合去重:dense/MoE × full/linear attn)只测代表层
    # 2) 跑 batch=1(两次,第一次只用来吃掉 cuBLAS workspace 这类一次性缓冲)与 batch=2
    for batch_size in (1, 1, 2):
        args, kwargs = _capture_layer_inputs(model, layer, batch_size, seq_len, device)
        peaks.append(_forward_peak_per_gpu(layer, args, kwargs))   # 每次 reset_peak 后测增量
    per_sample = two_peak - one_peak                              # 每样本的显存斜率
    # 3) 给"整轮校准集缓存的输入"留位(逐层要缓存本层与下一层的激活 ⇒ ×2)
    available = _available_gpu_bytes(gpu) - 2 * num_samples * cached_per_sample[gpu]
    fixed     = one_peak - per_sample                             # 与 batch 无关的固定开销
    return int((memory_fraction * available - fixed) // per_sample)   # memory_fraction 默认 0.5
```

三个细节都是踩过坑才有的:`batch=1` 要**跑两次**(第一次带上一次性 workspace 分配,算进斜率会低估 batch)、**斜率**而非绝对值(用 1 vs 2 的差,消掉固定开销)、探测完 `allgather` 取**所有 rank 的最小值**(多卡各算各的,小的那个才安全)。探测 OOM 时另有一条硬错误:逐层 kernel 是"replay 缓存 batch"的,拆小 batch 救不了,只能报错让人调小 `--batch_size`——这一点比默默重试有用得多。

实践结论不变,但理由要换:**要比两次运行的产物,就显式钉死 `--batch_size`**;公开数字里凡是"batch 变了但结果等价"的断言,先问它指的是权重还是激活 scale。这与 [PTQ 篇](ptq-gptq-awq-smoothquant.md)里"校准集决定一切"是同一条原则的工程版:**batching 也是校准协议的一部分**。(本段保留争议双方:公告原表述 vs #2945 的更正;以代码行为(#2945 所述 pad 在 batching 之前)为准更可信,但它仍是 open PR。)

## 6. 可复现:逐层 GPTQ 的最小配置

```yaml
# recipe 片段:layerwise 的开关是 export_dir 的存在,没有 CLI flag
quantize:
  algorithm:
    method: gptq            # 或 max / mse / local Hessian
    layerwise:
      enable: true
      get_qdq_activations_from_prev_layer: false   # 下一层看 FP 输入(与全模型 pass 一致)
      calib_mutates_weights: true                  # GPTQ 改权重,默认 true
      export_dir: /tmp/modelopt_layerwise_export    # 存在即开启;hf_ptq.py 会改写成 --export_path
      # checkpoint_dir 省略 → <export_path>.layerwise_resume
```

```bash
python examples/hf_ptq/hf_ptq.py \
  --pyt_ckpt_path <model> --recipe <layerwise recipe> --export_path <out> \
  --qformat nvfp4 --attn_implementation eager \
  --offload_folder <scratch> --max_gpu_memory_gb 120 --max_cpu_memory_gb 600 \
  --calib_size 512 --calib_seq 512 --batch_size 8
```

`--max_gpu_memory_gb` / `--max_cpu_memory_gb` 只作用于**权重放置**(它们给 accelerate 的 device map 定预算),激活、校准 buffer、CUDA context 都在这个数**之上**——公告实测峰值超出 GPU 预算 **21.7 GB**、主机 RSS 超出 4.8 GB。**别按机器精确容量吃饱**,两个都要留余量。

## 7. 生产数据:4.892 TB 在单张 GB300 上 3 h

| 项 | 值 |
|---|---|
| 模型 | `Qwen/Qwen3.8-2.4T-A95B`(2.4 T 参数,92 层 × 512 路由专家) |
| 源 | 4.892 TB BF16,213 个 shard |
| 硬件 | **1 × GB300(283 GB)** |
| 墙钟 | 3 h 04 m(11 026 s,≈120 s/层) |
| GPU 峰值 | **141.7 GB**(预算 120 GB,平均利用率 33.8%) |
| 主机 RSS 峰值 | 604.8 GB(预算 600 GB) |
| 产物 | **1.444 TB(3.4× 更小)**:92 个层 shard + tail + index |
| recipe 覆盖 | 路由专家 NVFP4(MSE 搜静态 weight scale + 动态 input scale)、self-attn q/k/v/o(23 层)FP8 W8A8、linear-attn/gated-delta 含 conv1d(69 层)FP8 W8A8、KV cache FP8 cast、MTP/router/shared experts/lm_head/embeddings 保持 BF16 |

量级参照(给"要不要上这条路"做判断用):**550–671B 级模型在单卡上约 40–47 分钟**跑完逐层校准+导出,2.4 T 是 3 h 04 m——也就是说墙钟大致随参数量线性涨,**单卡能不能扛取决于"最大的那一层",不是参数总量**。

两个值得记住的数字细节:
- 全部 **141,312**(92×512×3)个专家投影**各自带 `input_scale` 与 `weight_scale`**,vLLM 才会选 **FlashInfer 的 TRT-LLM NVFP4 MoE kernel** 而不是 emulation 回退——**scale 元数据的完整度直接决定服务端落到哪个 kernel**;
- GPU 显存随每层权重 materialize 上升、释放回落,**92 层无漂移**;这个"平顶包络"正是"峰值与深度无关"的证据,反过来说:**新模型跑前几层时如果地板持续爬升,说明有东西该释放没释放**。
- 正确性口径很窄也别放大:**逐层导出 == 全模型导出**。四个模型上两条路径逐张量、逐 config 一致(35B MoE 123,513 个张量、30B 74,163 个,0 mismatch),`weight_map` 每一项都指向持有该张量的 shard,vLLM 两边贪心生成完全相同。它**不**代表精度更好、也不代表这份 recipe 复现了官方 checkpoint 的数值。

## 8. 实践 checklist

**选型与预算**

- [ ] 先算 **一层专家 Hessian**:`num_experts × (2·hidden² + intermediate²) × 4 B`。> 单卡可用显存就必须 `layerwise`,没有别的选择。
- [ ] 再算**边界激活**:`样本数 × 序列长 × hidden × 2 B`。这是逐层校准唯一新增的常驻项,和层数无关。
- [ ] 用**最大单层 + 其校准激活**去对卡,而不是"整模型 / 卡数";逐层的下限就是这一层。

**跑起来之前**

- [ ] 记下你用的 **`--batch_size`**:默认值会随版本变(旧版硬编码 1 → #2864 起 8192 token → #2945 起改为实测探测、上限 32768 token),而它**改变激活 scale**。要对比两次运行,显式钉死。
- [ ] GPU/CPU 预算都留 10–20% 余量(实测会超预算 18%/0.8%)。
- [ ] 多卡且 `device_map="auto"` 时,确认 `fused: true` 的 NVFP4 GPTQ 不会跨卡:旧代码里 Triton 在**当前** CUDA 设备启动,而张量在别的卡上 → NVLink 机器报 `illegal memory access`(报错点还在无关算子上)、PCIe 机器报 `Pointer argument cannot be accessed from Triton`。修复是 `with torch.cuda.device(w_block.device):`——**自己写 Triton wrapper 时同理:launch 前先切到张量所在设备**。
- [ ] 原生 FP8 专家要 `unpack_weight()` 后再量化,否则原地更新丢失。

**判断"它是不是在偷懒"**

- [ ] 看日志里 Hessian 落在哪(新版会打位置)。若在 CPU 而卡上明明有空间,就是 65% 比例阈值在误判——临时办法是显式限制可见卡数/降低其他占用,根治等按 headroom 放置的实现。
- [ ] 单层耗时突然翻 3–5 倍,先查 Hessian 位置,别先怀疑数据。
- [ ] 逐层跑要开 `export_dir`:否则中断即全废;开了之后 shard 即产物,续跑原命令即可。

**结果校验**

- [ ] 逐层导出后核对:每层 shard 齐、manifest 里的提交边界覆盖全部层、`weight_map` 无悬空键;有条件就与同 recipe 的全模型导出**逐张量比对**。
- [ ] 服务端确认落到了正确的 MoE kernel(专家级 scale 元数据齐不齐是判据)。

## 9. 与库内其他篇的关系

- [PTQ:GPTQ / AWQ / SmoothQuant](ptq-gptq-awq-smoothquant.md)——**算法与推导**在本篇之外的这篇;本篇只讲 **MoE + 大规模工程的落地**,不讲闭式解的来历。
- [2-bit 权重量化前沿:QuIP# / AQLM](two-bit-weight-frontier.md)——IQ1/IQ2 recipe 的 2-bit 端,GTPQ 与 imatrix 谁管哪部分权重在那里有算法侧讨论;本篇给出"为什么 imatrix 单独不够"(只加权搜索、不做误差补偿,只能补 53–63%)。
- [structured-pruning-minitron-puzzletron](structured-pruning-minitron-puzzletron.md)——另一条也依赖"逐层"的路线:**逐层校准 vs 逐块蒸馏**,两者都把"整模型驻留"换成"一层/一块驻留",是本域之外同构的工程手法。
- [MoE 的专家并行 PTQ 校准](expert-parallel-ptq-calibration.md)——**多卡维度的正交问题**:本篇的逐层解决"显存 × 层数",EP 解决"显存 × 卡数 + 校准 token 的覆盖面"。两者可叠加——本篇逐层校准里每一层的前向仍然可以是 DP=EP;那篇的核心是"专家 amax 必须见到全局 token 池",正是本篇"专家是 MoE 绝大部分权重"的另一面。
- [vLLM fake-quant 验收](vllm-fakequant-eval-export.md)——本篇的下游:逐层导出的产物要在没有内核的引擎里评测时,走 `mtq.fold_weight` 把权重折进 BF16 + 两个 sidecar 的路径;两条链共用同一个 `fold_weight` 语义(顺序量化器/SVDQuant 折叠缺陷的修复同时影响两边)。
- [computer-vision/frameworks/onnx-tensorrt-deployment](../../computer-vision/frameworks/onnx-tensorrt-deployment.md)——部署侧视角:量化 checkpoint 如何被导出、被 TensorRT / vLLM 的 kernel 选中。

## 溯源

- 机制与内存公式、bit-exact 测试口径:ModelOpt PR #2840(2026-10-22 合并),正文与测试节。
- 三处白等的拆解、全部分阶段与端到端实测、KL 表:PR #2864(2026-10-23 合并)。数字均为 PR 自述,**未独立复现**;本文对其中可推导项(每层 Hessian 13.3/22.5 GiB、专家投影数 141,312、3.4× 压缩、120 s/层、8192/512=16)做了重算并标注一致。
- 逐层校准的 loop interchange、逐层导出/断点续跑、单卡 4.9 TB 生产数据:ModelOpt 官方公告(2026-09-28,2026-10-25 重抓核对正文)。
- 逐层 batch 默认值的三段演进、`_layer_batch_fit` 的探测算法:ModelOpt 源码 `hf_ptq.py` / `layerwise_calib.py`(commit `3c423a9f165b` 的 main 上是 8192 token 段;#2945 的 diff 给出 32768 token + 探测段)。"pad 不随 batch 变"的依据是 `modelopt/torch/utils/dataset_utils.py` 的 `tokenizer(all_samples, padding=True, ...)`,**不是**公告原话。
- 设备选择 bug 与修法:PR #2877(2026-10-23 合并);`map_location` 之外的"Triton 当前设备"是通用坑。
- 65% 阈值的问题与 headroom 方案、逐层 batch 探测:PR #2945(**open,未合并**)——文中凡引用它处均已标注"open",不要当已发布行为。
- AutoQuant 自动 batch 探测需复现打分负载:PR #2943(2026-10-23 合并),细节见 computer-vision 部署篇。
