---
title: vLLM fake-quant 验收:把权重折进 BF16、把激活量化留在运行时的量化 checkpoint 评测路径
type: concept
tags: [quantization, fake-quant, qad, nvfp4, fold_weight, vllm, megatron-bridge, modelopt, evaluation, checkpoint-export, ggml]
source:
  - "code: NVIDIA/Model-Optimizer PR #2889 *Export vLLM fake-quant checkpoints through Megatron-Bridge*(commit fb3ccd0e65bc,2026-10-24 合并;`mtq.fold_weight` 的 SequentialQuantizer/SVDQuant 折叠、`check_mcore_vllm_fakequant_quantizers` 前置校验、Megatron-Bridge 导出与 NLL 实测表)| https://github.com/NVIDIA/Model-Optimizer/pull/2889 | ↑primary"
  - "code: PR #2888 *vLLM reload of the versioned recipe*(已合并;`quant_recipe.yaml` 的版本化结构)| https://github.com/NVIDIA/Model-Optimizer/pull/2888"
  - "code: PR #2873 *在旧导出器上引入 `--vllm_fakequant_export`*(已合并,未发布;被 #2889 换掉实现)| https://github.com/NVIDIA/Model-Optimizer/pull/2873"
  - "code: 源码 `modelopt/torch/quantization/nn/modules/quant_module.py::QuantModule._fold_weight_quantizer`(折叠语义:逐成员折叠 / 禁用量化器 / 丢 buffer)| https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/nn/modules/quant_module.py"
  - "code: 源码 `modelopt/torch/export/plugins/vllm_fakequant_megatron.py`(`check_mcore_vllm_fakequant_quantizers` / `VllmFqGPTModelExporter` / `quant_recipe.yaml` 与 `quantizer_state.pth` 的 gather+merge)| https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/export/plugins/vllm_fakequant_megatron.py"
  - "code: 源码 `examples/vllm_serve/fakequant_worker.py`(vLLM 侧:恢复 quantizer state → `mtq.fold_weight` → 校验无残留启用的权重量化器)| https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/vllm_serve/fakequant_worker.py"
  - "code: 源码 `examples/megatron_bridge/export_quantized_megatron_to_hf.py --vllm_fakequant_export`(VLM 视觉塔从 HF 加载、TP/EP/PP 支持、清掉 `quantization_config`)| https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/export_quantized_megatron_to_hf.py"
  - "code: 上游依赖 NVIDIA-NeMo/Megatron-Bridge #6565(writes `quantizer_state.pth` + `quant_recipe.yaml`;ModelOpt CI 钉 `1d9caf7324`)| https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6565"
  - "web: ModelOpt 文档站 vllm_serve 示例 README(fake-quant reload 的两个 sidecar 与使用步骤,阅读 2026-10-25)| https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/vllm_serve/README.md"
date: 2026-10
---

# vLLM fake-quant 验收:把权重折进 BF16、把激活量化留在运行时的量化 checkpoint 评测路径

> **TL;DR**:量化 checkpoint 的"能不能用"通常要等 kernel——vLLM 没有 NVFP4(Hopper)/GGML IQ2/自研格式的内核时,你根本跑不起来,更别说评测 QAD(量化感知蒸馏)训出来的模型到底掉了多少精度。**fake quantization 这条路把"缺失的 kernel"换成"BF16 上的 QDQ 仿真"**,关键在于**按"静态/动态"切开两份东西**:
> ① **权重量化器可以离线折掉**——`mtq.fold_weight()` 把 `round(clamp(w/s))·s`(以及 pre-quant scale、旋转)直接写进 BF16 权重,量化器随即禁用、buffer 丢弃,于是**产物就是一个普通 BF16 模型**,连 stock `vllm serve` 都能加载;
> ② **激活 / KV-cache 量化器不能折**——它们依赖运行时输入,只能由 vLLM 侧的 `FakeQuantWorker` 在 forward 里做 QDQ,参数放在两个 sidecar 里:`quantizer_state.pth`(ranges)与 `quant_recipe.yaml`(版本化的按 HF 命名组织的设置)。
> 2026-10-24 的 PR #2889 把这条导出链从 ModelOpt 自家 Megatron exporter 换成 **Megatron-Bridge**(支持任意 TP/EP/PP 与 VLM),并修掉两个会"静默错"的折叠缺陷:**顺序量化器(W4A8 INT4+FP8)以前整个跳过、折叠后仍启用**(= 二次量化),**SVDQuant 的低秩修正项以前会在折叠时丢**。实测口径:**权重与旧导出器逐位相同、服务输出 8/8 greedy 一致(Δlogprob 0.0)**;跨引擎 per-token NLL 在 dense 模型上压到 0.0002(Nemotron-9B 2.3465 vs 2.3467),MoE 上因专家路由翻转放大到 0.016(Qwen3.6-35B-A3B 2.2844 vs 2.3008)。

## 1. 为什么需要这条路:量化评测的三重门槛

> 下图把三重门槛(格式有 kernel / 引擎能吃 / 口径一致)与 fake-quant 的绕法并排,回答「为什么要把权重折进 BF16、把激活量化留在运行时」。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-fakequant-three-gates.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="fake-quant 绕开的三重门槛"></iframe>
</div>



想回答"这个量化 checkpoint 掉了多少精度",正常路径要满足三件事同时成立:格式有 kernel、引擎能吃、评测口径一致。缺一样就卡住:

| 场景 | 卡在哪 | fake-quant 怎么绕 |
|---|---|---|
| QAD 训完的 NVFP4 checkpoint,手上只有 Hopper | **没有 NVFP4 内核**(NVFP4 的 tcgen05 路径在 Blackwell) | 折成 BF16 权重 + sidecar 里记激活/KV 的量化参数,Hopper 上也能跑 |
| GGML IQ2_XXS(2-bit 码本) | vLLM 侧只有部分 WNA16 内核,小模型/新架构常常缺 | 折叠后**没有任何量化器保持启用** → 不写 sidecar → stock `vllm serve` 当普通 BF16 加载 |
| 自研研究格式(KV `backend` + `backend_extra_args` 自定义) | 没有对应 kernel,也不会有 | sidecar 里连自定义 KV backend 与额外参数一起活下来 |
| 想对比"同一个 checkpoint 在 HF transformers / Megatron / vLLM 三处的数值" | 三处各自的量化实现 | 三处都吃同一份 `quant_recipe.yaml`,差异只剩框架与 kernel |

**这条路测量的是什么、不测量什么**,是使用前必须分清的一件事:

- **测量**:量化**参数**本身造成的精度损失——clipping 与 rounding 的误差、per-block scale 的粒度、KV cache 压到 FP8 后长上下文的退化。因为权重是"已经量化再反量化"的 BF16 值,激活是运行时真 QDQ 过的值;
- **不测量**:真实低精度 kernel **内部的数值行为**——tensor core 上的低精度累加、NVFP4 的 block scale 在每个 MMA 里怎么施加、FP4 乘法器的舍入。fake-quant 走的是 `dequant → BF16 matmul`,这一层差异只能靠真 kernel 才看得到。

> 一句话:**fake-quant 验的是"量化方案",不是"量化内核"**。QAD 训练完想快速判定"这套 recipe 值不值得上真 kernel",它是最便宜的一关;要出"实际部署精度"的结论,还得补一轮真 kernel 对照。

## 2. 机制①:权重折叠(`mtq.fold_weight`)——把量化器变成一次矩阵改写

> 下图把折叠的逐成员改写链画成数据流,回答「量化器怎么变成一次矩阵改写、三处修复分别修了什么静默错」。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-fakequant-fold-weight.dataflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="权重折叠 mtq.fold_weight"></iframe>
</div>



折叠的本质是一次**前向等价的原地改写**:把 `y = Q(w)·x`(Q 是 fake quant,即 `dequant(quant(w))`)改写成 `y = w'·x`,其中 `w' = Q(w)`。之后量化器被禁用,前向直接用存下来的权重——**产物在数值上等价于"量化后立刻反量化"**,但形式上是一个普通 BF16 权重。

```python
# modelopt/torch/quantization/nn/modules/quant_module.py(schema 化简)
def _fold_weight_quantizer(quantizer, weights, keep_attrs=False):
    members = list(quantizer) if isinstance(quantizer, SequentialQuantizer) else [quantizer]
    for weight in weights:                      # 可迭代:一个 per-tensor 量化器可被多个视图共享(fused MoE)
        folded = weight
        for member in members:                  # 顺序链逐成员折叠,每个成员的输出都回到 weight dtype
            folded = member(folded.float().contiguous()).to(weight.dtype)
        weight.data.copy_(folded)
    quantizer.disable()
    quantizer.disable_rotate()                  # 旋转(Hadamard 类)已烘进权重,不能再乘一次
    for member in members:
        if keep_attrs and hasattr(member, "_pre_quant_scale"):
            member._enable_pre_quant_scale = False   # scale 已在权重里,别再乘
        elif not keep_attrs:
            buffers = {n for n, _ in member.named_buffers(recurse=False)}
            for attr in buffers | {"_pre_quant_scale", "_amax"}:
                if hasattr(member, attr):
                    delattr(member, attr)      # 只留会导致"再算一遍"的状态,一律删
```

PR #2889 改的就是这段的三处细节,每一处都对应一个"折叠完仍然错"的真 bug:

| 缺陷 | 旧行为 | 后果 | 修法 |
|---|---|---|---|
| **顺序量化器** | `fold_weight` 只认 `TensorQuantizer`,`SequentialQuantizer` 被 `isinstance` 挡掉 | 链式格式(如 **W4A8:INT4 权重 + FP8 权重**)整条**没被折叠且仍启用** → 前向里权重被量化第二次,等价于"折叠 + 原量化器"叠加 | `members = list(quantizer)`,逐成员折叠;`fold_weight` 的收集条件放宽到 `(TensorQuantizer, SequentialQuantizer)` |
| **SVDQuant 的低秩修正** | 折叠时先读 `weight` 再丢 buffer,修正项(也是权重量化器的 buffer)来不及读 | 低秩纠正项丢失 → SVDQuant 精度白丢 | `SVDQuantLinear.fold_weight` **先读低秩因子再折叠** |
| **只删 `_amax` / `_pre_quant_scale`** | 硬编码两个属性名 | **GGML 的 pinned payload** 等其它 buffer 留在量化器上,`Megatron-Bridge` 保存折叠后模型时仍会带上它 / 或在前向里重算 | 改成删**全部** buffer(`named_buffers(recurse=False)` ∪ 两个历史属性) |

**通用教训**:折叠是"把一段计算搬进常量",所以**任何能在折叠后被重新应用的状态都必须一起清掉**——分工是"只删会重算的",而不是"删我记得的那两个"。用 `named_buffers()` 枚举比硬编码属性名可靠。

## 3. 机制②:sidecar——动不了的量化器怎么活到运行时

> 下图把 vLLM 侧恢复顺序画成流程,回答「sidecar 的恢复顺序为什么不能乱、最后那个 raise 为什么是范式」。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-fakequant-sidecar-restore.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="sidecar 恢复顺序"></iframe>
</div>



激活量化器作用于 forward 的中间张量,KV-cache 量化器作用于运行时写入的 cache,**都无法离线折进权重**。所以导出产物是"BF16 权重 + 一份按 HF 命名登记的量化器状态":

| sidecar | 内容 | 谁写 | 谁读 |
|---|---|---|---|
| `quantizer_state.pth` | 每个启用量化器的 ranges(`_amax`、常数 amax 等) | `gather_mcore_vllm_fq_quantized_state_dict`(跨 rank gather 后 merge) | `fakequant_worker.py` 的 `set_quantizer_state_dict` |
| `quant_recipe.yaml` | **版本化**的设置:每个 HF 量化器名一条 `quantizer_state`(格式、块粒度、是否旋转、KV 的 `backend`/`backend_extra_args`) | `gather_mcore_vllm_fq_quantizer_recipe` | `validate_quantizer_recipe_for_model` + 模型侧建量化器 |

vLLM 侧的顺序(见 `fqw.py` 的 `_fakequant_run_prolog_worker`)值得记,因为顺序错了会静默错:

```python
# 1) 读 sidecar,按 HF→vLLM 的名字映射转换后恢复到模型
modelopt_state = convert_modelopt_state_to_vllm(safe_load(path, map_location="cpu"), map_fun)
restore_from_modelopt_state_vllm(model, modelopt_state)          # 建出量化器并载入 ranges
mtq.utils.set_quantizer_state_dict(model, modelopt_weights)      # 再载 values
post_restore_vllm_parallel_linears(model)                        # 合并后的 ParallelLinear 修正
shard_pre_quant_scale_for_tp(model)                              # 必须在 post_restore 之后:靠 weight H_in 判 pqs 长度
# 2) 权重折叠,然后**硬校验没有权重量化器还活着**
mtq.fold_weight(model)
for name, module in model.named_modules():
    if is_weight_quantizer_state_key(name) and module.is_enabled:
        raise RuntimeError(f"Weight quantizer {name!r} is still enabled after fold_weight — "
                           "double-quantization would corrupt activations.")
```

这段最后那个 `raise` 是**把"折叠没生效"变成显式失败**的范式:量化器没折干净不会崩、也不会 NaN,只会让权重被量化两次、指标轻微变差——这类错误用数值比对根本抓不住,只能断言"该关的都关了"。另外注意 `restore_quantizer_state` 会跑 `modelopt_post_restore`,而 `set_quantizer_state_dict` **不跑**,所以 `post_restore_vllm_parallel_linears` 必须手动补一次,且 `shard_pre_quant_scale_for_tp` 要排在它之后(Tensor Parallel 下 pre-quant-scale 要按权重形状切分,而形状是 post_restore 才定的)。

**为什么"权重全折完就可以用 stock vLLM"?** 因为权重-only 格式(如 IQ2)在折叠后没有任何量化器保持启用 → 不写 sidecar → 导出物就是纯 BF16 权重。PR 实测 IQ2_XXS 的 Qwen3-0.6B 折叠前后 Megatron 侧 logits **逐位相同**,stock `vllm serve` 直接吃。这也说明**"没有 sidecar"不是缺产物,而是产物形态正确的信号**。

## 4. 机制③:为什么要换到 Megatron-Bridge 导出

> 下图把换 Megatron-Bridge 换来的三件能力与「校验前移」并排,回答「为什么换导出器、拒绝清单为何在动权重之前统一拒掉」。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-fakequant-mbridge-export.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="Megatron-Bridge 导出"></iframe>
</div>



旧实现(`#2873` 引入的 `--vllm_fakequant_export`)走 ModelOpt 自家 Megatron exporter,支持面窄。换到 Megatron-Bridge 后多出三件实质能力:

1. **任意并行度**:TP / EP / PP 全支持(`--tp_size` 是新增参数,**只允许与 `--vllm_fakequant_export` 同用**;旧的 unified exporter 仍要求 TP=1)。原因是 M-Bridge 的 `save_hf_pretrained` 直接存内存里已经 shard 好的模型;
2. **VLM**:QAD 的 checkpoint **只存语言模型**(视觉塔不参与量化训练),所以导出前要先从 HF 把视觉塔权重载入——代码里就是 `load_weights = args.vllm_fakequant_export and is_vlm_config(...)`;
3. **权重量化器折叠后的配置清理**:权重已是 BF16,若 `config.json` 里还留着源 checkpoint 的 `quantization_config`、或带着 `hf_quant_config.json`,下游会**按量化模型去加载一个 BF16 模型**。导出器在 rank 0 上显式 pop 掉并删除该文件。

```python
# examples/megatron_bridge/export_quantized_megatron_to_hf.py(核心 12 行)
def export_vllm_fakequant(bridge, model, export_dir):
    if not hasattr(MegatronModelBridge, "build_modelopt_quantizer_recipe"):
        raise RuntimeError("needs a Megatron-Bridge that writes the quantizer sidecars ...")
    for chunk in model:
        check_mcore_vllm_fakequant_quantizers(unwrap_model(chunk))   # 先整体校验,写盘前失败
        mtq.fold_weight(unwrap_model(chunk))                        # 权重折叠
    bridge.save_hf_pretrained(model, export_dir)                   # 权重 + 两个 sidecar 一起写
    if dist.is_master():
        config = json.loads((Path(export_dir) / "config.json").read_text())
        if config.pop("quantization_config", None) is not None:      # 别让 BF16 权重被当成量化模型
            (Path(export_dir) / "config.json").write_text(json.dumps(config, indent=2))
        (Path(export_dir) / "hf_quant_config.json").unlink(missing_ok=True)
```

**校验前移**是这次改动里最值得抄的一条工程实践。`check_mcore_vllm_fakequant_quantizers(model)` 把旧导出器散在各处的检查(MTP 量化、`lm_head` 量化、不支持的激活设置)收成一个函数,**在动任何权重之前**(`export_mcore_gpt_to_hf_vllm_fq` 的第一行、M-Bridge 路径的 fold 之前)统一拒掉;并且它是**集合式**的——任一个 rank 失败,所有 rank 都抛(`DistributedProcessGroup.get_dist_syncd_obj` + 取第一个非空 message)。理由很实在:**导出是集合操作,一个 rank 早退就是别的 rank 卡在 collective 上**。旧代码是在 gather 实现里各打一个 `_quantizer_validation_failure` 标记、然后"跳过不支持的 kernel 让对端跑完",结果是"部分写了、部分没写"的半成品目录。

拒绝清单(不是能力不足,是**语义上不能仿真**):

- **MTP 与 `lm_head` 的激活量化**:vLLM fake-quant reload 无法复现(注意判定条件:权重量化器在 tied embedding 情形下也不允许,因为 tied 时输出层走 embedding 的路径);
- **AWQ / SVDQuant**:需要全模型 pre-quant-scale 步骤,逐层做不到;
- **tied weights(`tie_word_embeddings`)**:改用 `export_hf_checkpoint()`;
- **`--export_extra_modules`(Medusa/EAGLE/MTP)与 fake-quant 导出互斥**。

## 5. 实测数据(PR #2889 自述,两轮口径)

> 下图把 per-token NLL 的 0.016 差异拆到根因,回答「为什么 dense 能压到 0.0002、MoE 上却被放大到 0.016」。


<div class="archify-figure">
<iframe src="../../../diagrams/qt-fakequant-nll-diff.tree.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="MoE 路由翻转放大差异"></iframe>
</div>



| 模型(NVFP4 W4A4 + FP8 KV) | 并行度一致性 | 与旧导出器/源权重 |
|---|---|---|
| Qwen3-0.6B | TP1 == TP2 | 服务输出与旧导出器**完全一致**(8/8 greedy,Δlogprob 0.0) |
| NVIDIA-Nemotron-Nano-9B-v2 | TP1 == TP2 == PP2(341 张量 / 128 ranges) | 权重**逐位相同**;服务输出 8/8 一致 |
| Qwen3-30B-A3B | EP2 == PP2(18,867 张量 / 18,720 ranges) | vLLM TP2 正常服务 |
| Qwen3.6-35B-A3B(GB300,vLLM 0.31) | EP4 == TP2×EP2 == PP2×EP2(1,045 张量 / 390 ranges / **410** recipe 条目,含 constant-amax KV) | 视觉塔与 MTP 权重**逐位等于源**;文本/图像 prompt 均正常 |

数值一致性(64 篇 CNN/DailyMail 测试文章,各取前 512 token,按 per-token NLL 比较):

| 模型 | Megatron vs vLLM fake-quant | 说明 |
|---|---|---|
| Nemotron-Nano-9B-v2 | 2.3465 vs 2.3467(BF16:2.2987 vs 2.2990) | 差距在 0.0002 量级 |
| Qwen3.6-35B-A3B | 2.2844 vs **2.3008**(HF transformers 同量化器:2.2901) | 单个量化器组各自都在 ±0.004 内;**全部叠加**才拉开 0.016 |

**这 0.016 从哪来**(PR 的解释,值得单独记):Qwen3.6 上三个引擎看到的是**同一批 410 个量化器与 ranges**,差异来自 **MoE 层被量化到 NVFP4 后,微小的 kernel 差异会翻转专家路由**——量化 MoE 时 per-token `|Δ|≈0.2`,不量化 MoE 时只有 0.07–0.1。所以"fake-quant 与真 kernel 不一致"在 MoE 上会被**路由不连续**放大,而不是线性放大。这也解释了为什么 dense 模型(Nemotron 9B)两边能压到 0.0002。

自检一下表里的口径:`1,045 张量` / `390 ranges` / `410 recipe 条目`是 PR 自己给的三个不同计数(权重张量 / 有 range 的量化器 / 设置条目),**不能互相换算**;而且 PR 正文另一处写的是"三个引擎看到同一批 **410 quantizers and ranges**",与表格里的 390 ranges 不完全一致——引用时按"张量 1,045、条目 410(含 constant-amax KV)、ranges 390"记录,别把这三个数写成同一个。Qwen3-30B 的 `18,720 ranges` 比 `18,867 张量` 少 147,PR 未解释差额来源,本文不做推断。

测试与 CI:

- 新测试 `test_vllm_fakequant_mbridge_export.py`:tiny Nemotron-H(NVFP4 + FP8 KV)与 Qwen3(W4A8,顺序量化器)两条导出路径**互相逐位比对**(权重逐位、ranges 相等、recipe 是旧 recipe 的超集);
- 示例测试没有新增(太慢),而是让 `test_quantize_export.py::test_quantize_and_export` **顺手用 `--vllm_fakequant_export` 再导一遍**它本来就要量化的每个 checkpoint(EP:Qwen3-MoE 逐专家命名;TP:Qwen3-VL 视觉塔逐位;**PP**:Nemotron-H),`test_qad.py` 只对 Qwen3.5-VL-MoE(packed experts)补这条——因为只有 QAD 会单独存 VLM 的语言模型;2 GPU 上 **6 passed / 25 min**(最慢 386 s);
- 回归范围:`tests/unit`、`tests/gpu/torch/{quantization,export}`、`tests/gpu_megatron/torch/{export,quantization}`、`tests/gpu_vllm`、`tests/examples/{megatron_bridge,vllm_serve}`;nemo:26.08 + M-Bridge `1d9caf73` 下 `gpu_megatron/torch/export` 139 passed、examples 81 + distill 21、prune/nas 62、vllm_serve 120 passed;
- CI 用新的 `.github/actions/megatron-from-source` 把容器里的 M-Bridge / Megatron-Core 源码换成 `1d9caf7324`(#6565 的 merge commit),等 NeMo 容器内置 #6565 后删掉——**测试不因旧 M-Bridge 而 skip**,避免 CI 悄悄失去覆盖。

## 6. 实践要点

**决定走不走 fake-quant**

- [ ] 先问一句:**我要的是"量化方案对不对"还是"部署精度多少"**。前者 fake-quant 够用且最省钱;后者必须补真 kernel 对照,尤其 MoE(NVFP4 下路由翻转会让 per-token 差异到 0.2)。
- [ ] 格式对照:权重-only(IQ2/GGML 系)折叠后**不需要 sidecar**,stock vLLM 直接跑;带激活/KV 量化的(NVFP4 W4A4、FP8 W8A8、FP8 KV)必须带 sidecar + `vllm_serve_fakequant.py`。
- [ ] NVFP4 QAD 在 Hopper 上评测是这条路最典型的用法(Hopper 无 NVFP4 内核)。

**导出前**

- [ ] 确认 Megatron-Bridge 够新:需要 `MegatronModelBridge.build_modelopt_quantizer_recipe`(会写两个 sidecar)。旧版会**明确报错**而不是悄悄产出错产物;#6565 尚未进 NeMo 容器,容器内要按 README 的 `MBRIDGE_DIR` 挂载 `1d9caf7324` 的 `src/megatron/bridge` 与其 `3rdparty/Megatron-LM`。
- [ ] 想清楚被拒的组合:**MTP / `lm_head` 的激活量化、AWQ、SVDQuant、tied embeddings、`--export_extra_modules`** 一律过不去;别指望"先导出来再说"。
- [ ] VLM 要保证 HF 侧视觉塔权重可读(QAD checkpoint 里只有语言模型)。

**导出后 / 服务前**

- [ ] 核对产物:两个 sidecar 在不在(`quantizer_state.pth`、`quant_recipe.yaml`)、`config.json` 里的 `quantization_config` 是否已被清掉、`hf_quant_config.json` 是否已删。**权重-only 格式没有 sidecar 是对的**,别当失败去补。
- [ ] 换并行度要重导吗:不必,**同一份导出**支持 TP/EP/PP 任意重组(PR 在 EP4/TP2×EP2/PP2×EP2 三种布局下得到一致的 recipe 与权重)。但两者的 ranges 比对要在**同一并行度**下做,否则 TP 切分会让 `pre_quant_scale` 的长度不同。
- [ ] 服务侧盯启动日志里的 `print_quant_summary`:权重量化器应当全部 disabled;若出现 `still enabled after fold_weight` 的异常,是折叠没生效(顺序量化器或自定义模块未走 `fold_weight`),不是 sidecar 问题。
- [ ] 需要对 NLL 时,**同一批 prompt、同一 token 切片、per-token NLL 配对**(PR 用的是 64 篇 CNN/DailyMail 前 512 token),否则 0.016 量级的差异会被评测噪声淹没。

## 7. 与库内其他篇的关系

- [PTQ:GPTQ / AWQ / SmoothQuant](ptq-gptq-awq-smoothquant.md)——本篇的上游:那里讲"怎么算出量化参数",本篇讲"算完之后怎么在没有 kernel 的引擎里复现这些参数的效果"。
- [FP8/FP4 硬件原生格式](fp8-fp4-formats.md)——NVFP4/FP8 的位布局与 scale 粒度决定了 sidecar 里要记什么(per-block scale 是设置,`_amax` 是 range)。
- [大规模 MoE 的 GPTQ 工程化](moe-gptq-layerwise-calibration.md)——同一条"导出/校准工程链"的另一半:那篇解决"怎么在单卡上把 4.9 TB 校准完并逐层导出",本篇解决"导出后的 checkpoint 怎么在没有内核的引擎里被评"。两篇共享同一个 `fold_weight` 语义(那篇的逐层导出也会调它)。
- [MoE 的专家并行 PTQ 校准](expert-parallel-ptq-calibration.md)——本篇的 TP/EP/PP 支持与那篇的 DP=EP 是**同一族并行话题的两端**(那端是校准期的 token 覆盖,这端是导出期的权重分片重组)。
- [vLLM 量化:GPTQ / AWQ / FP8](../../vllm/concepts/quantization-gptq-awq-fp8.md)——engine 视角的量化注册与内核分派;本篇是"绕过 kernel"的那条支路,对照读能看清"有 kernel"和"没有 kernel"两条验收链路分别测什么。
- [computer-vision 部署 howto](../../computer-vision/frameworks/onnx-tensorrt-deployment.md)——另一套"导出 + 数值回归"的验收范式(ONNX/TRT 侧),同样是"用参考实现比对导出产物"。

## 溯源

- 折叠语义的三处修复、校验前移、Megatron-Bridge 导出、全部并行度/NLL 实测与测试计数:ModelOpt PR #2889(commit `fb3ccd0e65bc`,2026-10-24 合并),PR 正文 + diff(`quant_module.py` / `vllm_fakequant_megatron.py` / `export_quantized_megatron_to_hf.py`)。
- 版本化 recipe 的 reload 侧:PR #2888(已合并);`--vllm_fakequant_export` 的来历:PR #2873(未发布)。
- sidecar 结构与使用步骤:`examples/vllm_serve/README.md`、`examples/megatron_bridge/README.md`(2026-10-25 实读)。
- 上游依赖:Megatron-Bridge #6565(写 sidecar 的能力),ModelOpt CI 钉 `1d9caf7324e608aca7181de40be26a6387976210`。
- 所有数字为 PR 自述,**未独立复现**;本文对可检查项做了重算与口径标注(`410 recipe 条目 ≠ 390 ranges ≠ 1,045 张量`;IQ2 无 sidecar = 形态正确;MoE 路由翻转放大差异)。
