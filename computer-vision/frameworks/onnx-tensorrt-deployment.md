---
title: 视觉模型部署：PyTorch → ONNX → TensorRT / ONNX Runtime
type: howto
tags: [computer-vision, inference, onnx, tensorrt, onnx-runtime, deployment, benchmark]
source:
  - "web: ONNX Runtime 官方文档 | https://onnxruntime.ai/docs/ | 阅读日期 2026-10-23(最新发布 1.31.0)"
  - "web: NVIDIA TensorRT 官方文档 | https://docs.nvidia.com/deeplearning/tensorrt/latest/ | 阅读日期 2026-10-23(当前版 11.4.0,支持 Vera Rubin 计算能力)"
  - "web: TensorRT Architecture Overview | https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html | 阅读日期 2026-09-27"
  - "web: NVIDIA Model Optimizer(仓库已由 TensorRT-Model-Optimizer 更名为 Model-Optimizer;新文档站 https://nvidia.github.io/Model-Optimizer/ 实测 200,旧站 nvidia.github.io/TensorRT-Model-Optimizer 已 404) + PR #2929(Wan I2V 导出探针静默跳融合) + PR #2467(远程 trtexec_safe 安全 benchmark) + PR #2816(Windows ONNX external data 损坏,依赖升 1.23.2)| https://github.com/NVIDIA/Model-Optimizer | 阅读日期 2026-10-25"
  - "code: onnx/onnx PR #7790 *Fix TOCTOU race in external data file open*(2026-04-06 合并:Windows `_O_BINARY` + Linux `openat2` 集中到 C++ `open_external_data()`);PR #8260 *Validate offset in save_external_data*(2026-08-05 合并,1.23.0 起)| https://github.com/onnx/onnx/pull/7790 | ↑primary"
  - "code: onnx/onnx 源码 external_data_helper.py 的 save_external_data(),v1.21.0 vs v1.22.0/v1.23.2 逐版本对比(核对 O_BINARY 与 64 KiB padding 守卫的引入版本)| https://github.com/onnx/onnx/blob/main/onnx/external_data_helper.py"
date: 2026-09
---

# 视觉模型部署：PyTorch → ONNX → TensorRT / ONNX Runtime

> **TL;DR**：部署不是“导出 ONNX 后就结束”。可靠流程是：固定模型与预处理 → 导出 → ONNX 图和数值验证 → 选择 runtime/backend → 构建并缓存 engine → 在目标输入形状上测准确率、延迟、吞吐、显存和稳定性。ONNX Runtime 适合作为跨平台基线和 Execution Provider 抽象；TensorRT 适合 NVIDIA GPU 上的深度优化，但 engine 与 TensorRT/CUDA/GPU 兼容性必须纳入发布物。

## 1. 两条部署路线

| 路线 | 定位 | 优点 | 注意事项 |
|---|---|---|---|
| ONNX Runtime | 跨平台 runtime + Execution Provider | API 稳定、可在 CPU/各类加速器间切换、适合作为基线 | 性能取决于 graph partition、EP 和算子覆盖；可能出现 CPU fallback |
| TensorRT | NVIDIA GPU 优化编译器和 runtime | FP16/BF16/FP8/INT8/FP4/INT4、动态 shape、融合、专用 kernel | engine 构建和兼容矩阵更复杂；不能把 plan 文件当成跨机器通用产物 |

ONNX Runtime 会先做图优化，再按可用硬件的 Execution Provider 对图进行分区；如果某些算子不能由目标 EP 执行，可能落回 CPU 或其他 provider，必须在日志和性能中确认，而不是只看“加载成功”。

TensorRT 的典型入口是 PyTorch/其他框架导出 ONNX，再由 ONNX parser 导入并构建 engine；也可以通过 TensorRT API 或 Torch-TensorRT 等路线接入。选择路线时优先看模型算子、动态 shape、量化需求和目标硬件，而不是只看导出格式。

> 下图是这条选择逻辑的分支形态:同一个 ONNX,先问「部署在哪」,再问「算子是否全覆盖」,两条答案指向完全不同的落地形态。

<div class="archify-figure">
<iframe src="../../../diagrams/cv-two-routes.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="两条部署路线怎么选:ORT 当基线,TRT 要深优化"></iframe>
</div>


## 2. 可复现部署流水线

> 下图把原文 9 步压成 5 段。注意「验证」出现两次(导出后、构建后),这是整条流水线最容易省掉、也最容易事后翻车的地方。

<div class="archify-figure">
<iframe src="../../../diagrams/cv-deploy-pipeline.dataflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="可复现部署流水线:五个阶段,每段都留版本信息"></iframe>
</div>


```text
训练 checkpoint + 预处理配置
          ↓
固定 export config / opset / dynamic axes
          ↓
导出 ONNX
          ↓
ONNX checker + shape inference + graph inspection
          ↓
PyTorch vs ONNX 数值回归
          ↓
ONNX Runtime baseline
          ↓
TensorRT parse/build/profile
          ↓
TensorRT 数值回归 + 目标硬件 benchmark
          ↓
打包 model / engine / config / runtime manifest
```

每个阶段都保存版本信息：模型 revision、opset、导出脚本 commit、Python/PyTorch/ONNX/TensorRT/CUDA 版本、GPU 型号、输入 shape 和预处理参数。

## 3. 导出阶段的关键决策

> 下图把这一节的三个决策排成一条链:契约 → 形状 → 算子缺口。顺序反了会反复返工——契约没定就配 profile,后面必然重做。

<div class="archify-figure">
<iframe src="../../../diagrams/cv-export-decisions.workflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="导出阶段三个决策:契约、动态 shape、算子缺口"></iframe>
</div>


### 输入输出契约

先写清楚：

- 输入张量名称、dtype、layout(NCHW/NHWC)、颜色顺序(RGB/BGR)
- 是否包含 resize、normalize、letterbox、padding 和后处理
- 输出是 logits、boxes、masks、embeddings 还是多尺度 feature maps
- batch、height、width 是否动态
- 检测/分割后处理由模型图、runtime 还是业务代码完成

最常见的“精度下降”不是 TensorRT 数学问题，而是预处理、坐标变换、NMS 阈值、颜色通道或 resize 对齐方式不一致。

### Dynamic shape

动态输入需要为 TensorRT optimization profile 指定 min/opt/max shape。`opt` 应靠近生产流量中心，而不是随便取中间值；过大的 max shape 会增加 build 搜索空间和 workspace 压力。若线上 shape 很离散，考虑按常见分辨率构建多个 profile 或多个 engine。

### 算子与控制流

遇到 parser 不支持的算子时，先区分：

1. 能否改写模型，使其落在 ONNX/TensorRT 常见算子集合内。
2. 是否可以在导出前做 constant folding、shape 推导或图简化。
3. 是否应写 TensorRT plugin。
4. 是否让该子图由 ONNX Runtime 或其他 backend 执行。

不要为了“全图 TensorRT”把一个简单、稳定的后处理强行改成复杂 plugin；端到端收益要用 benchmark 证明。

### 产物字节完整性:Windows 文本模式与 ONNX external data（2026-10）

上面讲的是**图和数值**层面的正确性;还有一类更隐蔽的错误在**文件字节**层面——导出报 success、模型能加载、checker 也过,但权重数据本身已经错位。2026-10 的实例(ModelOpt PR #2816,消费 ONNX 上游 PR #7790)值得记全链条,因为它是"平台特定 + 静默"的典型。

**为什么 external data 会错位。** >2GB 的 ONNX(量化大模型基本都超)必须走 external data:`TensorProto` 只保留 `{location, offset, length}`,真实字节写进一个共享 `.onnx_data` 文件。ONNX 的写法是**一个文件 + 逐张量 seek 追加**——每个张量 `seek(0, 2)` 到文件尾、写 `raw_data`、再回填自己的 `offset/length`:

```python
# onnx/external_data_helper.py :: save_external_data()(1.21 口径)
fd = os.open(external_data_file_path, os.O_CREAT | os.O_RDWR, 0o600)   # ← 没有 O_BINARY
with os.fdopen(fd, "r+b") as data_file:
    data_file.seek(0, 2)                    # 追加到文件尾
    offset = data_file.tell()
    data_file.write(tensor.raw_data)         # 前一个张量少写 1 字节 → 后一个整体前移
    set_external_data(tensor, info.location, offset, data_file.tell() - offset)
```

所以**任何一次写入少掉一个字节,后面所有张量的 offset 都错位**,而 `.onnx` 本身(protobuf + offset 表)仍然自洽。

**Windows 的文本模式。** Windows CRT 的 fd 分文本/二进制两种:文本模式下 `0x1A`(Ctrl-Z)被当 **EOF**、并且做 `\n ↔ \r\n` 翻译。`os.open()` 在 Windows 上默认开**文本模式**,要 `| os.O_BINARY` 才是二进制(POSIX 没有这个区分;CPython bpo-52646 记录过这条跨平台陷阱)。

ONNX 1.21.0 的写盘路径上那行 `os.open` 正是**没带 `O_BINARY`** 的(源码可核)。PR 给出的机制是:共享 external 文件被**逐张量重开追加**,当某个张量以 `0x1A` 结尾时,文本模式把它当 EOF 截断,于是下一个张量从 1023 而不是 1024 开始写,**覆盖掉前一个张量的最后一个字节**。`.onnx` 里的 offset 表由代码自己回填、内部自洽,所以 **checker 不报错**,崩的是 `raw_data` 的对应关系;症状不是崩溃,而是**模型能加载、输出崩坏**(PR 原文:错误输出 + 生成跑飞,直到把对话上下文撑爆)。

> **口径要诚实**:上面"截断 `0x1A`"是 PR 的陈述 + 测试的观测(1.21.0 上 `0x1A` 用例失败、`0x00`/`0x0A` 通过;1.23.2 全通过),我**没有在 Windows 上复现**(本机 Linux)。另有一点与该解释不完全吻合、我保留不加判词:CPython 的 `FileIO` 初始化在 `fd` 由外部传入时**也会**执行 `_setmode(fd, O_BINARY)`(`Modules/_io/fileio.c`,我逐版本核过 3.10/3.11/3.12/3.13 都有)。也就是说真实失败路径可能比"任何 `os.open` 都中招"更窄(可能与 `O_CREAT` 新建 + 后续重开的具体组合、或 1.21 内部某处别的 `open` 调用有关)。**实践上不必分辨**:规则永远是"二进制数据一律 `O_BINARY` / 用 `open(..., 'rb'/'wb')` / 直接升 onnx ≥ 1.23.2",这条在两种机制下都成立。

**上游怎么修的。** ONNX 把 external data 的所有打开动作收进 C++ `open_external_data()`:Linux 用 `openat2(RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS)`、BSD/macOS 用 `openat(O_RESOLVE_BENEATH|O_NOFOLLOW)`、Windows 用 `CreateFileW(FILE_FLAG_OPEN_REPARSE_POINT)` 再 `_open_osfhandle(_O_RDONLY|_O_BINARY)`,并顺带把 TOCTOU 补掉(PR #7790,2026-04 合并;python 侧从 **v1.22.0** 起就是 `_open_external_data_fd()` 调用它,我已对 v1.21.0/v1.22.0 源码逐一核对)。1.23.0 又给 save 加了二次守卫(PR #8260,2026-08 合并,已核对 v1.23.0/v1.23.1/v1.23.2 均有、v1.22.0 无),把"静默覆盖"变成硬报错:

```python
# onnx/external_data_helper.py :: save_external_data()（1.23.x）
padding = info.offset - data_file.tell()
if not 0 <= padding <= _MAX_EXTERNAL_DATA_PADDING:      # 64 KiB
    raise ValidationError(...)   # offset 落在已写数据里(会被覆盖)或远超文件尾(无界补零)都拒绝
```

**ModelOpt 侧的动作(PR #2816)。** 只给 Windows 升级依赖,Linux 保持不动(review 要求),因为损坏是平台特有的、Linux CI 本来就抓不到:

```toml
# pyproject.toml 的 onnx extra
- "onnx~=1.21.0",
+ "onnx~=1.21.0; platform_system != 'Windows'",
+ "onnx~=1.23.2; platform_system == 'Windows'",
```

配的回归测试直接把三种"尾部字节"摆在一起跑——`0x00`(零)、`0x0A`(换行)、`0x1A`(Ctrl-Z),两个 1024 字节张量写进同一个 external 文件后逐字节比:

```python
@pytest.mark.parametrize("last_byte", [0x00, 0x0A, 0x1A])
def test_save_external_data_preserves_tensor_boundaries(tmp_path, last_byte):
    first = np.full(1024, 0x41, dtype=np.uint8); first[-1] = last_byte
    second = np.full(1024, 0x42, dtype=np.uint8)
    ...  # save_onnx(model, path, save_as_external_data=True)
    assert (tmp_path / "model.onnx_data").read_bytes() == first.tobytes() + second.tobytes()
    # 再 onnx.load 回来逐张量 assert_array_equal
```

三例里 `0x00`/`0x0A` 在 1.21.0 上就通过、**只有 `0x1A` 失败**(byte 1023 被覆盖);1.23.2 三例全过。为什么这类 bug 难发现:触发条件要求某个张量的最后一个字节恰好是 `0x1A`——单个张量约 1/256 概率,但一个量化大模型有几百到几千个 external 张量,**几乎必然踩到**;而一旦踩到,崩的是"输出质量"而不是"加载",极易被误判成量化算法或 kernel 的问题。

**数据口径(PR 自述,已标注未独立复现)。** 用 1.23.2 wheel 重新导出 AWQ-lite INT4(32 校准样本):341 个 external 张量无重叠、无非法 scale;CUDA 两轮对话正常回答并停止。五 shot MMLU 全 57 科 14,042 例:重新导出的 INT4 **8,099/14,042(57.677%)** vs 干净 FP16 **8,374/14,042(59.635%)**,差 **1.96 个百分点**(我把两组数重算复核过:57.677 + 1.958 = 59.635)。注意 PR 明确说这批是**早先配置下的流水线结果,不是在当前 main 基线上完整跑出来的**;原始 CI 复现与完整测试套件作者也未跑。所以这组数字只能当"重新导出后模型没坏"的旁证,不能当精度基准。

**实践要点。**

- **导出侧的自检比"能加载"强得多**:`Σ 各张量 length == external 文件大小`、每个 `offset+length` 落在文件内、offset 两两不重叠——三条都是几十行 Python 就能查的不变量,建议进 CI。
- **跨平台产物在生产平台上验字节**,别在 Linux 导出、Linux 加载通过就发布;Windows 依赖门槛:`onnx ≥ 1.23.2`。
- **损坏不可修**:已错位的 external data 只能**从干净权重重新导出**,不要试图 patch offset。
- **症状指纹**:模型加载成功 + 输出崩坏/生成不收敛 → 先查权重字节完整性,再怀疑量化算法或 kernel。
- 与本节前面的 Wan I2V 那条是同一族问题的两个形态:**探针静默降级**(逻辑层少喂了输入)和**文件字节错位**(存储层少写了一个字节),共同教训是"导出 success 不是证据,产物必须被独立校验"。


## 4. 数值正确性验证

> 下图是五级比较的顺序。跳级比较等于把「导出误差」「EP 误差」「低精度误差」三种来源混在一起,定位不到根因。

<div class="archify-figure">
<iframe src="../../../diagrams/cv-numeric-layers.dataflow.html"
        style="width:100%;height:720px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="数值正确性的分层定位:五级比较,从图到任务指标"></iframe>
</div>

### 分层比较

按以下顺序定位误差：

1. 原始 PyTorch eager vs 导出 ONNX。
2. ONNX Runtime CPU vs ONNX Runtime GPU/目标 EP。
3. ONNX Runtime vs TensorRT FP32/FP16。
4. FP16/INT8/FP8/FP4 与高精度基线。
5. 单算子、单 batch、整图和真实数据集指标。

### 推荐指标

- 最大绝对误差、平均绝对误差、相对误差
- logits/embedding 的 cosine similarity
- 检测：mAP、precision/recall、NMS 后框一致性
- 分割：mIoU、边界质量
- 分类：top-1/top-5
- 端到端任务：真实业务成功率，而不是只比较中间 tensor

量化或低精度之后，少量 tensor 的误差变大不一定意味着任务失败；相反，平均误差很小也可能导致 NMS 或 top-k 排序翻转，所以必须加入任务级指标。

## 5. 性能测量规范

每次 benchmark 固定：

- GPU、驱动、CUDA、TensorRT/ORT 版本
- engine build 参数、workspace、precision 和 tactic cache
- 输入 shape、batch、预热次数、测量次数
- 是否包含 H2D/D2H、预处理、后处理和同步
- 单流/多流、并发数和功耗状态

至少记录：

| 指标 | 作用 |
|---|---|
| p50/p95/p99 latency | 观察平均和尾延迟 |
| throughput | 检查 batch/并发收益 |
| GPU memory peak | 判断部署容量和并发上限 |
| build time / engine size | 评估发布成本 |
| GPU utilization / power | 判断是否真正吃满硬件 |
| accuracy regression | 防止速度换来不可接受质量损失 |

预热后的 kernel latency 不能代表首次请求延迟；engine build 时间也不能混进在线请求延迟。两者都应该单独记录。

## 6. 量化与图优化

- FP16/BF16：通常先作为低风险基线。
- INT8/FP8：需要校准数据、scale 和硬件/算子支持；校准集必须接近真实输入分布。
- INT4/FP4：压缩收益更大，但算子覆盖、精度回归和 kernel 支持更敏感。
- Model Optimizer 可用于 PTQ、QAT、剪枝、蒸馏和 ONNX 相关优化；使用时要保存配置和校准数据版本。剪枝/蒸馏这条支线的机制(激活幅度打分、异构 MIP 搜索、蒸馏配方、Mamba hybrid 剪枝的状态一致性坑)见 [quantization/structured-pruning-minitron-puzzletron](../../quantization/concepts/structured-pruning-minitron-puzzletron.md)——与本域直接相关的一点:ModelOpt 的**剪枝**也支持 CV 模型(FastNAS mode,约束为 `flops`/`params`),它与 TRT-ViT 那类「架构层设计」是同一目标的两条路(手设架构 vs 搜架构)。
- **导出"成功"不等于产物能加载(2026-10 实例:Wan2.1 I2V)**:ModelOpt 导量化 checkpoint 时会先跑一次**探针前向**(用假输入走一遍模型)来触发 QKV scale 融合;如果探针**只喂了 text context、没喂 image features**,multi-modal 模型的 image-attention 分支直接抛异常,而外层捕获后**静默跳过融合**——导出照样报 success,但产出的 checkpoint 因 QKV scale 与权重不匹配**加载失败**。修复思路是补齐探针输入(512-token text context + 与模型 `image_dim` 匹配的 image features、按 projection bias 对齐 compute dtype、首尾帧 positional embedding),并开 **strict 融合模式**让探针失败直接抛错而不是跳过。**通用教训**:凡是"导出/转换脚本探针输入不完整"的场景,正确姿势是**让探针失败致命化**,而不是加 try/except 兜底——静默降级比直接报错危险得多(数据源:ModelOpt PR #2929)。
- **Model Optimizer 已更名并北移文档站**(2026-10 核实):仓库从 `NVIDIA/TensorRT-Model-Optimizer` 改为 **`NVIDIA/Model-Optimizer`**(旧地址 301 重定向,不会断但也别写死),文档站从 `nvidia.github.io/TensorRT-Model-Optimizer/`(现 **404**)搬到 **`nvidia.github.io/Model-Optimizer/`**;PyPI 包名仍是 `nvidia-modelopt`(稳定版 0.47.0,2026-09-23;仓库已有 0.48.0rc)。**含义**:它不再只是 TensorRT 的前处理,而是硬件无关的量化/蒸馏/剪枝 + NAS 工具链,导出目标同时覆盖 TensorRT-LLM、TensorRT、vLLM、SGLang。现有能力里与部署最相关的是 **AutoQuant**(按层敏感度 + 存储预算自动分配精度):GEMM 与 **KV cache** 两条搜索可分别设预算(如 Qwen3-8B 的 KV 案例:5.76 等效 bit 把 modeled KV payload 降 **64.0%**、RULER-128K 只掉 **1.04 个百分点**,而同预算均匀 NVFP4 掉 **6.54** 个百分点)——**"统一低精度"不如"按敏感度分配"**,这条与库内量化域的 HyQuant/KVTuner 是同一心法。
- **AutoQuant 的自动 batch 探测必须复现"打分时"的负载(2026-10,PR #2943)**:`--batch_size 0` 让 AutoQuant 自己探测 batch 时,旧实现一律用**开了梯度的普通前向**去量显存,而真正打分时会额外保留(梯度/Aumann-Shapley 的激活、KL 打分的 reference logits 与 FP32 log-prob)。于是探测挑出的 batch"装得下探测、装不下打分",跑到打分阶段才 OOM。修复是给 `get_max_batch_size` 加可选 `forward_step(model, inputs)` 回调,按 recipe 实际保留的内容去探(gradient / weight-KL / KV-KL / 组合各一套,OOM 重试保留);encoder-decoder 模型仍走生成式探测。**含义**:①自动 batch 与派生出的打分步数可能随版本变,**要比两次运行就显式钉死 `--batch_size`**;②这是一类通用坑——*探测用的前向必须与真实负载同构*,否则"探测通过"毫无信息量(与本节 Wan I2V 的"探针不完整导致静默降级"是同一族问题)。
- **逐层(layerwise)校准把显存下限从"整模型"降到"一层"**(2026-10 核实):`layerwise.export_dir` 一设就开启,校准完一层就写出该层 shard 并释放,断点续跑就是原命令重跑。量化超大规模 MoE 时这条是唯一可行的路径(Hessian 一层就 13–23 GiB),机制与全部踩坑见 [quantization/moe-gptq-layerwise-calibration](../../quantization/concepts/moe-gptq-layerwise-calibration.md)。注意 `--max_gpu_memory_gb` 只约束**权重放置**,激活与 CUDA context 在其之上(官方实测峰值超预算 21.7 GB)。
- 图融合和 constant folding 可能改变数值误差，优化前后都要跑回归。

量化不是部署的最后一个开关，而是模型格式、校准、kernel、输入分布和评测协议的共同结果。和 [PTQ：GPTQ / AWQ / SmoothQuant](../../quantization/concepts/ptq-gptq-awq-smoothquant.md) 一起阅读。

## 7. Engine 发布与兼容性

TensorRT serialized engine 对版本和硬件有兼容限制；不要默认一个 plan 文件可以跨 GPU、CUDA、TensorRT 版本复用。发布包至少包含：

> 下图是发布包的七项结构:产物本身、可追溯性、形状与精度、验证与回退。缺任何一类,下次升级就会变成考古。

<div class="archify-figure">
<iframe src="../../../diagrams/cv-engine-manifest.tree.html"
        style="width:100%;height:620px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="lazy" title="TensorRT 发布包里到底该装什么"></iframe>
</div>


- engine 或可重建 engine 的构建配置
- 原始 ONNX/checkpoint 的 revision
- TensorRT/CUDA/driver/GPU manifest
- dynamic shape profiles
- precision、workspace、plugins 和 tactic cache 信息
- 数值回归样本及结果
- build 命令和回滚用旧 engine

升级 TensorRT 时重新检查 API、plugin、engine 序列化和 calibration cache 兼容性。线上升级应先在独立端口完成 smoke test，再切换流量。

## 8. 实践检查清单

- [ ] 预处理和后处理已版本化，不依赖手工 notebook。
- [ ] PyTorch、ONNX Runtime、TensorRT 输出完成数值回归。
- [ ] 已验证真实输入 shape，而非只有随机 tensor。
- [ ] 已检查 ONNX Runtime 是否发生 CPU fallback。
- [ ] 已为动态 shape 配置合理 profile。
- [ ] 已记录 engine 的构建环境和硬件兼容性。
- [ ] 已分别测 warm-up、kernel、端到端和并发延迟。
- [ ] 已完成精度、显存、吞吐和尾延迟验收。
- [ ] **external data 字节完整性已自检**(`Σ length == 文件大小`、`offset+length` 不越界、offset 互不重叠);Windows 导出要求 `onnx ≥ 1.23.2`。
- [ ] 已保留旧 engine、旧配置和回滚路径。

## 相关条目

- [TRT-ViT：以 TensorRT 延迟为准绳](../concepts/trt-vit-inference.md)
- [CNN 重参数化 OREPA](../concepts/cnn-reparameterization.md)
- [CUDA GPU 性能基础](../../cutlass/concepts/cuda-performance-basics.md)
- [PTQ：GPTQ / AWQ / SmoothQuant](../../quantization/concepts/ptq-gptq-awq-smoothquant.md)
- [LLM 推理引擎选型](../../vllm/practices/inference-engine-selection.md)

## 溯源

- 主源:ONNX Runtime 官方文档(https://onnxruntime.ai/docs/)与 NVIDIA TensorRT 官方文档(https://docs.nvidia.com/deeplearning/tensorrt/latest/)。TensorRT 版本以 [release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes.html) 为准(2026-10 核实:11.4.0 为当前版,11.3.0 Enterprise 起支持 Vera Rubin 计算能力)。
- 交叉源:TensorRT Architecture Overview(https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html)、NVIDIA Model Optimizer(https://nvidia.github.io/Model-Optimizer/ 文档站 + https://github.com/NVIDIA/Model-Optimizer 仓库;旧地址 TensorRT-Model-Optimizer 的文档站已 404)。
- AutoQuant KV 案例数据:NVIDIA Model Optimizer 官方公告 *AutoQuant KV: Automatic Mixed-Precision KV Cache Quantization*(2026-10-08)— https://nvidia.github.io/Model-Optimizer/announcements/20261008-autoquant-kv.html
- Windows external data 损坏的机制、依赖改动与测试:ModelOpt PR #2816(2026-10-24 合并,commit `3c423a9f165b`;正文 + `pyproject.toml` / `tests/unit/onnx/test_external_data.py` diff)。上游根因与修法:onnx/onnx PR #7790(2026-04-06 合并)— https://github.com/onnx/onnx/pull/7790;save 侧 offset 守卫:PR #8260(2026-08-05 合并)。
- 版本引入点经**逐版本源码对比**核实(非推断):`_open_external_data_fd()` 在 v1.21.0 不存在、v1.22.0 起存在;`_MAX_EXTERNAL_DATA_PADDING` 守卫 v1.22.0 无、v1.23.0 起有。ONNX 当前 PyPI 版 1.23.2(2026-10-06 发布)。
- MMLU 数字为 PR 自述且 PR 声明「来自早先 ModelOpt 配置的流水线,非当前 main 基线的完整运行」,本文仅作旁证引用;百分比已重算复核(8,099/14,042 = 57.677%,8,374/14,042 = 59.635%,差 1.958 个百分点)。
