# obrain - 个人大脑

> 技术知识库根目录(原 `INDEX.md`,2026-09 并入 README)。每个领域一个文件夹,里面至少有一个 `index.md` 作为该领域索引。
> 定位:存档为主,平时不常看 → 目标是「扫 30 秒能找到,打开一篇能读懂」。

## 目录规约(自包含原则)

obrain 是自包含的知识库:只描述本文件夹内部结构;不引用任何设备的本地路径或文件名,只通过公开原始链接溯源;不依赖库外任何目录才能阅读。

```
obrain/
├── README.md             根入口(本文件)
├── SOURCE-POLICY.md      多来源登记策略
├── <domain>/             每个领域一个文件夹,至少含 index.md
│   ├── index.md          领域索引/导航起始页(type: domain)
│   ├── references.md     领域链接索引(可选,type: reference)
│   ├── concepts/         概念篇:机制原理(type: concept)
│   ├── frameworks/       工具上手篇(可选,type: howto)
│   └── practices/        实践清单/选型对比(可选,type: checklist/comparison)
└── profiles/             个人与家庭档案(非知识条目,豁免收纳规范)
```

> 例外:attention 域的深度篇分 `algorithm/`(算法层)与 `implementation/`(FA 内核实现层)两层,不用 `concepts/`。
> 个人原始资料归档(library)等目录属于各设备本地布局,不写入本文档。

## 领域列表(7 个独立领域)

| 领域 | 文件夹 | 一句话说明 | 最近更新 |
|---|---|---|---|
| Attention | [`attention/`](attention/index.md) | 注意力机制两层结构:算法层 algorithm/(AIAYN/StreamingLLM/LASER+SpAtten/MEGABYTE/Rabe&Staats,5 篇)+ 实现层 implementation/(FA 四代合成篇,1 篇) | 2026-09 |
| Model Architecture | [`model-architecture/`](model-architecture/index.md) | 通用 LLM 架构:归一化/激活/位置编码演进 + 训练哲学(Chinchilla/推理预算)+ GPT→LLaMA→Llama 2/3 定型线;MHA/MQA/GQA 注意力头谱系;RoPE 上下文扩展(PI/NTK/YaRN/LongRoPE);开源模型谱系与 MoE(Mistral/DeepSeek/Qwen/RWKV)(4 篇) | 2026-09 |
| CUTLASS | [`cutlass/`](cutlass/index.md) | 高性能 GEMM 模板库与 CuTe DSL:造内核的库(concepts 7 + frameworks 2,共 9 篇) | 2026-09 |
| GPU 随机数生成 | [`gpu-rng/`](gpu-rng/index.md) | counter-based(Philox)与 MTGP 两条 GPU PRNG 路线 + cuRAND 落地(1 篇) | 2026-09 |
| 量化与压缩 | [`quantization/`](quantization/index.md) | LLM 量化与压缩:PTQ(GPTQ/AWQ/SmoothQuant)、KV cache 量化、稀疏量化混合(SpQR)+ 低秩/剪枝路线(5 篇,论文来源通过公开链接追溯) | 2026-09 |
| vLLM | [`vllm/`](vllm/index.md) | 高吞吐 LLM 推理与服务引擎:PagedAttention、连续批处理、V1 架构、生产部署、观测与引擎选型等(10 篇) | 2026-09 |
| Computer Vision | [`computer-vision/`](computer-vision/index.md) | 视觉模型表示与部署效率:NeRF 视图合成、TRT-ViT 硬件感知架构、OREPA 重参数化、CCNN 连续卷积核、ONNX/TensorRT 部署(5 篇) | 2026-09 |

> 注:`profiles/` 为个人与家庭档案(非技术知识条目),不参与上述领域体系,豁免收纳规范。

## 跨域速查

- **三层依赖(CUTLASS → FA → vLLM)**:CUTLASS(造内核的库)→ attention 实现层(FlashAttention,注意力内核)→ vLLM(整机推理引擎),自底向上:vLLM 集成 FA;FA2/FA3 基于 CUTLASS 3.x 构建,升级任一层注意链路兼容。
- **架构→内核→引擎全链**:[`model-architecture/`](model-architecture/index.md)(LLM 整体架构:RMSNorm/SwiGLU/RoPE/GQA)→ [`attention/`](attention/index.md)(注意力算法层 algorithm/ + FA 内核实现层 implementation/)→ [`vllm/`](vllm/index.md)(引擎)。O(n²) 账单的三条路线:内核精确解(FA 四代合成篇)、架构分解(MEGABYTE)、专用硬件稀疏(SpAtten);StreamingLLM(cache 逐出策略)与 vLLM PagedAttention(cache 分页管理)正交互补;Llama 3 的 GQA(8 KV 头)直接缩小 vLLM 管的 KV cache。LLaMA 训练的 xformers attention = Rabe & Staats 前向 + FA1 反向。
- **量化双视角**:算法侧见 [`quantization/`](quantization/index.md),engine 侧(GPTQ/AWQ/FP8 注册分派)见 [`vllm/`](vllm/concepts/quantization-gptq-awq-fp8.md)。
- **硬件真实效率**:CUTLASS 的 tiling/Tensor Core 机制是理解「FLOPs≠延迟」的底层;应用侧实例见 [`computer-vision/concepts/trt-vit-inference.md`](computer-vision/concepts/trt-vit-inference.md)(以 TensorRT 实测延迟为架构设计准绳,TensorRT 内部以 CUTLASS 生成 GEMM kernel)。
- **精确 attention 省显存谱系**:Milakov online softmax → Rabe & Staats O(1) 显存 → FlashAttention IO-aware([`attention/algorithm/exact-attention-memory-lineage.md`](attention/algorithm/exact-attention-memory-lineage.md))。
- **FlashAttention 四代主线**:FA1 IO-aware tiling → FA2 work partitioning → FA3 Hopper 异步 → FA4 Blackwell,合成一篇 [`attention/implementation/flash-attention-implementation.md`](attention/implementation/flash-attention-implementation.md);底座机制见 [`cutlass/`](cutlass/index.md)(WGMMA/TMA/epilogue)。

---

## 阅读建议

- **临时查证** → 直接搜关键词(`PagedAttention`、`split-K`、`TiledMMA`、`sink token`)
- **回顾某领域** → 打开该领域 `index.md`,按索引跳
- **完整学习** → 打开领域 index.md,按深度篇目录(attention 域为 `algorithm/`+`implementation/`,其余域深度篇为 `concepts/`,`frameworks/` 为工具上手篇)导航顺序读(各篇文末『溯源』可回原始文献)

## 收纳规范(给自己定的,写这条是为了以后不纠结)

1. 每篇知识条目用 YAML frontmatter:`title` / `type` / `tags` / `source` / `date`(豁免:notes/ 源码笔记、各域 index/references、根目录 README/SOURCE-POLICY——详见 [SOURCE-POLICY.md](SOURCE-POLICY.md) §6/§7)
2. 每篇开头一段「TL;DR」,30 秒讲完结论
3. 概念篇管「懂」,案例篇管「查」,清单篇管「用」——三类分开,不混在一篇里(type 全集为 concept/case/checklist/howto/reference/domain/comparison 七类,comparison 专用于选型对比)
4. 引用外部资料只放链接索引(范式见 [attention/references.md](attention/references.md);cutlass/quantization/vllm 三域的链接索引即各篇 frontmatter 的 source 列表),不复制全文
5. **多来源登记**(ppt / pdf / 网页 / wiki …):每篇 `source` 写成「类型 + 标识符 + 链接」结构化形式;来源互不相同但内容重叠时,只收一条知识,`source` 里列全部出处 → 详见 [SOURCE-POLICY.md](SOURCE-POLICY.md)
6. 拿到新资料先问「这条知识是不是已在库里?」→ 是则补 source 去重,否则新开篇;不建两篇内容重叠的篇
7. **把某领域"做厚"**(升级到专业深度、多源综合)时,按领域 index.md 定位判断 + 主动网络多源综合,写成"打开一篇能读懂、有专业深度"的深度条目(机制原理/可复现细节/数据对比/实践要点/溯源完整)。
