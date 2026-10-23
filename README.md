# obrain - 个人大脑

> 技术知识库根目录(原 `INDEX.md`,2026-09 并入 README)。每个领域一个文件夹,里面至少有一个 `index.md` 作为该领域索引。
> 定位:存档为主,平时不常看 → 目标是「扫 30 秒能找到,打开一篇能读懂」。

## 目录规约(自包含原则)

obrain 是自包含的知识库:只描述本文件夹内部结构;不引用任何设备的本地路径或文件名,只通过公开原始链接溯源;不依赖库外任何目录才能阅读。

```
obrain/
├── README.md             根入口(本文件)
├── SOURCE-POLICY.md      多来源登记策略
└── <domain>/             每个领域一个文件夹,至少含 index.md
    ├── index.md          领域索引/导航起始页(type: domain)
    ├── references.md     领域链接索引(每域一篇,type: reference)
    ├── concepts/         概念篇:机制原理(type: concept)
    ├── frameworks/       工具上手篇(可选,type: howto)
    └── practices/        实践清单/选型对比(可选,type: checklist/comparison)
```

> 例外:attention 域的深度篇分 `algorithm/`(算法层)、`implementation/`(FA 内核实现层)与 `frameworks/`(用户接口/编译器后端层,如 FlexAttention)三层,不用 `concepts/`。
> 个人原始资料归档(library)等目录属于各设备本地布局,不写入本文档。

## 领域列表(10 个独立领域)

| 领域 | 文件夹 | 一句话说明 | 最近更新 |
|---|---|---|---|
| Attention | [`attention/`](attention/index.md) | 注意力机制三层结构:算法层 algorithm/(AIAYN/StreamingLLM/LASER+SpAtten/MEGABYTE/Rabe&Staats,5 篇)+ 实现层 implementation/(FA 四代合成篇,1 篇)+ 框架层 frameworks/(FlexAttention:score_mod/mask_mod 声明变体→torch.compile 落到 Triton 或 FA4/CuTeDSL,1 篇) | 2026-10 |
| Model Architecture | [`model-architecture/`](model-architecture/index.md) | 通用 LLM 架构:归一化/激活/位置编码演进 + 训练哲学(Chinchilla/推理预算)+ GPT→LLaMA→Llama 2/3 定型线;MHA/MQA/GQA 注意力头谱系;RoPE 上下文扩展(PI/NTK/YaRN/LongRoPE);开源模型谱系与 MoE(Mistral/DeepSeek/Qwen/RWKV + 2025-26 续篇)+ DeepSeek 网络结构演进四篇 + 混合线性注意力/SSM(Mamba/RWKV/Qwen3-Next)(10 篇) | 2026-10 |
| CUTLASS | [`cutlass/`](cutlass/index.md) | 高性能 GEMM 模板库与 CuTe DSL:造内核的库(concepts 8 + frameworks 2,共 10 篇) | 2026-10 |
| GPU 随机数生成 | [`gpu-rng/`](gpu-rng/index.md) | counter-based(Philox)与 MTGP 两条 GPU PRNG 路线 + cuRAND 落地(1 篇) | 2026-10 |
| 量化与压缩 | [`quantization/`](quantization/index.md) | LLM 量化与压缩:PTQ(GPTQ/AWQ/SmoothQuant)、KV cache 量化、稀疏量化混合(SpQR)、2-bit 前沿(QuIP#/AQLM)、KV 极低比特(KVQuant/KVTuner)、HyQuant 混合精度、FP8/FP4 硬件原生格式 + 低秩/剪枝路线(8 篇,论文来源通过公开链接追溯) | 2026-10 |
| vLLM | [`vllm/`](vllm/index.md) | 高吞吐 LLM 推理与服务引擎:PagedAttention、连续批处理、V1 架构、**Model Runner V2(执行层重设计,当前默认)**、生产部署、观测与引擎选型、投机解码、注意力后端选型、PD 分离;生态构件:全模态/路由集群/压缩/投机RL/硬件插件/评测(24 篇) | 2026-10 |
| Computer Vision | [`computer-vision/`](computer-vision/index.md) | 视觉模型基础/表示与部署效率:ViT 基础机制、目标检测与实例分割(Faster/Mask R-CNN、DETR、YOLO、SAM)、NeRF 视图合成、TRT-ViT 硬件感知架构、OREPA 重参数化、CCNN 连续卷积核、ONNX/TensorRT 部署(7 篇) | 2026-10 |
| DeepSeek Infra | [`deepseek-infra/`](deepseek-infra/index.md) | DeepSeek 算力基础设施栈:昇腾六件套(TileLang/DeepGEMM-Ascend/DeepEP-Ascend/TileKernels/FlashMLA/DeepSelect)+ 移植原则(ascend-port-principles)+ NVIDIA 原版对照(DeepGEMM/DeepEP)+ mHC 流形超连接(论文 + Mega mHC 内核)+ 并行/工具/存储(DeepJIT/Engram/DualPipe/EPLB+LPLB/ESFT/open-infra-index/3FS/smallpond)+ 训练并行编排 V3 + deepseek-recipe API 编解码 + **DeepInfer 生产推理引擎(V4.1-Flash,EPD 分离/ZOS/DSpark/YoCo 三级 KV)** + 部署清单(22 篇) | 2026-10 |
| DeepSeek Vision | [`deepseek-vision/`](deepseek-vision/index.md) | DeepSeek 多模态/视觉家族:图文理解(VL/VL2)、视觉压缩(OCR/OCR-2)、统一理解+生成(Janus 系列)、图生 3D(DreamCraft3D)(4 篇) | 2026-10 |
| DeepSeek Models | [`deepseek-models/`](deepseek-models/index.md) | DeepSeek 领域模型谱系:数学推理(Math/Math-V2,GRPO/自验证)、定理证明(Prover-V1/V1.5/V2,Lean 4)、代码智能(Coder/Coder-V2,FIM/MoE)、MoE 基座、推理 RL(R1-Zero→R1→Distill,GRPO 规模化/冷启动四阶段/蒸馏)、ESFT 专家专精微调(机制视角)+ agent 工具 deepseek-harness + DeepSpec 草稿训练 + dsh 内部 Node 插件(9 篇);与 model-architecture 网络结构篇互链 | 2026-10 |

## 跨域速查

- **三层依赖(CUTLASS → FA → vLLM)**:CUTLASS(造内核的库)→ attention 实现层(FlashAttention,注意力内核)→ vLLM(整机推理引擎),自底向上:vLLM 集成 FA;FA2 用 CuTe 原语写 tiling、FA3 基于 CUTLASS 3.x collective 抽象构建、FA4 走 CuTe DSL;升级任一层注意链路兼容。
- **架构→内核→引擎全链**:[`model-architecture/`](model-architecture/index.md)(LLM 整体架构:RMSNorm/SwiGLU/RoPE/GQA)→ [`attention/`](attention/index.md)(注意力算法层 algorithm/ + FA 内核实现层 implementation/ + FlexAttention 框架层 frameworks/)→ [`vllm/`](vllm/index.md)(引擎)。O(n²) 账单的三条路线:内核精确解(FA 四代合成篇)、架构分解(MEGABYTE)、专用硬件稀疏(SpAtten);StreamingLLM(cache 逐出策略)与 vLLM PagedAttention(cache 分页管理)正交互补;Llama 3 的 GQA(8 KV 头)直接缩小 vLLM 管的 KV cache。LLaMA 训练的 xformers attention = Rabe & Staats 前向 + FA1 反向。
- **量化双视角**:算法侧见 [`quantization/`](quantization/index.md),engine 侧(GPTQ/AWQ/FP8 注册分派)见 [`vllm/`](vllm/concepts/quantization-gptq-awq-fp8.md)。
- **硬件真实效率**:CUTLASS 的 tiling/Tensor Core 机制是理解「FLOPs≠延迟」的底层;应用侧实例见 [`computer-vision/concepts/trt-vit-inference.md`](computer-vision/concepts/trt-vit-inference.md)(以 TensorRT 实测延迟为架构设计准绳,TensorRT 内部以 CUTLASS 生成 GEMM kernel)。
- **精确 attention 省显存谱系**:Milakov online softmax → Rabe & Staats O(1) 显存 → FlashAttention IO-aware([`attention/algorithm/exact-attention-memory-lineage.md`](attention/algorithm/exact-attention-memory-lineage.md))。
- **FlashAttention 四代主线**:FA1 IO-aware tiling → FA2 work partitioning → FA3 Hopper 异步 → FA4 Blackwell,合成一篇 [`attention/implementation/flash-attention-implementation.md`](attention/implementation/flash-attention-implementation.md);底座机制见 [`cutlass/`](cutlass/index.md)(WGMMA/TMA/epilogue)。
- **架构第三主轴(混合线性注意力)**:softmax attention 之外,Mamba/RWKV/Gated DeltaNet→Qwen3-Next 的 hybrid 路线(少量 full attention + 多数线性/SSM 层)见 [`hybrid-linear-attention`](model-architecture/concepts/hybrid-linear-attention.md);与 FA/KV cache 杠杆正交——一个消灭序列状态,一个压缩/分页 KV。
- **vLLM 引擎↔生态**:引擎内部原理在 [`vllm/`](vllm/index.md) 的 concepts/,生态构件(全模态/路由/集群/压缩/插件/评测)在同域 `ecosystem/`——一个域读完引擎与围绕它的整圈。
- **低精度双路线**:硬件原生 FP8/FP4(见 [`fp8-fp4-formats`](quantization/concepts/fp8-fp4-formats.md))vs 算法压缩 2-bit(QuIP#/AQLM);训练侧 V3 FP8 recipe 与推理侧 FP8/FP4 KV 分别在 deepseek-infra/flashmla。
- **DeepSeek 结构↔领域模型双视角**:网络结构演进(MHA→MLA、dense→MoE)见 [`model-architecture/`](model-architecture/index.md) 的 deepseek 四篇;领域能力怎么做出来(数学 GRPO/自验证、定理证明 Lean 4、代码 FIM、MoE 数据评测、R1 推理 RL、ESFT 专家微调)见 [`deepseek-models/`](deepseek-models/index.md),两者互链互补——结构篇管「怎么变」,领域篇管「怎么做出来」。R1 是双视角的典型:结构侧(无新结构、V3-Base 底座)在 [deepseek-arch-v32-r1](model-architecture/concepts/deepseek-arch-v32-r1.md),训练策略侧(GRPO/冷启动/蒸馏)在 [deepseek-r1-reasoning](deepseek-models/concepts/deepseek-r1-reasoning.md)。ESFT 是跨 infra/领域双登记:操作流程在 deepseek-infra,机制/谱系在 [deepseek-esft](deepseek-models/concepts/deepseek-esft.md)。
- **引擎两条路**:通用引擎看 [`vllm/`](vllm/index.md)(PagedAttention/连续批处理/V1 架构,框架无关);DeepSeek 自家的生产引擎看 [deepinfer](deepseek-infra/concepts/deepinfer.md)——它基于 vLLM 改造,但把「单模型 + Wide-EP + 低精度 KV + EPD 分离」做到极致,两者的机制对照表在该篇 §14。读引擎时先分清「通用框架的工程心法」与「头部厂商为自家模型定制的取舍」。

---

## 阅读建议

- **临时查证** → 直接搜关键词(`PagedAttention`、`split-K`、`TiledMMA`、`sink token`)
- **回顾某领域** → 打开该领域 `index.md`,按索引跳
- **完整学习** → 打开领域 index.md,按深度篇目录(attention 域为 `algorithm/`+`implementation/`+`frameworks/`,其余域深度篇为 `concepts/`,`frameworks/` 为工具上手篇)导航顺序读(各篇文末『溯源』可回原始文献)

## 收纳规范(给自己定的,写这条是为了以后不纠结)

1. 每篇知识条目用 YAML frontmatter:`title` / `type` / `tags` / `source` / `date`(豁免:notes/ 源码笔记、各域 index/references、根目录 README/SOURCE-POLICY——详见 [SOURCE-POLICY.md](SOURCE-POLICY.md) §6/§7)
2. 每篇开头一段「TL;DR」,30 秒讲完结论
3. 概念篇管「懂」,案例篇管「查」,清单篇管「用」——三类分开,不混在一篇里(type 全集为 concept/case/checklist/howto/reference/domain/comparison 七类,comparison 专用于选型对比)
4. 引用外部资料只放链接索引(每域一篇 [references.md](attention/references.md) 为全量文献索引,范式见 attention;条目级溯源在 frontmatter source 与文末『溯源』),不复制全文
5. **多来源登记**(ppt / pdf / 网页 / wiki …):每篇 `source` 写成「类型 + 标识符 + 链接」结构化形式;来源互不相同但内容重叠时,只收一条知识,`source` 里列全部出处 → 详见 [SOURCE-POLICY.md](SOURCE-POLICY.md)
6. 拿到新资料先问「这条知识是不是已在库里?」→ 是则补 source 去重,否则新开篇;不建两篇内容重叠的篇
7. **把某领域"做厚"**(升级到专业深度、多源综合)时,按领域 index.md 定位判断 + 主动网络多源综合,写成"打开一篇能读懂、有专业深度"的深度条目(机制原理/可复现细节/数据对比/实践要点/溯源完整)。
8. **内容更新后必须同步检查**(用户明确要求):增/删/改任何知识条目后,同一轮内核对——(a) 所属域 `index.md` 条目索引与篇数、(b) 所属域 `references.md` 来源聚合、(c) 根 README 领域表「篇数/一句话说明/最近更新」、(d) 相关「扩展位」缺口是否该划掉、(e) 全库内链 0 断链。禁止只写正文不改索引。
