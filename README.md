# obrain - 个人大脑

> 技术知识库根目录(原 `INDEX.md`,2026-09 并入 README)。每个领域一个文件夹,里面至少有一个 `index.md` 作为该领域索引。
> 定位:存档为主,平时不常看 → 目标是「扫 30 秒能找到,打开一篇能读懂」。

## 目录规约(obrain vs library 判据)

`~/zhanbin/` 顶层约定:

```
~/zhanbin/
├── obrain/      知识库:提炼后的知识条目(md + frontmatter + 索引,按领域组织)——本目录
├── library/     原始资料只读存档(pdf/pptx/书籍,永不编辑):papers/ books/ manuals/ training/ personal/
├── projects/    自己写的代码 + 实验脚本:work/ patent/ scratch/
├── archive/     一次性产物 / 旧草稿:保留期 90 天,到期清理
├── scripts/     审查作业脚本(cron 调用)
├── tracker/     定时 wiki 同步管线(统一管理,含 vllm/cutlass/flash-attention-tracker)
└── repos/       只读上游克隆(vllm)
```

**判据**:`obrain` = 提炼后的知识条目(可检索、有索引、按领域,内容自足);`library` = 原始资料存档(永不编辑)。两者不混;obrain 引用 library/archive 文件用 `source:` 指向,不复制;obrain 知识不依赖 tracker 同步管线。

## 领域列表(6 个独立领域)

| 领域 | 文件夹 | 一句话说明 | 最近更新 |
|---|---|---|---|
| Attention | [`attention/`](attention/index.md) | 注意力机制与 Transformer:算法/架构层(AIAYN/StreamingLLM/LASER/SpAtten/MEGABYTE/LLaMA)+ GPU 内核/实现层(FA 四代/Rabe&Staats)(9 篇) | 2026-09 |
| CUTLASS | [`cutlass/`](cutlass/index.md) | 高性能 GEMM 模板库与 CuTe DSL:造内核的库(concepts 6 + frameworks 2,共 8 篇) | 2026-09 |
| GPU 随机数生成 | [`gpu-rng/`](gpu-rng/index.md) | counter-based(Philox)与 MTGP 两条 GPU PRNG 路线 + cuRAND 落地(1 篇) | 2026-09 |
| 量化与压缩 | [`quantization/`](quantization/index.md) | LLM 量化与压缩:权重/KV cache 量化 + 稀疏量化混合(SpQR)+ 低秩/剪枝路线(3 篇,原 papers/ 论文笔记域) | 2026-09 |
| vLLM | [`vllm/`](vllm/index.md) | 高吞吐 LLM 推理与服务引擎:PagedAttention、连续批处理、V1 架构等(8 篇) | 2026-09 |
| Computer Vision | [`computer-vision/`](computer-vision/index.md) | 视觉模型表示与部署效率:NeRF 视图合成、TRT-ViT 硬件感知架构、OREPA 重参数化、CCNN 连续卷积核(4 篇) | 2026-09 |

## 三层关系(CUTLASS → FA → vLLM)

**CUTLASS(造内核的库)→ attention 内核层(FlashAttention,注意力内核)→ vLLM(整机推理引擎)**,自底向上三层:vLLM 集成 FA;FA2 基于 CUTLASS 3.x 构建,升级任一层注意链路兼容。

## 跨域速查

- **架构→内核→引擎全链**:[`attention/`](attention/index.md)(算法/架构层 + FA 内核层,合并域)→ [`vllm/`](vllm/index.md)(引擎)。O(n²) 账单的三条路线:内核精确解(FA)、架构分解(MEGABYTE)、专用硬件稀疏(SpAtten);StreamingLLM(cache 逐出策略)与 vLLM PagedAttention(cache 分页管理)正交互补。LLaMA 训练的 xformers attention = Rabe & Staats 前向 + FA1 反向。
- **量化双视角**:算法侧见 [`quantization/`](quantization/index.md),engine 侧(GPTQ/AWQ/FP8 注册分派)见 [`vllm/`](vllm/concepts/quantization-gptq-awq-fp8.md)。
- **硬件真实效率**:CUTLASS 的 tiling/Tensor Core 机制是理解「FLOPs≠延迟」的底层;应用侧实例见 [`computer-vision/concepts/trt-vit-inference.md`](computer-vision/concepts/trt-vit-inference.md)(以 TensorRT 实测延迟为架构设计准绳,TensorRT 内部以 CUTLASS 生成 GEMM kernel)。
- **精确 attention 省显存谱系**:Milakov online softmax → Rabe & Staats O(1) 显存 → FlashAttention IO-aware([`attention/concepts/exact-attention-memory-lineage.md`](attention/concepts/exact-attention-memory-lineage.md))。

---

## 阅读建议

- **临时查证** → 直接搜关键词(`PagedAttention`、`split-K`、`TiledMMA`、`sink token`)
- **回顾某领域** → 打开该领域 `index.md`,按索引跳
- **完整学习** → 打开领域 index.md,按 `concepts/` 深度篇导航顺序读(各篇文末『溯源』可回原始文献)

## 收纳规范(给自己定的,写这条是为了以后不纠结)

1. 每篇用 YAML frontmatter:`title` / `type` / `tags` / `source` / `date`
2. 每篇开头一段「TL;DR」,30 秒讲完结论
3. 概念篇管「懂」,案例篇管「查」,清单篇管「用」——三类分开,不混在一篇里(type 全集为 concept/case/checklist/howto/reference/domain 六类)
4. 引用外部资料只放链接索引([references.md](cutlass/index.md) 域内各领域 references 为范式),不复制全文
5. **多来源登记**(ppt / pdf / 网页 / wiki …):每篇 `source` 写成「类型 + 标识符 + 链接」结构化形式;来源互不相同但内容重叠时,只收一条知识,`source` 里列全部出处 → 详见 [SOURCE-POLICY.md](SOURCE-POLICY.md)
6. 拿到新资料先问「这条知识是不是已在库里?」→ 是则补 source 去重,否则新开篇;不建两篇内容重叠的篇
7. **把某领域"做厚"**(升级到专业深度、多源综合)时,按领域 index.md 定位判断 + 主动网络多源综合,写成"打开一篇能读懂、有专业深度"的深度条目(机制原理/可复现细节/数据对比/实践要点/溯源完整)。
