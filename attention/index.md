---
title: Attention(算法层 algorithm/ + 实现层 implementation/ + 框架层 frameworks/)
type: domain
tags: [transformer, attention, self-attention, streamingllm, megabyte, laser, spatten, flashattention, flexattention, gpu, cuda]
source:
  - "pdf: Attention Is All You Need | arXiv:1706.03762 NIPS 2017"
  - "pdf: Efficient Streaming Language Models with Attention Sinks | arXiv:2309.17453 ICLR 2024"
  - "pdf: LASER: Attention with Exponential Transformation | arXiv:2411.03493 ICML 2025"
  - "pdf: SpAtten | arXiv:2012.09852 HPCA 2021"
  - "pdf: MEGABYTE | arXiv:2305.07185 NeurIPS 2023"
  - "pdf: Self-Attention does not Need O(n²) Memory (Rabe & Staats, FA1 前驱) | arXiv:2112.05682"
  - "pdf: FlashAttention FA1/FA2/FA3/FA4 | arXiv:2205.14135 / 2307.08691 / 2407.08608 / 2603.05451(详细 source 见 implementation/flash-attention-implementation.md frontmatter)"
  - "code: github.com/Dao-AILab/flash-attention | Star 24.7k"
date: 2026-10
---

# Attention(算法层 algorithm/ + 实现层 implementation/ + 框架层 frameworks/)

> 领域索引(导航起始页)。**领域知识本体分三层目录:`algorithm/`(理论/算法)、`implementation/`(GPU 内核/工程)与 `frameworks/`(用户编程接口/编译器后端)**,每篇一个专题、可独立打开读懂;本页放定位、关键事实与导航,不重复正文。外部链接索引见 [references.md](references.md)。
> 沿革:2026-09 由原 `attention-and-transformers` 与 `flash-attention` 两域合并;同月二次整合——concepts 平铺改为 algorithm/implementation 两层,三篇 FA 内核篇(io-aware-tiling/work-partitioning/fa3-hopper-async)合成为一篇 `implementation/flash-attention-implementation.md`,LLaMA 架构篇移出为独立域 [`model-architecture/`](../model-architecture/index.md)(扩展为通用 LLM 架构篇)。

## 一句话定位

以 Attention Is All You Need(2017)为根,覆盖注意力机制的三个层次:**算法层 `algorithm/`**(为什么这样设计、softmax 的数学性质与推论、长序列架构分解、省显存谱系)、**实现层 `implementation/`**(IO-aware tiling、online softmax、FlashAttention 四代高效内核——不改数学的精确解)与 **框架层 `frameworks/`**(用户编程接口+编译器后端:FlexAttention 如何用几行 Python 声明变体、自动落到 Triton 或 FA4/CuTeDSL)。模型整体架构(RMSNorm/SwiGLU/RoPE/GQA、训练配比)见 [`model-architecture/`](../model-architecture/index.md) 域。

## 算法层 algorithm/(5 篇)

| 条目 | 类型 | 一句话 | 状态 |
|---|---|---|---|
| [attention-is-all-you-need](algorithm/attention-is-all-you-need.md) | concept | Transformer 原始设计:√d_k 缩放的推导、多头为何免费、O(1) 路径 vs O(n²) 账单、base/big 训练配方与 WMT14 数据 | ✅ 已入库 |
| [attention-sinks-streaming](algorithm/attention-sinks-streaming.md) | concept | attention sink 因果链(softmax 和为 1 → 垃圾注意力倒给开头 token)+ StreamingLLM sink+滚窗:4M token 稳定 PPL、22.2× 加速、learnable sink token | ✅ 已入库 |
| [attention-variants-laser-spatten](algorithm/attention-variants-laser-spatten.md) | concept | 双线索:LASER 对 exp(V) 做 attention 修梯度饱和(训练侧,−9.4% 时间/−15.65% 参数)× SpAtten 级联 token/head 剪枝 + progressive quant(硬件侧,10× DRAM/162× GPU 加速) | ✅ 已入库 |
| [megabyte-multiscale](algorithm/megabyte-multiscale.md) | concept | 多尺度分解:Global(patch 间)+ Local(patch 内字节),attention O(T^4/3)、FFN per-patch → 同算力 P 倍大模型;tokenization-free 追平 subword | ✅ 已入库 |
| [exact-attention-memory-lineage](algorithm/exact-attention-memory-lineage.md) | concept | FA 前驱:Rabe & Staats lazy softmax+在线 max 重归一 → O(1) 存注意力;R&S 省"显存峰值"、FA 省"访存次数"(arXiv 2112.05682) | ✅ 已入库 |

## 实现层 implementation/(1 篇,原 3 篇合成)

| 条目 | 类型 | 一句话 | 状态 |
|---|---|---|---|
| [flash-attention-implementation](implementation/flash-attention-implementation.md) | concept | FA 四代主线合成篇:FA1 IO-aware tiling + online softmax(算法伪代码、Θ(N²d²/M) 推导、4d 块大小)→ FA2 work partitioning(减 non-matmul、warp 按 Q 行切、序列维并行)→ FA3 Hopper 异步(WGMMA/TMA/overlap/pingpong/FP8)→ FA4 Blackwell(CuTe DSL/UMMA/TMEM);反向重算、实践 checklist、全部溯源 | ✅ 已入库 |

## 框架层 frameworks/(1 篇)

| 条目 | 类型 | 一句话 | 状态 |
|---|---|---|---|
| [flexattention](frameworks/flexattention.md) | concept | FlexAttention 用户编程接口篇:score_mod/mask_mod 几行 Python 声明注意力变体(ALiBi/滑窗/document masking/softcap…),torch.compile 自动融合下降+自动反向;BlockMask 三类 block(full/partial/empty)吃稀疏;底层两条 codegen 路线 Triton vs CuTeDSL 对比;2026-03 新增 FA4(FLASH)后端(compute-bound 比 Triton 快 1.2–3.2×、~95% FA3)与全部限制清单 | ✅ 已入库 |

> 分层边界:`implementation/` 讲 FA 内核**怎么写**(四代机制),`frameworks/` 讲用户**怎么用接口声明变体、编译器怎么自动选/落后端**。两篇互补互链。

### FlashAttention 四代演进(速览,详见合成篇)

| 代次 | 时间 | 论文 | 目标 GPU | 核心实现技术 | MFU(前向) |
|---|---|---|---|---|---|
| FA1 | 2022.05 | NeurIPS 2022 | A100 | IO-aware tiling + online softmax | 25-40% |
| FA2 | 2023.07 | ICLR 2024 | A100+H100 | 更好 work partitioning, 减 non-matmul FLOPs | 50-73% |
| FA3 | 2024.07 | NeurIPS 2024 | H100 | WGMMA async, TMA, FP8 | ~75% FP16 / ~61-66% FP8 |
| FA4 | 2026.03 | arXiv:2603.05451 (MLSys 2026) | B200 | CuTe DSL, 异步 UMMA/tcgen05, 软 exp+软 softmax rescale, TMEM+2-CTA MMA | ~71% (BF16, up to 1613 TFLOPS) |

## 阅读顺序建议(一条主线读)

```
Attention Is All You Need (2017)  ← 根:QKV/多头/√d_k/位置编码/O(n²) 账单
        │
        ├── softmax 和为 1 的三个推论  [algorithm/]
        │     ├─ KV cache 侧: attention sink → StreamingLLM (ICLR 2024)
        │     ├─ 训练侧: 概率长尾 → 梯度饱和 → LASER (ICML 2025)
        │     └─ 推理硬件侧: 概率尖/稀疏 → SpAtten (HPCA 2021)
        │
        ├── 长序列架构分解: MEGABYTE (NeurIPS 2023)  ← FFN 才是大头  [algorithm/]
        │
        ├── 架构改进定型: LLaMA (2023)  ← 移出为独立域 model-architecture/
        │     (RMSNorm/SwiGLU/RoPE + 多数据小模型 + Llama 2/3 演进)
        │
        ├── O(n²) 账单的内核精确解  [algorithm/ 谱系 + implementation/ 主线]
              Rabe & Staats (2021, O(1) 显存前驱)
                → FA1 (2022, IO-aware tiling + online softmax)
                → FA2 (2023, work partitioning)
                → FA3 (2024, Hopper 异步 + FP8) → FA4 (2026, Blackwell)

        └── 变体爆炸的用户侧解法: FlexAttention (2024→2026)  [frameworks/]
              score_mod/mask_mod 声明变体 → torch.compile 融合下降
                → Triton 后端(默认) / FA4 CuTeDSL 后端(FLASH, 2026-03)
```

## 相关条目(跨域互链)

- **[`model-architecture/`](../model-architecture/index.md)**:LLM 整体架构域(LLaMA 三件套/位置编码演进/GQA/训练哲学),自本域 llama-architecture 篇扩展而来;attention 是其中被"改进定型"的组件,机制基础仍在本域。
- **[`vllm/`](../vllm/index.md)**:StreamingLLM(逐出谁)与 PagedAttention/KV cache 分页(装得下就全留,见 [`vllm/concepts/paged-attention.md`](../vllm/concepts/paged-attention.md))正交互补——长上下文/常驻流式服务的两个半边;vLLM 未内置 sink 逐出,TensorRT-LLM/HF 已内置(SGLang 主线仍支持 attention sink:sinks kwarg / learnable sink / speculative draft sink 等多个 backend)。vLLM 集成的注意力内核即本域 FA 谱系。
- **[`quantization/`](../quantization/index.md)**:SpAtten progressive quantization(attention 输入 MSB/LSB)与 LLM 权重量化(SpQR/GPTQ/AWQ)是量化谱系的两侧;FA3 的 FP8 block-quantize 是"量化进 attention kernel"的实例;attention sink 与量化 outlier 同源(softmax 和为 1 制造极端 token)。
- **[`cutlass/`](../cutlass/index.md)**:FA 内核的底座(FA2 用 CuTe 原语写 tiling、FA3 基于 CUTLASS 3.x collective 抽象构建);本域 SpAtten 代表"绕开通用 GEMM 库造专用 attention 硬件"的对照路线。
- **[`computer-vision/`](../computer-vision/index.md)**:NeRF/TRT-ViT 的「访存而非算力是瓶颈」判断,与本域 FA 的 IO-aware 分析同一方法论。

## 关键事实速记

**算法层**

- 原论文 Table 1:self-attention 每层 O(n²·d)、串行 O(1)、路径 O(1);n<d 时反而比 RNN 便宜——quadratic 当年是卖点;
- StreamingLLM:4 个 sink token 饱和(1~2 不够),PPL 5158→5.40,4M tokens 稳定;
- LASER:80% 注意力概率 <10⁻³、20% <10⁻⁷(Jacobian ∝ a 的直接后果);Log-Weighted-Sum-Exp trick 后 FlashAttention 零修改可用;
- SpAtten:attention 占 GPT-2 延迟 >50% / FLOPs ~10%,73% 时间在搬数据;仅 5.9% 样本需要 LSB;
- MEGABYTE:GPT-3 尺寸 attention 只占 1.4% FLOPs;P=T^{1/3} → O(T^{4/3});
- Rabe & Staats(FA 前驱):lazy softmax + 在线 max 重归一 → O(1) 存注意力;R&S 省"显存峰值"、FA 省"访存次数"——同为精确 attention,优化目标不同,后果是 FA 快 2-4×、R&S 与标准实现同速或略慢。

**实现层**

- 为什么慢:标准 attention 中 `S = Q@Kᵀ·scale`、`P = softmax(S)` 都是 N×N 矩阵,需写入 HBM 再读回;N=8192、d=128 时 S 占 ~128MB,远超 GPU SRAM(~192KB)。**瓶颈是内存搬运,不是计算**;
- IO 复杂度:标准 attention Θ(Nd + N²)(元素访问)→ FlashAttention **Θ(N²d²/M)**(M=SRAM 容量),显存 O(N²)→O(N),且不损失精度(精确);
- FA2 快 2× 的来源:减 non-matmul FLOPs、warp 按 Q 行切分免通信、序列维并行(切 Q 免归并,切 KV 才要 split-K 归并);
- FA3(H100):异步 WGMMA + TMA + softmax/matmul/load 重叠 pipeline + FP8;论文口径 BF16 相对 FA2 加速 1.5-2×,FP16 前向最高 740 TFLOPS(≈H100 dense 峰值的 75%);
- 显存 O(N) 的胜利来自"前向只存 LSE 统计量 + 反向重算"(约 2× 重复 FLOPs 换长序列可行)。

**框架层**

- FlexAttention = 两个扩展(score_mod 逐点改分 + BlockMask 块稀疏迭代)+ torch.compile 融合下降 + 自动反向;"软件彩票"问题的解法;
- mask 每元素应用有 15-20% 退化 → BlockMask 区分 full(免 mask)/partial(需 mask)/empty(整块跳过);causal ~50% 稀疏,不吃稀疏白慢 2×;
- FLASH(FA4 CuTeDSL)后端 2026-03 上线:compute-bound 比 Triton 快 1.2-3.2×、前向 ~95% FA3;代价是实验性+限制(SM100 BLOCK_SIZE=(256,128)、捕获标量须 dynamic=False、捕获 buffer 不支持求梯度)。

## 溯源

> 外部链接索引见 [references.md](references.md);精选论文:

- FA1: Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022, arXiv:2205.14135
- FA2: Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, ICLR 2024, arXiv:2307.08691
- FA3: Dao et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, NeurIPS 2024, arXiv:2407.08608
- FA4: *FlashAttention-4*, MLSys 2026, arXiv:2603.05451
- 前驱: Rabe & Staats, *Self-Attention does not Need O(n²) Memory*, arXiv:2112.05682(FA1 引用[66],附录 B.5 专节比较)→ [exact-attention-memory-lineage.md](algorithm/exact-attention-memory-lineage.md)
- FA 四代全部链接与博客口径见 [implementation/flash-attention-implementation.md](implementation/flash-attention-implementation.md) §7;算法层各篇论文链接见 [references.md](references.md)。
