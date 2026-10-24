---
title: 量化与压缩
type: domain
tags: [quantization, compression, llm-inference, kv-cache, low-rank, outlier]
source:
  - "pdf: HyQuant 论文全文 | arXiv:2608.27875v3(EMNLP 2026 Main;v1/v2 摘要未变,仅补录用标注)"
  - "pdf: SpQR 论文全文 | arXiv:2306.03078 (ICLR 2024)"
  - "pdf: On the Effectiveness of Low-Rank Matrix Factorization for LSTM Model Compression | arXiv:1908.09982 (PACLIC 2019)"
  - "pdf: GPTQ | arXiv:2210.17323"
  - "pdf: AWQ | arXiv:2306.00978"
  - "pdf: SmoothQuant | arXiv:2211.10438"
  - "pdf: KIVI | arXiv:2402.02750"
  - "pdf: Minitron(激活幅度结构化剪枝 + 知识蒸馏)| arXiv:2407.14679;实战版 arXiv:2408.11796"
  - "pdf: Puzzle(异构剪枝 NAS + MIP,即 Puzzletron 论文)| arXiv:2411.19146"
  - "code: NVIDIA/Model-Optimizer 剪枝示例与 PR #2922(2026-10-23,Megatron-Core 26.10 兼容)| https://github.com/NVIDIA/Model-Optimizer"
  - "code: NVIDIA/Model-Optimizer MoE 逐层 GPTQ 链(PR #2840 逐专家 Hessian → #2864 加速 → #2877 设备修复;#2943 AutoQuant 探测修复;commit f29bb60)| https://github.com/NVIDIA/Model-Optimizer/pull/2864;仓库 HEAD 已到 0638a27(2026-10-24)"
  - "code: NVIDIA/Model-Optimizer PR #2738 DP=EP 专家并行 PTQ 传输(2026-10-24 合并,21 个 PR 的第 1 块)| https://github.com/NVIDIA/Model-Optimizer/pull/2738"
  - "web: ModelOpt 公告 *Quantizing a 4.9 TB Qwen3.8 Model on a Single GPU*(2026-09-28,逐层校准/逐层导出/单卡生产数据)| https://nvidia.github.io/Model-Optimizer/announcements/20260928-single-gpu-ptq.html"
  - "web: vLLM Quantization | https://docs.vllm.ai/en/latest/features/quantization/ | 阅读日期 2026-09-27"
date: 2026-10
---

# 量化与压缩

## 一句话定位

> 领域索引。定位:**LLM/NN 的量化与压缩**——权重/KV cache 量化的算法侧(混合精度、低比特、误差分析方法)+ 权重压缩的两条结构路线(稀疏+量化混合表示/低秩分解 = 不动形状压表示;**结构化剪枝 + 蒸馏 = 直接改形状砍参数**)。2026-09 由原 `llm-inference` 域下的论文笔记目录(papers/)升级而来;同月批C扩充后定位从「量化算法」扩为「量化与压缩」(SpQR=量化+稀疏混合路线,LSTM 低秩=分解/剪枝路线);2026-10 补 Minitron/Puzzletron,把「剪枝」从「只有理论/低秩」补到「有完整工程链路(打分→剪→蒸馏→量化→部署)」。

## 条目索引

| 篇目 | 类型 | 一句话 |
|---|---|---|
| [HyQuant — LLM Attention 混合精度量化](concepts/hyquant-hybrid-precision-quantization.md) | concept | 竖线+局部窗口留全精度,其余 KV 压低比特(arXiv 2608.27875) |
| [SpQR — 稀疏+量化混合表示](concepts/spqr-sparse-quantized.md) | concept | ~1% 高敏感 outlier 存 16-bit(CSR)+ 其余 3-4bit + bilevel 量化统计量,首个近无损(<1% PPL)3-4bit 权重压缩(arXiv 2306.03078, ICLR 2024) |
| [LSTM 低秩压缩 — 分解 vs 剪枝](concepts/lstm-lowrank-compression.md) | concept | SVD/Semi-NMF vs 剪枝:压 W_h(天生低秩)远比压 W_i 安全,核范数是可压缩性先验(arXiv 1908.09982, PACLIC 2019) |
| [PTQ: GPTQ / AWQ / SmoothQuant](concepts/ptq-gptq-awq-smoothquant.md) | concept | 权重/激活后训练量化的三条路线:二阶误差补偿、激活感知权重保护、平滑离群值后做 W8A8;按显存、硬件和质量目标选型 |
| [KV Cache 量化](concepts/kv-cache-quantization.md) | concept | 把运行时 KV cache 从权重压缩问题单独拆出:按 token/channel 分组、对称/非对称量化、混合精度与长上下文权衡 |
| [2-bit 权重量化前沿:QuIP# / AQLM](concepts/two-bit-weight-frontier.md) | concept | ≤4-bit 权重 SOTA:QuIP# 旋转消 outlier+E8 格码+fine-tune,AQLM 多码本加和;3-bit 近无损、2-bit 有可感知退化(arXiv 2402.04396 / 2401.06118) |
| [KV cache 极低比特前沿:KVQuant / KVTuner](concepts/kvquant-kvtuner-frontier.md) | concept | KVQuant 3-bit 近无损(Pre-RoPE+NUQ+sparse,10M context);KVTuner layer-wise 精度对搜索(≈3.25-bit 近无损,+21.25% 吞吐) |
| [FP8/FP4 硬件原生格式](concepts/fp8-fp4-formats.md) | concept | E4M3/E5M2/E2M1 位布局与 scale 粒度(per-tensor/token;DeepSeek-V3:权重 block 128×128、激活 1×128)、Delayed vs JIT scaling;NVFP4/MXFP4/MXFP8 与 Blackwell tcgen05 路径;与 QuIP#/AQLM 算法压缩分野;训练 V3 recipe vs 推理 FP8 KV |
| [结构化剪枝 + 知识蒸馏:Minitron / Puzzletron](concepts/structured-pruning-minitron-puzzletron.md) | concept | 改结构而非改位宽:1024 样本纯前向激活幅度打分 → 按轴剪(深度/神经元/头/embedding/MoE/mamba)→ KL 蒸馏在 <3% token 量上恢复。15B→8B 只用 94B token(40× 更少)、MMLU +16%;Puzzletron 用块级局部蒸馏 + MIP 搜异构逐层架构(Llama-70B→51B,单卡 H100 2.17× 吞吐、保留 98.4%);含 Mamba hybrid 剪枝的三处状态一致性坑 |
| [大规模 MoE 的 GPTQ 工程化](concepts/moe-gptq-layerwise-calibration.md) | concept | 算法之外的落地三件事:HF fused MoE 逐专家 Hessian(旧版直接跳过专家,而专家是 MoE 的绝大部分权重)、一层 13–23 GiB/整模型 638–945 GiB 迫使逐层校准、逐层一开三处白等暴露(多余 max 前向 / Hessian 被 65% 阈值赶去 CPU / batch=1 每样本重跑专家循环)。逐层 GPTQ 20 h→44 min;4.892 TB 模型单卡 GB300 3 h 04 m、峰值 141.7 GB |
| [MoE 的专家并行 PTQ 校准(DP=EP all-to-all 传输)](concepts/expert-parallel-ptq-calibration.md) | concept | 复现要每 rank 整模型、FSDP2 又把专家 gather 回来;DP=EP 让稠密层走数据并行、专家分片、token 走 all-to-all,关键是**专家 amax 覆盖全局 token 池**。plain NCCL + 定序 combine 保证逐位可复现(旧 index_add_ 版 27% 的 input scale 与单卡不同);缓冲早释放峰值 2122→706 MiB |

**共同心法(十一篇一条线)**:压缩误差不是均匀分布的——HyQuant 按 attention mass 分精度、SpQR 按敏感度分位宽、LSTM 篇按谱结构(核范数)分压缩方式、GPTQ/AWQ/SmoothQuant 与 KV cache 篇按误差补偿/激活敏感/离群迁移分预算、QuIP#/AQLM 与 KVQuant/KVTuner 用旋转消 outlier / 多码本加和 / layer-wise 精度对把 2-bit 联合压缩推进到极限、Minitron/Puzzletron 按激活幅度与逐层重要性决定砍哪一维;**都是"把预算花在杠杆点上",而非全局均匀砍**。而 Minitron/Puzzletron 是唯一**改结构**的一条线(其余都在改表示精度),所以它的精度恢复不靠微调而靠 teacher 蒸馏。MoE GPTQ 工程化篇则是这九篇的**落地影子**:一旦模型大到"专家 Hessian 一层就是 22 GiB",算法本身的优劣退居其次,**显存预算与校准协议成了决定能不能跑完的变量**——这与 Minitron/Puzzletron 用"逐块"代替"整模型"是同一种工程直觉。专家并行 PTQ 校准篇(DP=EP all-to-all)是第二个落地影子,且它的心法不是"把预算花在杠杆点上",而是**校准统计量的覆盖面也是精度变量**:专家 amax 只看到一个 rank 的 token 切片就是系统性偏差——与"逐层 vs 整模型"是同一种工程直觉的另一个维度,逐层解决"显存 × 层数",EP 解决"显存 × 卡数 + 校准 token 的覆盖面",两者可叠加。

## 相关条目(跨域互链)

- [vLLM 量化:GPTQ / AWQ / FP8](../vllm/concepts/quantization-gptq-awq-fp8.md)——**engine 视角**:量化的注册/分派/内核集成如何在推理引擎里落地;本领域是**算法视角**(怎么量化、误差怎么控制),两者互补。
- [attention/flash-attention-implementation](../attention/implementation/flash-attention-implementation.md)——FA3 的 FP8 block-quantize(Q/K/V 分块量化 + incoherent processing)是"量化进 attention kernel"的实例;HyQuant 的 dequantize-in-attention 同属此路线。
- SpQR 的推理 kernel(稀疏 CSR matmul + dense 3/4-bit matmul)是 CUTLASS 域 GEMM 定制的应用场景(参见 [cutlass/](../cutlass/index.md))。

## 扩展位(待补充候选)

- 已补充:GPTQ / AWQ / SmoothQuant(算法侧统一篇)、KV cache 量化(KIVI + vLLM 工程视角)、2-bit 权重前沿(QuIP# / AQLM)、KV cache 极低比特前沿(KVQuant / KVTuner)。
- 已补:FP8/FP4 硬件原生格式(2026-10)、结构化剪枝 + 蒸馏 Minitron/Puzzletron(2026-10)。
- 待补充:BitsAndBytes/NF4/QLoRA 机制篇;GGUF/k-quant/imatrix 体系(2026-10 已部分覆盖:`imatrix` 的 `E[x²]` 加权搜索与 `[gptq, imatrix]` 组合语义见 [MoE GPTQ 工程化篇](concepts/moe-gptq-layerwise-calibration.md));激活量化独立篇(W4A8/LLM.int8() 离群值);QuaRot/SpinQuant/QTIP 旋转家族;W4A16 vs W8A8 kernel 成熟度(Marlin/Machete);BitNet b1.58 training-aware 对比;各 GPU 架构实测基准;**剪枝与量化叠加的联合搜索**(剪完再量化的 scale 重校准、剪枝感知的量化粒度选择);**稀疏化(Wanda/SparseGPT 的非结构化/Semi-structured 路线)**与剪枝的对比(一个是 mask 不动 shape、一个是真改 shape);**EP PTQ 的后续 20 个 PR**([4/21] DistributionPlan 与分片加载、[13/21] 起的模型 spec 与不 gather 导出)——本篇目前只覆盖已合并的传输层 [1/21]。

## 溯源

- 外部链接索引见 [references.md](references.md)。
