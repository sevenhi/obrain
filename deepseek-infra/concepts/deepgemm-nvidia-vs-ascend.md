---
title: DeepGEMM NVIDIA 版 vs Ascend 版 — 同 API 不同后端
type: concept
tags: [deepgemm, nvidia, cuda, cutlass, ascend, gemm, fp8, fp4, bf16, megamoe, deepjit, tma, wgmma, mad, fractal]
source:
  - "code: deepseek-ai/DeepGEMM(NVIDIA 原版)| https://github.com/deepseek-ai/DeepGEMM ↑primary"
  - "code: deepseek-ai/DeepGEMM-Ascend(昇腾版)| https://github.com/deepseek-ai/DeepGEMM-Ascend"
  - "code: deepseek-ai/DeepJIT(运行时 JIT)| https://github.com/deepseek-ai/DeepJIT"
date: 2026-10
---

# DeepGEMM NVIDIA 版 vs Ascend 版 — 同 API 不同后端

## TL;DR

**DeepGEMM 是 DeepSeek 的轻量高性能 tensor core 内核库**,把现代 LLM 的关键计算原语(FP8/FP4/BF16 GEMM、Mega MoE 融合 MoE、MQA 打分、HyperConnection)收进单一 CUDA 代码库。2026-09-30 起有两个后端:**NVIDIA 版**(deepseek-ai/DeepGEMM)与**昇腾版**(deepseek-ai/DeepGEMM-Ascend),**包名相同、API 完全兼容**,差别只在底层原语。

**同 API 不同后端的关键对应**(详见 [ascend-port-principles.md](ascend-port-principles.md)):

| 概念 | NVIDIA 版 | Ascend 版 |
|---|---|---|
| 矩阵乘原语 | Tensor Core(WGMMA/MMA) | **MAD(matrix multiply-add)** |
| 数据搬运 | TMA / async copy | **稀疏数据加载** |
| 流水机制 | persistent kernel | **协程(coroutine)流水线** |
| 片上布局 | shared memory tile | **fractal 分形布局**(库层隐藏) |
| JIT | DeepJIT(NVCC→CUBIN) | DeepJIT(bisheng+ld.lld→Ascend) |

**关键数据(⚠ 口径不同,不可直接横向比)**:NVIDIA H800 上 FP8 GEMM 最高 **1550 TFLOPS**(2025-04,内部 1024×1024 形状);昇腾 950DT 上密集 GEMM 利用率最高 **99.8%**(BF16)、FP8×FP4 M-Grouped MoE **856/862 TFLOPS**、FP8 inference **850 TFLOPS**。两者硬件代际与测试口径都不同,列在一起只为说明「各自吃满自家硬件的近峰值」,不做跨硬件 PK。

---

## 1. 机制:NVIDIA 版的「轻量 + 极致」

### 1.1 设计哲学

DeepGEMM 的设计目标是一条:**单一 kernel、极致性能、代码可读**(开源周公告原话「core logic at ~300 lines」,来自 open-infra-index,本地未核)。它借鉴 CUTLASS/CuTe 的 **tiling / layout / MMA atom** 概念,但**刻意不依赖它们的模板与代数**(avoid heavy reliance on their templates or algebras),只用有限几个核心 kernel 函数,是学习 NVIDIA GPU kernel 优化的干净读本。性能却匹配甚至超过专家调优库(如 cuBLAS/CUTLASS)在多种矩阵形状上的表现。

### 1.2 JIT:安装时零 CUDA 编译

所有 kernel 都通过 **DeepJIT** 在**运行时**编译(NVCC → CUBIN → CUDA Driver API 加载),所以**安装阶段不需要编译 CUDA**。这带来两个好处:
- 安装轻(不要求 `TORCH_CUDA_ARCH_LIST`、不要求可见 GPU);
- kernel 可按当前设备形状/精度即时生成(配合 `DG_JIT_CACHE_DIR` 磁盘缓存,编译一次复用)。

JIT 缓存键由:extra_signature + NVCC 版本 hash + 生效编译 flag + post_hook 路径/内容 hash + 源文件与 tracked include 树 hash 组成(见 [deepjit.md](deepjit.md))。

### 1.3 GEMM 接口命名

约定 `D = C + A @ B`,输入形状为 **NT**(A 非转置、B 转置):

- **SM90(Hopper)** 只支持 NT 布局(row-major A, col-major B);**SM100(Blackwell)** 支持 NT/TN/NN/TT 全部布局。例如 `fp8_gemm_nt` 计算 `D = C + A @ B.T`。
- **scaling factor 格式分代**:SM90 用 FP32;SM100 用打包 **UE8M0** 格式(4 个 UE8M0 打包进一个 `torch.int`)。LHS scaling factor 都要求 TMA-aligned 且转置布局。
- 转置、FP8 cast 等由用户自己处理/融合进前序 kernel,库只优化 GEMM 本体。

### 1.4 grouped GEMM(为 MoE 专家量身)

不同于 CUTLASS 的传统 grouped GEMM,**DeepGEMM 只 group M 轴**(N、K 必须固定)——因为 MoE 里各 expert 共享同一 hidden/intermediate 形状,只有 token 数(即 M)不同:

- **contiguous layout**:训练前向 / 推理 prefill 时,把各 expert 的 token 拼成一张连续张量(`m_grouped_fp8_gemm_nt_contiguous`),每个 expert 段需对齐到 `get_mk_alignment_for_contiguous_layout()` 的 M block 大小;
- **masked layout**:解码阶段启用 CUDA graph 后 CPU 不知道各 expert 收到的 token 数,用 mask 张量只算有效段(`m_grouped_fp8_gemm_nt_masked`),典型输入来自 DeepEP 低延迟 kernel;
- **K-grouped**:MoE 权重反向用(`k_grouped_fp8_gemm_tn_contiguous`,M、N 固定)。

### 1.5 Mega MoE:单 mega-kernel 融合通信与计算

Mega MoE 把 **EP dispatch + linear 1 + linear 2(FP8×FP4 或 FP8×FP8)+ SwiGLU + EP combine** 融合进一个 mega-kernel,**重叠 NVLink 通信与 tensor core 计算**。需要多进程启动 + 对称内存(PyTorch ≥ 2.9):

```python
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk, hidden,
    intermediate_hidden, mma_type='fp8xfp4')
transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)
# (可选)把权重 localize 进 locality domain 以贴近使用它的 GPU
buffer.x[:num_tokens].copy_(x_fp8)   # … 填入 topk_idx / topk_weights …
y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)
```

这是 DeepGEMM 与 DeepEP 分工里**「计算侧吃掉通信」**的一环:Mega MoE 在 DeepGEMM、原始 all-to-all 原语在 DeepEP(见 [deepep-nvidia-vs-ascend.md](deepep-nvidia-vs-ascend.md))。

### 1.6 MQA scoring kernels(lightning indexer)

为 V3.2 的 lightning indexer 提供 scoring kernel(`fp8_fp4_mqa_logits` 非分页 / `fp8_fp4_paged_mqa_logits` 分页),算每个 query 对 KV 的加权 ReLU 压缩 logits:`out = Σ_j relu(q_i @ (kv_j · kv_sf_j)) · weights[i,:]`。这是 DSA 链路里出索引前的打分环节(库内 [deepselect.md](deepselect.md) 消费这些 logits 做 TopK)。

### 1.7 稀疏 MQA logits(NVIDIA 09-30 引入、10-14 对齐修复;Ascend 2026/10 新增)

**NVIDIA 侧 2026-09-30(v2.8.1,#462)已引入「稀疏 MQA 打分」内核族,10-14 v2.8.2 做 stride 对齐等修复;Ascend 侧 Public Release 2026/10 加入**,扩展 MQA 评分到**稀疏 KV 上下文**场景(DSA/Lightning Indexer 的索引环节,配合 DeepSelect 选出的稀疏块):

- **非分页** `fp8_fp4_sparse_mqa_logits` + **分页** `fp8_fp4_paged_sparse_mqa_logits`;
- 每个 query 只 attend 选中的**稀疏 KV 块**(`sparse_kv_block_indices`,`sparse_block_kv=8/16`,最多 `num_max_sparse_blocks` 块),而非全量 KV —— 由 prepare 元数据 `get_sparse_mqa_logits_metadata` / `get_paged_sparse_mqa_logits_metadata` 生成调度信息,logits 调用传 `metadata` 复用;
- 支持 **MXFP4 / MXFP8**(KV cache 量化),`head_dim=128`,`page_kv` 典型 64(NVIDIA 侧仅要求 `page_kv % sparse_block_kv == 0`);NVIDIA 侧 SM100(Blackwell)专属(`get_arch_major()==10` 才测),Ascend 侧走 950 + MAD/协程原语;
- **FIRST/REUSE 状态(Ascend 版 paged wrapper 特有,见 DeepGEMM-Ascend `__init__.py`)**:首次调用缓存 KV 快照,后续同 metadata 复用该快照只换当前 Q/weights;KV 内容变化须重新生成 metadata —— NVIDIA 版纯 C++ binding 无 is_first/FIRST-REUSE 状态;
- **split-KV 归并**:NVIDIA SM100 侧 `kSplitKV=512/640`(fp8/fp4,来自 `get_sparse_mqa_split_kv`)是**单个 CTA 内部的 KV 切分粒度**,归并(reduce)由 CTA 内 math warpgroups 累加(`__hadd2_rn`)完成,非跨 SM/core;Ascend 侧则是**按 core 切分处理**:`num_ai_cores` 把 Q 块/KV split 调度到各 AI core,metadata 中 `QBlockSplitKVOutputInfo` 的 `output_ranges`/`masks` 记录每块输出范围与有效 mask(每 core 写自己负责的段),两端不要混。

> 稀疏 MQA 是 **DSA 注意力链路里「打分」环节的稀疏化版本**:DeepSelect(TopK,见 [deepselect.md](deepselect.md))选稀疏块 → 稀疏 MQA 只对这些块打分 → 产出的稀疏 logits 供 indexer 用。与 FlashMLA sparse 是不同环节(见 [flashmla.md](flashmla.md))。注意区分两个阶段:§1.6 的**全量 MQA 打分**(`fp8_fp4_mqa_logits`,对全量 KV)产 logits → DeepSelect TopK 选稀疏块;§1.7 的**稀疏 MQA 重打分**(`fp8_fp4_sparse_mqa_logits`)只对已选出的稀疏块打分,两个打分方向相反、串行衔接,不是同一环节。

---

## 2. 与 Ascend 版的对照(同 API,不同后端)

昇腾版新建独立仓 **DeepGEMM-Ascend**,**包名 `deep_gemm` 不变、API 对齐**,把同一套「轻量 GEMM 内核」搬到昇腾。机制细节见 [deepgemm-ascend.md](deepgemm-ascend.md),这里只列对照结论:

| 维度 | NVIDIA | Ascend |
|---|---|---|
| 计算原语 | WGMMA/MMA(Tensor Core) | MAD 矩阵乘加 |
| 搬运 | TMA/async copy | 稀疏数据加载 |
| 流水 | persistent kernel | 协程流水线 |
| 片上布局 | shared memory(显式) | fractal 分形(库层隐藏) |
| scaling factor | SM90 FP32 / SM100 UE8M0 | 每对 UE8M0 沿 K 维打包进 `int16`,MN-major |
| 支持算子 | BF16/FP8/FP4、Mega MoE、MQA logits、HC | BF16/FP8/FP4、MQA logits、MegaMoE、HC prenorm(依赖 tilelang) |
| 环境 | SM90/SM100 + CUDA 12.9+ | 昇腾 950 + CANN 9.2.0(README 写作「9.20」)+ torch_npu |

> **支持算子补注(2026 新增)**:README News 2026-09-10 起新增 **Sparse Indexer / Mega Gate / Mega mHC** 算子族(同批含 DeepJIT、MoE/Indexer 优化,PR #432);**PDL**(Programmatic Dependent Launch,`set_pdl`/`get_pdl`)为 2026-04-16 批次引入(PR #304);2026-09-30 另增 locality domain 特性(PR #462,与 DeepGEMM-Ascend 同日发布)。

**两条 P0 结论**:
1. **「API 兼容」是移植的锚**:开发者按 NVIDIA 版写 kernel 调用,换昇腾只改环境不改代码——这是 DeepSeek 把算子生态从「NVIDIA 单一后端」变成「双平台」的核心手段(与 DeepEP-Ascend、TileKernels v2.0.0 同一策略)。
2. **scaling factor 格式两边不通用**:昇腾用 `int16` 打包且 MN-major,FP8 场景传 SF 时别沿用 NVIDIA 的 FP32/UE8M0 格式——这是最容易踩的坑。

---

## 3. 数据(⚠ 各标口径,不做跨硬件 PK)

| 场景 | 数值 | 环境口径 |
|---|---|---|
| FP8 GEMM(H800) | **1550 TFLOPS** | NVIDIA H800,2025-04 PR #74/#78/#81/#86,内部形状 |
| FP8 GEMM(Hopper) | 1350+ TFLOPS | 2025-02 开源周 Day 3 公告,Hopper(来自 open-infra-index,本地未核) |
| 密集 GEMM 利用率(Ascend) | 99.8% | 950DT,BF16×BF16,M=4096/N=7168/K=16384 |
| M-Grouped MoE(Ascend) | 856 / 862 TFLOPS | 950DT,FP8×FP4,4/8 组专家 |
| FP8 inference(Ascend) | 850 TFLOPS | 950DT,M=4096/N=4096/K=7168 |

> ⚠ NVIDIA 的「1550/1350 TFLOPS」与昇腾的「99.8%/856/862」**不是同一套形状、同一代硬件**,不能直接比「谁快」。它们各自说明:两边都吃到了自家硬件的近峰值。

---

## 4. 实践要点 / 坑

- **submodule 必克隆**:`git clone --recursive`,否则 JIT/依赖缺失。NVIDIA 版 `./develop.sh`(链接头 + 构建 C++ 扩展)、`./install.sh`(装 wheel)。
- **NVIDIA 版环境**:SM90/SM100 + Python 3.8+ + C++20 `<format>` + CUDA 12.9+ + PyTorch 2.3+ + CUTLASS 4.0+(submodule)。
- **JIT 环境变量**:`DG_JIT_*` 优先、回落到 `DJ_JIT_*`(DeepJIT 全局);常用 `DG_JIT_CACHE_DIR`(默认 `$HOME/.dj`)、`DG_JIT_DEBUG`、`DG_JIT_CHECK_NO_SPILLS`、`DG_JIT_DUMP_SASS` 等。
- **scaling factor 别混**:NVIDIA SM90=FP32、SM100=UE8M0 打包;昇腾=`int16` 打包 MN-major。跨后端复用时显式重算 SF,别沿用。
- **grouped GEMM 只 group M**:N/K 必须固定;contiguous 布局要对齐 `get_mk_alignment_for_contiguous_layout()`,masked 布局靠 mask 张量标有效段。
- **「1550/1350/99.8%/856/862/850」别张冠李戴**:1550/1350 是 NVIDIA H800/Hopper FP8;99.8% 是昇腾 BF16 密集;856/862 是昇腾 M-Grouped;850 是昇腾 FP8 inference。

---

## 溯源

- **DeepGEMM 原版 README**(主源:接口命名/scaling factor 分代/grouped GEMM/Mega MoE/MQA/1550 TFLOPS/JIT 环境变量):https://github.com/deepseek-ai/DeepGEMM
- **DeepGEMM-Ascend README**(昇腾版对照,MAD/fractal/协程/scaling factor int16/CANN 9.20):https://github.com/deepseek-ai/DeepGEMM-Ascend
- **DeepJIT**(运行时 JIT):https://github.com/deepseek-ai/DeepJIT
- **开源周 Day 3 公告**(1350+ TFLOPS、~300 行核心逻辑;来自 open-infra-index,本地未核):https://github.com/deepseek-ai/open-infra-index
- **库内交叉**:[deepgemm-ascend.md](deepgemm-ascend.md)(昇腾版机制全文)、[ascend-port-principles.md](ascend-port-principles.md)(NVIDIA→昇腾抽象对照)、[deepjit.md](deepjit.md)(JIT 机制)、[../../cutlass/index.md](../../cutlass/index.md)(CUTLASS 造内核方法论)、[deepep-nvidia-vs-ascend.md](deepep-nvidia-vs-ascend.md)(Mega MoE 的通信侧)。
