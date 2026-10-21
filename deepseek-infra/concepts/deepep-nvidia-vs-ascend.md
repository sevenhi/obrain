---
title: DeepEP NVIDIA 版 vs Ascend 版 — MoE 专家并行通信库
type: concept
tags: [deepep, nvidia, cuda, nccl, gin, rdma, nvlink, ascend, moe, expert-parallel, dispatch, combine, fp8, engram, deepjit, epbuffer]
source:
  - "code: deepseek-ai/DeepEP(NVIDIA 原版;2026-10-21 核对于 f6aa38fe,PR #788 submodule 转 HTTPS)| https://github.com/deepseek-ai/DeepEP ↑primary"
  - "code: deepseek-ai/DeepEP-Ascend(昇腾版)| https://github.com/deepseek-ai/DeepEP-Ascend"
  - "code: deepseek-ai/DeepJIT(运行时 JIT)| https://github.com/deepseek-ai/DeepJIT"
date: 2026-10
---

# DeepEP NVIDIA 版 vs Ascend 版 — MoE 专家并行通信库

## TL;DR

**DeepEP(Deep Expert Parallelism,专家并行通信库)** 是 DeepSeek 的 MoE **专家并行(EP)通信库**,提供 EP all-to-all 的 **dispatch**(token → 负责其 expert 的 rank)与 **combine**(expert 结果 → token 原 rank)两类高性能 GPU kernel,支持 FP8 dispatch、冗余专家(load balancing)的 NVLink 权重/梯度交换,以及 PP/CP/DP/Engram 的试验性原语。**2026-09-30 起有 NVIDIA 版(deepseek-ai/DeepEP)与昇腾版(deepseek-ai/DeepEP-Ascend),同一 API 跑两套后端**。

**关键机制结论**:
1. **通信内核运行时 JIT 编译**(DeepJIT),安装时零 CUDA 编译;
2. **V2 从 NVSHMEM 迁到轻量 NCCL Gin 后端**,统一 `EPBuffer` 接口 + 解析式 SM/QP 估计(不再 autotune),支持更大 scale-up/scale-out 域,SM 占用从 24 降到 4–6 且峰值反升 1.3×;(⚠ 该 1.3×/24→4–6 为**旧 V2 公告口径,当前 README 已移除性能表**,见 §3)
3. **EP 通信要吃 GPU SM**(zero-SM RDMA EP 已不支持),靠「通信+计算重叠」(deferred epilogue、`EventOverlap`)把通信藏进专家 GEMM 的间隙;
4. **同 API 不同后端**:NVIDIA 侧 = NCCL Gin + IBGDA(GPUDirect Async)+ NVLink/RDMA;昇腾侧 = Ascend C + HCCL/HCOMM + UBMEM/URMA(见 [deepep-ascend.md](deepep-ascend.md))。

---

## 1. 机制:EP 通信为什么难、DeepEP 怎么解

### 1.1 MoE 的 all-to-all 账单

MoE 里每个 token 由 router 选 top-k expert。专家并行(EP)下 expert 分布在不同 GPU/rank,于是每个 MoE 层都要做两次不规则 all-to-all:

- **dispatch**:token 从「所在 rank」发往「负责其 expert 的 rank」;
- **combine**:expert 算完,结果从「expert rank」归回「token 原 rank」。

难点在于**它是 token 级、细粒度、GPU 发起**的通信:消息又小又多、路由不规则,传统 NCCL 的 bulk 集体通信语义不匹配。DeepEP 是第一个开源的高性能 EP 通信库,核心用 **GPU 直接发起 RDMA**(IBGDA / GPUDirect Async)绕过 CPU,把每个 token 写进目标 rank 的缓冲。

### 1.2 V2 架构:NCCL Gin 后端 + 统一 EPBuffer

V2 是一次完整重构,关键变化:

| 维度 | V1(旧) | V2(现) |
|---|---|---|
| 后端 | NVSHMEM | **NCCL Gin**(轻量设备侧通信 API,可复用现有 NCCL communicator) |
| 接口 | 高吞吐 / 低延迟两套 API | 统一 `EPBuffer`(EPv2),expand 布局适配 grouped expert GEMM |
| SM/QP 配置 | 需 autotune | **解析式计算**,无需调参 |
| scale-up/scale-out | 小 | 更大(旧版至 EP2048) |
| SM 占用(V3 式训练) | 24 | **4–6**,峰值反升 ~1.3× |
| zero-SM RDMA EP | 支持 | **不支持**(EP dispatch/combine 需 GPU SM) |

`EPBuffer` 的 dispatch/combine 支持 **deferred epilogue + `async_with_compute_stream=True`**:调用立即返回 `EventOverlap`,在专家 GEMM 前 `.wait()` 才真正跑 epilogue 并吐结果——把通信的尾段藏进计算间隙,实现通信-计算重叠。解码侧 `do_cpu_sync=False` 走 GPU 侧 receive count(配合 CUDA graph),复用 routing handle 避免反复 CPU 同步。

### 1.3 冗余专家负载均衡原语

DeepEP 暴露 `EPBuffer.lb_prefetch_weights` / `lb_reduce_grads`,支撑动态冗余专家(把重载 expert 复制到更多 GPU 上跑)——这是 MoonEP、UltraEP 探索的 expert-replication 路线所需的通信原语:前向把原 expert 权重推到冗余槽(NVLink),反向把冗余专家的 FP32 梯度累加回原 expert。复制计划由调用方(如 LPLB,见 [lplb-epb-load-balancing.md](lplb-epb-load-balancing.md))给出。

### 1.4 其他原语(试验性)

- **Bucket collectives**(`BucketBuffer`):CP/DP 的 batched all-gather / reduce-scatter / all-reduce;
- **Engram**(`EngramBuffer`):多 GPU/CPU 层的远端内存查找,供 Engram 稀疏记忆(见 [engram.md](engram.md))在推理时做 RDMA fetch;
- **PP**(`PPBuffer`):RDMA-only 流水线组的相邻 rank send/recv。

---

## 2. 同 API 不同后端:NVIDIA vs Ascend

昇腾版新建独立仓 **DeepEP-Ascend**,公开 buffer API 与 NVIDIA 版对齐。对照:

| 维度 | NVIDIA 版 | Ascend 版 |
|---|---|---|
| 通信内核 | CUDA 内核,JIT(NVCC) | Ascend C 内核,JIT(bisheng/ld.lld) |
| 集合通信 | NCCL Gin(复用 NCCL communicator) | HCCL(对标 NCCL)+ HCOMM(控制面 API) |
| 片上/远程 | shared memory / NVLink | UBMEM(片上统一缓冲)+ URMA(Unified Remote Memory Access) |
| RDMA | IBGDA(GPUDirect Async),GPU 直写 NIC | —(超节点内部 UBL128 互联) |
| 依赖 | NCCL ≥ 2.32.3 + NVLink + RDMA | CANN 9.2.0 + UBMEM 连通 + UBC_CTP/URMA 通道 |
| JIT | DeepJIT(NVCC→CUBIN) | DeepJIT(bisheng→Ascend) |

**两条 P0 结论**:
1. **「同 API」让上层框架(训练/推理)无感切换**:dispatch/combine 的 `EPBuffer` 语义两边一致,框架只改环境不改调用。
2. **通信后端是各自硬件的「亲儿子」**:NVIDIA 侧深度绑定 NCCL Gin + IBGDA;昇腾侧深度绑定 HCCL/HCOMM + URMA。**不要**把 NCCL/NVSHMEM 概念直接套到昇腾版上(昇腾版已无 NVSHMEM 依赖)。

---

## 3. 数据(⚠ 口径不同)

> ⚠ **NVIDIA 侧数字时效标注**:下表 1.3×/4×、90/643/675 GB/s 均出自**旧 V2 公告 README 性能表**,当前 main README 已移除该性能表(V2.5 news 仅描述「Analytical SM & QP calculation」)。引用时按「旧 V2 公告口径」标注,勿当现行 main 一手数字。

| 场景 | 数值 | 口径 |
|---|---|---|
| dispatch/combine 峰值(V2 vs V1) | 峰值 up to **1.3×**、SM 省 up to **4×** | NVIDIA,旧 V2 公告口径(现行 README 已移除性能表) |
| dispatch(FP8,CX7,EP 8×2) | 90 GB/s(RDMA)/ 643 GB/s(NVLink,SM100 EP8) | NVIDIA,旧 V2 公告口径(现行 README 已移除性能表) |
| combine(NVLink,SM100 EP8) | 675 GB/s | NVIDIA,旧 V2 公告口径(现行 README 已移除性能表) |
| dispatch(昇腾,EP≤32) | 物理载荷带宽 **90–95%** | Ascend 950DT + CANN 9.2.0 + PoC HDK,netlayer 1(见 [deepep-ascend.md](deepep-ascend.md)) |

> ⚠ NVIDIA 侧 90/643 GB/s 是特定 CX7/NVLink 拓扑的 bottleneck 带宽;昇腾侧 90–95% 是「达物理载荷带宽的百分比」,两者口径不同,不横向 PK。

---

## 4. 实践要点 / 坑

- **环境(NVIDIA)**:Hopper+ GPU、CUDA 13.1+、PyTorch 2.10+、NCCL 2.32.3+(先 `pip install "nvidia-nccl-cu13>=2.32.3" --no-deps`)、NVLink(节点内)+ RDMA(节点间)。安装 `git submodule update --init --recursive && bash install.sh`。
- **EP 通信要吃 SM**:zero-SM RDMA EP 已不支持;`EPBuffer` 会用解析式估计的 SM 数,也可手动 `num_sms` 覆盖。别再按 V1 的「0 SM」预期。
- **engram/PP/Bucket 是试验性**:Engram 需带 `ncclGinOptFlagsWarpGet` 的 NCCL build;PP 需 RDMA-only 组;别当生产稳定 API 用。
- **术语雷区(易写错)**:NCCL Gin ≠ NVSHMEM(V2 已弃 NVSHMEM);dispatch/combine 的反向互为对偶(dispatch 反向是 combine,combine 反向是 dispatch);FP8 dispatch 的 scaling factor 有 TMA-aligned col-major 要求。
- **NVIDIA 版 ≠ 昇腾版**:昇腾版无 NVSHMEM、无 IBGDA,后端是 HCCL/HCOMM/URMA(见 [deepep-ascend.md](deepep-ascend.md) 的 PoC HDK vs 商用 HDK 坑)。
- **网络调优**:流量隔离靠 InfiniBand VL(`sl_idx` / `EP_OVERRIDE_RDMA_SL` / `EP_DEFAULT_RDMA_SL`),带宽型负载建议关 congestion control,PCIe 环境 `mlxconfig ... PCI_ATOMIC_MODE=4` 提 RDMA atomic 性能。
- **submodule 必须走 HTTPS(2026-10-21 起)**:`f6aa38fe`(PR #788)把 `.gitmodules` 里 `third-party/deep_jit` 的 URL 从 `git@github.com:` 改成 `https://github.com/`。含义有两层:① DeepJIT 是**必需** submodule(JIT 编译通信内核),漏 init 会在 import 期才炸;② 以前靠外层仓库 `-c url."https://github.com/".insteadOf="git@github.com:"` 兜底的构建机(只开 HTTPS 出口、无 SSH key)现在不依赖这层 rewrite 也能过——DeepInfer 的 PR #157 就是同步删掉自家 rewrite 的(见 [deepinfer.md](deepinfer.md) §15)。反过来,若你的 fork 里还有别的 `git@` 地址,这层兜底没了就会报 `Permission denied (publickey)`,用 `git config --global url."https://github.com/".insteadOf "git@github.com:"` 自行补。
- **同一天还删了 lint 工具链**:`b5a33eef`(PR #781)移除 `.clang-format` / `.github/workflows/format.yml` / `format.sh` / `pyproject.toml` / `requirements-lint.txt` 五个文件。**纯工程卫生,无内核变化**——贡献者本地 `format.sh` 会 404,自己接 clang-format/yapf 即可;使用者无感。

---

## 溯源

- **DeepEP 原版 README**(主源:V2/NCCL Gin/EPBuffer/负载均衡原语/Bucket/Engram/PP/环境变量/性能表;2026-10-21 核对于 `f6aa38fe`):https://github.com/deepseek-ai/DeepEP
- **DeepEP commit `f6aa38fe`**(submodule URL 改 HTTPS,PR #788):https://github.com/deepseek-ai/DeepEP/commit/f6aa38fe06cd4a0489e7730365d4e0114970a2f3
- **DeepEP PR #781**(移除 clang-format/yapf lint 工具链):https://github.com/deepseek-ai/DeepEP/pull/781
- **DeepEP-Ascend README**(昇腾版对照,HCCL/HCOMM/UBMEM/URMA/PoC HDK):https://github.com/deepseek-ai/DeepEP-Ascend
- **DeepJIT**(运行时 JIT):https://github.com/deepseek-ai/DeepJIT
- **⚠ DeepEP 无独立 arXiv 论文**:公开引用是 GitHub 仓库(bibtex `deepep2025`);候选号 arXiv 2609.01649 经核为**无关的直播异常检测论文**(cs.SI),不采用。
- **开源周 Day 2 公告**:https://github.com/deepseek-ai/open-infra-index
- **库内交叉**:[deepep-ascend.md](deepep-ascend.md)(昇腾版机制全文)、[deepgemm-nvidia-vs-ascend.md](deepgemm-nvidia-vs-ascend.md)(Mega MoE 计算侧)、[lplb-epb-load-balancing.md](lplb-epb-load-balancing.md)(冗余专家复制的规划侧)、[engram.md](engram.md)(Engram 稀疏记忆)、[../../model-architecture/concepts/open-model-landscape.md](../../model-architecture/concepts/open-model-landscape.md)(MoE 架构背景)。
