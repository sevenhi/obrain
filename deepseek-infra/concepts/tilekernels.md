---
title: TileKernels — TileLang 算子库合集
type: concept
tags: [tilekernels, tilelang, moe, quantization, fp8, fp4, engram, mhc, ascend, npu]
source:
  - "code: deepseek-ai/TileKernels | https://github.com/deepseek-ai/TileKernels ↑primary"
  - "code: pypi.org/project/tile-kernels(v2.0.0,昇腾支持说明)| https://pypi.org/project/tile-kernels/2.0.0/"
  - "code: tile-ai/tilelang(底层 DSL)| https://github.com/tile-ai/tilelang"
date: 2026-10
---

# TileKernels — TileLang 算子库合集

## 版本更新(2026-10)

- **v2.1.0(2026-10-14,HEAD c1cf914a)**:量化(norm_forward/backward、SwiGLU、cast)、Engram gate 前后向的**内核/测试 bug 修复与再优化**(含昇腾 Asc 后端同步),无新算子族、无公开 API 变更;昇腾支持延续 v2.0.0 口径。

## TL;DR

**TileKernels** 是构建在 TileLang 之上的 **LLM 算子库**(Python),封装了一批训练/推理高频算子:MoE routing、量化(FP8/FP4/E5M6 + fused SwiGLU+quant)、Transpose、Engram(融合 RMSNorm 前后向 + 权重梯度归约)、Manifold HyperConnection(mHC,Sinkhorn 归一化 + mix 拆分)、Modeling(autograd.Function 高层封装)。多数算子接近硬件算力/带宽极限,部分已用于内部训练与推理。

**⚠ 昇腾支持(本域 P0 坑)**:main 分支已合入 v2.1.0(HEAD c1cf914a),README 现含 Ascend 950 + CANN 9.2.0 + Python 3.12/PyTorch 2.13/TileLang 0.1.15 双平台版本;PyPI `tile-kernels` v2.0.0(2026-09-30)同样明确 "Added Huawei Ascend support… same Python APIs run on both NVIDIA GPUs and Huawei NPUs"。写作按「v2.0.0 起支持昇腾」记。昇腾后端运行时自动选择,同一 Python API 跑 NVIDIA GPU + 华为 NPU。

---

## 1. 机制:基于 TileLang 的算子封装

TileKernels 用 TileLang 描述每个算子的 tile 级内核,再包成 PyTorch 可调用的算子。目录结构:

```txt
tile_kernels/
├── moe/        # MoE routing 相关内核(Gating / token-to-expert mapping)
├── quant/      # FP8/FP4/E5M6 量化(per-token/block/channel + fused SwiGLU+quant)
├── transform/  # Transform 算子(RoPE)
├── rand/       # Rand 算子(randn/seed)
├── engram/     # Engram gating 内核(融合 RMSNorm、前后向、权重梯度归约)
├── mhc/        # Manifold HyperConnection(Sinkhorn 归一化 + mix 拆分/应用)
├── modeling/   # autograd.Function 高层封装(engram gate、mHC pipeline)
├── torch/      # PyTorch 参考实现
└── testing/    # 测试与基准工具
```

**核心算子族**:
- **Gating / MoE Routing**:MoE 的 top-k expert 选择与打分、token→expert 映射、融合 expansion/reduction + 权重归一化;
- **Quantization**:per-token / per-block / per-channel 的 FP8/FP4/E5M6 cast,带 fused SwiGLU+quantization;
- **Engram**:gating 内核,融合 RMSNorm,前向/反向 + 权重梯度归约;
- **Manifold HyperConnection(mHC)**:hyper-connection 内核,含 Sinkhorn 归一化与 mix 拆分/应用——**不是普通全连接,是流形超连接**;
- **Modeling**:`torch.autograd.Function` 高层封装,把底层内核组合成可训练层。

---

## 2. 数据对比 / 性能口径

- **多数算子接近硬件算力/带宽极限**(README 原文口径);
- **单指令熔合减延迟**:如 fused SwiGLU+quant 把激活+量化融进一次 kernel 调用,省中间往返。

> 具体数字:README 未给逐算子基准表;如需精确定量,走仓库 `tests/` 的 `--run-benchmark`。

---

## 3. 实践要点(⚠ 版本要求已统一,main = v2.1.0)

main 已合入 v2.1.0(HEAD c1cf914a),版本要求不再分「main vs PyPI」两套:

| 维度 | 要求(main = PyPI v2.1.0) |
|---|---|
| Python | **3.12+** |
| PyTorch | **2.13+** |
| TileLang | **0.1.15+** |
| 硬件 | NVIDIA SM90/SM100 + **Ascend 950**(双平台) |
| CUDA/CANN | CUDA 13.1+ / **CANN 9.2.0+** |

- **坑 1**:别把「main README 只写 NVIDIA」当成「不支持昇腾」——v2.0.0 起 main 与 PyPI 同口径,双平台。
- **坑 2**:漏写双平台版本要求差异(NVIDIA 侧 CUDA 13.1+,昇腾侧 CANN 9.2.0+、Python 3.12/PyTorch 2.13/TileLang 0.1.15)。
- **坑 3**:mHC(流形超连接)不是普通全连接,是 Sinkhorn 归一化 + mix 拆分/应用的 hyper-connection。
- **安装**:本地开发 `pip install -e ".[dev]"`;发布版 `pip install tile-kernels`(v2.0.0 起含昇腾后端)。
- **运行时**:昇腾后端自动选择,同一 Python API 在 NVIDIA GPU 与华为 NPU 上运行。

---

## 溯源

- **TileKernels 仓库**(主源:算子族、目录结构、main 版 requirements、MIT license):https://github.com/deepseek-ai/TileKernels
- **PyPI tile-kernels v2.0.0**(昇腾支持声明、昇腾版 requirements):https://pypi.org/project/tile-kernels/2.0.0/
- **TileLang**(底层 DSL):https://github.com/tile-ai/tilelang
- **库内交叉**:[tilelang.md](tilelang.md)(同一域,底层 DSL)、[../../model-architecture/index.md](../../model-architecture/index.md)(Engram/mHC 架构背景)。
