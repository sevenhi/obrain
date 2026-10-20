---
title: FlashMLA — MLA 注意力内核库(dense + sparse)
type: concept
tags: [flashmla, mla, dsa, sparse-attention, dense-attention, kv-cache, fp8, fp4, ascend, npu, hopper]
source:
  - "code: deepseek-ai/FlashMLA(main;昇腾版 2026-09-30 已合入 main,commit 2e5429f;2026-10-15 ABI 稳定化 commit 0f11e17)| https://github.com/deepseek-ai/FlashMLA ↑primary"
  - "code: deepseek-ai/FlashMLA/docs/20260930-ascend-prefill-deep-dive.md(昇腾稀疏 prefill 机制)| https://github.com/deepseek-ai/FlashMLA"
  - "code: deepseek-ai/FlashMLA/docs/20250929-hopper-fp8-sparse-deep-dive.md(FP8 sparse decoding)| https://github.com/deepseek-ai/FlashMLA"
  - "code: deepseek-ai/FlashMLA/docs/20250422-new-kernel-deep-dive.md(dense decoding)| https://github.com/deepseek-ai/FlashMLA"
date: 2026-09
---

# FlashMLA — MLA 注意力内核库(dense + sparse)

## TL;DR

**FlashMLA** 是 DeepSeek 的**注意力内核库**,支撑 DeepSeek-V4.1 在 NVIDIA GPU 与华为昇腾 NPU 上的推理。它含**两套内核**:

- **dense 内核**:普通 MLA 的 **prefill + decoding + backward**(开源周 Day 1 定位即 MLA 高效解码;2025-04 deep-dive 的 dense decoding 660 TFLOPS 属此内核);
- **sparse 内核**:token 级稀疏注意力的 prefill + decoding(**decoding 用 FP8/FP4 KV cache**),**服务 DSA(DeepSeek Sparse Attention)**;另有融合 Q-norm、Q-RoPE、attention、O-RoPE(共轭)、FP8 cast 的 fused kernel。

**三层关系务必分清(本域 P0)**:DSA(稀疏注意力**机制/架构层**)→ 由 **DeepSelect**(TopK)选出每个 token 该 attend 哪些 KV 的 **indices** → 再由 **FlashMLA sparse** 按 indices 算稀疏注意力。**FlashMLA 是实现 DSA 的内核库,不是 DSA 本身**。

**⚠ 昇腾版 breaking change**:昇腾版 **2026-09-30 已合入 main(commit 2e5429f)**。该版本**移除 Hopper 架构支持、移除 V3/V3.2/V4.0 旧模型、改 FP8/FP4 KV cache 格式**,只服务 DeepSeek-V4.1(含 NVIDIA + 昇腾);**旧模型/旧 KV cache 切旧 commit `ba89a346`**——三条 breaking change、旧 commit 号与昇腾 deep-dive 文档均已由 README 明写,无需待核。

**⚠ "410 TFLOPS" 双义(必须标平台)**:NVIDIA sparse decoding 是 410 TFLOPS(H800),昇腾 sparse prefill 也是 410 TFlops(950)——同名数字,平台完全不同,引用时必须标注。

---

## 1. 机制:MLA 与 DSA 链路

### 1.1 MLA 机制回顾(为什么省 KV cache)

MLA(Multi-head Latent Attention)把每 token 的 K/V 联合低秩压缩成一个 latent 向量,解码时只 cache 这个 latent,用时再上投影还原各头 K/V——相比 MHA 省 **~93% KV cache**(DeepSeek-V2 口径)。完整机制见库内 [../../model-architecture/concepts/attention-heads-mha-mqa-gqa.md](../../model-architecture/concepts/attention-heads-mha-mqa-gqa.md) §3.3。FlashMLA 就是把这个机制做成高性能内核。

### 1.2 DSA 链路(三层关系)

```
DSA(机制:每个 token 只 attend 最"重要"的 KV)
   │
   ├─ 前置 TopK:DeepSelect 选出 indices(该 token attend 哪些 KV)
   │     └─ [deepselect.md](deepselect.md)
   │
   └─ 按 indices 算稀疏注意力:FlashMLA sparse
         └─ prefill:在线 softmax + rescale
         └─ decoding:FP8/FP4 KV cache + bfloat16 矩阵乘
```

### 1.3 sparse prefill 内核:在线 softmax + rescale

昇腾 sparse prefill 内核(2026-09-30 deep-dive)计算 DSA 前向,单 Q token 视角:

```
gathered_kv = KV.index_select(indices, dim=0)   # [topk, d_qk]
P = Q @ gathered_kv^T                            # [h_q, topk]
S = softmax(P * sm_scale, dim=-1)
out = S @ gathered_kv[:, :d_v]
```

因为 softmax 有全局依赖,内核用**与 FlashAttention 相同的 online softmax** 算法,维护 `running_max` / `running_max_for_softmax` / `running_sumexp` 三个 `[h_q]` 量 + `out_accum [h_q, d_v]`(全 float32):

1. 把 topk indices 切成 `B_TOPK`(通常 64 或 96)大小的块;
2. 按块稀疏读 KV(据 indices),算 `P = Q @ gathered_kv^T`,求每行 max;
3. 更新 `running_max`;当某行 `running_max[i] - running_max_for_softmax[i] > RESCALE_THRES`(超参,内核里通常设 6)时触发 online softmax 更新:`out_accum ← out_accum * exp(running_max_for_softmax - running_max)` 后继续累加、重算 lse。

要点:**B_TOPK** 控制块大小,**RESCALE_THRES** 控制 rescale 触发频率(减少 rescale 开销、提升数值稳定性的折中)。

### 1.4 调用签名

```python
# sparse prefill(无 batch 维;多 batch 需 reshape + 调 indices)
flash_mla_sparse_fwd(q, kv, indices, sm_scale)   # 返回 out / max_logits / lse

# decoding(FP8 KV cache)
flash_mla_with_kvcache(..., is_fp8_kvcache=True, indices=...)
```

---

## 2. 数据对比(带平台口径)

> ⚠ **"410" 出现两次,必须标平台**:NVIDIA sparse decode 与昇腾 sparse prefill 都是 410,平台不同。

### NVIDIA 侧(Hopper / B200)

> **版本锚点**:下表 H800 数字(660/3000/410/640)来自 2025 开源周/2025-04 deep-dive 时代(旧 commit `ba89a346` 及更早),2026-09-30 合并后 main 已移除 Hopper 支持,无法在当前 main 复现,引用时须标注「旧 commit 时代」;B200 数字(1460/1000、1350/1024)为当前 main 口径(CUDA 13.3)。

| 场景 | 数值 | 环境口径 |
|---|---|---|
| dense MLA decoding | **660 TFLOPS**(compute-bound)/ **3000 GB/s**(memory-bound) | H800 SXM5 + CUDA 12.8 |
| sparse MLA decoding | **410 TFLOPS** | H800 SXM5 + CUDA 12.8;FP8 KV cache + bfloat16 矩阵乘 |
| MHA prefill(SM100) | **1460 TFLOPS fwd / 1000 TFLOPS bwd** | B200,NVIDIA 报告 |
| sparse MLA prefill | 640 TFLOPS | H800 SXM5 + CUDA 12.8 |
| sparse MLA prefill | 1350 TFLOPS | B200 + CUDA 13.3(当前 main) |
| sparse MLA decoding | 1024 TFLOPS | B200 + CUDA 13.3(当前 main) |

### 昇腾侧(Ascend 950,2026-09-30)

| 场景 | 数值 | 峰值占比 |
|---|---|---|
| sparse prefill | **410 TFlops** | **95%** 硬件峰值 |
| sparse decoding | **360 TFlops** | **83%** 硬件峰值 |

昇腾版另有 **fused norm-RoPE-attn-RoPE-cast kernel**,decoding 下再优化 **10–15%**。

---

## 3. breaking change(2026-09-30,必写)

- **移除 Hopper 架构支持**;
- **移除旧模型 V3 / V3.2 / V4.0**;
- **改 FP8/FP4 KV cache 格式**(与旧版不兼容)。

→ 只服务 DeepSeek-V4.1(NVIDIA + 昇腾)。**需要跑旧模型或旧 KV cache 格式时,切旧 commit `ba89a346`**。

---

## 4. 实践要点 / 坑

- **别把 FlashMLA 当 DSA 本身**:FlashMLA 是实现 DSA 的内核库;DSA 是机制(见 §1.2)。
- **"410" 混平台的坑**:NVIDIA sparse decode(410,H800)与昇腾 sparse prefill(410,950)同名,标注平台再引用。
- **dense vs sparse 别混表**:660 是 dense、410 是 sparse(decode 场景)。
- **漏写 breaking change 的坑**:昇腾版(已合入 main)移除 Hopper/旧模型、改 KV cache 格式,旧用户要切 `ba89a346`。
- **多 batch 的 sparse prefill**:内核无 batch 维,多 batch 需 reshape + 调 indices。
- **2026-10-15 ABI 稳定化(构建层,机制不变)**:commit `0f11e17` 只动了构建/绑定层——Python 扩展由 pybind 迁移到 **TORCH_LIBRARY(稳定 Torch ABI,兼容 CPython/abi3 + PyTorch 2.10 stable ABI)**,并把加载方式改成 setup.py 生成 `flash_mla/_build_info.py` 记录平台与扩展文件名、import 时精确加载(不再 glob 目录猜 `cuda*`/`npu*` so);昇腾流桥接头拆到 `csrc/api/ascend_stream.{h,cpp}`。**对使用者几乎无感知,唯一影响**:编译/打包后的扩展在 CPython 多版本间更稳、Ascend 构建缺 torch_npu 时报明确 ImportError。无内核/性能变化,现有数据全保留。另有 `sparse_decode_fwd` 可选缓冲区(tile_scheduler_metadata/num_splits)的 **in-place 变异 schema 标注修正**(Tensor(a!)?/Tensor(b!)?),不改内核数值。

---

## 溯源

- **FlashMLA 仓库(昇腾版 2026-09-30 已合入 main,commit 2e5429f)**:dense/sparse 内核、昇腾 410/360 + 10–15% fused kernel 优化 —— https://github.com/deepseek-ai/FlashMLA
- **昇腾 sparse prefill deep-dive**(在线 softmax + rescale、B_TOPK、RESCALE_THRES、410/360 峰值占比):https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20260930-ascend-prefill-deep-dive.md (中文 `-zh.md`)
- **旧 FP8 sparse decoding deep-dive**(2025-09-29,410 TFLOPS):https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md
- **dense decoding deep-dive**(2025-04-22,660 TFLOPS):https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md
- **库内交叉**:[../../model-architecture/concepts/attention-heads-mha-mqa-gqa.md](../../model-architecture/concepts/attention-heads-mha-mqa-gqa.md)(MLA 低秩压缩)、[../../attention/implementation/flash-attention-implementation.md](../../attention/implementation/flash-attention-implementation.md)(FA 内核,online softmax 前身)、[../../vllm/index.md](../../vllm/index.md)(集成)、[deepselect.md](deepselect.md)(同域 DSA 链路)。
