---
title: DeepGEMM-Ascend — 昇腾 GEMM 库
type: concept
tags: [deepgemm, ascend, gemm, mad, fractal, fp8, fp4, bf16, moe, npu]
source:
  - "code: deepseek-ai/DeepGEMM-Ascend | https://github.com/deepseek-ai/DeepGEMM-Ascend ↑primary"
  - "code: deepseek-ai/DeepGEMM(原版,API 对齐基准)| https://github.com/deepseek-ai/DeepGEMM"
date: 2026-10
---

# DeepGEMM-Ascend — 昇腾 GEMM 库

## TL;DR

**DeepGEMM-Ascend** 是 DeepGEMM 的**昇腾移植版**:新建独立仓、**API 完全兼容**(同包名),把 DeepGEMM 的「轻量、可读、极致性能 GEMM 内核」思路搬到华为昇腾。它对昇腾的 **MAD(matrix multiply-add)矩阵乘加原语**做了一层轻量抽象,**隐藏了 fractal 分形布局、对齐约束、地址计算、冗长低层参数**这些繁重细节,让 GEMM kernel 既简洁又高效。

**昇腾特有优化**:稀疏数据加载(sparse data loading)+ 协程(coroutine)流水线——对标原版 Hopper 上的 TMA/persistent 机制。支持算子:BF16/FP8/FP4 GEMM、MQA logits、MegaMoE;其中 HC prenorm kernel 依赖 tilelang(本域 [tilelang.md](tilelang.md))。

**关键数据(出自 README Performance 表)**:密集 GEMM 利用率最高 **99.8%**(BF16×BF16,M=4096/N=7168/K=16384)、FP8×FP8 约 **99.5%**、FP8×FP4 约 **99.5%**;M-Grouped GEMM(FP8×FP4,BF16 output,DeepSeek MoE 专家)4 组最高 **856** / 8 组最高 **862 TFLOPS**;FP8 inference(M=4096/N=4096/K=7168)达 **850 TFLOPS**。环境口径:**950DT + CANN 9.20 + bench_msprof(cold L2)**。

---

## 1. 机制:MAD 原语上的轻量抽象

### 1.1 与 NVIDIA 原版的对应

DeepGEMM 原版(deepseek-ai/DeepGEMM)是 DeepSeek 的轻量 GEMM 库:单一 kernel、Hopper TMA + persistent 风格,靠「tiling + 汇编级手工调度」把 FP8 GEMM 压到近峰值。昇腾版保持了同一条设计哲学,但换了底层原语:

| 概念 | NVIDIA 版(DeepGEMM) | 昇腾版(DeepGEMM-Ascend) |
|---|---|---|
| 矩阵乘原语 | Tensor Core(WGMMA/MMA) | **MAD(matrix multiply-add)** |
| 数据搬运 | TMA / async copy | **稀疏数据加载** |
| 流水机制 | persistent kernel | **协程(coroutine)流水线** |
| 片上布局 | shared memory tile | **fractal 分形布局**(被库层隐藏) |

### 1.2 fractal 分形布局(库层藏起来的部分)

昇腾的矩阵数据按 **fractal 分形布局**存放(分形排列),不是简单的行/列主序。直接用 MAD 原语要手工处理:对齐约束(矩阵块要对齐到特定粒度)、地址计算(fractal 布局下的元素地址换算)、冗长低层参数。DeepGEMM-Ascend 在 MAD 上做轻量抽象,把这套细节**全部隐藏**,开发者按普通 tiling 思路写即可。

> ⚠ **fractal 是数据布局,不是算子**——别写成「分形算子」。

### 1.3 支持算子

- BF16 / FP8 / FP4 GEMM;
- MQA logits(多查询注意力 logits 计算);
- **稀疏 MQA logits(2026/10 Public Release 新增)**:`fp8_fp4_sparse_mqa_logits`(非分页)+ `fp8_fp4_paged_sparse_mqa_logits`(分页),只对选中的稀疏 KV 块打分,支持 MXFP4/MXFP8(见 [deepgemm-nvidia-vs-ascend.md](deepgemm-nvidia-vs-ascend.md) §1.7);Ascend 侧分页稀疏 MQA 通过 Python wrapper 提供 FIRST/REUSE 状态管理(首次调用缓存 KV 快照,后续同 metadata 复用);
- MegaMoE(MoE 的大规模专家 GEMM 场景);
- HC prenorm kernel(依赖 tilelang)。

---

## 2. 可复现代码:安装与接口

```bash
# submodule 必须克隆
git clone --recursive https://github.com/deepseek-ai/DeepGEMM-Ascend.git
cd DeepGEMM-Ascend

# 链接头文件 + 构建 C++ 扩展
./develop.sh

# 安装
pip install . --no-build-isolation
```

**kernel API**:参照 DeepGEMM 原版的 interfaces(见 [DeepGEMM#interfaces](https://github.com/deepseek-ai/DeepGEMM#interfaces)),昇腾版完全对齐。库另提供工具函数 `deep_gemm.set_num_sms` / `get_num_sms`(设置/获取 kernel 可用的 AI core 数)等。

---

## 3. 数据对比(带环境口径)

> ⚠ **出处标注(P0)**:以下数字出自 DeepGEMM-Ascend README `## Performance` 表(950DT + CANN 9.20 + bench_msprof/cold L2)。

| 场景 | 数值 | 环境口径 |
|---|---|---|
| 密集 GEMM(最高利用率) | **99.8%** | BF16×BF16,M=4096 / N=7168 / K=16384 |
| FP8×FP8 GEMM | ~**99.5%** | FP8×FP8 |
| FP8×FP4 GEMM | ~**99.5%** | FP8×FP4 |
| M-Grouped GEMM(FP8×FP4,BF16 output) | **856 / 862 TFLOPS** | 4 组 / 8 组专家(DeepSeek MoE 场景) |
| FP8 inference | **850 TFLOPS** | M=4096 / N=4096 / K=7168 |
| MQA logits | prefill FP8 **681** / FP4 **743**;decode FP8 **565** / FP4 **708 TFLOPS** | #Q=4096/256,#K=8192,32 heads,dim 128 —— MQA 是 **FIX-pipe bound**(FIX 管道 99% 饱和),非 compute bound |

> ⚠ **此表为 dense MQA(全量 KV)性能**,稀疏 MQA logits 暂无独立公开性能数字。

---

## 4. 实践要点 / 坑

- **环境要求**:昇腾 950 系列 NPU + CANN 9.20(**实为 9.2.0**,DeepGEMM README 写法不规范;需提供 `bin/bisheng`、`bin/ld.lld`)+ `torch_npu` + Python 3.10+ + C++20 编译器 + `tilelang`(HC prenorm kernel 依赖)+ `tree-sitter`/`tree-sitter-cpp`(源码构建生成 Python type stub)。
- **scaling factor 格式与 NVIDIA 不同(易踩)**:昇腾上每对 **UE8M0** scaling factor 沿 K 维打包进一个 **`int16`**,并按 **MN-major** 顺序存储(为硬件效率)。FP8 场景传 scaling factor 时别沿用 NVIDIA 的格式。
- **"CANN 9.20" 当独立版本的坑**:不是独立版本,就是 CANN 9.2.0,统一按 9.2.0 写。
- **"850 / 856 / 862" 别混**:850 是 FP8 inference M=4096/4096/7168 行;M-Grouped(FP8×FP4)是 4 组 856 / 8 组 862 TFLOPS;99.8% 是 BF16×BF16 密集 GEMM,别张冠李戴。

---

## 溯源

- **DeepGEMM-Ascend README**(主源,MAD 抽象/fractal/稀疏加载/协程/scaling factor/CANN 9.20/支持算子):https://github.com/deepseek-ai/DeepGEMM-Ascend
- **DeepGEMM 原版**(API 对齐基准、interfaces):https://github.com/deepseek-ai/DeepGEMM
- **性能数字出处**(99.8% / 99.5% / 856 / 862 / 850 TFLOPS):DeepGEMM-Ascend README `## Performance` 表(950DT + CANN 9.20 + bench_msprof cold L2)。
- **库内交叉**:[../../cutlass/index.md](../../cutlass/index.md)(NVIDIA 侧造 GEMM 内核方法论,昇腾侧对照)、[../../vllm/concepts/quantization-gptq-awq-fp8.md](../../vllm/concepts/quantization-gptq-awq-fp8.md)(FP8 量化背景)、[tilelang.md](tilelang.md)(HC prenorm 依赖)。
