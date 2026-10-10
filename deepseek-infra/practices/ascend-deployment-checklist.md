---
title: DeepSeek 昇腾部署检查清单:环境两线与算子跑通顺序
type: checklist
tags: [deepseek, ascend, deployment, cann, tilelang, deepgemm, deepep, flashmla, deepselect, torch-npu, 950, checklist]
source:
  - "code: deepseek-ai/DeepEP-Ascend README(validated stack) | https://github.com/deepseek-ai/DeepEP-Ascend ↑primary"
  - "code: deepseek-ai/FlashMLA README(昇腾构建) | https://github.com/deepseek-ai/FlashMLA"
  - "code: deepseek-ai/DeepSelect README(昇腾限制) | https://github.com/deepseek-ai/DeepSelect"
  - "code: tile-ai/tilelang README + tilelang/ascend/README.md(950 native backend) | https://github.com/tile-ai/tilelang"
  - "code: tile-ai/tilelang-ascend docs/TileLang-Ascend Programming Guide.md(A2/A3 环境) | https://github.com/tile-ai/tilelang-ascend"
date: 2026-10
---

# DeepSeek 昇腾部署检查清单:环境两线与算子跑通顺序

## TL;DR

DeepSeek 昇腾基础设施栈的部署要**先把环境分成两条线**——950 超节点线与 A2/A3 旧线——别把两套版本要求混在一起(本域此前最易踩的坑)。**950 线(DeepEP-Ascend validated stack)**:CANN 9.2.0 + Python 3.12 + PyTorch 2.13.0+cpu + torch_npu 2.13.0rc1;**A2/A3 线(只适用 tilelang-ascend 独立仓)**:CANN 8.3.RC1+ / torch-npu 2.6.0.RC1+。算子跑通顺序:**TileLang(编程指南)→ DeepGEMM-Ascend → DeepEP-Ascend → FlashMLA → DeepSelect**。机制细节见各 concepts 篇,本篇只列可执行的检查清单。

---

## 1. 环境两线分开(关键修正)

### 1.1 950 线(Ascend 950 超节点,目标平台)

| 组件 | 版本 | 出处 |
|---|---|---|
| CANN | **9.2.0** | DeepEP-Ascend README / DeepGEMM README「9.20」 |
| Python | **3.12** | DeepEP-Ascend validated stack |
| PyTorch | **2.13.0+cpu** | DeepEP-Ascend validated stack |
| torch_npu | **2.13.0rc1** | DeepEP-Ascend validated stack |
| 硬件 | Ascend 950DT(DeepEP 基准)、UBMEM/URMA 通道 | 各仓 README |

> 这是 **DeepEP-Ascend 的 validated stack**(README「The validated stack is Ascend 950DT, CANN 9.2.0, Python 3.12, PyTorch 2.13.0+cpu and torch_npu 2.13.0rc1」),是 950 线的权威版本组合。

### 1.2 A2/A3 线(只适用 tilelang-ascend 独立仓)

| 组件 | 版本 | 出处 |
|---|---|---|
| CANN | **8.3.RC1+** | tilelang-ascend Programming Guide(「至少 8.3.RC1 版本」) |
| torch-npu | **2.6.0.RC1+** | 同上(「至少 2.6.0.RC1 版本」) |

> ⚠ **这条只适用 `tile-ai/tilelang-ascend` 独立仓的 A2/A3 后端**;**不要安到 950**。注意:此条只适用独立仓 A2/A3 后端,不是通用要求。

### 1.3 主仓 tilelang 的 Ascend 950 native backend

- 主仓 `tile-ai/tilelang` 现支持 **Ascend 950 native backend**(2026-09-30 起),但 **README 未标具体 CANN/torch_npu 版本**;
- 构建:`USE_ASCEND=ON USE_CUDA=OFF python -m pip install -v .`;默认编译目标 **`dav-3510`**(backend defaults to dav-3510 for Ascend 950);
- 与 A2/A3 的 tilelang-ascend 独立仓是**两个东西**,别混(分层见 [tilelang.md](../concepts/tilelang.md))。

---

## 2. 算子跑通顺序 + 实操坑

### 顺序:语言层 → 计算库 → 通信库 → 注意力算子

```
① TileLang         编程指南里的 elementwise/GEMM 先跑通(语言层地基)
② DeepGEMM-Ascend  GEMM(依赖 tilelang 的 HC prenorm kernel)
③ DeepEP-Ascend    MoE EP all-to-all(依赖 DeepJIT 运行时 JIT)
④ FlashMLA         dense + sparse 注意力内核
⑤ DeepSelect       DSA 的 TopK(出 indices,喂给 FlashMLA sparse)
```

### 2.1 TileLang

- **环境分线**:950 走主仓 native backend(dav-3510);A2/A3 走 tilelang-ascend(CANN 8.3.RC1+ / torch-npu 2.6.0.RC1+);
- 安装:主仓 `USE_ASCEND=ON USE_CUDA=OFF pip install -v .`;tilelang-ascend 走 `install_ascend.sh` + `source set_env.sh`(先 `source {cann}/ascend-toolkit/set_env.sh`);
- 无 NPU 硬件可用 **camodel** 软件仿真验正确性(CANN 9.0.0+ 含 `Ascend950PR_9599` 镜像),**只验逻辑、别当性能数字**。

### 2.2 DeepGEMM-Ascend

- 环境:950 系列 NPU + CANN 9.2.0(README 写作「9.20」)+ `bin/bisheng`、`bin/ld.lld` + torch_npu + Python 3.10+ + C++20;
- 依赖 tilelang(HC prenorm kernel)+ tree-sitter(生成 type stub);
- 安装:`git clone --recursive` + `./develop.sh` + `pip install . --no-build-isolation`;
- **scaling factor 格式与 NVIDIA 不同**:每对 UE8M0 scaling factor 沿 K 维打包进 `int16`、MN-major 存储(FP8 场景别沿用 NVIDIA 格式)。

### 2.3 DeepEP-Ascend

- 环境(validated):950DT + CANN 9.2.0 + Python 3.12 + PyTorch 2.13.0+cpu + torch_npu 2.13.0rc1 + PoC HDK;
- 设 `ASCEND_HOME_PATH`(或 `ASCEND_TOOLKIT_HOME`);设备内核首次使用时编译,运行时需 CANN + bisheng;
- ⚠ **PoC HDK vs 商用 HDK**:README 基准用「手工配置的 PoC HDK,未公开分发」;Atlas 850E 商用 HDK(~2026-10 中旬)才含完整带宽配置,性能有差异。

### 2.4 FlashMLA(昇腾版)

- **环境要求**:CANN 9.2.0+(提供 bisheng 编译器和 Ascend C 头);
- 安装:`pip install -v . --no-build-isolation` —— **`--no-build-isolation` 必选**(setup.py 构建时 import torch/torch_npu,本仓未把它们声明为 PEP 518 build 依赖);
- **`FLASH_MLA_BUILD_TARGET_PLATFORM=ASCEND`**:目标平台自动检测(`/dev/davinci_manager` = Ascend),可显式覆盖为 `ASCEND`;
- **`ASCEND_HOME_PATH`** 必须指向 CANN 安装根(默认 `/usr/local/Ascend/ascend-toolkit/latest`);
- ⚠ **breaking change**:昇腾版 2026-09-30 合入 main,移除了 Hopper 支持、移除了 V3/V3.2/V4.0 旧模型、改了 FP8/FP4 KV cache 格式,**只服务 V4.1**;旧模型/旧 KV cache 需切旧 commit `ba89a346`。

### 2.5 DeepSelect

- **昇腾 dtype 限制**:昇腾实现**仅支持 bfloat16**(Sampling 场景的 float32 只有 CUDA 支持);
- **`indices_type` 仅 `torch.int32`**(README 注释「The Ascend implementation only supports torch.int32」);
- 参数建议:禁用 `sorted_index`、`return_value=False`(更快);`topk≤4096`,主优化 `topk=512`。

### 2.6 UBL128(待核标注)

- 128 卡超节点 UBL128(全互联)**沿用 index 的「待核」标注**,勿当已核事实(据昇腾官方资料待核)。

---

## 3. 一页速查表

| 组件 | 环境关键点 | 最高频坑 |
|---|---|---|
| TileLang(950) | 主仓 native backend,默认 dav-3510,未标版本 | 与 A2/A3 独立仓混 |
| TileLang(A2/A3) | CANN 8.3.RC1+ / torch-npu 2.6.0.RC1+ | 把这套安到 950 |
| DeepGEMM-Ascend | CANN 9.2.0 + tilelang 依赖 | scaling factor 格式沿用 NVIDIA |
| DeepEP-Ascend | CANN 9.2.0 + PyTorch 2.13.0 + torch_npu 2.13.0rc1 | PoC HDK 当商用 |
| FlashMLA | CANN 9.2.0+ / `--no-build-isolation` / `FLASH_MLA_BUILD_TARGET_PLATFORM=ASCEND` | 忘 breaking change,跑旧模型 |
| DeepSelect | 昇腾仅 bfloat16 / indices_type 仅 int32 | float32 当昇腾支持 |

---

## 4. 溯源

- **DeepEP-Ascend README**(validated stack:950DT/CANN 9.2.0/Python 3.12/PyTorch 2.13.0+cpu/torch_npu 2.13.0rc1/PoC HDK):https://github.com/deepseek-ai/DeepEP-Ascend
- **FlashMLA README**(CANN 9.2.0+、`--no-build-isolation` 必选、`FLASH_MLA_BUILD_TARGET_PLATFORM`、`ASCEND_HOME_PATH`、breaking change):https://github.com/deepseek-ai/FlashMLA
- **DeepSelect README**(昇腾仅 bfloat16、indices_type 仅 int32、topk≤4096):https://github.com/deepseek-ai/DeepSelect
- **tilelang 主仓**(Ascend 950 native backend、默认 dav-3510、USE_ASCEND=ON):https://github.com/tile-ai/tilelang (含 tilelang/ascend/README.md)
- **tilelang-ascend**(A2/A3 环境 CANN 8.3.RC1+ / torch-npu 2.6.0.RC1+):https://github.com/tile-ai/tilelang-ascend/blob/HEAD/docs/TileLang-Ascend%20Programming%20Guide.md
- **库内交叉**:[tilelang.md](../concepts/tilelang.md)、[deepgemm-ascend.md](../concepts/deepgemm-ascend.md)、[deepep-ascend.md](../concepts/deepep-ascend.md)、[flashmla.md](../concepts/flashmla.md)、[deepselect.md](../concepts/deepselect.md)、[ascend-port-principles.md](../concepts/ascend-port-principles.md)。
