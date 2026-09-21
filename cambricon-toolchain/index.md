---
title: 寒武纪工具链
type: domain
tags: [cambricon, mlu, bang-c, cngdb, cncc, cnmon, mim, toolchain]
source:
  - "ppt: CNGDB培训 v0.3 | ~/zhanbin/library/training/CNGDB培训-v0.3.pptx"
  - "ppt: MLU系列MIM介绍材料 v0.1 (2022-02) | ~/zhanbin/library/training/MLU系列MIM介绍材料_v0.1.pptx"
date: 2026-09
---

# 寒武纪工具链 (cambricon-toolchain)

> 领域索引。主题:寒武纪 MLU 软件栈里"开发者直接摸的工具"——CNGDB 调试器、MIM 算力切分(后续可扩:CNCC 编译器、CNProfiler、cnmon 等)。
> 定位:**howto 为主**(内部 PPT 正文薄,重上手命令与排错,不做无据扩写)。

## 一句话核心

> CNGDB = GDB 命令语义 + MLU focus 机制(命令只作用于 focus 的 core,断点对所有 core 生效);MIM = MLU 版 MIG(PI/CI 两级物理切分,MLU370-X8 最大 4 拆,IPU/VPU/JPU+内存硬隔离)。

## 内容索引(2 篇)

| 专题 | 文件 | 定位 | 状态 |
|---|---|---|---|
| CNGDB 调试器 | [frameworks/cngdb-debugging.md](frameworks/cngdb-debugging.md) | 用:依赖链、cncc -g/-gline-directives-only 编译、focus 机制、断点四类、print/x、CNCORE coredump 解析、排错顺序 | ✅ |
| MIM 算力切分 | [frameworks/mlu-mim-partitioning.md](frameworks/mlu-mim-partitioning.md) | 用:MIG 参照系(Streams/MPS/MIG 对比表)、PI/CI 术语、最大 4 拆分、容器化/SRIOV 场景、上手路径与坑 | ✅ |
| 参考资料索引 | [references.md](references.md) | 溯源:公开手册/上游仓库链接 | ✅ |

## 速记

1. 想看**变量值**:必须 `cncc -g -O0`;`-O3` release 用 `-gline-directives-only` 只能定位行
2. 多核调试:`info cngdb all` → `cngdb focus <core>`;单核模式单步只动 focus core,多核模式单步动所有 core
3. 断点对**所有 core** 生效;条件断点可引用内建变量 `coreId`/`taskIdX`
4. coredump = `.cndump` CNCORE 文件;要详细信息/反汇编用**内部版**(红区)
5. CNGDB 依赖 `/tmp` 读写权限 + CNDrv/CNBin/CNRT/libcngdb_backend 版本对齐
6. MIM:PI≈MIG 的 GI(按内存容量/带宽切),CI≈CI(在 PI 内再分 IPU/VPU/JPU);**busy 时不可重配置**,实例间 P2P/跨 PI IPC 受限

## 相关条目(跨域)

- [gpu-memory-lifecycle](../gpu-memory-lifecycle/index.md):MIM 切分不改变实例内 host-vs-device 显存生命周期规则(两域互链);CNGDB 调试的 cnrtQueueSync 语义同源
- [cpp-language](../cpp-language/index.md):CNCC 编译 BANG C 的模板/符号链接基础
- [vllm](../vllm/index.md) / [cutlass](../cutlass/index.md):NVIDIA 侧对照(MIG 之于 MIM、cuda-gdb 之于 CNGDB)
