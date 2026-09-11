---
title: TFRA prepare async thread 踩内存 — 事故复盘
type: case
tags: [tfra, mlu, cnco, memory, debugging]
source: library/training/framework_cuda_memory_lifecycle_training_with_tfra_case_v2.pptx
date: 2026-09
---

# 案例:TFRA prepare async thread 踩了 allocator 复用的内存

## TL;DR

不是 BFC 算法不安全,而是 **allocator 的逻辑 lifetime 没和真实 device queue lifetime 对齐**:
prepare thread 用**自己的 queue**,不在 TF runtime / TF queue 的顺序保护里,导致地址 P 被复用但两个 queue 没有 happens-before。

## 场景

```
remote PS table ──> device table + host table
                      (prepare async thread 提前拉取未来会用到的 KV)
```

- **prepare async thread**:提前把未来要用的 KV 从 PS 拉到 device + host table
- **扩容路径 `cncoInsertAndEvict`**:触发 device table 扩容时,内部通过 **TF BFC allocator** 申请 device memory
- **风险点**:prepare thread 使用自己的 queue,不在 TF runtime / TF queue 的顺序保护里

## 踩踏时序

```
Host / runtime          TF queue            prepare async queue
P 分配给 TF tensor
                        enqueue K_tf(P)
Host 侧逻辑 free
BFC allocate 返回 P
写 P                                        cncoInsertAndEvict(P)
                        K_tf(P) pending/running
─ race window:TF queue 与 prepare queue 无顺序 ─
```

**根因**:不是 "BFC 复用 P" 本身错误;错误是 **P 的两次实际访问没有建立跨 queue 的 happens-before**。

## 为什么 CUDA Merlin 没踩

现场规避是**全局同步**:扩容前后 `cudaDeviceSynchronize()`。

- 全局同步会等 **所有 stream / 所有 host thread** 的 preceding commands 完成,旧 TF queue 上不再有会访问 P 的 pending work → BFC 再返回 P 才安全
- 正确,但**代价是打断异步 pipeline**(粗粒度止血,非长期设计)
- 正确方向:精确建立 dependency / safe frontier,而不是所有场景都全局同步兜底

## 为什么 MLU/CNCO 踩了

现场用 `cnrtQueueSync`:

- 只同步 **CNCO queue 自己的**任务完成
- **不能证明 TF queue 已经停止访问 P** → `cnrtQueueSync(cnco_queue) ≠ cudaDeviceSynchronize()`
- 错误路径:CNCO queue sync 完成后直接 `write P`,但 TF queue 上 K_tf(P) 仍 pending

## 修复原则(把真实访问顺序暴露给 allocator/runtime)

1. **扩容前**:prepare queue wait TF frontier / TF stream event
2. **使用后**:让 allocator 知道 CNCO queue 访问过 P
3. **过渡止血**:device-level sync 或所有相关 queue 的 event 依赖
4. **长期**:prepare async thread 纳入 TF runtime / stream executor 管理

## Code review 问题

> 谁声明了这块内存在**所有可能访问它的 queue 上**都完成了?

## 暴论 4 条(MLU 视角,供参考)

1. 框架上下文里出现同步内存申请/释放(`cnrtMalloc`)→ 99% 写错;应改用异步原语(`cnrtMallocAsync`)
2. 忽略 queue 时序用同步内存操作语义(`cnrtMemcpy` / `cnrtMemset`)操作框架 allocator/tensor 内存 → 100% 写错
3. 异步 H2D/D2H 拷贝要谨慎管理 src/dst 生命周期:拷贝实际发生时内存必须有效;拷贝结束时资源能释放
4. 有 cuda 基线先照抄一份,跑通了再叠加自己的理解;照抄也跑不对,是别的问题(参考老版本 cnrtMemcpyAsync vs cudaMemcpyAsync)
