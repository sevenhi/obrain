---
title: TensorFlow 算子开发 + BFC allocator
type: concept
tags: [tensorflow, bfc, allocator, OpKernelContext]
source:
  - "ppt: 显存生命周期培训 v2 (primary) | framework_cuda_memory_lifecycle_training_with_tfra_case_v2.pptx | ~/zhanbin/library/training/"
  - "code: TF BFCAllocator(best-fit with coalescing / safe frontier / freed_at_count) | github.com/tensorflow/tensorflow tensorflow/core/common_runtime/bfc_allocator.h"
  - "code: TF allocator.h(AllocationAttributes / freed_by_func) | github.com/tensorflow/tensorflow tensorflow/core/common_runtime/allocator.h"
  - "code: TF op_kernel.h(OpKernelContext / allocate_output / allocate_temp) | github.com/tensorflow/tensorflow tensorflow/core/framework/op_kernel.h"
date: 2026-09
---

# TensorFlow 算子开发规则 与 BFC

## TL;DR

BFC 是**二次管理**(best-fit with coalescing);自定义 op 应走 `OpKernelContext` 和框架执行上下文,`allocate_output` / `allocate_temp` 区分两类执行期分配。BFC 的 free 是 allocator 视角的 free,**不是所有 GPU 访问已结束的证明**。

> 本篇与概念篇 [concepts/host-vs-device-lifetime.md](../concepts/host-vs-device-lifetime.md) 的分工:概念篇 §5.2 已讲 BFC 在"caching allocator 落点"中的位置与 `safe frontier` 的存在性;本篇只补 **BFC 特有的机制增量与写法**,通用心智模型(两条时间线 / happens-before / 5 问判定)见概念篇,不在此重复。

## 正确写法

```cpp
void Compute(OpKernelContext* ctx) override {
  Tensor* out = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &out));

  Tensor scratch;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, tmp_shape, &scratch));

  // launch on framework/device stream
}
```

- `allocate_output`:op 结束后要**保留**的结果
- `allocate_temp`:kernel 执行期间需要的 **scratch storage**(不保留)
- 不要把 raw pointer 生命周期**独立于 Tensor 对象**保存

## BFC 特有增量(概念篇未展开的部分)

概念篇 §5.2 只点到"BFC free ≠ GPU 访问结束"。BFC 内部的复用判定增量在这三处:

| 机制 | 作用 | 对照(二级 suballocation vs 池化复用) |
|---|---|---|
| **safe frontier** | 一个单调递增的 allocation count 分界线;free 时 chunk 记下当时的 count | 与 PyTorch caching allocator 的"按 stream 事件等待"不同:BFC 用**计数时间戳**而非 stream 事件做保护 |
| **freed_at_count / timestamped list** | `freed_at_count` 晚于 safe frontier 的 chunk 进特殊列表,**free 后不立即 merge**,防止过早复用 | 这是"延迟复用"的 TF 实现;PyTorch 侧对应 `record_stream` 把 stream 信息补进 bookkeeping |
| **二级 suballocation** | BFC 从 CUDA 拿大 chunk,再 suballocate 切给 Tensor;free 只标 free、还回 free list,**底层大 chunk 不还给 CUDA** | 与"每次 cudaMalloc/cudaFree 裸管"对比:后者绕过二次管理,安全边界与复用率都变差 |

> 不需要记 bin/split/merge 的细节;要记住的是"**保护靠 timing 计数,复用靠 free list,而两者都只在 runtime 把真实依赖传进 BFC 时才有效**"。

## 边界

BFC / runtime 只能保护**它知道的分配、stream 和 timing frontier**。绕过框架,统计、复用、安全边界都会变差。

## 官方原话要点

- TensorFlow BFCAllocator 源码(best-fit with coalescing,假设大多数 allocation 都经过这个接口): `tensorflow/core/common_runtime/bfc_allocator.h`
- `allocator.h` 的 `AllocationAttributes / freed_by_func`:按 timing count 限制哪些 free chunk 可被返回
- `op_kernel.h`:执行期分配区分 allocate_output(结果) / allocate_temp(scratch)
