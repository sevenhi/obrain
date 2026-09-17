---
title: TensorFlow 算子开发 + BFC allocator
type: concept
tags: [tensorflow, bfc, allocator, OpKernelContext]
source:
  - "ppt: 显存生命周期培训 v2 (primary, 来源已提炼入库)"
  - "code: TF BFCAllocator(best-fit with coalescing / safe frontier / freed_at_count) | github.com/tensorflow/tensorflow bfc_allocator.h"
  - "code: TF allocator.h(AllocationAttributes / freed_by_func) | github.com/tensorflow/tensorflow allocator.h"
  - "code: TF op_kernel.h(OpKernelContext / allocate_output / allocate_temp) | github.com/tensorflow/tensorflow op_kernel.h"
date: 2026-09
---

# TensorFlow 算子开发规则 与 BFC

## TL;DR

BFC 是**二次管理**(best-fit with coalescing);自定义 op 应走 `OpKernelContext` 和框架执行上下文,`allocate_output` / `allocate_temp` 区分两类执行期分配。BFC 的 free 是 allocator 视角的 free,**不是所有 GPU 访问已结束的证明**。

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

## BFC 只需要理解到这一层

1. Tensor 临时对象结束 → BFC chunk 标成 free
2. 可能进入 free list / timestamped list
3. 后续申请可能复用

**重点**:BFC 的 free 是 allocator 视角的 free,不是"所有 GPU 访问已经结束"的证明。
`safe frontier` / `freed_at_count` 这类机制说明:allocator 会**把 timing 信息纳入复用判断**,但前提是 runtime/算子把真实依赖传进来了。

> BFCAllocator = best-fit with coalescing;freed_at_count 晚于 safe frontier 的 chunks 会放在特殊列表,free 后不立即 merge。不需要记 bin/split/merge 细节。

## 边界

BFC / runtime 只能保护**它知道的分配、stream 和 timing frontier**。绕过框架,统计、复用、安全边界都会变差。

## 官方原话要点

- TensorFlow BFCAllocator 源码:best-fit with coalescing,假设大多数 allocation 都经过这个接口
- `allocator.h` 的 `AllocationAttributes / freed_by_func`:按 timing count 限制哪些 free chunk 可被返回
- `op_kernel.h`:执行期分配区分 allocate_output(结果) / allocate_temp(scratch)
