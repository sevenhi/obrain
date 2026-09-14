---
title: host-side lifetime vs device-side execution lifetime(核心心智模型)
type: concept
tags: [cuda, memory-lifecycle, happens-before, async, stream]
source:
  - "ppt: 显存生命周期培训 v2 (primary, 来源已提炼入库)"
  - "web: PyTorch CUDA semantics | https://pytorch.org/docs/stable/notes/cuda.html"
  - "web: torch.Tensor.record_stream | https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html"
  - "code: TF BFCAllocator | github.com/tensorflow/tensorflow bfc_allocator.h"
date: 2026-09
---

# 显存生命周期:逻辑所有权 vs 实际执行

## TL;DR

框架 allocator 管的是**逻辑所有权**(host 侧谁拥有这块地址);CUDA stream/queue 管的是**实际访问顺序**(device 侧谁会访问它)。
不要用「对象作用域」(C++ 析构 / Tensor 引用归零 / temp 出作用域)判断显存安全——要用「访问这块地址的所有 queue/stream 是否有顺序关系」判断。

## 两个时间线

**Host / Runtime** 侧:alloc / free 只是逻辑状态变化,CPU 会**提前于 GPU** 跑(CPU ahead-of-GPU)。
**GPU / Stream** 侧:kernel 真正访问发生在 stream 上,按 enqueue 顺序异步执行。

> 同一个事件在 CPU 侧「发生了」,不代表 GPU 侧已经执行到那里。即使 Host 已经把 P 给了下一个 Tensor,stream 仍保证 K1 先于 K2。

## 三层拆解(判断安全的依据)

1. **逻辑所有权**:Tensor / resource handle / allocator bookkeeping → 决定 P 能否被再次分配
2. **执行顺序**:stream / queue / event → 决定旧访问和新访问之间是否有 happens-before
3. **真实安全性**:只有所有访问 P 的队列都完成,或新访问被正确排在旧访问之后,P 才安全

## 关键辨析:异步 ≠ 并发

| | 定义 | 对显存安全的意义 |
|---|---|---|
| **异步** | 提交者不用等执行者完成(API 返回 ≠ GPU 完成) | 不等于并发,也不等于乱序 |
| **并发** | 两个执行窗口在 device 时间线上**重叠** | 若访问同一地址且至少一写 → data race / 踩踏 |

- 同一 stream 上的多个 kernel:异步提交但**按 FIFO 顺序执行** → 通常是安全复用的基础。
- 不同 stream/queue 且无依赖:可能重叠 → 复用同一地址就是危险区。
- 同步点之前已发生的并发,不会被「之后的 sync」修复;sync 只能当边界,不能当依赖设计。

## stream 上的 contract

保证来自 **enqueue order**,不是来自地址 P 这个对象:
1. Allocator 返回地址 P
2. Kernel launch 被放进 stream
3. Host 可以继续释放/复用 P
4. Stream 按顺序执行旧访问再执行新访问

把「内存生命周期」看成两部分:**host 侧谁拥有地址** + **device 侧哪些 stream 还会访问**。
只管第一部分不管第二部分,就是 bug 来源。

## 结论

- **framework free 是逻辑释放**,不是 GPU 完成证明(≠ cudaFree,底层内存通常仍被进程/池持有)
- **同一 stream 上提前复用地址可以是安全的**——靠队列顺序
- **side stream 必须显式建立 dependency**
- **allocator 只能保护它知道的 stream / lifetime**
- **正确修复是精确同步**,不是 `cudaDeviceSynchronize` 全局兜底

> 一句话:allocator 管逻辑所有权;stream 管实际访问顺序。相同结论的速记版见领域 `README.md` 与清单篇。
