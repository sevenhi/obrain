---
title: host-side lifetime vs device-side execution lifetime(核心心智模型)
type: concept
tags: [cuda, memory-lifecycle, happens-before, async, stream, record-stream, caching-allocator]
source:
  - "ppt: 显存生命周期培训 v2 (primary) | framework_cuda_memory_lifecycle_training_with_tfra_case_v2.pptx | ~/zhanbin/library/training/"
  - "web: PyTorch CUDA semantics | https://pytorch.org/docs/stable/notes/cuda.html"
  - "web: torch.Tensor.record_stream | https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html"
  - "code: TF BFCAllocator | github.com/tensorflow/tensorflow bfc_allocator.h"
  - "web: CUDA Programming Guide — Stream-Ordered Allocator | https://docs.nvidia.com/cuda/cuda-programming-guide/index.html"
date: 2026-09
---

# 显存生命周期:逻辑所有权 vs 实际执行

## TL;DR

框架 allocator 管的是**逻辑所有权**(host 侧谁拥有这块地址);CUDA stream/queue 管的是**实际访问顺序**(device 侧谁会访问它)。
**不要用「对象作用域」(C++ 析构 / Tensor 引用归零 / temp 出作用域)判断显存安全——要用「访问这块地址的所有 queue/stream 是否有顺序关系」判断。** 判断准则是:**最后一次旧访问 和 第一次新访问 之间,是否在所有相关 stream 上有 happens-before 关系**。

> **一个反转直觉,先记住**:框架里 `free` 之后,底层显存**不是被释放,而是被换了个"主人"**。安全与否不取决于"谁不再拥有",而取决于"谁还会访问、访问是否排在新主人之后"。CUDA 的 caching allocator(BFC / PyTorch caching allocator)正是靠这个语义做"提前复用"——它赌的是 stream 顺序,不是对象作用域。

---

## 1. 两个时间线:CPU ahead-of-GPU

**Host / Runtime 侧**:alloc / free 只是逻辑状态变化,CPU 会**提前于 GPU** 跑(CPU ahead-of-GPU)。`cudaFree(p)` 返回时 GPU 可能还在用 p 里的数据。
**GPU / Stream 侧**:kernel 真正访问发生在 stream 上,按 enqueue 顺序异步执行。

```
CPU 时间线:   alloc(p) ── launch K1 ── free(p) ── alloc 复用 p ── launch K2
                │             │            │            │             │
GPU 时间线:   (还没跑)      K1(p)      (CPU 已 free)   (CPU 已复用)  K2(p)  ← 顺序执行
                                                        ▲
                                                  GPU 视角: K1 先于 K2 完成
```

> 同一个事件在 CPU 侧「发生了」,不代表 GPU 侧已经执行到那里。即使 Host 已经把 P 给了下一个 Tensor,stream 仍保证 K1 先于 K2。

## 2. 真正的判断依据:三层拆解

1. **逻辑所有权**:Tensor / resource handle / allocator bookkeeping → 决定 P 能否被**再次分配**
2. **执行顺序**:stream / queue / event → 决定旧访问和新访问之间是否有 **happens-before**
3. **真实安全性**:只有所有访问 P 的队列都完成,或新访问被正确排在旧访问之后,P 才安全

**核心是第 2 层**。内存安全 = "所有 old 访问 P 的 (queue, command) 都先于所有 new 访问 P 的 (queue, command)"。这个"先于"就是编程语言/并发里的 happens-before,它是**可传递的**:`A 在 S1 上先于 B`,`B 通过 event/wait 在 S2 上先于 C` ⇒ `A 可证先于 C`。判断安全就是**构造这条传递链**。

### 2.1 谁在什么时序访问 P(view 模型)

```
old 访问集 done(running)     新访问集 new
        │                         │
        └────── 需要 happens-before 闭合 ──────┘
  每个 old (queue,cmd) 都须在某个 new (queue,cmd) 之前完成
  任意一个 old 与新之间无顺序 → race window
```

安全 = 所有 old 与所有 new 之间构成传递链;不安全 = 存在一对 old/new 无顺序。

## 3. 关键辨析:异步 ≠ 并发

| | 定义 | 对显存安全的意义 |
|---|---|---|
| **异步** | 提交者不用等执行者完成(API 返回 ≠ GPU 完成) | 不等于并发,也不等于乱序 |
| **并发** | 两个执行窗口在 device 时间线上**重叠** | 若访问同一地址且至少一写 → data race / 踩踏 |

- 同一 stream 上的多个 kernel:异步提交但**按 FIFO 顺序执行** → 通常是安全复用的基础。
- 不同 stream/queue 且无依赖:可能重叠 → 复用同一地址就是危险区。
- 同步点之前已发生的并发,不会被「之后的 sync」修复;sync 只能当**边界**,不能当**依赖设计**。

### 3.1 为什么 `cudaDeviceSynchronize` 只能当边界、不能当修复

`cudaDeviceSynchronize()` 等待"当前所有已提交到 device 的 work 完成"。它能**清空**已存在的 race window(把 old 访问推到完成),但不能**创建**未来访问间的顺序。

- **作为边界**:`sync; 复用 p` 时,sync 保证 old 全完成 → 这次复用安全。✅
- **作为依赖**:`sync; launch K_new(p)`,若 K_new 之后还有别的 stream 也要用 p,sync 没给它们建立顺序 → **隐患仍在**。❌
- **性能代价**:sync 强制 CPU 等 GPU,把异步运行时退化成串行,打断 pipeline 重叠。

一句话:**sync 是"已经发生的事"的灭火器,不是"将要发生的事"的红绿灯**。

## 4. stream 上的 contract(为什么同 stream 复用安全)

保证来自 **enqueue order**,不是来自地址 P 这个对象:

1. Allocator 返回地址 P
2. Kernel launch 被放进 stream
3. Host 可以继续释放/复用 P
4. Stream 按顺序执行旧访问再执行新访问

```
// 同一 stream S 上的两次访问,靠 FIFO 顺序保护,P 可提前复用
kernel< K1 ><<<grid, block, 0, S>>>(p);   // old 访问
// ... host 侧 p 被"释放"并重新分配 ...
kernel< K2 ><<<grid, block, 0, S>>>(p);   // new 访问
// ✅ 安全:K2 一定在 K1 之后执行,因为同一 stream FIFO
```

**跨 stream 就不行了**——两个 stream 之间没有自动顺序,必须显式 `wait_stream` / `event` / `record_stream` 建立。

### 4.1 record_stream:把真实访问顺序暴露给 allocator

```python
# PyTorch 侧:side stream 用完后,让 caching allocator 知道它在 s 上被用过
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())   # 等主 stream 的数据就绪
with torch.cuda.stream(s):
    y = kernel(x)                              # 在 s 上算
    x.record_stream(s)                         # 告诉 allocator:x 被 s 访问过
# 之后 x 被释放时,allocator 会等待 s 上 x 的 work 完成,才允许复用 x 的内存
```

`record_stream` 的语义(官方原话):当 tensor 被释放时,确保该 tensor 内存在 stream 上**当时已排队的工作完成之前**不会被另一 tensor 复用。**本质:把"这个 tensor 在 side stream 上也用过"这个信息补进 allocator 的 bookkeeping**——因为 allocator 默认只知道创建 stream,不知道 side stream 的访问。

## 5. PyTorch / TF 两个 caching allocator 的落点

### 5.1 PyTorch Caching Allocator

- 用 `at::empty` / `torch::empty_like` 走 ATen allocator;`tensor.data_ptr()` 拿地址。
- **不要**绕过 allocator 直接 `cudaMalloc/cudaFree`——会脱离 allocator 的复用与 stream 语义,变成"裸指针 + 人工生命周期管理",正是踩踏的温床。
- 侧 stream 用法见 §4.1;核心是让 allocator 知道**所有**访问过 tensor 的 stream。

### 5.2 TensorFlow BFC Allocator

- BFC = best-fit with coalescing,**二次管理**:Tensor 临时对象结束 → chunk 标 free → 进 free list / timestamped list → 后续申请可能复用。
- **BFC 的 free 是 allocator 视角的 free,不是"所有 GPU 访问已结束"的证明**。`safe frontier` / `freed_at_count` 机制说明 allocator 会把 **timing 信息**纳入复用判断——**但前提是 runtime/算子把真实依赖传进来了**。
- 自定义 op 走 `OpKernelContext`: `allocate_output`(op 结束保留的结果)/ `allocate_temp`(kernel 执行期 scratch)。**两类的生命周期语义不同,别混**。
- **边界**:BFC / runtime 只能保护**它知道的分配、stream 和 timing frontier**。绕过框架,统计、复用、安全边界都会变差。

## 6. 判定准则(把心法变成可执行判断)

**每次要复用 / 释放一块 buffer,问 5 问**:

| # | 问题 | 正确方向 | 错在哪 |
|---|---|---|---|
| 1 | buffer 从哪申请? | 框架 allocator / Tensor API | 裸 `cudaMalloc` → 脱离复用语义 |
| 2 | 哪个 stream 创建/填充? | 输入 ready 关系已建? | 没 wait 就消费未就绪数据 |
| 3 | 哪些 stream 会访问? | 所有非创建 stream 都要被记录或同步 | 只记创建 stream,侧 stream 漏了 |
| 4 | 什么时候允许回池? | **最后一次 GPU 访问之后**,不是 C++ 作用域 | 按作用域 free → 提前复用 |
| 5 | 错误如何定位? | Nsight / event / blocking 诊断 | 用全局 sync 当修复(掩盖) |

**判断一个 Case 安全否的裁剪准则**:对每个 buffer,列出所有 old 访问 P 的 (queue, cmd) 和所有 new 访问 P 的 (queue, cmd),检查每一对之间是否存在传递性的 happens-before:

- **Case A**:`K1(P) on S0 → free → K2(P) on S0` → ✅ 安全(同 stream FIFO)
- **Case B**:`K1(P) on S1 → free → K2(P) on S0`(无 event/wait)→ ❌ 不安全(两 stream 无依赖)
- **Case C**:`K1(P) on S1 → record_stream(P, S1) → free → allocator 延迟复用` → ✅ 安全但有性能/可预测性代价

## 7. 实践要点(checklist)

1. **精神模型切换**:忘掉"对象作用域",记住"谁还会访问这台地址"。判断显存安全 = 问访问顺序。
2. **跨 stream 显式建依赖**:side stream 必须 `wait_stream`(用前)+ `record_stream`(用完);**只 sync 自己的 queue 证明不了别的 queue 不用这块内存**。
3. **sync 是边界不是修复**:`cudaDeviceSynchronize` / `cudaStreamSynchronize` 用于清空已发生的窗口,不能给未来跨 stream 访问建顺序;滥用会打断异步 pipeline。
4. **绕过 allocator = 自杀**:裸 `cudaMalloc/cudaFree`、长期保存 raw pointer,会让 allocator 对这块内存"失明",安全边界归零。
5. **debug 工具定位,不是修复**:`CUDA_LAUNCH_BLOCKING=1` / Nsight 用于定位异步错误,不是修 lifetime bug 的设计手段。
6. **正确修复是"精确同步"**:把真实访问顺序暴露给 allocator(record_stream / event / wait_stream),而不是全局兜底。**让 allocator 知道它该保护谁,而不是把整个设备停摆**。

---

## 8. 溯源

- 原始来源:显存生命周期培训 deck v2(30 页 + 讲者备注)
- PyTorch CUDA semantics(异步、stream、caching allocator 复用 / record_stream): https://pytorch.org/docs/stable/notes/cuda.html
- torch.Tensor.record_stream: https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html
- TensorFlow OpKernelContext / BFCAllocator / allocator.h: github.com/tensorflow/tensorflow (op_kernel.h / bfc_allocator.h / allocator.h)
- CUDA Stream-Ordered Allocator(cudaMallocAsync/cudaFreeAsync 的 stream-order 语义): https://docs.nvidia.com/cuda/cuda-programming-guide/index.html

> 事故案例(TFRA prepare async thread)+ code review 清单见 [领域 index](../index.md)。本文是"懂"的心智模型,那两篇是"查"和"用"的实体。
