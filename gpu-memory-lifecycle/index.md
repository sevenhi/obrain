# GPU / 框架 显存生命周期

> 领域索引。主题:框架运行时里,host 侧逻辑所有权与 device 侧实际执行生命周期是不同的两套时间线。
> 面向:TensorFlow / PyTorch 算子开发(含 Cambricon MLU)。来源培训 deck + 内部事故。

---
title: GPU / 框架 显存生命周期
type: domain
tags: [cuda, memory-lifecycle, gpu, tensorflow, pytorch]
source:
  - "ppt: 显存生命周期培训 v2 (primary, 来源已提炼入库)"
date: 2026-09
---

## 一句话核心

> allocator 管逻辑所有权(host 侧);stream/queue 管实际访问顺序(device 侧)。
> "free 了" ≠ GPU 已经不用了。判断显存安全看**访问顺序**,不看**对象作用域**。

## 内容索引

| 章节 | 文件 | 定位 |
|---|---|---|
| 核心心智模型(必读) | [concepts/host-vs-device-lifetime.md](concepts/host-vs-device-lifetime.md) | 懂:两条时间线、异步≠并发、happens-before 传递性、sync=边界非修复、record_stream 落点、5 问判定 + Case A/B/C |
| PyTorch 规则 | [frameworks/pytorch-stream-rules.md](frameworks/pytorch-stream-rules.md) | 用:side stream 写法 |
| TensorFlow / BFC | [frameworks/tensorflow-bfc.md](frameworks/tensorflow-bfc.md) | 懂+BFC 边界 |
| TFRA 事故案例 | [cases/tfra-prepare-async-thread.md](cases/tfra-prepare-async-thread.md) | 查:踩内存根因复盘 |
| Code review 检查清单 | [practices/memory-lifecycle-checklist.md](practices/memory-lifecycle-checklist.md) | 用:评审 + 反模式 |
| 参考资料索引 | [references.md](references.md) | 溯源:官方文档链接 |

## 三层速记

1. **framework free 是逻辑释放**,不是 GPU 完成证明(不 = cudaFree,底层内存仍被池持有)
2. **同一 stream 上提前复用地址可以是安全的**——靠 enqueue 顺序保护
3. **跨 queue/stream 必须显式建立 dependency**(wait_stream / event / record_stream);只 sync 自己的 queue 证明不了别的 queue 不用这块内存

## 事故记忆锚点(万一以后再踩)

- 症状:偶发 illegal memory access / 随机数值错 / Nsight 见两 stream overlap / 加 `cudaDeviceSynchronize` 就"好了"
- 根因:**P 的两次访问没有跨 queue 的 happens-before**,不是 allocator 复用 P 本身错
- 现场差异:`cnrtQueueSync(cnco_queue) ≠ cudaDeviceSynchronize()`——只能证明自己的 queue,不能证明 TF queue 已停用 P
