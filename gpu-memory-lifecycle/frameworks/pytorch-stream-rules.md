---
title: PyTorch 算子开发:流与显存生命周期规则
type: howto
tags: [pytorch, record_stream, stream, cuda]
source: "ppt: 显存生命周期培训 v2 (primary, 来源已提炼入库)"
date: 2026-09
---

# PyTorch 算子开发规则

## TL;DR

不要绕过 ATen allocator;不要随手创建无人知晓的 stream。跨 stream 使用必须让框架/allocator 知道——`wait_stream` / `event` / `record_stream` 的本质是让「真实访问顺序」可见。

## 分配

用 `at::empty` / `torch::empty_like` / Tensor API,避免普通路径直接 `cudaMalloc/cudaFree`。

## stream

kernel launch 到 **PyTorch current CUDA stream**;不要默认假设 legacy default stream。

## side stream(核心)

- **使用前建立依赖**:side stream 要 `wait_stream` 创建/填充 stream,避免读到尚未 ready 的数据
- **使用后保护释放**:释放前让 creation stream wait side stream,或 `record_stream`,让 allocator 延迟复用
- **回到框架语义**:优先用框架 current stream 和框架分配 API;只有需要 overlap 时才引入 side stream

```python
# PyTorch side stream 模式
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    kernel(x)
    x.record_stream(s)

# Manual event 模式
record event on side_stream
creation_stream.wait_event(event)
# then allow tensor deallocation
```

`record_stream` 定义:当 tensor 被释放时,确保该 tensor 内存在 stream 上**当时已排队的工作完成之前**不会被另一 tensor 复用。

## debug

`CUDA_LAUNCH_BLOCKING=1` 只能帮助定位异步错误,**不是修复 lifetime bug 的设计手段**。

## 官方原话要点

> Caching allocator 只知道 tensor 的创建 stream;跨 stream 使用可能导致内存被意外复用。
> CPU 侧可能已经重新分配 tensor,而旧 tensor 上的 CUDA kernels 仍在进行;如果它们都在同一 stream,这没问题(官方称 counterintuitive 但 fine)。—— PyTorch CUDA semantics / record_stream 文档

## 反模式(单测能过,实战必炸)

- 每个 op 尾部 `cudaDeviceSynchronize()`
- 自建 stream 但不 record / wait
- `cudaMalloc/cudaFree` 绕过框架 allocator
- 这些在训练图、动态图、CUDA Graph、通信重叠、多 stream pipeline 里会变成隐性性能或正确性问题
