---
title: 显存生命周期 - 参考资料索引
type: reference
tags: [pytorch, tensorflow, cuda]
source: "ppt: 显存生命周期培训 v2 (primary) | framework_cuda_memory_lifecycle_training_with_tfra_case_v2.pptx | ~/zhanbin/library/training/"
date: 2026-09
---

# 参考资料索引

> 溯源用。只放链接与「看什么」,不复制全文。

## 官方文档

| 主题 | 链接 | 看什么 |
|---|---|---|
| PyTorch CUDA semantics | pytorch.org/docs/stable/notes/cuda.html | GPU op 异步、stream 顺序、caching allocator 复用 / record_stream |
| torch.Tensor.record_stream | pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html | allocator 只知道创建 stream;跨 stream 使用需记录/同步,避免提前复用 |
| NVIDIA CUDA Programming Guide — Stream-Ordered Allocator | docs.nvidia.com/cuda/cuda-programming-guide/index.html | cudaMallocAsync / cudaFreeAsync 与 stream-order semantics |
| TensorFlow OpKernelContext | github.com/tensorflow/tensorflow (op_kernel.h) | allocate_output / allocate_temp 执行期分配语义 |
| TensorFlow BFCAllocator source | github.com/tensorflow/tensorflow (bfc_allocator.h) | best-fit with coalescing;safe frontier / timestamped chunks |
| TensorFlow allocator.h | github.com/tensorflow/tensorflow (allocator.h) | AllocationAttributes / freed_by_func:按 timing 限制可复用 chunk |

## 内部资料

- Cambricon wiki CUDA 相关页(PPT 内引用的两个页面,内部 wiki 地址明文 pageId 不外引)
- Cambricon MLU-OPS BANG-C guide:Queue / cnrtQueueSync 等接口,用于并行流水任务通信和同步
- 内部案例:TFRA prepare async thread + cncoInsertAndEvict 扩容踩内存(见 `../cases/tfra-prepare-async-thread.md`)

## 阅读顺序建议

培训版主线(概念 → 案例):核心心智模型 → PyTorch 规则 → TF/BFC → TFRA 案例 → checklist 当评审尺

## 来源

- 原始来源:显存生命周期培训 deck v2(30 页 + 讲者备注,来源已提炼入库) | framework_cuda_memory_lifecycle_training_with_tfra_case_v2.pptx | ~/zhanbin/library/training/
