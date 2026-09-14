---
title: 显存生命周期 Code Review 检查清单
type: checklist
tags: [code-review, memory, cuda, tensorflow, pytorch]
source: "ppt: 显存生命周期培训 v2 (primary, 来源已提炼入库)"
date: 2026-09
---

# 显存生命周期 - Code Review 检查清单

## TL;DR

跨 stream/queue 使用必须显式建立 dependency;看到 `cudaStreamCreate` / `cudaMalloc` / `cudaFree` / raw pointer cache,就触发 lifetime 审查。

## 每个临时 buffer 问 5 问

| # | 问题 | 正确方向 |
|---|---|---|
| 1 | buffer 从哪里申请? | framework allocator / Tensor API,不是普通路径 raw cudaMalloc |
| 2 | 哪个 stream 创建/填充? | 输入 ready 关系是否已建立 |
| 3 | 哪些 stream 会访问? | 所有非创建 stream 都要被记录或同步 |
| 4 | 什么时候允许回池? | **不是 C++ 作用域**,而是最后一次 GPU 访问之后 |
| 5 | 错误如何定位? | Nsight / event / blocking 诊断,不用全局同步当修复 |

## 触发审查的代码特征

- `cudaStreamCreate`(自建 stream)
- `cudaMalloc` / `cudaFree`(绕过框架)
- 长期保存 raw pointer(生命周期独立于 Tensor)
- 异步 callback / async stream 使用

## 常见反模式(看起来修好,其实同步掩盖了)

- ❌ `cudaDeviceSynchronize()` 放在每个 op 尾部 → 把异步运行时退化成串行
- ❌ 自建 stream 但不 `record` / `wait`
- ❌ `cudaMalloc/cudaFree` 绕过框架 allocator

这些在单测里可能工作,但在训练图、动态图、CUDA Graph、通信重叠、多 stream pipeline 里会变成隐性性能或正确性问题。

## 课堂练习(判断准则)

判断准则:**最后一次旧访问 和 第一次新访问 是否有 stream-order dependency?**

- **Case A**:`K1(P) on S0` → free → `K2(P) on S0` → ✅ **安全**(同 stream 顺序保护)
- **Case B**:`K1(P) on S1` → free → `K2(P) on S0`(无 event/wait)→ ❌ **不安全**(两 stream 无依赖)
- **Case C**:`K1(P) on S1` → `record_stream(P, S1)` → free → allocator 延迟复用 → ✅ 安全但有性能/可预测性代价

## 5 句话带走(即评审标准)

1. **framework free 是逻辑释放,不是 GPU 完成证明**
2. **同一 stream 上提前复用地址可以是安全的**(靠队列顺序)
3. **side stream 必须显式建立 dependency**
4. **allocator 只能保护它知道的 stream / lifetime**
5. **正确修复是精确同步,不是 cudaDeviceSynchronize 全局兜底**

> 事故版收尾(案例篇):跨 queue/stream 必须显式建立 dependency;只 sync 自己的 queue 不能证明别的 queue 不用这块内存。TFRA case 的本质就是第 3、4 条。
