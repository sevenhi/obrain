---
title: DeepSelect — DSA/Sampler 的 TopK 算子
type: concept
tags: [deepselect, topk, dsa, sparse-attention, sampler, lightning-indexer, bfloat16, ascend, npu]
source:
  - "code: deepseek-ai/DeepSelect | https://github.com/deepseek-ai/DeepSelect ↑primary"
  - "code: deepseek-ai/DeepSelect/docs/DeepSelect-deep-dive.md(算法分析)| https://github.com/deepseek-ai/DeepSelect"
date: 2026-09
---

# DeepSelect — DSA/Sampler 的 TopK 算子

## TL;DR

**DeepSelect** 是为 **DSA(DeepSeek Sparse Attention)** 和 **Sampler(采样器)** 定制的高性能 **TopK 算子**:支持 NVIDIA CUDA 与华为昇腾双平台,相比原生 `torch.topk` 加速 **2–20×**。**口径是有效内存带宽(不是 FLOPs)**——TopK 不做浮点运算,FLOP 率无意义。

**两场景**:
- **Lightning Indexer**:bfloat16,batch/vocab 大小任意,**topk≤4096(主优化 topk=512,即 DeepSeek V4 的 topk)**;
- **Sampling**:float32,vocab ~128K,topk≤4096。

**昇腾限制(易踩)**:Sampling 场景的 float32 **只有 CUDA 支持**;昇腾实现**仅支持 bfloat16**。在 DSA 链路里,DeepSelect 是**前置 TopK**:先选出每个 token 该 attend 哪些 KV 的 indices,再交给 FlashMLA sparse 按 indices 算稀疏注意力。

---

## 1. 机制:阈值 + 随机块扫描 + 分批压缩

DeepSelect 的算法核心是「**维护 top-k 阈值 T + 随机块顺序扫描 + 分批 compact**」三步循环:

```text
DeepSelectTopk(x[0:N), k, B, B2):
    topk_candidate = []                     # shared memory
    topk_threshold = -inf                   # 当前 top-k 阈值
    p = random_permutation(ceil_div(N, B))  # 与 x 独立的随机块顺序

    for block in p:
        lo, hi = block * B, min((block + 1) * B, N)
        for j in range(lo, hi):             # Filter:只收仍可能进 top-k 的
            if x[j] > topk_threshold:
                topk_candidate.append((x[j], j))
        if len(topk_candidate) >= k + B2:   # Compact:分批压缩
            topk_candidate = RadixSelectTopK(topk_candidate, k)
            topk_threshold = min(v for v, _ in topk_candidate)

    return RadixSelectTopK(topk_candidate, k)
```

要点:
1. **Scan**:按**随机块顺序**(random permutation,与输入无关)逐块读入;
2. **Filter**:用当前 top-k 阈值 T 过滤,只把 `> T` 的元素进候选缓冲;
3. **Compact**:候选缓冲变大时,在 shared memory 里跑 **radix-select 版 TopK** 压缩回 k 个,并把 T 更新为这 k 个里的最小值。

**为什么随机块顺序**:阈值 T 单调不减,后续元素通过筛选的概率越来越低,候选缓冲增长越来越慢;随机顺序避免极端输入下的性能退化。**空间复杂度 O(k+B+B₂)**(可放 shared memory);`k=512` 时合理配置 `B=B₂=1024`。

---

## 2. 使用(usage)

```python
import torch
import deep_select

# input: [batch, vocab] 的 logits/score
# topk=512(DeepSeek V4 的 topk)
# 建议:sorted_index 关掉(除非输出必须按 index 排序,开它耗性能)
#       return_value=False(不需要值时跳过 value 输出,更快)
```

**参数建议**:禁用 `sorted_index`(除非必须按 index 有序);设 `return_value=False`(不要值时更快)。安装:`git clone` + `git submodule update --init --recursive` + `pip install -v .`。

---

## 3. 数据对比(带口径)

- **加速比**:vs `torch.topk` 为 **2–20×**;
- **口径:有效内存带宽(effective memory bandwidth)**——TopK 无浮点运算,FLOP 率不适用(README 明确)。
- **场景限制**:Lightning Indexer 与 Sampling 的 `topk` 都必须 **≤4096**(更大不支持),主优化 `topk=512`。

---

## 4. 实践要点 / 坑

- **"2–20×" 别当 FLOPs 加速**:口径是有效内存带宽(TopK 无浮点运算)。
- **昇腾 dtype 限制(P1)**:CUDA 支持 float32;**昇腾仅 bfloat16**(Sampling 场景的 float32 只有 CUDA 支持)。
- **topk≤4096 限制**:不是无限制,更大 topk 不支持。
- **与 FlashMLA 的 DSA 配合**:DeepSelect 出 indices → FlashMLA sparse 消费 indices(见 [flashmla.md](flashmla.md))。DeepSelect 是 DSA 链路的前置 TopK。

---

## 溯源

- **DeepSelect 仓库**(主源:双平台、两场景参数、2–20×、有效内存带宽口径、昇腾 bfloat16 限制、usage):https://github.com/deepseek-ai/DeepSelect
- **DeepSelect deep-dive**(算法:阈值+随机块扫描+radix-select 压缩、期望上界分析):https://github.com/deepseek-ai/DeepSelect/blob/main/docs/DeepSelect-deep-dive.md
- **库内交叉**:[flashmla.md](flashmla.md)(同域 DSA 链路)、[../../vllm/concepts/sampling-penalties.md](../../vllm/concepts/sampling-penalties.md)(采样场景背景)。
