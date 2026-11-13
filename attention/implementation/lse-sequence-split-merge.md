---
title: Attention 的 LSE 序列分块合并:split-K、online softmax 与可验证实现
type: concept
tags: [attention, softmax, lse, logsumexp, split-k, sequence-parallel, flashattention, online-softmax, distributed-attention, numerical-stability]
source:
  - "pdf: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao et al. arXiv:2205.14135"
  - "pdf: Online normalizer calculation for softmax | Milakov & Gimelshein arXiv:1805.02867"
  - "code: github.com/Dao-AILab/flash-attention | FlashAttention 官方实现"
date: 2026-11
---

# Attention 的 LSE 序列分块合并

## 速览

> **先记结论**：如果拆的是 **KV 的 sequence 维度**，每个 chunk 先得到局部输出 `O_c` 和局部 `LSE_c`，再用 `global_lse = logsumexp(LSE_c)` 恢复全局 softmax，最后按 `exp(LSE_c - global_lse)` 加权合并 `O_c`。如果拆的是 **Q 的 sequence 维度**，不同 query 行彼此独立，结果直接沿 sequence 维度拼接，不需要跨块 LSE 加权。

这正是 split-K、FlashAttention 的 online softmax、以及把 KV 序列分散到多张卡时的共同数学基础。LSE 不是额外的近似统计量，而是 softmax 分母的对数表示；保留它就保留了跨块重建全局归一化所需的信息。

## 1. 先分清两种“拆 seq”

Attention 的单个 query 行写作：

\[
s_{ij}=q_i k_j^T / \sqrt d,
\qquad
o_i=\frac{\sum_j\exp(s_{ij})v_j}{\sum_j\exp(s_{ij})}.
\]

因此有两种完全不同的切分：

| 切分对象 | 每个分片看到什么 | 合并方式 | 是否需要 LSE 跨片归约 |
|---|---|---|---|
| **KV sequence（split-K）** | 同一个 `q_i`，不同的 `K_c,V_c` | 先合并 softmax 分母，再加权 `O_c` | **需要** |
| **Q sequence** | 不同的 query 行，完整 `K,V` | 沿 query 维 `cat` 输出和 LSE | **不需要** |

本页的主线是第一行：**KV sequence 分块**。完整流程如下：

<div class="archify-figure">
<iframe src="../../../diagrams/at-lse-sequence-merge.workflow.html"
        style="width:100%;height:760px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="Attention KV 序列分块与 LSE 合并流程图"></iframe>
</div>

## 2. LSE 到底保存了什么

对某个 query 行 `q_i`，定义全局 LSE：

\[
L_i=\operatorname{LSE}(s_i)=\log\sum_j\exp(s_{ij}).
\]

于是全局 softmax 可以写成：

\[
p_{ij}=\exp(s_{ij}-L_i),
\qquad
o_i=\sum_j p_{ij}v_j.
\]

现在把 KV 序列分成不相交的 chunk：

\[
\{0,\ldots,N_k-1\}=C_0\cup C_1\cup\cdots\cup C_{C-1}.
\]

第 `c` 个 chunk 的局部量是：

\[
L_{i,c}=\log\sum_{j\in C_c}\exp(s_{ij}),
\]

\[
O_{i,c}=\frac{\sum_{j\in C_c}\exp(s_{ij})v_j}
                 {\sum_{j\in C_c}\exp(s_{ij})}.
\]

这里 `O_{i,c}` 已经在 chunk 内部归一化，但它还不是全局输出。关键观察是：

\[
\exp(L_{i,c})=\sum_{j\in C_c}\exp(s_{ij}),
\]

也就是说，`exp(LSE_c)` 正好是该 chunk 对全局 softmax 分母的贡献。

## 3. 全局 LSE 与加权输出

因为各个 chunk 的 token 集合不重叠，完整分母可以相加：

\[
\sum_j\exp(s_{ij})
=\sum_c\sum_{j\in C_c}\exp(s_{ij})
=\sum_c\exp(L_{i,c}).
\]

因此全局 LSE 是：

\[
\boxed{L_i=\operatorname{logsumexp}(L_{i,0},L_{i,1},\ldots,L_{i,C-1})}.
\]

第 `c` 个局部输出在全局结果中的权重为：

\[
\alpha_{i,c}=\exp(L_{i,c}-L_i).
\]

把局部分子改写为 `O_{i,c}` 乘上局部分母，就得到：

\[
\begin{aligned}
o_i
&=\frac{\sum_c\exp(L_{i,c})O_{i,c}}
        {\sum_c\exp(L_{i,c})}\\
&=\boxed{\sum_c\exp(L_{i,c}-L_i)O_{i,c}}.
\end{aligned}
\]

这解释了为什么不能直接平均局部输出：平均默认每个 chunk 的分母贡献相同，而真实的 `α_c` 由 logits 决定。

### 两个 chunk 的直观形式

若只有两个 chunk：

```text
global_lse = log(exp(lse0) + exp(lse1))
weight0    = exp(lse0 - global_lse)
weight1    = exp(lse1 - global_lse)
output     = weight0 * output0 + weight1 * output1
```

如果 `lse0` 比 `lse1` 大很多，chunk 0 的分母贡献更大，最终输出自然更偏向 `output0`。

## 4. 数值稳定的 `(m, l, acc)` 形式

直接计算 `exp(score)` 会溢出，所以 kernel 通常不直接保存 `LSE`，而是保存一个可合并的稳定状态：

\[
m_c=\max_{j\in C_c}s_{ij},
\]

\[
l_c=\sum_{j\in C_c}\exp(s_{ij}-m_c),
\]

\[
a_c=\sum_{j\in C_c}\exp(s_{ij}-m_c)v_j.
\]

局部输出和局部 LSE 为：

\[
O_c=a_c/l_c,
\qquad
LSE_c=m_c+\log l_c.
\]

两个状态合并时先统一参考最大值：

\[
m=\max(m_a,m_b),
\]

\[
\tilde l_a=\exp(m_a-m)l_a,\qquad
\tilde l_b=\exp(m_b-m)l_b,
\]

\[
l=\tilde l_a+\tilde l_b,
\qquad
a=\exp(m_a-m)a_a+\exp(m_b-m)a_b.
\]

最后：

\[
O=a/l,\qquad LSE=m+\log l.
\]

多个 chunk 可以用同样的二元合并做树形归约或 ring 归约。这个状态具有结合式的合并结构，正是它适合 GPU 分块和跨卡通信的原因。

```python
# kernel-like pseudocode: one query row, one KV chunk
m_chunk = scores.max(dim=-1).values
p = torch.exp(scores - m_chunk[..., None])
l_chunk = p.sum(dim=-1)
acc_chunk = p @ v_chunk

# merge state_a and state_b
m = torch.maximum(m_a, m_b)
scale_a = torch.exp(m_a - m)
scale_b = torch.exp(m_b - m)
l = scale_a * l_a + scale_b * l_b
acc = scale_a[..., None] * acc_a + scale_b[..., None] * acc_b
output = acc / l[..., None]
lse = m + torch.log(l)
```

实际 kernel 通常让 `m/l/acc` 在 FP32 中累积，即使输入和输出是 FP16 或 BF16；这样能减少长序列归约中的误差。

## 5. 可运行 PyTorch 演示

下面的脚本同时验证两件事：

1. KV 分块后，用局部 `output + lse` 合并，结果接近完整 attention；
2. 用 `(m,l,acc)` 稳定状态合并，结果同样接近完整 attention；
3. Q 分块时只需 `torch.cat`，不需要 LSE 加权。

```python
import math

import torch


torch.manual_seed(0)
torch.set_printoptions(precision=6, linewidth=120)

dtype = torch.float64  # 用高精度突出数学等价性；GPU kernel 常用 FP32 累积
q = torch.randn(3, 4, dtype=dtype)   # [query_len, head_dim]
k = torch.randn(11, 4, dtype=dtype)  # [kv_len, head_dim]
v = torch.randn(11, 5, dtype=dtype)  # [kv_len, value_dim]
scale = 1.0 / math.sqrt(q.shape[-1])


def full_attention(q, k, v):
    scores = (q @ k.T) * scale
    lse = torch.logsumexp(scores, dim=-1)
    output = torch.softmax(scores, dim=-1) @ v
    return output, lse


def local_attention(q, k_chunk, v_chunk):
    scores = (q @ k_chunk.T) * scale
    lse = torch.logsumexp(scores, dim=-1)
    output = torch.softmax(scores, dim=-1) @ v_chunk
    return output, lse


def merge_local_outputs(local_results):
    outputs = torch.stack([item[0] for item in local_results], dim=0)
    local_lse = torch.stack([item[1] for item in local_results], dim=0)

    # local_lse: [num_chunks, query_len]
    global_lse = torch.logsumexp(local_lse, dim=0)
    weights = torch.exp(local_lse - global_lse[None, :])

    # weights[..., None]: [num_chunks, query_len, 1]
    output = (weights[..., None] * outputs).sum(dim=0)
    return output, global_lse


def local_state(q, k_chunk, v_chunk):
    scores = (q @ k_chunk.T) * scale
    m = scores.max(dim=-1).values
    p = torch.exp(scores - m[..., None])
    l = p.sum(dim=-1)
    acc = p @ v_chunk
    return m, l, acc


def merge_states(states):
    # states[i] = (m_i, l_i, acc_i), where acc_i is unnormalized
    maxima = torch.stack([state[0] for state in states], dim=0)
    m = maxima.max(dim=0).values

    l = torch.zeros_like(m)
    acc = torch.zeros_like(states[0][2])
    for local_m, local_l, local_acc in states:
        rescale = torch.exp(local_m - m)
        l = l + rescale * local_l
        acc = acc + rescale[..., None] * local_acc

    return acc / l[..., None], m + torch.log(l)


full_output, full_lse = full_attention(q, k, v)

# KV sequence split: [0:4], [4:8], [8:11]
boundaries = [0, 4, 8, k.shape[0]]
chunks = [(k[start:end], v[start:end]) for start, end in zip(boundaries, boundaries[1:])]

local_results = [local_attention(q, k_chunk, v_chunk) for k_chunk, v_chunk in chunks]
merged_output, merged_lse = merge_local_outputs(local_results)

stable_states = [local_state(q, k_chunk, v_chunk) for k_chunk, v_chunk in chunks]
stable_output, stable_lse = merge_states(stable_states)

print("KV split max |output error|:", (merged_output - full_output).abs().max().item())
print("KV split max |LSE error|:", (merged_lse - full_lse).abs().max().item())
print("state merge max |output error|:", (stable_output - full_output).abs().max().item())
print("state merge max |LSE error|:", (stable_lse - full_lse).abs().max().item())

torch.testing.assert_close(merged_output, full_output, rtol=1e-12, atol=1e-12)
torch.testing.assert_close(merged_lse, full_lse, rtol=1e-12, atol=1e-12)
torch.testing.assert_close(stable_output, full_output, rtol=1e-12, atol=1e-12)
torch.testing.assert_close(stable_lse, full_lse, rtol=1e-12, atol=1e-12)

# Q sequence split: each query block sees the complete K/V, so concatenate.
q_blocks = torch.chunk(q, 2, dim=0)
q_split_results = [full_attention(q_block, k, v) for q_block in q_blocks]
q_split_output = torch.cat([item[0] for item in q_split_results], dim=0)
q_split_lse = torch.cat([item[1] for item in q_split_results], dim=0)
torch.testing.assert_close(q_split_output, full_output, rtol=1e-12, atol=1e-12)
torch.testing.assert_close(q_split_lse, full_lse, rtol=1e-12, atol=1e-12)
print("Q split: concatenate passed")
```

输出误差只来自浮点运算顺序；在 GPU 上使用 BF16/FP16 输入时，应该把误差验收放宽，并让 `m/l/acc` 使用 FP32 累积。

## 6. Mask、causal 与分布式边界

### 6.1 mask 必须在局部 LSE 之前应用

对无效位置先写成 `-inf`：

```python
scores = scores.masked_fill(~valid_mask, float("-inf"))
local_lse = torch.logsumexp(scores, dim=-1)
```

每个 chunk 的 `valid_mask` 必须与完整 attention 的 mask 语义一致。不能先对未 mask 的局部块算 LSE，再在合并阶段试图补回 causal 或 padding 约束。

### 6.2 空 chunk

如果一个 query 在某个 chunk 中没有任何有效 key，则：

```text
local_lse = -∞
local weight = exp(-∞ - global_lse) = 0
```

实现时要避免产生 `0 × NaN`；空块的局部输出应保持为零或由 kernel 的有效行谓词保护。

### 6.3 分布式 KV sequence parallel

当不同 GPU 持有不同 KV 分片时，每张卡可以先计算：

```text
local_output_rank
local_lse_rank
```

然后做：

```text
global_lse = logsumexp(local_lse_rank, dim=rank)
weight_rank = exp(local_lse_rank - global_lse)
global_output = Σ weight_rank * local_output_rank
```

通信实现可以是：

- 先 `all-reduce(max)`，再 `all-reduce(sum)` 的稳定两阶段归约；
- ring attention 中逐块维护 `(m,l,acc)`；
- 树形归约，把二元状态合并算子应用到多个 rank。

### 6.4 dropout 与训练

上面的等式针对没有 dropout 的 attention，最直接适用于推理。训练时如果 dropout 在 softmax 后应用，不能把已经使用不同随机 mask 的局部输出当作普通 `O_c` 直接合并；需要保持与完整 attention 相同的 dropout 语义，或合并 dropout 前的未归一化统计量并使用一致的随机数映射。

## 7. 工程检查清单

- **先问切的是谁**：切 `K/V` 要 LSE 合并，切 `Q` 只要拼接。
- **确认缩放一致**：所有 chunk 使用相同的 `1/√d`、softcap、ALiBi 或其他 score bias。
- **确认 mask 一致**：causal、padding、document mask 必须在局部分数上生效。
- **优先保存稳定状态**：kernel 内部用 `(m,l,acc)`，不要直接累加未平移的 `exp(score)`。
- **累积精度提高**：FP16/BF16 输入通常用 FP32 保存 `m/l/acc`。
- **处理空块**：`LSE=-∞`，并避免无效输出参与 `0×NaN` 运算。
- **验收两项结果**：同时比较 `output` 和 `LSE`，只比较最终 output 可能漏掉归一化错误。
- **分布式先对齐协议**：各 rank 的序列边界、mask 和位置偏置必须与单卡参考实现一致。

## 8. 与 FlashAttention 的关系

FlashAttention 的 online softmax 在遍历 KV tile 时维护 running `m/l/O`，本页的跨 chunk LSE 合并是同一思想的“分片归约”形式：

- tile 内部：遇到新的最大值时，对旧的 `l` 和 `acc` 做 rescale；
- tile 之间：用 `(m,l,acc)` 或 `(LSE,O)` 组合多个局部结果；
- 最后只做一次 `acc/l`，避免物化完整的 `N_q×N_k` softmax 矩阵。

更完整的 IO-aware tiling、FA1–FA4 演进和 online softmax 伪代码，见 [FlashAttention 实现全解](flash-attention-implementation.md)。

## 溯源

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Online normalizer calculation for softmax — [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
- FlashAttention 官方实现 — [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- 本域实现主线 — [FlashAttention 实现全解](flash-attention-implementation.md)
