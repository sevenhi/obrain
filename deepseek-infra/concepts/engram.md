---
title: Engram — 条件记忆与可扩展查找(一条新的稀疏轴)
type: concept
tags: [engram, conditional-memory, ngram, sparse, moe, memory-lookup, deepseek, offloading, scaling-law]
source:
  - "web: Conditional Memory via Scalable Lookup(Engram 论文,arXiv:2601.07372)| https://arxiv.org/abs/2601.07372 ↑primary"
  - "code: deepseek-ai/Engram | https://github.com/deepseek-ai/Engram"
date: 2026-10
---

# Engram — 条件记忆与可扩展查找(一条新的稀疏轴)

## TL;DR

**Engram 是 DeepSeek 提出的一条新的稀疏轴——「条件记忆(conditional memory)」**,它把经典的 N-gram embedding 现代化成 **O(1) 查找**模块,给 Transformer 补上一个原生缺失的原语:知识查找。MoE 用条件计算扩容量,Engram 用**静态记忆扩容量**,两者正交互补。

**关键结论(论文)**:在**严格 iso-parameter + iso-FLOPs** 约束下,Engram-27B 对 MoE baseline 全面领先——MMLU +3.4、CMMLU +4.0(知识)、BBH +5.0、ARC-C +3.7(推理)、HumanEval +3.0、MATH +2.4(代码/数学)。机制分析:Engram 把早期层的**静态模式重建**外包出去,变相加深了网络的有效深度;并把局部依赖交给查找,腾出注意力容量给全局上下文,长上下文检索大涨(Multi-Query NIAH 84.2→97.0)。

**系统侧**:Engram 用**确定性寻址**,使海量 embedding 表可以 offload 到 host memory 且推理开销近乎为零(靠运行时预取),这是「记忆」这个新轴能落地的工程前提。

---

## 1. 机制:把 N-gram 记忆做成 Transformer 原语

### 1.1 为什么需要「记忆」这条轴

MoE 通过条件计算扩容量(每个 token 只激活部分 expert),但 Transformer **没有原生的知识查找原语**——它被迫用计算去「模拟」检索(把知识压进权重、前向时算出来)。Engram 的思路是:知识/静态模式其实可以**查表**得到,不必每次都用计算。于是把 N-gram embedding 现代化,做成一个**记忆模块**:

- 输入 token 序列的 N-gram(如 3-gram)作为**键**;
- 每个 N-gram 映射到一个 embedding 向量(记忆条目),O(1) 查出来;
- 查出的静态记忆与动态 hidden state **融合**(fuse),喂给骨干网络。

这就是「conditional memory」:记忆是**静态**的(不随 hidden 变),但**条件化**于输入 token 序列(N-gram 由当前输入决定读哪条)。

### 1.2 稀疏分配(U-shaped scaling law)

论文提出 **Sparsity Allocation** 问题:一个稀疏模型的总容量有限,该分多少给神经计算(MoE)、多少给静态记忆(Engram)?答案是 **U 形 scaling law**——存在一个最优分配点,过多或过少记忆都次优。这条 law 指导 Engram 缩到 27B 参数。

### 1.3 确定性寻址 → 可 offload(工程关键)

与 MoE 的 soft routing(每个 token 动态算 expert 分数)不同,Engram 的寻址是**确定的**:输入 token 序列直接决定查哪几条记忆,**不需要读任何权重来算地址**。这让系统可以:

1. **预取**:知道当前 token 序列,就能预判下一步要查的记忆条目,从 host memory 预取到片上,推理时命中率极高;
2. **offload**:海量 embedding 表(记忆库)放在 host memory,不占 GPU 显存,推理开销近乎为零。

这是「记忆轴」区别于「计算轴」的**系统红利**:MoE 扩容要加算力,Engram 扩容主要加存储。

---

## 2. 可复现细节

### 2.1 数据流(简化伪代码)

```python
# 概念伪代码(仓库 engram_demo_v1.py 是演示版,mock 了 Attention/MoE/mHC)
def engram_layer(x, tokens):
    # 1) 对当前 token 序列取 N-gram 作为键,确定性寻址
    keys = make_ngram_keys(tokens, n=N)          # 每步 token 序列唯一决定键
    # 2) O(1) 查静态记忆条目
    memory_vecs = mem_table[keys]                 # [num_ngrams, mem_dim], 可预取
    # 3) 与动态 hidden state 融合
    fused = fuse(x, memory_vecs)                  # 如加权和 / 拼接 / gating
    # 4) 融合结果喂给骨干(MoE/Attention)
    return backbone(fused)
```

### 2.2 仓库结构

- 论文 PDF 在仓库根(`Engram_paper.pdf`);
- `engram_demo_v1.py`:独立演示实现,只演示 Engram 数据流,Attention/MoE/mHC 都 mock;
- `drawio/Engram.drawio`、`figures/`:架构图、scaling law、27B 实验结果、long-context 结果;
- 依赖仅 `pip install torch numpy transformers sympy`;license 为 Model License(模型权重)。

> ⚠ 仓库是**演示版 + 论文**的组合,不是生产训练栈;Engram 的生产内核在 TileKernels 的 `engram/` 子目录(融合 RMSNorm 前后向 + 权重梯度归约,见 [tilekernels.md](tilekernels.md)),通信侧在 DeepEP 的 `EngramBuffer`(见 [deepep-nvidia-vs-ascend.md](deepep-nvidia-vs-ascend.md))。

---

## 3. 数据对比(论文,iso-parameter/iso-FLOPs)

| 指标 | 增益 | 域 |
|---|---|---|
| MMLU | +3.4 | 知识 |
| CMMLU | +4.0 | 知识(中文) |
| BBH | +5.0 | 通用推理 |
| ARC-Challenge | +3.7 | 推理 |
| HumanEval | +3.0 | 代码 |
| MATH | +2.4 | 数学 |
| Multi-Query NIAH | 84.2 → 97.0 | 长上下文检索 |

**反直觉点**:最大增益不在知识检索(直觉上记忆该帮知识),而在**通用推理/代码/数学**——因为 Engram 解放了早期层,让网络对复杂推理更「深」。

---

## 4. 实践要点 / 坑

- **Engram ≠ MoE 的替代,是互补**:MoE=条件计算、Engram=静态记忆,论文的 U 形 law 讲的是两者**容量分配**,不是二选一。
- **N-gram 键由 token 序列唯一决定**:是确定性寻址,不是学出来的 router——别和 DeepSelect/MoE gating 的 soft routing 混淆。
- **「近零推理开销」的前提是确定性寻址 + 预取**:靠 host-memory offload + 运行时预取实现;如果实现丢了这个特性,记忆表塞回 GPU 显存就失去了系统红利。
- **仓库是演示**:生产内核见 TileKernels `engram/`(融合 RMSNorm + 梯度归约)与 DeepEP `EngramBuffer`(RDMA 多 GPU/CPU 层查找),别把 demo 当生产。
- **论文日期/号**:arXiv **2601.07372**(2026-01 提交),ACL 2026 long(aclanthology 2026.acl-long.226);已排除错误候选论文号,以 2601.07372 为准。

---

## 溯源

- **Engram 论文**(主源,arXiv:2601.07372):https://arxiv.org/abs/2601.07372 (ACL 2026 long;作者北京大学 + DeepSeek-AI)
- **Engram 仓库**(demo + 论文 PDF + 图):https://github.com/deepseek-ai/Engram
- **库内交叉**:[tilekernels.md](tilekernels.md)(Engram 生产内核)、[deepep-nvidia-vs-ascend.md](deepep-nvidia-vs-ascend.md)(EngramBuffer 通信)、[../../model-architecture/concepts/open-model-landscape.md](../../model-architecture/concepts/open-model-landscape.md)(MoE 稀疏轴背景)、[deepselect.md](deepselect.md)(soft routing 的对照)。
