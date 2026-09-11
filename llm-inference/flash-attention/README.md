---
title: FlashAttention
type: domain
tags: [flashattention, attention, gpu, cuda]
source:
  - "wiki: flash-attention wiki | pageId 561077705 space ~zhanbin"
  - "code: github.com/Dao-AILab/flash-attention | Star 24.7k"
  - "pdf: FlashAttention FA1 | arXiv:2205.14135 NeurIPS 2022"
  - "pdf: FlashAttention-2 FA2 | arXiv:2307.08691 ICLR 2024"
  - "pdf: FlashAttention-3 | library/papers/FlashAttention-3...pdf"
  - "doc: flash-attention-tracker/flash-attention-annual-report.md (canonical)"
date: 2026-09
---

# FlashAttention

> 领域索引。完整内容见 canonical 文档 `flash-attention-tracker/flash-attention-annual-report.md`(735 行)。
> 本页只放定位、关键事实与导航,不重复正文。

## 一句话定位

FlashAttention 是 Tri Dao 主导的开源高效 Attention 实现,核心思想是 **IO-aware tiling**:把 attention 分块(tile),减少 GPU HBM 与 SRAM 间内存读写次数,在不损失精度前提下**加速 + 省显存**。从 2022 起已四代演进。

## 四代演进

| 代次 | 时间 | 论文 | 目标 GPU | 核心实现技术 | MFU(前向) |
|---|---|---|---|---|---|
| FA1 | 2022.05 | NeurIPS 2022 | A100 | CUTLASS 3.x tiling, online softmax | 25-40% |
| FA2 | 2023.07 | ICLR 2024 | A100+H100 | 更好 work partitioning, 减 non-matmul FLOPs | 50-73% |
| FA3 | 2024.07 | blog | H100 | WGMMA async, TMA, FP8 | ~75% FP16 / ~80% FP8 |
| FA4 | 2025.03 | TBD | H100+B200 | CuTe DSL, Python kernel generation | TBD |

## 为什么快(标准 Attention 的问题)

标准 attention 中 `S = Q@Kᵀ·scale` 和 `P = softmax(S)` 都是 N×N 矩阵,需写入 HBM 再读回:

- N=8192, d=128 时,`S` 占 `8192²×2B ≈ 128MB`,远超 GPU SRAM(~192KB)
- IO 开销:标准 attention 需 **O(N²·d)** 次 HBM 读写(完整 N×N 矩阵实例化两次)
- FA 通过 tiling + online softmax,把 HBM 访问降到 **O(N²)**(与 sequence length 线性相关的下界)

## 内容导航(canonical 章节)

1. 总体概况(四代演进表)→ 2. 各代算法原理与实现细节(标准问题/FA1/FA2/FA3/FA4)→ 3. 技术对比与性能... → 参考链接

## 相关文档与论文(本地副本)

- `archive/flash-attention-技术分析笔记.md` — 技术分析笔记(2026-08-28,已归档)
- `library/papers/FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.pdf` (FA1 论文)
- `library/papers/FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.pdf` (FA3)
- `flash-attention-tracker/dynamic_causal_analysis.md` / `dynamic_causal_implementation_deep_dive.md` — dynamic causal 专题
- `archive/flash_attention_wiki.html` — wiki 页面快照(已归档)

## 更新与同步

- 更新检查:`flash-attention-tracker/check_updates.sh`
- Wiki 同步:`flash-attention-tracker/sync_wiki.py --push`(pageId 561077705,surgical 增量)
- **新提交审查闭环**:先 `submit_wiki_review.sh` 审查 → `review_passed: YES/NO`;NO 时走 `discuss_wiki_review.sh` 专家讨论后重审 → 仍 NO 则停下汇报用户(详见 AGENTS.md)
- 每日报告:`flash-attention-tracker/daily_reports/`
