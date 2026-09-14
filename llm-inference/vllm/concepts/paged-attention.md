---
title: PagedAttention 与分页 KV Cache:vLLM 的显存管理心法
type: concept
tags: [vllm, pagedattention, kv-cache, memory, block-table, gpu, llm-serving]
source:
  - "pdf: Efficient Memory Management for LLM Serving with PagedAttention | Kwon et al. 2023 | arXiv:2309.06180 (SOSP 2023)"
  - "web: vLLM Blog — Easy, Fast, and Cheap LLM Serving with PagedAttention | https://blog.vllm.ai/2023-06-20/vllm.html"
  - "web: zeroentropy.dev — PagedAttention: vLLM's OS-style KV cache memory manager | https://zeroentropy.dev/concepts/paged-attention/"
  - "code: vLLM V1 源码(commit b74ac0bc6)notes 一手实读 | notes/vllm-core-architecture-notes.md"
  - "doc: notes/vllm-core-architecture-notes.md (库内源码笔记)"
date: 2026-09
---

# PagedAttention 与分页 KV Cache

## TL;DR

**PagedAttention 把 KV cache 当成 OS 的虚拟内存来管**:物理块固定(默认 16 token/层),每个序列一张 **block table** 把逻辑位置映射到物理块,按需从全局块池懒分配。这一层"间接寻址"把 KV cache 的利用率从 naive 连续分配的 **~33% 提到 ~84%**(zeroentropy 玩具演示,对应 ~2.5× 并发),并让 vLLM 相对 FasterTransformer/Orca 拿到 **2-4× 吞吐提升**(论文实测,同延迟口径),把跨请求前缀共享变成页表上的一个引用计数而不是架构改动。它同时是 vLLM 吞吐神话的**地基**(连续批处理、prefix caching、抢占全建立在"块可独立分配/释放"之上)。

> **先记住一个反转直觉**:注意力计算谁都会,真正决定 LLM serving 吞吐上限的,是**怎么在显存里安放 KV cache**。PagedAttention 的核心不是新的数学,而是一个**内存管理思想(分页)**。

---

## 1. 它解决的分片问题:naive 连续分配的白白浪费

naive 的 KV cache 为每个在跑序列**预留一段连续内存**,大小按 `max_seq_len` 封顶:

```
naive 连续 slab(每个请求一条,按最大序列长预留):
pos  0 ──────── 实际长度 ──┐  max_seq_len ────────────────────
│ 实际用到的 KV │   闲置的尾部(stranded)                   │
└──────────────────────────┴──────────────────────────────────┘
```

- 一个 8K 上下文的服务器里,只产出 100 token 答案的请求,整条生命周期内都占着一个 **8K 形状的 slab**——闲置尾巴 stranding;
- 异构 batch(不同请求长度差距大)时,**60-70% 的显存浪费在调度器收不回的空闲**上;
- **可达到的 batch 不是被 GPU 算力卡住,而是被"放得下多少 slab"卡住**。

**规模感的墙**:70B 模型 / 8K 上下文 / 32 条在跑序列 → naive 预分配约 **80GB KV cache**,在**一个有用 token 落地之前**就占掉了(zeroentropy 估算)。

> 关键认知:KV cache 在长上下文 + 大 batch 下是**显存的第一大消费者**(常占 30-50% 总量),它去哪、留多少,batch 就装多少。

---

## 2. 分页:把"一条连续 slab"变成"一堆可独立管理的块"

跟 OS 一脉相承:物理内存不够连续/不够用时,就用**分页 + 页表间接寻址**。

```
PagedAttention · 全局 16-token 物理块池,按需懒分配,block table 作逻辑→物理映射:

req 1 (22 tok):  L0→P0  L1→P3               block table
req 2 (41 tok):  L0→P1  L1→P5  L2→P9
req 3 (59 tok):  L0→P2  L1→P6  L2→P11 L3→P14
req 4 (25 tok):  L0→P7  L1→P13

物理块池:  P0 P1 P2 P3(尾)P4 P5 P6 P7 P8 P9(尾)P10 P11 P12 P13(尾)P14(尾) ...free 块收回到池
```

- **物理块形状因 backend/版本而异**(本文所述的是 **V1 + FlashAttention 路径**:`[num_blocks, num_kv_heads, block_size, 2*head_size]`,K、V 拼在最后一维;V0+FA 常见 `[num_blocks, num_kv_heads, head_size, block_size, 2]`),全局一个池;
- 每个序列一张 **block table**:逻辑位置 L → 物理块 P;
- **按需分配**:需要多少 token 才分多少块,不再预留给最大长度——尾巴不在物理上占用块;
- 内存利用率 naive ~33% → paged ~84%(zeroentropy 玩具演示),对应 **~2.5× 并发请求**(demo 场景);论文实测的 2-4× 是**同延迟下吞吐**提升(相对 FasterTransformer/Orca),两个口径分开看。

---

## 3. 为什么"间接寻址"是解放一切的前提

一旦 KV cache 由"连续 slab"变成"可独立操作的一堆块",下面这些 vLLM 招牌特性全是**白送**:

| 特性 | 建立在分页上的机制 |
|---|---|
| **连续批处理** | 块按 token 数精确分配/释放 → 调度器每步可按可用块接纳/移除请求 |
| **Prefix caching** | 相同前缀的块在物理上可**共享**(下表)→ 命中即复用,跳过 prefill 计算 |
| **抢占(preemption)** | 块不足时,默认抢占 **running 队尾(FCFS 下最低优先)请求**,也可按 PRIORITY 策略抢低优先级;**释放的块进 LRU 缓存**,被抢占请求退回 waiting(重排时可被 prefix cache 命中) |
| **共享前缀(shared prefix)** | 前缀共享 = 页表上的一个 **ref_cnt**,而非深层次架构改动 |
| **copy-on-write** | 多序列引用同一块时,写入前先复制,避免污染(cache.py CoW) |

> **心法**:引用计数 + copy-on-write 是 OS 页表的老朋友。PagedAttention 把"分享 prefix"这个看似复杂的问题,降维成**给同一物理块加一个引用**——这就是"借 OS 思路"的力量:一套成熟的内存语义,直接搬进 KV cache。

---

## 4. 反直觉的代价:非连续访存怎么算注意力?

分页让显存高效了,但注意力 kernel 碰到的 K/V 是**非连续、散落各块**的——传统 `S@V` 没法直接算。PagedAttention 的注意力 kernel 核心处理:

```
传统:  K/V 是一整块连续 [N, d] → matmul 直接吃
Paged: K/V 按 (block, slot) 散在多个物理块,block table 索引
        → kernel 遍历 block table,把每块的 K/V 与对应 S 分数做局部 matmul+softmax
        → 用 block-scaling 系数(每块一个)在写回前 rescale,保持数值一致
```

- 分页注意力的**计算语义与标准注意力完全等价**(不是近似),只是**访存走 block table**;
- 每个物理块独立计算局部注意力 + 累积 softmax rescale。**rescale 机制**正是 FlashAttention online softmax 的直接延伸:每块维护局部 max 和 sum,块间用 `exp(old_max − new_max)` 做 rescale 累积——这正是"分页 + 分块不损失数值精度"的关键;
- 论文还区分 **V1 / V2 两套 kernel**:V1 针对 decode(单 token query,块内 K/Q 点积 + V 加权)、V2 针对 prefill(多 token query,paged K 与连续 Q 的块状 matmul)——这解释了 vLLM 在 prefill 与 decode 阶段性能特征不同的根源;

> ⚠ **V0→V1 的落地差异(读代码必看)**:上面是论文原始 / V0 手写 `paged_attention` kernel 的行为(遍历 block table 做局部 matmul+softmax)。**V1 主路径已改用 `flash_attn_varlen_func`**——它原生支持"变长序列 + block table"形式的分页 K/V,**不再用旧版手写 paged_attention kernel**(后者仅保留在 `csrc/attention/` 非主路径)。V1 里"PagedAttention 的体现" = 分页物理存储(BlockPool)+ block table 索引 + flash_attn_varlen 的变长分页注意力。别按 §4 现在时去源码里找旧的 paged_attention kernel 会踩坑。

---

## 5. V1 里的落地:BlockPool + 引用计数 + 内容哈希

V1 把分页分成**底层块池**与**高层管理**两层(read notes §2 有 path:line 级一手细节):

- **BlockPool**(`block_pool.py`):物理块的 free list + 按块内容哈希的 `BlockHashToBlockMap` + LRU 淘汰 + ref_cnt 回收;
- **KVCacheManager**(`kv_cache_manager.py`):Scheduler 操作 KV cache 的**唯一入口**,`allocate_slots`(分三阶段:释放冗余 → 处理 prefix token → 为新 token 分块)、`get_computed_blocks`(prefix 查询)、`free`/`evict_blocks`;
- **块布局分区**(allocate_slots 注释):
  ```
  | < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
    已计算      新命中         外部(connector)命中  待计算   speculative lookahead
  ```
- **prefix caching**(默认开,`enable_prefix_caching: bool = True`):每个满块由 token ids + **前驱块哈希**构成**哈希链**,新请求查哈希链命中 → ref_cnt++ 复用物理块,跳过该段 prefill 计算(system prompt / 共享前缀省算力)。

**KV cache 怎么被填满(写回 kernel)**:每步前向后,新算出的 K/V 按 `slot_mapping`(把 (block_id, slot_in_block) 展平成的一维索引)散列写进分页张量:
- `reshape_and_cache_flash()`(`_custom_ops.py:2631`,FlashAttention 路径,按 slot_mapping 散列写 `key_cache`/`value_cache`);
- `swap_blocks()`(`_custom_ops.py:2903`):CPU↔GPU 块交换(offloading 用)。
这正是"非连续显存"的上半场:块怎么安放(PagedAttention)之外,新数怎么散列写进去(reshape_and_cache)——两者合力才让分页 cache 转起来。

---

## 6. 实践要点(checklist)

1. **先量化 KV cache 占多少显存**:`max_seq_len × 2 × num_layers × num_kv_heads × d_head × dtype_bytes` 是天花板;长上下文下它常是第一大消费者——决定 batch 容量的不是显存总量,而是 KV cache 余量。
2. **block_size 默认 16,别乱调**:更小块减少内部碎片但增加 block table 行数与 kernel 间接访存开销;默认 16 是吞吐/碎片平衡点(paper 里也测过 32/64)。
3. **gpu_memory_utilization 默认 0.92(当前 main 源码 `config/cache.py:111` 实测;旧版曾为 0.9)**:分配 KV cache 前的显存余量由 profiling 峰值决定;调高可给 KV cache 更多空间(风险:模型峰值超了 OOM)。
4. **enable_prefix_caching 默认开**:长共享前缀(如 system prompt)收益显著;但每块要算内容哈希(CPU 开销),纯随机上下文可关。
5. **共享前缀 = ref_cnt,不是复制**:多请求共享 prefix 块只在页表加引用;要改写的共享块走 CoW——理解这点才懂为什么 vLLM 对重复 system prompt 的预填充几乎免费。
6. **跟分片/连续批处理联动**:块管理是调度器的命根子(allocate_slots 返回 None 触发抢占);调吞吐时优先看**有没有把 KV cache 块耗光**,而不是 batch 数字本身。

---

## 7. 溯源

- **PagedAttention 论文**(分页思想、内存利用率、吞吐基准): Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023, arXiv:2309.06180
- **vLLM 博客**(吞吐提升 2-4×、KV cache 显存占比场景图——"每请求 KV cache 占显存比例"图,与 §1 的"60-70% 浪费在 slack"是不同口径): https://blog.vllm.ai/2023-06-20/vllm.html
- **zeroentropy 解读**(naive 33%→paged 84% 玩具演示、~2.5× 并发;70B/8K/32 req 预分配 80GB): https://zeroentropy.dev/concepts/paged-attention/
- **V1 源码一手实读**: notes/vllm-core-architecture-notes.md §2(BlockPool / KVCacheManager / 块布局 / prefix caching)

> **路径基准**：正文与溯源里的相对路径(`notes/...`)均相对知识库根 `~/zhanbin/obrain/`。条目移动时先确认基准。
