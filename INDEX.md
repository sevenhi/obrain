# 知识库索引 (INDEX) — obrain

> 本文件是所有领域的统一入口。每个领域一个文件夹,里面至少有一个 `README.md` 作为该领域索引。
> 定位:存档为主,平时不常看 → 目标是「扫 30 秒能找到,打开一篇能读懂」。
> 本知识库根目录 2026-09 由 `knowledge-base` 更名为 **`obrain`**。

## 目录规约(obrain vs library 判据)

`~/zhanbin/` 顶层约定:

```
~/zhanbin/
├── obrain/      知识库:提炼后的知识条目(md + frontmatter + 索引,按领域组织)——本目录
├── library/     原始资料只读存档(pdf/pptx/书籍,永不编辑):papers/ books/ manuals/ training/ personal/
├── projects/    自己写的代码 + 实验脚本:work/ patent/ scratch/
├── archive/     一次性产物 / 旧草稿:保留期 90 天,到期清理
├── scripts/     审查作业脚本(cron 调用)
├── tracker/     定时 wiki 同步管线(统一管理,含 vllm/cutlass/flash-attention-tracker)
└── repos/       只读上游克隆(vllm)
```

**判据**:`obrain` = 提炼后的知识条目(可检索、有索引、按领域);`library` = 原始资料存档(永不编辑)。两者不混;obrain 引用 library/archive/tracker 文件用 `source:` 指向,不复制。

## 领域列表

| 领域 | 文件夹 | 一句话说明 | 最近更新 |
|---|---|---|---|
| GPU/框架 显存生命周期 | `gpu-memory-lifecycle/` | 框架 allocator vs CUDA stream 的生命周期心智模型 + TFRA 事故案例 | 2026-09 |
| LLM 推理引擎栈 | `llm-inference/` | vLLM / FlashAttention / CUTLASS 三层索引 + concept/howto 深度条目(cutlass 6 篇、FA 3 篇、vLLM 3 篇,均多源综合+bot 审批) + vllm 源码 notes/ | 2026-09 |

---

## 阅读建议

- **临时查证** → 直接搜关键词(`happens-before`、`record_stream`、`BFC`、`cnrtQueueSync`)
- **回顾某领域** → 打开该领域 `README.md`,按索引跳
- **完整学习** → 按各篇底部「延伸阅读」的链式顺序读

## 收纳规范(给自己定的,写这条是为了以后不纠结)

1. 每篇用 YAML frontmatter:`title` / `type` / `tags` / `source` / `date`
2. 每篇开头一段「TL;DR」,30 秒讲完结论
3. 概念篇管「懂」,案例篇管「查」,清单篇管「用」——三类分开,不混在一篇里
4. 引用外部资料只放链接索引(`references.md`),不复制全文
5. **多来源登记**(ppt / pdf / 网页 / wiki …):每篇 `source` 写成「类型 + 标识符 + 链接」结构化形式;来源互不相同但内容重叠时,只收一条知识,`source` 里列全部出处 → 详见 `SOURCE-POLICY.md`
6. 拿到新资料先问「这条知识是不是已在库里?」→ 是则补 source 去重,否则新开篇;不建两篇内容重叠的篇
