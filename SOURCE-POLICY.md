---
title: 多来源登记策略 (SOURCE-POLICY)
type: reference
tags: [knowledge-base, sourcing, dedup]
date: 2026-09
---

# 多来源登记策略

> 知识来源不限于一种文件:ppt / pdf / 网页链接 / wiki / 论文 / 官方文档 / 内部脑图 ……都可能。
> 多来源带来的核心问题是**溯源**与**去重**。本文定义统一规则。

## 1. 来源类型与标识符格式

`source` 统一写成结构化字符串(可逐条列表,每篇按需写 1~N 条):
`类型: 描述 | 标识符 | 位置/链接`

| 类型 | 标识符写法 | 示例 |
|---|---|---|
| `ppt` | 文件名(建议带版本号) | `ppt: 显存生命周期培训 v2 (primary)` |
| `pdf` | 标题 + 出处 | `pdf: FlashAttention FA1 | arXiv:2205.14135 NeurIPS 2022` |
| `web` | 页面标题 + URL | `web: PyTorch CUDA semantics | https://pytorch.org/docs/...` |
| `wiki` | 页面标题 + pageId | `wiki: <内部 wiki 页标题> | pageId <NNNNNNNN>`(内部 wiki 地址含明文 pageId,只登记不公开展示) |
| `code` | 仓库路径 + 文件/行 | `code: TF BFCAllocator | github.com/tensorflow/tensorflow bfc_allocator.h` |
| `technote` | 内部文档/脑图 | `technote: 某某设计文档 | <存放路径>` |

标识符的意义:同一类型文件可能多次出现(不同版本 ppt)、不同位置(本地/远端 wiki),标识符让你能精确区分、避免混淆。

## 2. 内容重叠 → 只收一条,合并 source

规则 5 的核心动作。出现「同一条知识」的多种来源时:

- **只保留一篇内容页**,`source` 下列出**全部出处**(按可信度/新旧排序)
- `source` 里可加 `↑primary` 标注主来源(权威/最新)
- 好处:改一条知识不必同步多条;溯源时一眼看全所有出处

反例:某知识点同时出现在 wiki 和 pdf → 建了两篇 → 两篇内容漂移 → 找不到该信哪个。
正确做法:一篇内容,`source` 两行。

## 3. 下载型资料(重要,存档场景)

- **ppt / pdf 这类可下载文件**:优先**保留一份本地副本**在知识库的 `assets/` 下(或引用你自己的归档目录),URL/在线 wiki 可能失效或改版。
  - 副本路径写进 `source`(`本地副本:` 前缀),与原始链接并存
- **web / wiki 在线页**:在线会更新,通常**不复制全文**,只在 `source` 记链接 + 记录「抓取/阅读日期」,便于判断时效
- 取舍:价值高且稳定 → 存副本;动态更新 → 只记链接

## 4. 新资料入库流程(套用即省心)

1. 提炼出「知识条目」,不复制原文
2. 查重:该条是否已在库里?→ 已在 → 补写 `source` 并标日期,不新开篇
3. 不在 → 判断归属领域与 type(concept/case/checklist/howto/reference/domain,实际六类;目录↔type 默认映射与已知例外见 §5)
4. 落篇 + 更新该领域 index.md 索引 + 若新领域则登记进全局 INDEX

## 5. 目录 ↔ type 默认映射(惯例 + 已知例外)

新条目落哪个目录、type 标什么,按下面的**默认惯例**走;例外是历史现状,登记在案、不强行回改:

| 目录 | 默认 type | 约定强度 |
|---|---|---|
| `concepts/` | concept(机制原理) | 强约定 |
| `cases/` | case(事故复盘) | 强约定 |
| `practices/` | checklist(清单) | 强约定 |
| `notes/` | 源码一手实读笔记(**豁免** TL;DR/frontmatter 要求,见 §6) | 强约定(豁免) |
| `frameworks/` | howto(工具上手) | 弱约定 |
| `references.md`(领域根) | reference(链接索引) | 补充映射 |
| `index.md`(领域根) | domain(领域导航页) | 补充映射 |

> 原则:目录是组织,type 是内容属性;两者默认对齐,冲突时以实际内容定 type、并在此表登记例外,避免新规范自己引入新矛盾。

## 6. notes/ 豁免条款

`notes/` 目录定位是**源码一手实读笔记**(path:line 级参考材料),**豁免**知识条目的 TL;DR 与 YAML frontmatter 要求——它不是知识条目,是源码参考。给 notes 逐篇补 frontmatter/TL;DR 属于误用规范,不要做。

## 对现有库的说明

本库的多来源合并已示范在多个领域(如 FlashAttention 篇的 source 同时含 arXiv 论文多个来源、quantization 篇含论文 + 库内 PDF)。后续每接一个新来源,按本策略补 `source` 即可。
