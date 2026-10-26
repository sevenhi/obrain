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

`source` 统一写成**跨设备可用的结构化字符串**(可逐条列表,每篇按需写 1~N 条):
`类型: 标题 | 稳定标识符或原始 URL | 阅读/抓取日期 | 角色`

原则：`source` 是知识的公开溯源，不是本机文件清单。优先使用 DOI、arXiv、出版社、官方文档、官方仓库或项目主页等跨设备可访问的原始链接。

| 类型 | 标识符写法 | 示例 |
|---|---|---|
| `ppt` | 标题 + 稳定公开链接 | `ppt: 显存生命周期培训 v2 | https://example.org/slides` |
| `pdf` | 标题 + DOI/arXiv/出版社链接 | `pdf: FlashAttention FA1 | https://arxiv.org/abs/2205.14135` |
| `web` | 页面标题 + URL | `web: PyTorch CUDA semantics | https://pytorch.org/docs/...` |
| `wiki` | 页面标题 + 稳定页面 URL | `wiki: 某某页面 | https://wiki.example.org/...` |
| `code` | 仓库/项目 URL + 文件路径 | `code: TF BFCAllocator | https://github.com/tensorflow/tensorflow/...` |
| `technote` | 标题 + 可访问 URL | `technote: 某某设计文档 | https://...` |
| `paper` | 论文(arXiv/期刊均可,与 pdf 混用) | `paper: DeepSeek-V3 | https://arxiv.org/abs/2412.19437` |
| `arXiv` | arXiv 预印本专指 | `arXiv: DeepSeek-OCR | https://arxiv.org/abs/2510.15266` |

> 登记说明(2026-10 补):早期 8 篇条目已大量使用 `paper:`(20 条)/`arXiv:`(8 条)前缀,按「登记例外、不回改」原则收录进本表。新写条目两者皆可用;与 `pdf:` 的差异:`pdf:` 强调载体是 PDF 文件(含 arXiv PDF 里的章节定位),`paper:`/`arXiv:` 只做文献身份登记。

标识符的意义:同一类型资料可能多次出现(不同版本、不同发布页面),稳定标识符和原始 URL 让不同设备、不同时间都能重新定位来源。不要在 source 中写本机绝对路径、家目录路径、挂载点、文件名或“本地副本”字段。

## 2. 内容重叠 → 只收一条,合并 source

规则 5 的核心动作。出现「同一条知识」的多种来源时:

- **只保留一篇内容页**,`source` 下列出**全部出处**(按可信度/新旧排序)
- `source` 里可加 `↑primary` 标注主来源(权威/最新)
- 好处:改一条知识不必同步多条;溯源时一眼看全所有出处

反例:某知识点同时出现在 wiki 和 pdf → 建了两篇 → 两篇内容漂移 → 找不到该信哪个。
正确做法:一篇内容,`source` 两行。

## 3. 来源只登记公开链接(跨设备)

- **不在 obrain 保存或依赖本地原始文件**。正文只保留提炼后的知识；`source` 只登记 DOI、arXiv、出版社、官方文档、官方仓库或项目主页等原始链接。
- PDF 论文优先登记 arXiv/DOI/出版社页面；网页登记 canonical URL；代码登记官方仓库或具体文件 URL。
- 每条来源可附「阅读/抓取日期」和 `primary` 角色，但不要写本机路径、挂载点、家目录、文件名或本地副本说明。
- 原始链接失效时，优先寻找同一来源的稳定替代链接并更新 source；不要用某台设备的临时下载路径替代。

## 4. 新资料入库流程(套用即省心)

1. 提炼出「知识条目」,不复制原文
2. 查重:该条是否已在库里?→ 已在 → 补写 `source` 并标日期,不新开篇
3. 不在 → 判断归属领域与 type(concept/case/checklist/howto/reference/domain/comparison,实际七类;目录↔type 默认映射与已知例外见 §5)
4. 落篇 + 更新该领域 index.md 索引 + 若新领域则登记进全局 INDEX(README.md 域表)

## 5. 目录 ↔ type 默认映射(惯例 + 已知例外)

新条目落哪个目录、type 标什么,按下面的**默认惯例**走;例外是历史现状,登记在案、不强行回改:

| 目录 | 默认 type | 约定强度 |
|---|---|---|
| `concepts/` | concept(机制原理) | 强约定 |
| `cases/` | case(事故复盘) | 强约定 |
| `practices/` | checklist(清单) | 强约定;例外:vllm/practices/inference-engine-selection.md 为 comparison(选型对比);vllm/practices/attention-backend-selection.md 为 comparison(选型对比) |
|| `implementation/`(attention 域) | implementation | 弱约定;目录独有 |
|| `algorithm/`(attention 域) | algorithm | 弱约定;目录独有 |
|| `ecosystem/`(vllm 域) | concept(生态构件深读) | 弱约定;2026-10 合入 vllm-project 域后沿用 concepts 同款 |
| `notes/` | 源码一手实读笔记(**豁免** TL;DR/frontmatter 要求,见 §6) | 强约定(豁免) |
| `frameworks/` | howto(工具上手) | 弱约定;例外:attention/frameworks/flexattention.md 为 concept(编程模型+编译器机制深读,2026-10 登记) |
| `references.md`(领域根) | reference(链接索引) | 补充映射 |
| `index.md`(领域根) | domain(领域导航页) | 补充映射 |

> 原则:目录是组织,type 是内容属性;两者默认对齐,冲突时以实际内容定 type、并在此表登记例外,避免新规范自己引入新矛盾。

## 6. notes/ 豁免条款

`notes/` 目录定位是**源码一手实读笔记**(path:line 级参考材料),**豁免**知识条目的 TL;DR 与 YAML frontmatter 要求——它不是知识条目,是源码参考。给 notes 逐篇补 frontmatter/TL;DR 属于误用规范,不要做。

## 7. 索引/政策/链接索引页的 frontmatter 豁免

以下页面不是知识条目,豁免 `source` 字段;README.md 作为根入口额外豁免全部 frontmatter:

- 各域 `index.md`(type: domain)
- 各域 `references.md`(type: reference,自身即链接索引)
- 根目录 `SOURCE-POLICY.md`
- 根目录 `README.md`(豁免全部 frontmatter)

## 对现有库的说明

本库的多来源合并已示范在多个领域(如 FlashAttention 篇的 source 同时含 arXiv 论文多个来源、quantization 篇含论文来源)。后续每接一个新来源,按本策略补 `source` 即可。
