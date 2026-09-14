# obrain 知识库"做厚"流程手册(固化版)

> 2026-09-14 由 CUTLASS(6篇)/ FlashAttention(3篇)/ vLLM(3篇)三轮实战固化。
> 目标:给定一个领域,把它从"README 几句话 + notes 研究笔记"升级成"打开一篇能读懂、有专业深度、可沉淀分享"的知识库条目。

---

## 0. 定位(先确认做厚的是哪类内容)

| obrain 现状 | 是否要做厚 | 怎么做 |
|---|---|---|
| 领域 README 只有导航/几句话 | ✅ | 新增 concept/howto 深度篇(本流程) |
| notes/ 是源码研究笔记(path:line 级) | ⚠️ 不完全 | 提炼成知识范式(TL;DR/机制/数据/实践),notes 保留作源码参考 |
| 已有深度 concept 篇 | 否 | 不重复,只补缺 |

> vLLM 教训:notes 已是高质量源码实读,缺的不是"从零写",而是"提炼成可沉淀范式"。先读 README + notes,判断差距再动手。

## 1. 深度标准(concept 篇四要素,缺一不可)

1. **机制原理**:回答"为什么这样设计/底层怎么工作",有因果推导(算力≫带宽→必须 tiling→mainloop/epilogue)。
2. **可复现细节**:至少一段代码/伪代码/结构图,能照着写出或读懂真实实现(CUTLASS 三层 tiling、FA 伪代码、KV cache 分页图)。
3. **数据 + 对比**:数字、基准、横向对比(Stream-K 6.7×、FA2 2×、vLLM 2-4× 归因),非空谈。
4. **实践要点**:可执行 checklist / 坑 / 决策准则(决定"能不能沉淀分享")。

**TL;DR 首屏 30 秒讲清结论 + 一个"心法/反直觉"金句**(PagedAttention 篇的"注意力谁都会,显存怎么放才是吞吐上限")。

## 2. 主动多源综合(给定来源是引子,不是全部)

1. 拿一个来源 → `web_search`(官方文档/原论文/官方博客/优质社区,中英都找);
2. `web_extract` 拉 2~5 个权威源全文交叉核对;
3. 提炼:每源给什么独有信息?公认事实 vs 某源一家之言(标 primary)?
4. 综合成一篇:机制推导为骨架,多源数据互相印证,歧义在文中标注;
5. 落篇前所有来源登记进 `source:`(类型+标识符+链接)+ references/溯源节。

## 3. 写成(结构与规范)

- YAML frontmatter:`title` / `type: concept|howto|...` / `tags` / `source`(结构化多来源)/ `date`
- 正文:TL;DR → 机制(因果)→ 可复现细节(代码/结构图)→ 数据对比 → 实践 checklist → 溯源
- **正文只写事实,绝不带内部审批标记**(`2026-09 review 校正` 这类会被 doc-writer 判 blocking,曾全局清理)
- 溯源路径标注基准(相对知识库根 `~/zhanbin/obrain/`;库内以 obrain 为根,内容自足)

## 4. 审批(用 review_obrain_batch.sh)

**主流程即批量脚本**,只跑 doc-writer + quick-assistant(实测可靠的两个 bot):

```
bash ~/zhanbin/scripts/review_obrain_batch.sh <篇1.md> [篇2.md] [篇3.md] ...
```

- 逐篇、每篇两 bot 并行、600s timeout、输出到 `各篇目录/.review/*.clean`;
- **code-reviewer 不用于知识文档**(SOUL 是审 PR/diff 的,遇单篇文档全库探索/挂死,验证过);
- 脚本(共 batch / triple 两个):
  | 脚本 | 用途 | 何时用 |
  |---|---|---|
  | `review_obrain_batch.sh` | 2 bot(doc-writer+quick-assistant)批量 | **默认主流程** |
  | `review_obrain_triple.sh` | 3 bot(code-reviewer 也上)长 timeout 单篇深审 | 极罕见,code-reviewer 健康时 |

## 5. 修正闭环(核心:数轮收敛到 YES)

bot 意见分两类:
- **硬伤**(approved: NO 的必修):技术/事实/API 名/归因/URL 错误 → 必须修;
- **建议**(不构成 NO):深度增强 → 视价值采纳。

每个 bot 意见 → 逐条 patch 修 → 重跑。**收敛纪律**:
- 三轮复审仍 NO → 停止,汇报用户(不无限重试);
- **quick-assistant 对长文档可能超时**(600s RC=124):超时 → 用 doc-writer 复审意见 + quick-assistant 首审覆盖来兜底,判通过,不为 qa 复审反复重跑(重跑大概率再超时);
- **权威数字竞争时用一手核实**:bot grep 被引 PDF 全文比 web 缓存可靠(FA3"840 TFLOPS"被 qa grep 出只在 ISBN 里、论文只写 740)。Python/源码实测 > bot 猜(如 vLLM `gpu_memory_utilization` 源码实测 0.92,doc-writer 给过时 0.9);
- URL 先 curl 验证(博客 `blog.vllm.ai` 非 github.io;文档站 `docs.vllm.ai`);
- **修完笔记(src 一手资料)里同样错误**(vLLM counted 抢占源头在 notes,正文修完 notes 也回修)。

## 6. 收尾(达标即入库)

1. 校验:正文无 review 元信息残留、代码围栏数偶(闭合)、多篇无矛盾;
2. 更新领域 README(加 entries 导航,一句定位);
3. 更新全局 `INDEX.md`(领域描述反映已做厚篇数);
4. git commit(work 区干净);
5. **沉淀进 skill `knowledge-base-ingest`**:该领域的高频坑追加到 skill(FA/CUTLASS/vLLM 各有一块),供下次写同类内容直接查。

## 7. 高频坑速查(写这三大领域必看,详见 skill)

- **CUTLASS**:CuTe divide 系列 API 名(别杜撰 `partitioned_into_TV`)、scheduler 默认 persistent、bias layout 易反、Stream-K 确定性默认值。
- **FlashAttention**:IO 复杂度标准是 Θ(Nd+N²) 非 Θ(N²d)、FA1 伪代码外层 K/V、FA2 grid 与 FA1 相同(改进在 block 内 warp)、online-softmax m/l 是行向量、FA 输出非 bitwise、FA3 FP8 是 block-quantize+incoherent 非"V 保持 BF16"。
- **vLLM**:2-4× 归因主要来自分页非调度、counted 抢占不存在(V1 只有 recompute)、step/step_with_batch_queue 两路径、Worker 进程模型随 Executor、Ray 类名、URL。
