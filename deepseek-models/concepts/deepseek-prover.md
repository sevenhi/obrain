---
title: DeepSeek 定理证明谱系 — Prover-V1 → V1.5 → V2(Lean 4 形式化)
type: concept
tags: [deepseek, theorem-proving, lean4, formal-math, rlpaf, monte-carlo-tree-search, synthetic-data, subgoal-decomposition]
series: deepseek-prover
source:
  - "pdf: DeepSeek-Prover V1 | arXiv:2405.14333 | https://arxiv.org/abs/2405.14333 | 2026-10-09"
  - "pdf: DeepSeek-Prover-V1.5 | arXiv:2408.08152 | https://arxiv.org/abs/2408.08152 | 2026-10-09 | ↑primary"
  - "pdf: DeepSeek-Prover-V2 | arXiv:2504.21801 | https://arxiv.org/abs/2504.21801 | 2026-10-09 | ↑primary"
  - "code: DeepSeek-Prover-V1.5 | https://github.com/deepseek-ai/DeepSeek-Prover-V1.5"
  - "code: DeepSeek-Prover-V2 | https://github.com/deepseek-ai/DeepSeek-Prover-V2"
date: 2026-10
---

# DeepSeek 定理证明谱系:Prover-V1 → V1.5 → V2(Lean 4 形式化)

> **TL;DR**:DeepSeek-Prover 是「用 LLM 做**形式化定理证明**(formal theorem proving)」的三代进化,目标语言是 **Lean 4** 证明助手(证明由编译器级内核验证,100% 可靠)。三代各解决一个瓶颈:**V1** 解决「训练数据不足」——用 8M 条合成证明数据喂模型;V1.5 解决「证明搜索太笨」——引入 RLPAF(从证明助手反馈做 RL)和 RMaxTS(蒙特卡洛树搜索);**V2** 解决「难题一步证不出来」——用 DeepSeek-V3 把难题**拆成子目标**,7B 模型逐个证子目标,再 RL。miniF2F 通过率 46.3% → 63.5% → **88.9%**。核心结论:**形式化证明的难点不在「模型多强」,而在「数据怎么造」和「搜索/分解怎么组织」——验证靠 Lean 内核白给,难度全在生成侧。**

---

## 0. 背景:为什么用 Lean 4

**形式化数学**(formal math)把数学证明写成编程语言代码,由**证明助手的类型检查内核(proof kernel)**逐行验证。Lean 4 是当前主流证明助手(微软研究院主导,mathlib4 是它的数学库)。对 LLM 的意义:

- **验证是免费的、绝对的**:proof kernel 通过 = 证明严格成立,不存在「看起来对」的灰色地带(对比自然语言证明只能靠人/模型评)。
- **但生成极难**:LLM 要写出「类型检查能通过」的 Lean 代码,任何一步类型不匹配都编译失败——reward 极其稀疏(要么全对要么全错)。

所以 DeepSeek-Prover 三代的核心问题都是「**怎么提高生成出『能过 Lean 内核』证明的概率**」。

---

## 1. Prover-V1(2405.14333):用合成数据喂饱模型

### 1.1 问题与解法

形式化证明最大的瓶颈是**训练数据匮乏**(数学库里的形式化定理太少,没法直接训 LLM)。V1 的答案是**迭代自举合成数据**:把自然语言数学题翻译成 Lean 形式化命题,让模型自己证明,通过 Lean 内核验证的「命题+证明」对就成了新训练数据。

### 1.2 四步合成管道(核心机制,可复现)

```
① 形式化(autoformalization):把自然语言竞赛题翻译成 Lean 4 形式化命题
② 过滤:模型打分 + 假设拒绝(hypothesis rejection),筛掉低质量/不可证的命题
③ 证明:用 DeepSeek-Prover 生成证明,Lean 4 内核验证(leanprover/lean4:v4.7.0-rc2)
④ 回填:验证通过的「命题+证明」进合成数据 → 微调 DeepSeek-Prover → 回到 ①
        (循环直到收益边际递减)
```

**关键技巧——命题与否定并发证**:为加速、省资源,对每个命题同时搜两个方向 `Γ⊢P` 和 `Γ⊢¬P`(命题的否定)。**哪个先证出来,另一个就确定不可证**,立刻终止另一个方向的搜索。这是「用逻辑对称性快速丢弃不可证命题」的经典技巧。

### 1.3 结果(有据数字)

- 合成数据规模:**800 万条**形式化命题+证明。
- 底座:DeepSeekMath 7B(微调而来)。
- miniF2F-test:64 采样 46.3%,cumulative 52.0%(超 GPT-4 的 23.0%、树搜索 RL 方法的 41.0%)。
- FIMO(形式化 IMO)148 题:100 采样证出 4 题(GPT-4 0 题),4096 采样证出第 5 题。
- 关键规律:**性能随合成数据量指数增长**——证明「数据规模是形式化证明的第一杠杆」。

---

## 2. Prover-V1.5(2408.08152):RLPAF + RMaxTS 搜索

### 2.1 两大升级

V1 是「单次整证生成」(一次性吐完整证明)。V1.5 在**训练和推理两侧**都升级:

1. **RLPAF(Reinforcement Learning from Proof Assistant Feedback)**:从「证明助手的反馈」做 RL——即用 Lean 内核的验证结果(过/不过)当 reward 信号。这是 V1 只有 SFT 的关键增量:让模型在「证明能不能过内核」这个明确目标上做 RL。
2. **RMaxTS(RMax Tree Search)**:一种蒙特卡洛树搜索(MCTS)变体,用「**内在奖励驱动探索**」生成多样化的证明路径,替代 V1 的单次整证。

### 2.2 机制:RLPAF 与 RMaxTS 为什么有效

- **RLPAF** 解决「reward 稀疏但绝对可靠」:Lean 内核反馈是二元的(对/错),比自然语言 reward 干净得多——RL 直接在「可验证性」上优化,不引入 reward 模型误差。
- **RMaxTS** 解决「单次采样命中率低」:形式化证明一步错就全错,单次生成命中率低;MCTS 式的树搜索在「证明状态」上分支探索,靠**内在奖励**(探索新状态)而非仅靠最终 reward 引导,能产生更多样化的证明路径。

### 2.3 结果(有据数字)

miniF2F-test 63.5%、ProofNet 25.3%,**均刷新当时 SOTA**。分阶段拆解(Base 42.2% → SFT 57.4% → RL 60.2% → RL+RMaxTS **63.5%**)清晰显示每步增量,尤以「+RMaxTS」贡献最大。底座 7B。

---

## 3. Prover-V2(2504.21801):子目标分解 + RL

### 3.1 问题与解法

V1/V1.5 都让 7B 模型**端到端**证整题。难题一步证不出来。V2 的思路:**让强模型(DeepSeek-V3)拆题,让 7B 模型证子目标**,再把非形式推理(CoT)与形式化证明(Lean)合流到统一模型。

### 3.2 递归子目标分解管道(核心机制)

```
① 冷启动:prompt DeepSeek-V3 把复杂定理拆成一串子目标(subgoals),
   同时把这些证明步骤形式化成 Lean 4 → 得到子目标序列
② 子目标证明:用更小的 7B 模型对每个子目标做证明搜索(省算力)
③ 合成冷启动数据:难题被拆开的步骤都证完后,把「完整分步形式证明」
   +「V3 的逐步推理 CoT」配对 → 冷启动推理数据
④ RL:用合成冷启动数据微调后,做 RL(二元 对/错 reward,标准推理模型目标),
   强化「非形式推理 ↔ 形式证明」的桥接
```

**机制本质——非形式 + 形式的合流**:V3 擅长**非形式**(自然语言逐步推理、拆解),7B 擅长**形式**(证具体小步骤)。V2 把「V3 的分解 CoT」和「7B 逐子目标证出的形式证明」拼成一条链,让模型同时学会「怎么想(非形式)」和「怎么写(形式)」——这正是它相对 V1/V1.5 的能力飞跃来源。

### 3.3 底座与结果(有据数字)

- 两个规格:**V2-7B**(基于 Prover-V1.5-Base,上下文扩到 32K)和 **V2-671B**(基于 DeepSeek-V3-Base)。
- **miniF2F-test 88.9%**(SOTA);PutnamBench 658 题解出 47(README 写 49,以论文 arXiv 摘要 47 为准——README 数字与论文摘要冲突处标此)。
- 新增 **ProverBench** 基准:325 道形式化题(含 AIME 24/25 的 15 题,其余来自教材),覆盖代数/数论/微积分/实变/复变/泛函/概率。
- 那 15 道 AIME 题 V2 解出 6 题;对比 DeepSeek-V3 用多数投票解 8 题——**形式化与自然语言推理的差距在大幅收窄**。

---

## 4. 三代总表与演进逻辑

| 代 | 底座 | 核心增量 | miniF2F-test | 解决瓶颈 |
|---|---|---|---|---|
| V1(2405.14333) | DeepSeekMath 7B | 8M 合成数据 + 命题否定并发证 | 46.3% | 数据不足 |
| V1.5(2408.08152) | Prover-V1 | RLPAF + RMaxTS 树搜索 | 63.5% | 搜索太笨 |
| V2(2504.21801) | V1.5-Base / V3-Base | V3 子目标分解 + RL 合流 | **88.9%** | 难题证不出 |

**演进逻辑一句话**:形式化证明的杠杆按「**数据 → 搜索 → 分解**」递进——先造够数据(量变),再组织搜索(质变),最后把「非形式推理能力」借进来拆题(范式升级)。验证一直是 Lean 内核的确定性能力,DeepSeek 三代的功夫全花在**生成侧**。

---

## 实践要点

- **reward 可靠性是 RL 的第一优先级**:RLPAF 用 Lean 内核二元反馈,免 reward model 误差——任何「验证免费且绝对」的场景(代码编译通过、证明内核通过、测试用例通过)都应优先用这类**确定性 reward**,别上易漂移的 reward model。
- **「命题+否定并发证」省资源**:对不可证命题,证其否定等价于否定其可证性,一条搜索就终结另一条——形式化/逻辑场景的经典加速技巧。
- **强模型拆题 + 弱模型证子目标**:V2 的「V3 分解、7B 证明」是通用范式——用贵的模型做规划/分解,用便宜的模型做执行,把「一步证整题」降成「逐子目标证」。
- **合成数据要进「内核验证」闭环**:V1 的每轮数据都经 Lean 内核过滤,保证数据「绝对正确」——这是形式化合成数据相对自然语言合成数据的独特优势(后者要额外想办法保证质量)。

## 溯源

- Prover-V1:https://arxiv.org/abs/2405.14333
- Prover-V1.5:https://arxiv.org/abs/2408.08152
- Prover-V2:https://arxiv.org/abs/2504.21801
- 仓库:https://github.com/deepseek-ai/DeepSeek-Prover-V1.5 、 https://github.com/deepseek-ai/DeepSeek-Prover-V2

## 相关条目(库内互链)

- 底座结构(DeepSeekMoE/V2/V3):见 [model-architecture/concepts/deepseek-arch-v1-moe-v2.md](../../model-architecture/concepts/deepseek-arch-v1-moe-v2.md)、[deepseek-arch-v3.md](../../model-architecture/concepts/deepseek-arch-v3.md)。
- 数学推理谱系(自然语言证明路线,与 Lean 形式化互补):[deepseek-math.md](deepseek-math.md)——V2 的「V3 拆题 + CoT」正是 Math-V2 自然语言证明与 Prover 形式化证明两条路的合流点。
