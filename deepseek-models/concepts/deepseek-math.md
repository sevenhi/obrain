---
title: DeepSeek 数学推理谱系 — DeepSeekMath 与 DeepSeekMath-V2(GRPO / 自验证)
type: concept
tags: [deepseek, math, grpo, rl, verifier, self-verification, data-pipeline, domain-model]
series: deepseek-math
source:
  - "pdf: DeepSeekMath | arXiv:2402.03300 | 2026-10-09 | ↑primary"
  - "pdf: DeepSeekMath-V2 | arXiv:2511.22570 | 2026-10-09 | ↑primary"
  - "code: DeepSeek-Math | https://github.com/deepseek-ai/DeepSeek-Math"
  - "code: DeepSeek-Math-V2 | https://github.com/deepseek-ai/DeepSeek-Math-V2"
date: 2026-10-09
---

# DeepSeek 数学推理谱系:DeepSeekMath → DeepSeekMath-V2

> **TL;DR**:DeepSeek 的数学能力走两步——**DeepSeekMath 7B**(2024)用「120B 数学 token 数据管道 + GRPO 强化学习」在 MATH 上做到 51.7%,开创了「领域数据续训 + RL」的范式;两年后 **DeepSeekMath-V2**(2025)把数学推到「**自验证**」:不再只追最终答案正确率,而是训练一个 verifier(验真器)判定证明对不对,再用 verifier 当 reward model 训 generator,让模型生成前自己审自己的证明。结果:IMO 2025 金牌、CMO 2024 金牌、Putnam 2024 118/120(超人类最高 90 分)。**核心结论:数学推理的瓶颈从「答案对不对」转移到「推理可不可验证」,自验证是测试时算力扩展(开放问题无已知答案)的关键。**

---

## 1. DeepSeekMath 7B:数据管道 + GRPO 的开山

### 1.1 定位

DeepSeek 第一个数学专项模型。结构**无创新**(就是 DeepSeek-Coder-Base-v1.5 7B 的架构),价值全在**两件事**:① 怎么挖出 120B 高质量数学 token;② 用 GRPO 强化学习提升推理。一句话:**领域模型 = 底座复用 + 数据工程 + RL,不发明结构**——这是理解 DeepSeek 领域模型家族的第一性原理。

### 1.2 数据管道(为什么能挖到 120B 数学 token)

从 Common Crawl(全网爬虫语料)里筛数学语料,靠一个**迭代自举的 fastText 分类器**循环,五步(论文写四轮迭代):

```
Step 1  选 OpenWebMath(高质量数学网页集)做种子 → 训 FastText 分类器
Step 2  用 FastText 在去重后的 Common Crawl 里检索数学网页
Step 3  统计分析,识别「潜在数学域名」
Step 4  人工标注这些域名里与数学相关的 URL
Step 5  把标注 URL 链到、但还没采集的网页加进种子 → 回到 Step 1
        (循环 4 轮,最终 35.5M 数学网页 / 120B tokens)
```

**机制要点**:FastText 是轻量文本分类器,`fastText 分类 → 域名统计 → 人工标注 → 回填种子` 的闭环让分类器越训越准、语料越滚越大。这是「用公开网络数据挖出领域数据」的可复现范式——同类手法后被广泛借鉴(领域模型做数据都绕不开「种子→分类→回填」自举)。

### 1.3 GRPO:去掉 critic 的 PPO(核心机制)

DeepSeekMath 训练阶段用了 **Group Relative Policy Optimization(GRPO)**,它是本论文最重要的方法贡献,后成为 DeepSeek-R1 的 RL 核心,也是开源推理 RL 的事实标准。

**为什么不用 PPO?** 标准 PPO 需要训练一个**价值函数 critic**(估计每个状态的期望回报),而 critic 的规模通常和策略模型一样大 → 内存翻倍。数学 RL 场景里 reward 只有「对/错」二元,不依赖过程价值,没必要养个 critic。

**GRPO 做法**(相对优势估计):

```
对每个问题 q,采样 G 个回答 {o₁…o_G},每个得 reward rᵢ(通常是规则判定,如答案对不对)
对这组回答:
  Aᵢ = (rᵢ - mean(r)) / std(r)          ← 组内标准化 = 优势估计,替代 critic
目标(带 KL 惩罚):
  max E[ min( π/π_old · Aᵢ, clip(π/π_old, 1±ε)·Aᵢ ) − β·D_KL(π || π_ref) ]
```

**与 PPO 的本质差异**:PPO 的优势 = `r + γV(s') − V(s)`(需要 critic);GRPO 的优势 = **同一问题的多个回答彼此比较**(组内相对排名),天然对齐「这个回答比组内平均好还是差」,既省 critic 内存,又更适合 reward 稀疏的数学场景。GRPO 的 "Group Relative" 就是指「相对同组其他样本」。

### 1.4 结果(有据数字)

- **MATH benchmark 51.7%**(不用外部工具、不用投票),逼近 Gemini-Ultra 与 GPT-4;自一致性 64 采样到 60.9%。
- 相对开源模型:few-shot CoT 下比当时开源 base 模型**绝对提升超 10%**,且超过 Minerva 540B。
- 三档发布:base / instruct(SFT)/ rl(GRPO);RL 版带工具可达 ~60% MATH。
- 因为底座是 Coder-v1.5,DeepSeekMath 还继承了**写代码解题**能力(tool use),这为后面 Prover 系列铺路。

---

## 2. DeepSeekMath-V2:从「答案对」到「推理可验证」

### 2.1 动机:最终答案 reward 的死穴

RL 只奖励「最终答案对不对」能把 AIME/HMMT 这类**有唯一数值答案**的比赛刷饱和(一年内从很差到接近满分)。但它有两个结构性局限:

1. **答案对 ≠ 推理对**:追最终答案准确率,不保证中间推导是对的。
2. **定理证明类任务没有数值答案**:证明题的 reward 无法用「答案对不对」判定。

所以 V2 的命题是:**与其追「答案对」,不如训练模型「自己验证证明的完备性与严谨性」**——即「self-verifiable mathematical reasoning」。这尤其对**开放问题(没有已知答案)**有意义,因为那时唯一能 scale 的测试时算力就是「多验证几遍」。

### 2.2 方法:verifier → generator 的闭环(核心机制)

V2 的贡献是一条**「验证能力 ⇄ 生成能力」互相增强的训练回路**,四步(按论文贡献):

```
① 训 verifier:训练一个准确、忠实的 LLM 验证器,判定一个(自然语言)证明对不对
② meta-verification:再上一层「元验证」,大幅削减 verifier 的幻觉性挑错(否则 verifier
   会瞎报问题),保证验证质量
③ 训 generator:用 verifier 当 reward model 训证明生成器,激励它在定稿前
   「自己找出并解决尽量多的问题」——生成前先自审
④ scale verification compute:随着 generator 变强、它的证明越来越难验证,
   自动给「难验证的证明」打标,造训练数据反过来再提升 verifier——
   维持「生成-验证差距」(generation-verification gap)
```

**机制本质**:这是**互为对手的循环**。generator 越强 → 生成的证明越难 → 逼 verifier 变强 → verifier 更强 → 能当更好的 reward 训出更强的 generator。第 ④ 步是维持循环不塌的关键(否则 generator 一旦强到 verifier 判不了,reward 信号就失效)。

### 2.3 底座与结果(有据数字)

- **底座**:DeepSeek-V3.2-Exp-Base(不是 7B,直接上了 671B 级主模型底座)。
- **IMO 2025**:6 题解 5 题 = **金牌**;CMO 2024 金牌(4 题全解 + 1 题部分分)。
- **Putnam 2024**(本科生数学竞赛):11/12 题完整解 + 1 题有小错 = **118/120**,超过人类参赛者最高分 90。
- **IMO-ProofBench**:基础集超 DeepMind 的 DeepThink(IMO-Gold),进阶集仍具竞争力,远超其他 baseline。
- 论文自述:**完全解出的题通过全部 64 次验证尝试;未完全解出的题,generator 也能识别出自己证明里的真实问题**——这是「自验证」成立的直接证据。

---

## 3. 谱系关系与演进逻辑

| 维度 | DeepSeekMath(2024) | DeepSeekMath-V2(2025) |
|---|---|---|
| 底座 | DeepSeek-Coder-Base-v1.5 7B | DeepSeek-V3.2-Exp-Base |
| 方法 | 数据管道 + GRPO(组内相对 RL) | verifier + generator 自验证闭环 |
| 目标 | 答案正确率(MATH 51.7%) | 推理可验证(IMO 金牌/Putnam 118) |
| 结构创新 | 无 | 无 |
| 范式意义 | 领域数据 + RL 范式开创 | 从「答案 reward」到「自验证」的转向 |

**演进逻辑一句话**:数学推理的评测标尺从「答案对不对」(可规则判定、已被刷饱和)升级到「推理可不可信」(无法规则判定、只能靠模型自己学会验证)——V2 的突破不在结构,在**把「验证」本身变成可训练的、可 scale 的能力**。

---

## 实践要点(能不能沉淀分享)

- **领域模型优先复用底座,别想结构创新**:数学/编码模型的 90% 工作量在数据与训练策略;结构直接拿 Coder/V3 底座。选底座时优先「已含相关能力」的(如 Math 选 Coder 底座是为继承代码/工具能力)。
- **GRPO 是稀疏 reward 场景首选 RL**:reward 只能规则判定(答案对错)时,GRPO 用组内相对优势替代 critic,省一半内存且更稳;需要密集过程 reward 的场景才考虑带 critic 的 PPO。
- **「数据自举管道」可复用**:种子语料 → 分类器 → 域名统计 → 人工标注 → 回填,是挖领域数据的通用闭环;关键是每轮回填让分类器变强。
- **自验证的坑**:verifier 会**幻觉性挑错**(乱报不存在的错误)→ 必须加 meta-verification 层把关,否则 reward 信号被污染。
- **测试时算力 scale 的新轴**:V2 表明,对开放问题,scale 的方向从「多采样投票」转向「生成→验证→改进」的迭代自审。

## 溯源

- DeepSeekMath:https://arxiv.org/abs/2402.03300
- DeepSeekMath-V2:https://arxiv.org/abs/2511.22570
- 仓库:https://github.com/deepseek-ai/DeepSeek-Math 、 https://github.com/deepseek-ai/DeepSeek-Math-V2

## 相关条目(库内互链)

- 底座结构:DeepSeek-V3 的架构见 [model-architecture/concepts/deepseek-arch-v3.md](../../model-architecture/concepts/deepseek-arch-v3.md);DeepSeekMoE 结构见 [deepseek-arch-v1-moe-v2.md](../../model-architecture/concepts/deepseek-arch-v1-moe-v2.md)。
- 定理证明谱系(与 Math-V2 的自然语言证明互补的 Lean 形式化路线):[deepseek-prover.md](deepseek-prover.md)。
- GRPO 后续被 R1 复用为推理 RL 核心:见 [model-architecture/concepts/deepseek-arch-v32-r1.md](../../model-architecture/concepts/deepseek-arch-v32-r1.md)。
