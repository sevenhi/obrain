---
title: DeepSeek-R1 推理谱系 — 训练策略视角(R1-Zero → R1 → 蒸馏)
type: concept
tags: [deepseek, r1, reasoning, grpo, rl, cold-start, rejection-sampling, distillation, cot, aha-moment, domain-model]
series: deepseek-r1
source:
  - "pdf: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | arXiv:2501.12948 | https://arxiv.org/abs/2501.12948 | 2026-10-14 | ↑primary"
  - "code: deepseek-ai/DeepSeek-R1(README:模型表/评测表/使用建议)| https://github.com/deepseek-ai/DeepSeek-R1 | 2026-10-14"
date: 2026-10
---

# DeepSeek-R1 推理谱系:训练策略视角(R1-Zero → R1 → 蒸馏)

> **TL;DR**:DeepSeek-R1 回答的问题是「**推理能力能不能不靠人工标注、纯靠强化学习(RL)逼出来?**」——答案是能。三步谱系:**① R1-Zero**:直接在 DeepSeek-V3-Base 上做大规模 RL(GRPO + 纯规则 reward),不用任何 SFT,AIME 2024 从 15.6% 一路涨到 71.0%,并自发涌现反思、长思维链(CoT)和「aha moment」,但可读性差、语言混杂;**② R1**:四阶段流水线(冷启动 SFT → 推理 RL → 拒采样 SFT → 全场景 RL),修好可读性,做到 AIME 79.8%、MATH-500 97.3%,对标 OpenAI-o1-1217;**③ 蒸馏**:用 R1 生成的 800k 样本对小 dense 模型只做 SFT,Distill-Qwen-32B 在 AIME 上 72.6% 反超 o1-mini,且**蒸馏打赢了对同规模模型直接做 RL**(72.6 vs 47.0)。**核心结论:推理 RL 的事实标准配方 = GRPO + 规则 reward + 冷启动;小模型学推理,抄大模型的推理轨迹比自己 RL 更划算。**
>
> **视角分工(先读这条)**:本篇讲 R1 的**训练策略/RL 谱系**(怎么训出来的)。R1 **没有任何新网络结构**(底座就是 V3-Base 的 MLA+MoE),结构侧解读见 [model-architecture/concepts/deepseek-arch-v32-r1.md](../../model-architecture/concepts/deepseek-arch-v32-r1.md)(该篇结论同样是「R1 无新结构,是 V3-Base 上的纯 RL 推理时扩展」),两篇不重复。

---

## 1. 为什么需要 R1:推理 RL 的动机(新手向)

在 R1 之前,让模型「会推理」的主流做法是 **SFT(监督微调)**:人工收集/标注一批带推理过程的示范答案,让模型模仿。这条路有三个痛点:

1. **标注贵**:高质量推理示范(数学/代码解题过程)要人写,规模上不去;
2. **模仿有天花板**:模型只能学到示范者会的套路,学不出示范里没有的策略;
3. **过程难定义**:「哪一步推理是对的」很难细粒度标注(这正是论文放弃 PRM 过程奖励模型的原因,见 §7)。

R1 的赌注是:**只给模型「最终答案对/错」这种便宜的规则信号,让它自己在海量试错中发明推理策略**。这就像不给学生看例题、只批改最终答案,学生自己摸索出解题套路——前提是试错次数(RL 步数)足够多。

**谱系位置**:这套「规则 reward + RL」的算法基础(GRPO)是 DeepSeekMath 先提出的(见 [deepseek-math.md](deepseek-math.md) §1.3),R1 把它从「数学 7B 模型的 RL 收尾步骤」放大成「671B 主模型的核心训练范式」,并验证了**不用 SFT 冷启动也能涌现推理**。

---

## 2. 算法底座:GRPO(R1 论文公式 (1)–(3))

下图是一个 GRPO 训练步的完整数据流:同题采样 G 个回答 → 各自算规则 reward → 组内标准化成 Aᵢ → 裁剪目标 + KL。整条路径里没有 critic,这就是它能在 671B 上跑起来的原因。

<div class="archify-figure">
<iframe src="../../../diagrams/dm-r1-grpo-step.workflow.html"
        style="width:100%;height:730px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="GRPO 单个训练步的数据流"></iframe>
</div>


GRPO(Group Relative Policy Optimization)与 [deepseek-math.md](deepseek-math.md) 篇口径一致:**去掉 PPO 的 critic(价值函数),用「同一问题的 G 个回答组内标准化」做优势估计**。R1 论文 §2.2.1 原文:

- 对每个问题 q,从旧策略 π_old 采样一组输出 {o₁…o_G},每个得 reward rᵢ;
- 优势:`Aᵢ = (rᵢ − mean({r₁…r_G})) / std({r₁…r_G})`(公式 (3))——「比组内平均好多少」;
- 目标(公式 (1)):PPO 同款裁剪目标 + KL 惩罚:`(1/G) Σᵢ [ min(ratioᵢ·Aᵢ, clip(ratioᵢ, 1−ε, 1+ε)·Aᵢ) − β·D_KL(π_θ ‖ π_ref) ]`,其中 ratioᵢ = π_θ(oᵢ|q)/π_old(oᵢ|q);
- KL 用无偏估计(公式 (2)):`D_KL = π_ref(oᵢ|q)/π_θ(oᵢ|q) − log(π_ref/π_θ) − 1`。

**为什么省**:critic 通常和策略模型一样大,671B 场景下养一个 671B critic 内存直接翻倍;数学/代码 reward 只有「对/错」,不需要逐状态估值——组内相对比较就够了。

### 2.1 GRPO 训练循环伪代码(按 R1 论文 §2.2 整理)

```python
# GRPO 训练循环伪代码(参考 DeepSeek-R1 论文公式 (1)–(3))
for step in range(num_rl_steps):
    prompts = sample_prompts(train_set)        # 只用答案可规则判定的题(数学/代码/逻辑)
    for q in prompts:
        outputs = pi_old.sample(q, G)          # 旧策略采样 G 个回答
        rewards = []
        for o in outputs:
            r  = accuracy_reward(o)            # 正确性:如 \boxed{} 内答案匹配、编译器跑测试用例
            r += format_reward(o)              # 格式:思考过程必须在 <think>…</think> 内
            rewards.append(r)
        A = (rewards - mean(rewards)) / std(rewards)   # 组内标准化 = 优势(替代 critic)

    ratio = pi_theta(outputs) / pi_old(outputs)        # token 级重要性比
    loss  = -mean(min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A))
    loss += beta * (pi_ref / pi_theta - log(pi_ref / pi_theta) - 1).mean()  # KL(公式 2)
    loss.backward()               # 逐问题累积梯度(记:此处 A 以组内标准化标量呈现,实作中按 token 广播)
optimizer.step()                  # 一个 step 内所有问题累积完再更新
```

### 2.2 Reward 设计:纯规则,不用神经 reward model(论文 §2.2.2)

- **accuracy reward**:数学题要求把最终答案放进指定格式(如 `\boxed{}`)以便规则核对;LeetCode 类题用编译器 + 预定义测试用例给反馈;
- **format reward**:强制思考过程写在 `<think>`/`</think>` 标签之间;
- **明确不用**结果/过程神经 reward model——大规模 RL 下神经 RM 会被 reward hacking,且重训 RM 增加流水线复杂度(原文 §2.2.2)。

### 2.3 训练模板(论文 Table 1)

只约束结构、不注入内容先验(不强制「必须反思」或特定解题策略),以便观察 RL 自然演化:

```
<think> reasoning process here </think><answer> answer here </answer>
```

---

## 3. R1-Zero:纯 RL、零 SFT 的对照实验(论文 §2.2)

下图把三条 reward 路线并列:规则 reward(R1 选的)、神经 reward model(明确不用)、PRM / MCTS(试过但放弃)。每组下面标出具体的失效机制——读懂这张图,就懂了 R1 为什么长成这样。

<div class="archify-figure">
<iframe src="../../../diagrams/dm-r1-reward-choices.tree.html"
        style="width:100%;height:820px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="训练信号的三种来源与取舍"></iframe>
</div>


**做法**:DeepSeek-V3-Base 直接上 GRPO,不做任何 SFT。这是「推理能否纯靠 RL 激励出来」的第一个公开验证。

**结果(有据数字,论文 Table 2 / Figure 2)**:

| 指标 | 数值 | 对照 |
|---|---|---|
| AIME 2024 pass@1 | **15.6% → 71.0%**(RL 过程中持续上涨) | OpenAI-o1-0912 = 74.4 |
| AIME 2024 cons@64(多数投票) | **86.7%** | 超过 o1-0912 的 83.3 |
| MATH-500 pass@1 | 95.9% | o1-0912 = 94.8 |
| GPQA Diamond | 73.3% | o1-mini = 60.0 |
| LiveCodeBench | 50.0% | o1-mini = 53.8 |
| Codeforces rating | 1444 | o1-mini = 1820 |

> 对照基线说明:表中 GPQA/LiveCodeBench/Codeforces 用 o1-mini,AIME/MATH-500 用 o1-0912——两条基线混用。若统一对更强的 **o1-0912**(GPQA 77.3 / LiveCodeBench 63.4 / Codeforces 1843),R1-Zero 在这三项上**仍落后**,只在 AIME/MATH 的 pass@1 口径上打平或略超;真正全面超过 o1 的是后续完整 R1(+冷启动 SFT 后)。读表时留意「对照强基线则未全胜」。

**涌现行为(论文 §2.2.4)**——这部分是 R1 论文最有影响力的观察:

- **思考时间自发变长**:训练中平均响应长度持续增长,模型自己学会「多想一会儿」(test-time compute 扩展);
- **反思(reflection)**:自发回头重估之前的步骤、换解法;
- **aha moment**:中间版本模型在 CoT 里写出 *"Wait, wait. Wait. That's an aha moment I can flag here."*(论文 Table 3,§2.2.4)然后重估解题路径——没人教它这么做,是 RL 激励出的元认知行为。

**缺陷(为什么还需要 R1)**:无穷重复、可读性差、语言混杂(中英夹杂)——纯 RL 只管「对」,不管「好看」。

---

## 4. R1 正式流水线:四阶段(论文 §2.3)

下图是四阶段的状态转移,右侧旁支是 R1-Zero:它只走第 ② 步,所以能力到但话说不清。每次 SFT/RL 交替都在补上一步的短板——这是 R1 与 R1-Zero 的全部差别。

<div class="archify-figure">
<iframe src="../../../diagrams/dm-r1-four-stages.lifecycle.html"
        style="width:100%;height:820px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="R1 四阶段流水线与 R1-Zero 旁支"></iframe>
</div>


| 阶段 | 做什么 | 关键数字/细节 |
|---|---|---|
| ① 冷启动 SFT | 数千条长 CoT 数据微调 V3-Base,作为 RL 起点 | 输出格式 `\|special_token\|<reasoning_process>\|special_token\|<summary>`;人写可读 pattern(带结尾摘要),过滤不友好回答 |
| ② 推理向 RL | 与 R1-Zero 同款大规模 GRPO,专注数学/代码/科学/逻辑 | 新增**语言一致性 reward**(CoT 中目标语言词占比,直接加到 accuracy 上);消融显示略降性能但更可读 |
| ③ 拒采样 + SFT | 用②的 checkpoint 拒采样生成新 SFT 数据,重训 V3-Base | 推理数据 ~600k(多采样留对的;引入 DeepSeek-V3 做生成式 reward 判分)+ 非推理数据 ~200k(写作/事实 QA/自认知/翻译,复用 V3 管道);共 ~800k 样本训 **2 个 epoch** |
| ④ 全场景 RL | 第二次 RL,兼顾推理 + 人类偏好对齐 | 推理数据仍用规则 reward;通用数据用 reward model。**helpfulness 只评 summary**(不干扰推理过程),**harmlessness 评全文**(含 CoT) |

**与 R1-Zero 的差异一句话**:R1-Zero 证明「纯 RL 可行」,R1 用「少量高质量冷启动 + 迭代式 RL/SFT 交替」换来更好的收敛起点、可读性和通用能力。冷启动的价值(论文 §2.3.1):防 RL 早期不稳定、可读性、以及人类先验带来的性能潜力。

**R1 最终成绩(README 评测表,pass@1,温度 0.6 采样 64 次)**:AIME 2024 **79.8**(o1-1217 = 79.2)、MATH-500 **97.3**、LiveCodeBench **65.9**、Codeforces **96.3 百分位 / 2029 rating**、MMLU 90.8、AlpacaEval2.0 **87.6**、ArenaHard **92.3**——数学/竞赛代码对标 o1-1217,工程向代码(SWE Verified 49.2、Aider 53.3)仍落后。

---

## 5. 蒸馏:小模型的推理能力从哪来(论文 §2.4、§4.1)

下图是同底座 Qwen-32B 的两条路对照:自己跑 10K+ 步 RL 得 47.0,吃 R1 的 800k 轨迹只做 SFT 得 72.6。差的 25.6 个点说明「推理模式」本身就是资产。

<div class="archify-figure">
<iframe src="../../../diagrams/dm-r1-distill-vs-rl.workflow.html"
        style="width:100%;height:730px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="蒸馏 vs 直接 RL:同底座的对照"></iframe>
</div>


**做法**:用 R1 当 teacher 生成 800k 推理样本,对 6 个开源 dense 模型**只做 SFT、不做 RL**(Qwen2.5-Math-1.5B/7B、Qwen2.5-14B/32B、Llama-3.1-8B、Llama-3.3-70B-Instruct)。

**结果(README 蒸馏评测表)**:

- Distill-Qwen-32B:AIME **72.6** / MATH-500 94.3 / LiveCodeBench 57.2,超 o1-mini(63.6)且创 dense 模型纪录;
- Distill-Qwen-7B:AIME 55.5,超过 QwQ-32B-Preview(44.0);Distill-14B(69.7)大幅领先 QwQ-32B;
- Distill-Qwen-1.5B:AIME 28.9,数学上超 GPT-4o(9.3)和 Claude-3.5-Sonnet(16.0)。

**蒸馏 vs RL 的直接对照(论文 Table 6,新手最值得记的实验)**:对 Qwen-32B-Base 直接做 10K+ 步大规模 RL(数学/代码/STEM 数据)得到 R1-Zero-Qwen-32B,AIME 只有 **47.0**;而同底座吃 R1 蒸馏数据的 Distill-Qwen-32B 是 **72.6**。论文结论:

1. 小模型自己跑大规模 RL 费算力且打不过蒸馏——**大模型发现的推理模式才是关键资产**;
2. 蒸馏经济有效,但**突破智能边界仍要靠更强底座 + 大规模 RL**(蒸馏只能继承,不能超越)。

---

## 6. 上手指南(本地跑 Distill 模型,README §6)

R1/R1-Zero 本体是 671B MoE,本地跑走 DeepSeek-V3 仓库的推理栈;**新手从 Distill 系列入手**(用法同 Qwen/Llama):

```bash
# vLLM 起服务(README 原命令)
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager
```

**官方使用建议(README §6 Usage Recommendations,直接影响发挥)**:

1. temperature 设 **0.5–0.7(推荐 0.6)**——太低会无穷重复/语无伦次;
2. **不要加 system prompt**,所有指令放 user prompt;
3. 数学题加一句 `Please reason step by step, and put your final answer within \boxed{}.`;
4. 评测跑多次取平均;模型偶尔会输出空 `<think>\n\n</think>` 跳过思考——**强制以 `<think>\n` 开头**可保证进入推理模式。

---

## 7. 局限与失败尝试(论文 §4.2、§5)

- **工程向代码弱**:SWE Verified / Aider 落后 o1-1217,官方归因「相关 RL 训练数据还很少」;
- **PRM(过程奖励模型)被放弃**,三个理由(原文):① 通用推理的「细粒度步骤」难定义;② 判定中间步对错本身很难(模型标注不可靠、人工标注不可扩展);③ 模型化 PRM 必然被 reward hacking,重训 RM 又贵。PRM 只适合做 top-N 重排/引导搜索,不值得进大规模 RL 主循环;
- **MCTS(蒙特卡洛树搜索)没跑通**:token 生成的搜索空间指数级大于棋类;细粒度 value model 难训,难以迭代自举;
- **蒸馏的天花板**:蒸馏模型只有 SFT、没加 RL(官方留白给社区),且能力上限被 teacher 锁死。

---

## 8. 谱系小结与互链

下图是本文谱系小结的可视化版本:GRPO 来自 DeepSeekMath → R1-Zero 验证纯 RL → R1 四阶段补可读性 → 蒸馏压进 dense 模型;下方标出被放弃的 PRM / MCTS 两条路。

<div class="archify-figure">
<iframe src="../../../diagrams/dm-r1-lineage.timeline.html"
        style="width:100%;height:740px;border:1px solid rgba(0,0,0,.08);border-radius:14px"
        loading="eager" title="R1 推理谱系时间轴"></iframe>
</div>


```
DeepSeekMath(2024,提出 GRPO,数学 7B RL)
        │  GRPO + 规则 reward 被复用并放大
        ▼
DeepSeek-R1-Zero(V3-Base + 纯 RL,验证「无 SFT 也能推理」)
        ▼
DeepSeek-R1(冷启动 → 推理 RL → 拒采样 SFT → 全场景 RL,对标 o1-1217)
        │  800k 蒸馏样本,只 SFT
        ▼
R1-Distill 系列(1.5B–70B dense,32B 超 o1-mini)
```

- **GRPO 起源与公式推导细节**:[deepseek-math.md](deepseek-math.md)(§1.3「去掉 critic 的 PPO」)——本篇与其完全同口径:DeepSeekMath 提出、R1 复用为推理 RL 事实标准。
- **结构视角(R1 为什么不需要新结构、V3.2 DSA 与 R1 的关系)**:[model-architecture/concepts/deepseek-arch-v32-r1.md](../../model-architecture/concepts/deepseek-arch-v32-r1.md);V3-Base 底座结构(MLA/256 专家 MoE/MTP)见 [deepseek-arch-v3.md](../../model-architecture/concepts/deepseek-arch-v3.md)。
- **数学 RL 的另一支(自验证)**:[deepseek-math.md](deepseek-math.md) §2 的 Math-V2——R1 用规则 reward 判「答案对」,Math-V2 训 verifier 判「证明对」,互补。
- **形式化定理证明的 RL(RLPAF/MCTS 成功用例)**:[deepseek-prover.md](deepseek-prover.md)——注意 R1 论文里 MCTS 失败是指「通用推理 + 自举训练」场景,Prover 的 RMaxTS 是「有编译器反馈的形式化」场景,不矛盾。
- **推理产物的部署**(Distill 模型 vLLM/SGLang 服务):[../../vllm/index.md](../../vllm/index.md)。

## 溯源

- 论文(本地 `DeepSeek_R1.pdf` 全文核对):https://arxiv.org/abs/2501.12948 — GRPO 公式 (1)–(3) §2.2.1、reward 设计 §2.2.2、模板 Table 1、R1-Zero 成绩 Table 2、aha moment Table 3(§2.2.4)、四阶段 §2.3、蒸馏 §2.4、**蒸馏 vs RL 对照 Table 6(§4.1 Discussion)**、失败尝试 §4.2
- 仓库(本地 README 核对):https://github.com/deepseek-ai/DeepSeek-R1 — 模型表(671B/37B/128K)、R1 评测表、蒸馏评测表、使用建议、vLLM/SGLang 命令
- 模型下载:https://huggingface.co/deepseek-ai/DeepSeek-R1 、 https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
