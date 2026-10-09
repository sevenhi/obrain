---
title: DeepSeek 代码模型谱系 — Coder → Coder-V2(FIM 填充 / MoE 代码模型)
type: concept
tags: [deepseek, code, fim, fill-in-the-middle, moe, code-model, humaneval, repo-level]
series: deepseek-coder
source:
  - "pdf: DeepSeek-Coder | arXiv:2401.14196 | https://arxiv.org/abs/2401.14196 | 2026-10-09 | ↑primary"
  - "pdf: DeepSeek-Coder-V2 | arXiv:2406.11931 | https://arxiv.org/abs/2406.11931 | 2026-10-09 | ↑primary"
  - "code: DeepSeek-Coder | https://github.com/deepseek-ai/DeepSeek-Coder"
  - "code: DeepSeek-Coder-V2 | https://github.com/deepseek-ai/DeepSeek-Coder-V2"
date: 2026-10
---

# DeepSeek 代码模型谱系:Coder → Coder-V2(FIM 填充 / MoE 代码模型)

> **TL;DR**:DeepSeek 代码模型两代,走了一条「从 dense 专项 → MoE 通用」的路:**DeepSeek-Coder**(2024)是 1B~33B 的 dense 代码专项模型,从零训 2T tokens(87% 代码),杀手锏是 **FIM(fill-in-the-middle)填充**——不是只往后补,而是「中间挖洞、两头给上下文」的项目级补全;**DeepSeek-Coder-V2**(2024)换成 **DeepSeekMoE 底座**(16B/236B,激活 2.4B/21B),续训 6T tokens,编程语言从 86→338,上下文 16K→128K,HumanEval 90.2 打平 GPT-4-Turbo。核心结论:**代码模型的核心差异在「训练数据组织 + FIM 任务设计」,结构复用主模型(MoE)后反而更强、更省算力。**

---

## 1. DeepSeek-Coder:从零的 dense 代码专项

### 1.1 定位与规模

DeepSeek 第一个代码模型,从零训练,规模 1B / 5.7B / 6.7B / 33B。核心是**项目级(project-level)代码补全**能力,而非单文件续写。

### 1.2 数据组织(为什么「项目级」)

代码不是孤立的单文件——函数定义可能在别的文件里。Coder 的数据管道四步(可复现):

```
① 从 GitHub 采代码,套用 StarCoder Data 的过滤规则
② 解析同一仓库内文件的依赖关系,按依赖重排文件顺序
③ 把有依赖关系的文件拼成一个训练样本,用 repo 级 minhash 去重
④ 再滤掉低质量代码(语法错误、可读性差的)
```

**机制要点**:`依赖重排 + 多文件拼接` 让模型在训练时就看到「跨文件上下文」,这是「项目级补全」能力的数据来源——单文件语料只能训出「单文件续写」。

### 1.3 训练配方

- 数据配比:87% 代码 + 10% 代码相关语言(GitHub Markdown、StackExchange)+ 3% 非代码中文。
- **两阶段**:先 1.8T tokens + 4K 窗口预训练,再追加 200B tokens + 16K 窗口(**长上下文续训**),得到 Base。
- 指令微调:2B tokens instruction data → Instruct。

### 1.4 FIM:fill-in-the-middle 填充(核心机制)

这是 Coder 相对纯自回归代码模型的核心差异。传统自回归只能「给定前缀补后缀」;FIM 支持**中间挖洞**——给定洞的前后文,填中间。

**训练格式**(特殊 token 标记洞):

```
<｜fim▁begin｜> 前缀代码 <｜fim▁hole｜> 目标(被挖掉的中间) <｜fim▁end｜> 后缀代码
```

推理时:给 `<｜fim▁begin｜>前缀<｜fim▁hole｜>后缀`,模型生成中间部分(如 README 示例:补出 `for i in range(1, len(arr)):` 那行)。

**机制**:FIM 是**改训练目标**——让模型学会「根据上下文双向信息生成中间片段」,这是 IDE 里「光标在中间补代码」场景(代码插入/重构)的刚需。DeepSeek-Coder 是早期把 FIM 做进开源代码模型的开创者之一。

### 1.5 结果(有据数字)

- DeepSeek-Coder-Base-33B 显著超过当时开源代码模型;比 CodeLlama-34B 在 HumanEval Python/Multilingual、MBPP、DS-1000 上分别领先 7.9% / 9.3% / 10.8% / 5.9%。
- **DeepSeek-Coder-Base-7B 达到 CodeLlama-34B 的水平**——「小模型打大模型」。
- Instruct-33B 在 HumanEval 超 GPT-3.5-turbo,MBPP 打平。
- 支持 86 种编程语言。

---

## 2. DeepSeek-Coder-V2:MoE 底座的通用代码模型

### 2.1 定位

V2 的转变:**放弃 dense 专项,改用 DeepSeek-V2 的 MoE 底座**(基于 DeepSeekMoE 框架),从 DeepSeek-V2 的中间 checkpoint 续训 6T tokens。结果:代码能力大涨,同时**保留通用语言能力**(不再「只会写代码」)。

### 2.2 规格

| 模型 | 总参 | 激活参 | 上下文 |
|---|---|---|---|
| Coder-V2-Lite-Base/Instruct | 16B | 2.4B | 128K |
| Coder-V2-Base/Instruct | 236B | 21B | 128K |

关键变化:编程语言 86→**338**;上下文 16K→**128K**。

### 2.3 结果(有据数字)

- **代码生成**(HumanEval):V2-Instruct 90.2,超过 GPT-4-Turbo-1106(87.8),打平 GPT-4o-0513(91.0);MBPP+ 76.2 最高。
- **代码修复**(SWE-Bench/Aider):V2-Instruct SWE-Bench 12.7、Aider 73.7,超 CodeStral 等开源,接近/超部分闭源。
- **数学推理**:GSM8K 94.9、MATH 75.7——代码模型的数学能力也被 MoE 底座带起来了。
- **通用 NLU**:MMLU 79.2、BBH 83.9——相对 V2-Lite 全面上涨,证明「通用+代码」双强。
- 128K 上下文经 NIAH 测试全长度表现良好。

### 2.4 MoE 底座的意义(机制)

V2 证明了一件事:**代码能力不必靠专用 dense 模型,直接续训通用 MoE 底座反而更强**。原因:

- MoE 的**稀疏激活**(236B 只激活 21B)让大容量代码模型在**推理算力**上可负担——代码生成是高频、低延迟场景,激活参数小 = 快且省。
- **通用底座的知识迁移**:从 DeepSeek-V2 续训,继承了通用语言/推理能力,所以 V2 不再「只会代码」,数学/通用 NLU 同步上涨。
- 这与数学谱系(Math 从 Coder 底座继承代码能力)方向相反却逻辑一致:**底座越通用,领域越省事**。

---

## 3. 谱系关系与演进逻辑

| 维度 | Coder(2024) | Coder-V2(2024) |
|---|---|---|
| 底座 | 从零 dense | DeepSeek-V2 MoE 续训 |
| 规模 | 1B~33B dense | 16B/236B MoE(激活 2.4B/21B) |
| 语言数 | 86 | 338 |
| 上下文 | 16K | 128K |
| 核心创新 | FIM 填充 + 项目级数据 | MoE 通用代码模型 |
| HumanEval | 33B ~79(开源 SOTA) | **90.2**(打平 GPT-4-Turbo) |

**演进逻辑一句话**:代码模型从「dense 专项(FIM/数据工程立身)」走向「MoE 通用(底座复用 + 规模)」——FIM/数据组织是 Coder 时代的差异化能力,到 V2 变成「标配 + MoE 规模红利」。

---

## 实践要点

- **FIM 是代码补全的必修课**:只要做「中间补全/插入/重构」场景,就得训 FIM(特殊 token 挖洞 + 前后文条件生成),纯自回归补全覆盖不了。IDE 集成务必用 FIM 版模型。
- **项目级数据组织**:按「依赖重排 + 多文件拼接 + repo 级去重」组织代码语料,是「跨文件补全」能力的来源;单文件语料训不出这个。
- **代码模型推理用 vLLM/SGLang**:Coder/Coder-V2 官方都给了 vLLM 集成示例;MoE 代码模型用 SGLang 的 MLA 优化 + FP8 KV cache 可拿到最好延迟/吞吐。
- **底座选择**:要纯代码专项且算力敏感 → dense(7B 就很能打);要「通用+代码」双强 → 直接续训 MoE 底座(激活参数小,推理反而便宜)。

## 溯源

- DeepSeek-Coder:https://arxiv.org/abs/2401.14196
- DeepSeek-Coder-V2:https://arxiv.org/abs/2406.11931
- 仓库:https://github.com/deepseek-ai/DeepSeek-Coder 、 https://github.com/deepseek-ai/DeepSeek-Coder-V2

## 相关条目(库内互链)

- MoE 底座结构(DeepSeekMoE 细粒度+共享专家):见 [model-architecture/concepts/deepseek-arch-v1-moe-v2.md](../../model-architecture/concepts/deepseek-arch-v1-moe-v2.md);本域 [deepseek-moe.md](deepseek-moe.md) 补数据/评测。
- 推理引擎(官方推荐 vLLM/SGLang):[vllm/index.md](../../vllm/index.md)。
- 数学谱系(DeepSeekMath 底座就是 Coder-v1.5):[deepseek-math.md](deepseek-math.md)。
