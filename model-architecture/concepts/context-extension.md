---
title: 上下文扩展机制:RoPE 外推为何失败与 PI/NTK/YaRN/LongRoPE
type: concept
tags: [context-extension, rope, position-interpolation, pi, ntk-aware, yarn, longrope, long-context, llama, extrapolation, attention, frequency]
source:
  - "pdf: Extending Context Window of Large Language Models via Position Interpolation (PI, primary) | arXiv:2306.15595 (Chen et al., Meta AI, 2023-06) | https://arxiv.org/abs/2306.15595"
  - "pdf: YaRN: Efficient Context Window Extension of Large Language Models | arXiv:2309.00071 (Peng et al., Nous Research/EleutherAI, 2023-08) | https://arxiv.org/abs/2309.00071"
  - "pdf: LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens | arXiv:2402.13753 (Ding et al., Microsoft Research, 2024-02) | https://arxiv.org/abs/2402.13753"
  - "pdf: RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE 原始定义) | arXiv:2104.09864 (Su et al. 2021) | https://arxiv.org/abs/2104.09864"
  - "pdf: The Llama 3 Herd of Models (θ=500k + 六阶段渐进长上下文的工业实践) | arXiv:2407.21783 | https://arxiv.org/abs/2407.21783"
  - "code: jquesnelle/yarn (YaRN 官方实现与 Llama-2 7B/13B 64k/128k checkpoint) | https://github.com/jquesnelle/yarn"
  - "code: microsoft/LongRoPE (LongRoPE 官方实现) | https://github.com/microsoft/LongRoPE"
date: 2026-09
---

# 上下文扩展机制:RoPE 外推为何失败与 PI/NTK/YaRN/LongRoPE

## TL;DR

RoPE 用「按位置旋转 Q/K」编码相对距离,但**旋转角频率 θ_d = base^(−2d/D) 在训练时只见过 [0, L) 内的相位分布**——位置超过训练长度后,低频维度的旋转角落入模型从未见过的区间,attention 分数出现灾难性离群值,直接外推会让模型崩坏(PI 论文实测:LLaMA 在超出训练长度的位置 PPL 爆炸)。朴素长微调也无效:直接加长微调 10000+ batch,有效窗口只从 2048 爬到 2560。正确做法是**改位置映射而非硬外推**,四代方法逐级精细化:① **PI(位置插值,Meta 2023-06)**:把新位置线性压缩回旧区间 `m′ = m·L/L′`,理论上界比外推小 ~600×,1000 步微调即可把 LLaMA 扩到 32k;② **NTK-aware**(社区,bloc97):不均匀缩放——高频维度基本不动、低频维度大幅拉伸,等价于调大 RoPE base,**零微调**就能用(代价是精度);③ **YaRN(Nous 2023-08)**:NTK-by-parts 按波长分段处理(r<α 的维外推、r>β 的维插值、中间线性过渡,Llama 系 α=1/β=32)+ 注意力温度补偿(RoPE 嵌入乘 √t),**比 PI 少 10× tokens、2.5× 步数**扩到 128k 且 PPL 更优;④ **LongRoPE(MS 2024-02)**:进化搜索逐维 rescale 因子(非均匀插值)+ 渐进两阶段(256k→2048k),≤1k 微调步扩到 **2M tokens**,再用 8k 回调恢复短窗口性能。工业主线另有一条:从头/继续预训练直接改 base(Llama 3:θ 10k→500k,8k→128k 六阶段渐进 ~800B tokens)——**有算力就改预训练,没算力就 YaRN/LongRoPE 微调**。

---

## 1. 机制:RoPE 为什么难外推

### 1.1 RoPE 的频率结构(回顾)

RoPE(数学推导见 [llm-architecture](llm-architecture.md) §4.3)把 head_dim 维 D 切成 D/2 个二维平面,第 d 个平面按位置 m 旋转角度 mθ_d:

```
θ_d = base^(−2d/D)          # base 常取 10000
波长 λ_d = 2π·base^(2d/D)   # 转一整圈需要多少 token
```

- **低维 d 小 → θ 大 → 波长短**(高频,分辨相邻位置);**高维 d 大 → θ 小 → 波长长**(低频,编码远程距离)。base=10000、D=128 时最长波长 ≈ 2π·10⁴ ≈ 62.8k tokens——**大部分维度的波长远超 2k/4k 的训练窗口**,即训练时这些维度连一整圈都没转过,只见过相位 [0, m·θ_d) 的一个小区间;
- attention 分数 q·k 依赖旋转角的**三角函数**,模型对相位的响应是在训练见过的相位分布上拟合出来的。位置超出 L 后,低频维相位进入**从未见过的角度区间**,内积出现训练分布外的值——PI 论文称之为「catastrophically high attention scores that completely ruin the self-attention」,softmax 被离群 logit 支配,输出崩坏;
- PI 的理论分析:attention logit 的增长上界,**插值至少比外推小 ~600×**——这就是「压缩回熟悉区间」优于「闯进陌生区间」的数学根据。

### 1.2 为什么朴素微调不行

PI 论文 Table 4 实测:把 LLaMA 直接在长序列上微调 **10000+ batch,有效上下文只从 2048 → 2560**。原因:新位置的旋转角仍是分布外输入,模型要靠大量数据重新学「陌生相位的响应」,收敛极慢。**必须先改位置映射,把问题从「学新相位」变成「重标定旧相位」,微调才有效**(PI 只需 ~1000 步)。

### 1.3 四代方法的统一视角

所有方法都在改两个函数(YaRN 论文 §2.3 的统一记号):位置映射 `g(m)` 与频率缩放 `h(θ_d)`:

```
旋转角 = g(m) · h(θ_d)

PI:          g(m) = m·L/L′        h(θ) = θ            # 均匀压缩位置,频率不动
NTK-aware:   g(m) = m             h(θ_d) = θ_d/s^(...)  # 位置不动,改 base:
             base′ = base·s^(D/(D−2))                   #   s=L′/L,高频维几乎不缩、低频维大幅缩
NTK-by-parts(YaRN): h(θ_d) = (1−γ_d)·θ_d/s + γ_d·θ_d   # 按波长分段:
             γ_d = ramp(r_d; α, β),  r_d = L/λ_d        #   r<α(高频,窗口内转过多圈):外推不插值
                                                        #   r>β(低频,连一圈都没转够):纯插值
                                                        #   中间:线性过渡;Llama 系 α=1, β=32
YaRN 额外:   q,k 嵌入乘 √t(t>1)  # 注意力温度补偿:插值拉长上下文后
                                  # softmax 分布熵下降(过度尖锐),放大 logit 恢复熵;
                                  # 用「RoPE 嵌入长度缩放 √t」实现,免改 attention 代码
LongRoPE:    h(θ_d) = 逐维搜索出的 rescale 因子  # 进化搜索,不再用解析式
             + 前 n̂ 个 token 位置不做插值(n̂∈{0,1,2,4,…,256},Finding 2)(利用「前 n̂ 个 token 位置不插值可保护短窗口性能(Finding 2)」的非均匀性①)
             + 渐进:先 256k 微调,再二次搜索因子外推到 2048k(非均匀性②:免微调可扩 8×)
             + 8k 回调:再搜一组因子恢复短窗口性能,推理按序列长度选因子
```

直觉对比:**PI 一刀切**(所有维度均匀压缩,高频维被过度压缩→局部分辨率受损);**NTK/YaRN 认识到「高频维管近距离、低频维管远距离」,应该区别对待**——近距离信息(高频)不该被压缩,只有远程相位(低频)需要拉伸;**LongRoPE 干脆不信任何解析公式,逐维搜出来**。

---

## 2. 可复现细节

### 2.1 PI(最小实现)

```python
# 原 RoPE: angle = pos * theta_d ,  theta_d = base^(-2d/D)
# PI (扩到 L' = s·L): 位置下标先除以 s,再照常算
pos_pi = pos / s                    # s = L'/L,如 2048→32768 时 s=16
angle = pos_pi * theta_d            # 全部旋转角落回训练时见过的 [0, L·θ_d) 区间
# 然后照常微调 ~1000 步(PI 论文:LLaMA 7B~65B 扩 32k 的设置)
```

HF Transformers 的 `rope_scaling={"type": "linear", "factor": s}` 即 PI。

### 2.2 NTK-aware(免微调)

```python
# 等价于把 base 调大:高频维几乎不变、低频维波长拉长
base_new = base * s ** (D / (D - 2))    # s = L'/L
theta_d = base_new ** (-2*d/D)
# 例:base=10000, D=128, s=2 → base_new ≈ 10000·2^(128/126) ≈ 20220
```

HF `rope_scaling={"type": "dynamic"}` 是它的动态版(推理时按当前长度实时算 s)。Code Llama(100k)用的是 NTK 路线(YaRN 论文 Table 1 中以 "NTK" 对照)。

### 2.3 YaRN(NTK-by-parts + 温度)

```python
r_d = L / lambda_d                       # 该维度在训练窗口内转了几圈
gamma_d = clip((r_d - alpha) / (beta - alpha), 0, 1)   # alpha=1, beta=32 (Llama 系)
theta_new_d = (1 - gamma_d) * theta_d / s + gamma_d * theta_d
# 温度:q,k 的 RoPE 输出乘 sqrt(t);论文拟合 t ≈ 0.1·ln(s) + 1
#    (s=16 时 √t ≈ 1.13,即 logit 放大约 13%)
```

官方实现与 checkpoint:[github.com/jquesnelle/yarn](https://github.com/jquesnelle/yarn)(Llama-2 7B/13B 的 64k/128k YaRN 版权重)。HF `rope_scaling={"type": "yarn", "factor": s}`。

### 2.4 工业路线:改 base + 渐进继续预训练(Llama 3)

```
θ: 10000 → 500000          # 最长波长 62.8k → ~3.1M,给 128K 留足相位空间
context: 8192(初始预训练)→ 六阶段渐进拉到 131072
长上下文档:~800B tokens(占总 15T 的 ~5%),混入长文档比例逐步提高
```

(Llama 3 论文 §3.2;细节与数字见 [llm-architecture](llm-architecture.md) §5。)θ 调大 = NTK-aware 的「训练时版」:与其事后插值,不如预训练就让低频维见过足够相位。

---

## 3. 数据对比

### 3.1 YaRN 论文 Table 1(滑窗 PPL,Proof-pile 128k 文档截断评测,越低越好)

来源:arXiv:2309.00071 Table 1(2026-09-21 arXiv HTML 原文核对):

| 模型(7B) | 扩展方法 | 8k | 32k | 64k | 96k | 128k |
|---|---|---|---|---|---|---|
| Together LLaMA-2-7B-32K | PI | 3.50 | 2.64 | >10² | >10³ | >10⁴ |
| Code Llama 100k | NTK | 3.71 | 2.74 | 2.55 | 2.54 | 2.71 |
| YaRN s=16 | YaRN | 3.51 | 2.65 | **2.42** | >10¹ | >10¹ |
| YaRN s=32 | YaRN | 3.56 | 2.70 | 2.45 | **2.36** | **2.37** |

- PI-32k 模型在 64k 就爆炸(>10²);YaRN-128k 在 128k 处 PPL 2.37,且**在微调数据没见过的 96k/128k 上仍能外推**(YaRN 论文强调的 extrapolation beyond fine-tuning dataset 能力);
- 13B 上结论同型(YaRN s=32:128k 处 2.24 vs Code Llama NTK 2.54);
- 训练成本:YaRN 比 PI 少 **10× tokens、2.5× 步数**(摘要口径)。

### 3.2 LongRoPE(2048k)

- LLaMA2-7B/Mistral-7B 上:**≤1000 微调步**(在 ≤256k 训练长度内)扩到 **2048k**;非微调场景(纯改 rescale 因子)可扩 **8×**(如 4k→32k);
- 关键消融:均匀插值(LongRoPE w/o 非均匀)性能明显差于逐维搜索的非均匀插值;不做 8k 回调则短窗口 PPL 受损(摘要与 §方法,arXiv:2402.13753);
- 定位:证明「解析公式(PI/YaRN)→ 搜索出的逐维因子」还能再挤性能,2M 窗口是 2024 初的公开上限。

### 3.3 各方法一览(取舍速查)

| 方法 | 是否要微调 | 典型扩展比 | 优点 | 缺点 |
|---|---|---|---|---|
| 直接外推 | 否 | — | 零成本 | 崩坏(分布外相位) |
| 朴素长微调 | 要,很久 | ~1.25×(2048→2560/10k+ batch) | 无假设 | 极慢,PI 论文反例 |
| PI | 要(~1000 步) | 16×(2k→32k) | 简单、稳、理论界 600× | 高频维被过度压缩,短窗口能力略降 |
| NTK-aware | **可免微调** | ~2-8× | 零训练成本,改 base 即用 | 精度不如微调路线;超远扩展仍崩 |
| YaRN | 要(比 PI 少 10× tokens) | 32-64×(→128k) | PPL 最优、可外推过微调长度、HF 原生 | 超参(α/β/温度)需按模型族调 |
| LongRoPE | 要(≤1k 步)+搜索 | 512×(4k→2M) | 公开最远;短窗口回调 | 需进化搜索基础设施 |
| 改 base+渐进预训练 | 要(继续预训练,如 Llama 3 的 ~800B tokens) | 16×(8k→128k) | 质量最稳(能力来自真训练) | 最贵,只有模型方做得起 |

---

## 4. 实践要点

- **部署侧先分清「谁做的扩展」**:模型方原生 128k(Llama 3.1、Qwen2.5)= 预训练+渐进,最可靠;社区 YaRN 版(如 NousResearch/Yarn-Llama-2-7b-128k)= 微调扩展,PPL 好但下游任务能力需自测;自己 HF 参数里开 `rope_scaling` 而没微调 = NTK 免微调路线,只能应急;
- **HF 配置对照**:`rope_scaling.type` → `linear`=PI、`dynamic`=动态NTK、`yarn`=YaRN、`llama3`=Llama-3.1 的分段插值(对高频/低频维区别处理+平滑过渡,思想同 NTK-by-parts);`factor`=s;**只改 factor 不微调,长文能力不会凭空出现**;
- **θ(base)与目标长度匹配**:经验上让最长波长 ≫ 目标上下文(Llama 3:θ=500k → λ_max≈3.1M ≫ 128k)。看到 θ=10k 的模型宣称 32k+,一定用了某种插值/NTK,问清楚;
- **YaRN 超参**:Llama 系 α=1、β=32、温度 t≈0.1·ln(s)+1 是论文给的起点;换模型族(如 Mistral/Qwen)要重新验证——α/β 论文自己就说「tuned in a case-by-case basis」;
- **短窗口回归测试不能省**:所有插值方法都动全局位置映射,PI/YaRN/LongRoPE 都报告过短上下文性能损失(并由「回调/readjust」补偿)——扩展后必须重测原窗口内的基准;
- **长上下文 ≠ 位置编码一件事**:attention 本身 O(n²)(FA 缓解的是显存与速度,不改变复杂度,见 [flash-attention-implementation](../../attention/implementation/flash-attention-implementation.md));KV cache 随长度线性涨(128k 单序列 16GB 量级,GQA/MLA/KV 量化是配套手段,见 [attention-heads-mha-mqa-gqa](attention-heads-mha-mqa-gqa.md) 与 [`vllm/concepts/paged-attention.md`](../../vllm/concepts/paged-attention.md));长文里「中间信息丢失」是检索层问题(attention sink 研究见 [attention-sinks-streaming](../../attention/algorithm/attention-sinks-streaming.md));
- **SWA(sliding window)是正交的另一条路**:Mistral 7B 用 W=4096 的局部窗口 + rolling buffer cache(32k 序列 cache 省 8×),靠层间递归把感受野堆到 W×层数(32 层 → 理论 ~131k),但**单 token 直接可见范围仍是 W**——与全局注意力的 RoPE 扩展路线互补,不解决「精确远程检索」;
- **选型决策树**:要 128k 生产级 → 选原生长上下文模型(Llama 3.1/Qwen2.5);只有 4k/8k 老模型且预算小 → YaRN 微调或直接用社区 YaRN 版权重;临时应急/免训练 → NTK-aware(dynamic);研究 2M 级 → LongRoPE 路线。

## 溯源

- **PI 主源**(arXiv:2306.15595,2026-09-21 ar5iv 全文核对):线性下缩放定义、1000 步微调、32768 上限、外推崩坏表述(catastrophically high attention scores)、理论上界 ~600×、朴素微调 10000+ batch 仅 2048→2560(Table 4)、7B~65B passkey/摘要实验范围;
- **YaRN 主源**(arXiv:2309.00071 HTML v1 全文核对):NTK-aware/NTK-by-parts/动态 NTK 三节结构、波长 λ_d 与 r=L/λ_d 定义、ramp 函数 γ_d、α=1/β=32(Llama 系)、温度 √t 缩放技巧与「免改 attention 代码」、10× tokens/2.5× 步数、Table 1 全部 PPL 数字(Together PI 32k / Code Llama NTK / YaRN s=16,32)、64k/128k checkpoint 发布;
- **LongRoPE 主源**(arXiv:2402.13753 HTML v1 全文核对):2048k、≤1k 微调步、两种非均匀性(前 n̂ 个 token 位置不插值/免微调 8×)、渐进 256k→2048k 两阶段、8k 回调恢复短窗口、LLaMA2+Mistral 实验对象;
- **RoPE 定义**(arXiv:2104.09864)与 **Llama 3 工业实践**(arXiv:2407.21783:θ=500k、8k→128k 六阶段、长上下文档 ~800B tokens)与 [llm-architecture](llm-architecture.md) 共享来源,细节已交叉核对;
- **Mistral SWA 数字**(arXiv:2310.06825 HTML v1 核对):W=4096、32 层理论感受野 ~131k、rolling buffer 32k 序列省 8× cache、FA/xFormers 改造 2× 提速(16k 序列);
- 代码仓库 yarn/LongRoPE curl 200 实测(2026-09-21);NTK-aware 插值原始出处为社区(Reddit /u/bloc97,YaRN 论文引用 [4][5] 记录在案,无独立论文,故不单列 source)。
- 关联条目:[llm-architecture](llm-architecture.md) §4(RoPE/ALiBi 位置编码谱系)、[attention-heads-mha-mqa-gqa](attention-heads-mha-mqa-gqa.md)(长上下文的 KV cache 账单)、[open-model-landscape](open-model-landscape.md)(各模型的上下文长度与扩展方式)。
