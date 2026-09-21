---
title: Attention Sinks 与 StreamingLLM — 无限长流式生成的 KV cache 策略
type: concept
tags: [streamingllm, attention-sink, kv-cache, streaming-inference, sliding-window, iclr-2024]
source:
  - "pdf: Efficient Streaming Language Models with Attention Sinks (primary) | arXiv:2309.17453 ICLR 2024 | ~/zhanbin/library/papers/Efficient Streaming Language Models With Attention Sinks.pdf"
  - "web: StreamingLLM 项目页 (MIT Han Lab) | https://hanlab.mit.edu/projects/streamingllm"
  - "code: mit-han-lab/streaming-llm | https://github.com/mit-han-lab/streaming-llm"
  - "web: OpenReview ICLR 2024 | https://openreview.net/forum?id=NG7sS51zVF"
date: 2026-09
---

# StreamingLLM:attention sink + 滚动 KV cache

## TL;DR

**现象**:自回归 LLM 会把大量注意力分给最初几个 token,哪怕它们语义无关(把开头 4 个 token 换成 "\n" 后模型照样重度关注它们)——这些 token 是 **attention sink(注意力汇聚点)**。**根因**:softmax 强制注意力权重和为 1,当 query 与所有历史 token 都不匹配时,多余注意力必须"倒"进某个地方;开头 token 因对所有后续 token 可见,在预训练中最容易被训练成这个垃圾桶。**后果**:纯滑窗(window attention)一旦把开头 token 的 KV 逐出 cache,PPL 从 5.4 爆炸到 5158(Llama-2-13B)。**方案**:StreamingLLM = **保留开头 4 个 token(sink)+ 最近窗口 token 的 KV**,零微调让有限窗口训练的模型稳定处理 400 万+ token 流,比"滑窗重计算"基线快至多 22.2×。已被 HuggingFace Transformers、NVIDIA TensorRT-LLM、MLC LLM 等集成。**边界**:它不扩展上下文——被逐出的中间 token 找不回来,只保证"用窗口内内容稳定续写"。

---

## 1. 机制原理

### 1.1 流式部署的两个矛盾

1. **KV cache 无限增长**:解码阶段每个新 token 都要 attend 全部历史,历史 K/V 必须驻留显存 → 多轮对话/常驻服务场景显存撑不住;
2. **长度外推失败**:模型只在训练窗口内(如 Llama-2 的 4K)可靠,超出后 dense attention 直接崩(PPL 爆炸)。

三种基线各死一头:
- **Dense attention**:O(T²) 计算 + 无限 cache,超过预训练长度就崩;
- **Window attention**(只留最近 L 个 KV):高效,但**开头 token 被逐出的瞬间崩溃**;
- **滑窗 + 重计算**(每来一个新 token 用最近 L 个 token 重建 KV):质量好,但每 token O(L²) 重算,慢到不实用。

### 1.2 attention sink 的因果链

```
softmax 要求 Σ attention = 1
   → query 与所有 token 都不强匹配时,注意力无处可去
   → 必须倒给某些 token("sink")
   → 谁最有资格? 开头 token: autoregressive 下它们对所有后续 token 可见,
     预训练中被反复选中承担这个角色
   → 深层网络(除最底 1-2 层是局部模式外)所有层所有头都重度 attend 开头 token
```

**两个判别实验**(证明 sink 与语义无关):
1. 把开头 4 个 token 换成 "\n":模型照样把注意力倒给这几个 "\n",PPL 恢复到与原开头 token 相当(5.60 vs 5.40)→ **重要的是"绝对位置在开头",不是内容**;
2. 只逐出开头 token 的 KV(其余全保留):PPL 立即爆炸 → **崩溃的充分必要条件是 sink 消失**。

数学上:设 x₁ ≫ x_j (j≥2),softmax 分母由 sink token 主导;把它从分母里删掉,整个注意力分布剧烈偏移,后续所有层的输入分布全变 → 崩。

### 1.3 为什么要 4 个而不是 1 个 sink

消融(Table 2,400K token PG19):0+4096(纯窗)Llama-2-7B PPL=3359.95;1+4095→11.88;2+4094→10.51;**4+4092→9.59**;8+4088→9.54。1~2 个不够、4 个基本饱和、再加边际收益趋零。原因:Llama-2 虽然每段前有 `<s>`,但它在文本分块**之前**加上,导致第 0 位 token 实际近乎随机——**预训练没有统一的开头 token,模型只好把多个开头 token 都训练成 sink**。MPT(ALiBi)/Falcon/Pythia 结论一致。

### 1.4 与相对位置编码的配合(实现关键)

StreamingLLM 要求模型用**相对位置编码**(RoPE/ALiBi),且位置编号按 **cache 内位置**而非原文位置:cache 里是 token [0,1,2,3,6,7,8]、正解码第 9 个时,分配位置 [0,1,2,3,4,5,6,7]。RoPE 实现细节:**缓存旋转前的 Key**,每步解码时对滚动 cache 里的 K 重新施加旋转变换(否则被逐出 token 造成的位置空洞会让相对距离错乱)。ALiBi 更直接:用连续线性 bias 即可。

## 2. 可复现细节

### 2.1 cache 布局

```
KV cache (总大小 = 预训练窗口, 如 2048):
┌──────────────┬────────────────────────────────┐
│ sink: 前 4 个 │ rolling: 最近 (cache_size-4) 个  │
│ token 的 KV   │ token 的 KV (FIFO 滚动逐出)      │
└──────────────┴────────────────────────────────┘
新 token 到来 → 逐出 rolling 区最旧的一个 → 追加新的
位置编码按 cache 内槽位重新编号 (RoPE: 存旋转前 K, 每步重旋)
```

### 2.2 核心代码(官方 repo 的逻辑,~15 行)

```python
def get_slim_paged_cache(cache, sink_size=4, recent_size=1020):
    # cache: (batch, heads, seq_len, dim) 的全量 KV
    return torch.cat([cache[:, :, :sink_size, :],
                      cache[:, :, -recent_size:, :]], dim=2)

# 解码循环中: 每步 append 新 KV 后重新裁剪;
# 模型前向时 position_ids = torch.arange(cache_len)  # cache 内编号!
```

官方仓库 `mit-han-lab/streaming-llm`(HuggingFace Transformers 接口,支持 Llama-2/MPT/Falcon/Pythia);TensorRT-LLM、HF Transformers 主线均已内置(项目页 News 记录:2023-12 HF 集成、2024-01 TensorRT-LLM 集成);SGLang 早期版本曾支持,现主线已移除(server_args 无 attention-sink 参数),以官方文档为准。

### 2.3 预训练侧改进:learnable sink token

给所有训练样本开头加一个可学习占位 token,让它专职当 sink。160M 模型对照实验(Table 3,PPL):

| cache 配置 | 0+1024 | 1+1023 | 2+1022 | 4+1020 |
|---|---|---|---|---|
| Vanilla | 27.87 | 18.49 | 18.05 | 18.05 |
| Zero Sink(SoftMax-off-by-One) | 29214 | 19.90 | 18.27 | 18.01 |
| **Learnable Sink** | 1235 | **18.01** | 18.01 | 18.02 |

- **Learnable sink 只需 1 个 token** 即达稳定 PPL(且略优于 vanilla);
- **Zero Sink**(把 softmax 换成 SoftMax1:分母 +1,等价于前置一个全零 K/V 的虚拟 token)反而不如 vanilla——它缓解了"必须倒垃圾"的压力,但模型仍依赖真实开头 token 当 sink,0+1024 时崩得更惨(29214)。**结论:显式可训练 sink token 优于改 softmax**。
- 7 项 NLP 基准(ARC/HellaSwag/LAMBADA/OBQA/PIQA/WinoGrande)zero-shot 加 sink token 无损甚至略好(如 ARC-c 18.6→19.6)。

## 3. 数据对比

### 3.1 PPL(20K token 长文,Llama-2-13B,PG19 首书)

| 方法 | PPL | 复杂度/token |
|---|---|---|
| Dense attention | 5641(超窗后崩) | O(T²) |
| Window attention | 5158(sink 被逐出后崩) | O(TL) |
| 滑窗+重计算 | 5.43 | O(TL²) |
| **StreamingLLM** | **5.40** | O(TL) |

StreamingLLM 达到 oracle(重计算)水平,复杂度却是线性的。4M token 规模上 Llama-2-[7,13,70]B、MPT-[7,30]B、Falcon-[7,40]B、Pythia-[2.8,6.9,12]B 全家族 PPL 稳定(波动只来自书籍切换边界)。

### 3.2 速度/显存(A6000,vs 滑窗重计算,Figure 10)

每 token 解码延迟随 cache size(256→4096)的变化:
- **Llama-2-7B**:重计算 63→103→223→523→1411ms(平方增长);StreamingLLM 31→31→35→45→65ms(线性缓增);
- **Llama-2-13B**:重计算 99→169→361→860→2355ms;StreamingLLM 48→52→60→75→106ms。cache=4096 时 2355/106 ≈ **22.2× 加速**(即论文头条数字的来源)。

显存两者相当(7B:13-21GB;13B:25-36GB,均由窗口大小决定)——StreamingLLM 省的是时间,不是相对重计算的显存;省显存是相对 dense attention(无限 cache)而言。

### 3.3 流式 QA(Table 5,ARC 拼接流,Llama-2-Chat)

| | Arc-E (7B) | Arc-C (7B) | Arc-E (70B) | Arc-C (70B) |
|---|---|---|---|---|
| One-shot 基线 | 71.25 | 53.16 | 91.29 | 78.50 |
| Dense | OOM | OOM | OOM | OOM |
| Window | 3.58 | 1.39 | 0.12 | 0.32 |
| **StreamingLLM** | **71.34** | **55.03** | **91.37** | **80.20** |

window attention 在长流上输出接近随机;StreamingLLM 追平逐样本 one-shot。

### 3.4 cache 大小的反直觉结论(Table 6)

增大 cache 并不单调降 PPL(Falcon-7B:4+1020→12.34,4+2044→12.84;Llama-2-7B:4+2044→9.08,4+4092→9.59)——**模型没有充分利用给它的上下文**。这与 LongEval/"lost in the middle"一脉相承:窗口大≠用得好。

## 4. 实践要点

- **判断该不该用**:StreamingLLM 适合**无限流 + 只关心近期内容**的场景(常驻对话、日志续写、StreamEval 式"每 10 行查最近信息")。**需要全文检索/回忆被逐出内容时它无能为力**——那是长上下文扩展(RoPE 插值/YaRN)或 RAG 的地盘,两者正交、可叠加(32K 扩展模型 + StreamingLLM 把流式 cache 上限扩到 32K)。
- **sink 数量**:默认 4 个;若模型预训练带统一 BOS(或以 sink token 训过),1 个够。
- **位置编码**:必须按 cache 内槽位重编号;RoPE 要存旋转前的 K。用绝对位置编码的模型不适用。
- **别把 StreamingLLM 当上下文扩展卖**:论文自己强调"efficiently generates coherent text from tokens within the KV cache without extending the LLMs' context length"。
- **与 vLLM 的关系**(跨域,见 [`vllm/concepts/paged-attention.md`](../../vllm/concepts/paged-attention.md)):vLLM 用分页解决 KV cache 的**碎片与利用率**(能装下就全保留,支持 prefix caching),StreamingLLM 解决**装不下时逐出谁**——sink+recent 的逐出策略与 PagedAttention 的 block 管理正交,服务引擎做长会话常驻时可组合(固定 sink block 不回收 + rolling block LRU)。vLLM 主线未内置 StreamingLLM 式逐出,但 TensorRT-LLM/HF 已内置(SGLang 早期版本曾支持,现主线已移除,以官方文档为准)。
- **量化视角的旁证**:sink 与量化 outlier 同源(SmoothQuant/AWQ 观察到的激活离群值),都指向"softmax 和为 1 的约束制造了少数极端 token"——见 [`quantization/`](../../quantization/index.md) 域;SoftMax-off-by-One 是同一根因的 softmax 侧修法,但本论文实验证明它不如显式 sink token。

## 溯源

- 主源:本地 PDF(arXiv v4,2024-04-07,ICLR 2024 会议版),全部表格数字直接核对原文。
- 交叉源:MIT Han Lab 项目页(集成时间线)、官方 GitHub repo(实现)、OpenReview(评审版)。URL 均已 curl 验证存活(2026-09-21);openreview API 有反爬挑战,forum 页 HTTP 200。
- 关联条目:[attention-is-all-you-need](attention-is-all-you-need.md)(softmax 分母与 O(n²) 的来源)、[`vllm/`](../../vllm/index.md)(服务引擎侧 KV cache 管理)、[io-aware-tiling](io-aware-tiling.md)(窗口内 attention 的高效内核)。
