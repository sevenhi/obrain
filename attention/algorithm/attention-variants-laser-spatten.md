---
title: 注意力变体双线索 — LASER(激活替换/训练视角)与 SpAtten(稀疏剪枝/硬件架构视角)
type: concept
tags: [laser, spatten, attention-variants, gradient-saturation, token-pruning, head-pruning, progressive-quantization, accelerator, icml-2025, hpca-2021]
source:
  - "pdf: LASER: Attention with Exponential Transformation (primary) | arXiv:2411.03493 ICML 2025 (PMLR 267:14902-14917) | ~/zhanbin/library/papers/LASER: Attention with Exponential Transformation.pdf"
  - "pdf: SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning (primary) | arXiv:2012.09852 HPCA 2021 | ~/zhanbin/library/papers/SpAtten-- Efficient Sparse Attention Architecture with Cascade Token and Head Pruning.pdf"
  - "web: LASER PMLR 官方页 | https://proceedings.mlr.press/v267/duvvuri25a.html"
  - "code: mit-han-lab/spatten (SpinalHDL/Verilog + PyTorch 算法) | https://github.com/mit-han-lab/spatten"
date: 2026-09
---

# 注意力变体:LASER 激活替换(训练侧)× SpAtten 稀疏剪枝(推理硬件侧)

## TL;DR

同一个 softmax(QKᵀ/√d)V,两篇论文从相反方向动刀:
- **LASER**(ICML 2025,Google/UT Austin):**训练视角、改 V 不改打分**。发现 softmax 的 Jacobian 与注意力概率成正比,而 LLM 里 ~80% 概率 <10⁻³、~20% <10⁻⁷ → 反向传播经过 attention 时梯度被极度缩小(vanishing gradient 在 attention 内部的对应物)。改法:对 **exp(V)** 做标准 attention 再取 log:`laser(X)=log(softmax(QKᵀ)·exp(V))`(log-sum-exp 结构,可看作带注意力权重的 soft-max 检索)。Jacobian 上限从 a_j 级变成 ~1−a₁ 级(不饱和)。用 Log-Weighted-Sum-Exp trick 防溢出,**底层 attention kernel 一行不改**(FlashAttention 直接可用)。2.2B 模型达标 test loss 2.327 的训练时间 27.22h→24.88h(−9.4%);同等 loss 2.347 省 15.65% 参数;下游 one-shot 平均 63.39% vs 62.34%。
- **SpAtten**(HPCA 2021,MIT Han Lab):**推理视角、剪 token/head 不改权重**。attention 在 GPU/CPU 上占 GPT-2 端到端延迟 >50% 但只占 ~10% FLOPs,73% 时间耗在数据搬运 → 设计专用加速器:cascade token pruning(累计重要性分数逐层剪 token,全局级联)+ cascade head pruning + local value pruning + progressive quantization(先取 MSB 算,概率分布平坦才取 LSB 重算)。30 个 benchmark 平均 DRAM 访问降 10.0× 无精度损失,对 TITAN Xp GPU 162× 加速/1193× 能效。

一句话对比:**LASER 让训练学得更好(改激活的函数形式),SpAtten 让推理跑得更快(改计算的稀疏结构)**;两者共同点是都把 softmax 概率分布的"尖锐性"当作核心变量——LASER 因为它梯度消失而绕开它,SpAtten 因为它可预测剪枝安全而利用它。

---

## 1. 线索一:LASER — softmax 梯度饱和与指数变换

### 1.1 机制原理:梯度为什么在 attention 里变小

softmax 的反向传播(Lemma 3.1):前向 a = softmax(ã),反向 ∂ℓ/∂ã = (diag(a) − aaᵀ)·∂ℓ/∂a,**Jacobian 元素 ∂a_j/∂ã_i = a_j(1{i=j} − a_i)——每一项都乘了概率 a**。序列长 N 时典型 a ~ O(1/N) 且实测极度长尾:预训练一个 2.2B 模型时观察到 **~80% 的注意力概率 <10⁻³,~20% <10⁻⁷** → Jacobian 相应地小,W_Q/W_K/W_V 及更底层的梯度信号被系统性削弱。N=2 特例最直观:∂o₁/∂ã₁₁ = (v₁−v₂)·σ(ã₁₁−ã₁₂)(1−σ(ã₁₁−ã₁₂)),σ 饱和到 0/1 时梯度归零——经典 sigmoid 饱和问题的 attention 版。

### 1.2 LASER 的定义与低饱和证明

```
exp(laser(X)) = softmax(QKᵀ) · exp(V)      # 对 exp(V) 做标准 attention
⟹ laser(X) = log( softmax(QKᵀ) · exp(V) )  # 名字来源: LogArithm of
                                            # Summed Exponentials of Representations
```

N=2、d=1 时输出 o₁ = log(α₁e^{v₁} + (1−α₁)e^{v₂}),Jacobian:

∂o₁/∂ã₁₁ = (e^{v₁}−e^{v₂})·α₁(1−α₁) / (α₁(e^{v₁}−e^{v₂}) + e^{v₂})

极限情形 e^{v₁} ≫ e^{v₂} 时 ≈ **1−α₁**,不再有 α 乘子——概率再小,梯度也不随之消失(低饱和)。且 log-sum-exp 是 max 的可微逼近(Lemma 3.2):max(v₁+log a₁₁, v₂+log a₁₂) ≤ o₁ ≤ 同式 + log2 → **LASER ≈ "带注意力偏置的 soft-max 检索"**,从值集合里挑最大者而非加权平均;这也解释了它与 Veličković et al. 的 sharp-distribution 表达力研究同源(尖分布利于检索,但标准 softmax 的尖分布以梯度消失为代价——LASER 两者兼得)。

### 1.3 可复现细节:Log-Weighted-Sum-Exp trick

exp(V) 直接算会溢出,log-sum-exp trick 又需要改 attention 内部。**LASER 的解法:只改输入输出,不改 kernel**:

```
O_ij = ( log( softmax(QKᵀ) · exp(V̂) ) )_ij + m_j
其中 m_j = max_i V_ij (按列取最大, stop_gradient), V̂_ij = V_ij − m_j
```

JAX 实现(官方伪代码,标准 attention 函数 f 可以是 FlashAttention):

```python
m = jnp.max(value, axis=1, keepdims=True)
m = jax.lax.stop_gradient(m)          # 阻断 m 的梯度
exp_value = jnp.exp(value - m)        # 平移后 exp 不溢出
attention_out = f(key, query, exp_value)   # 任意现成 attention kernel
out = jnp.log(attention_out) + m      # 加回最大值
```

实验规模与配置(PAX/JAX on TPUv5):C4 数据集,batch 1024×1024 tokens,160K iters ≈ 167.8B tokens;301M 基座(32 层/8 头/head 128/MLP 2048)到 2.2B(32 层/dim 2048/MLP 8192)。**不用 trick 时 2.2B 训练直接数值溢出失败**。

### 1.4 数据对比

- **test loss(301M,C4)**:32 层/MLP 2048:LASER 2.595 vs 标准 2.641;32 层/MLP 4096:2.555 vs 2.575(至多 1.74% 相对改进);16 层也有增益(2.673 vs 2.681)——**MLP 占比越大(attention 参数越少)增益仍在**,说明不是简单加参数;
- **训练效率(2.2B)**:达到 loss 2.327,标准 27.22h vs LASER 24.88h(**−9.4%**);达到 loss 2.347,LASER **省 15.65% 参数**(幂律拟合 f(n)=anᵇ,234M~2.2B 五点);
- **LAMB 消融**:LASER 训练中梯度范数确实更高,但换用按权重范数归一化更新的 LAMB 后增益仍在(test 2.741 vs 2.758)→ **增益不只是"梯度更大",是架构内在的**;
- **下游(2.2B,one-shot 16 任务)**:平均 63.39% vs 标准 62.34%;调过温度的标准 attention 62.49%、per-dim temp 62.56%,LASER+per-dim temp 63.52%(单项至多 +3.4%);SuperGLUE 微调平均 +1.65%(44.01 vs 42.35);
- **跨模态**:BERT 2.2B MLM 误差相对 −0.93%;ViT/ImageNet、Conformer/Librispeech(准确率 +1.2%、验证 WER −0.2%)均有效 → 不是 LLM 特化。

### 1.5 实践要点(LASER)

- **改动成本极低**:输入端 `exp(V−colmax)`、输出端 `log(·)+colmax`,attention kernel(含 FlashAttention)零修改;colmax 要 stop_gradient;
- **必须用 Log-Weighted-Sum-Exp**:大模型无 trick 直接溢出,这不是可选优化;
- **对比基线要公平**:标准 attention 调温度(τ、per-dim temp)后 LASER 仍有净增益——评估任何 attention 变体时先调温度基线,否则容易把"更尖的分布"误判为"新机制的功劳";
- **收益量级要有预期**:是 ~1-2% loss / 10-15% 算力或参数的稳定增益,不是数量级革命;适合作为新预训练的默认尝试项而非改造旧模型的动机。

## 2. 线索二:SpAtten — 级联稀疏剪枝与专用加速器

### 2.1 机制原理:为什么 attention 需要专用硬件

GPT-2 剖析:attention 占端到端延迟 >50%,却只有 ~10% FLOPs;其中 **~73% 时间花在数据搬运**(split heads、K/V 拼接、reshape、transpose)——低算术强度 + 复杂访存模式,GPU/CPU 的 matmul 优化完全用不上(TITAN Xp 上生成 30 个 token 的 attention 要 370ms;树莓派 ARM CPU 要 43s)。定位:**attention 协处理器**(FC 层留给主处理器)。

三个算法观察:
1. **token 冗余**:语言高度冗余,大部分 token 对输出影响小,且"不重要"随层加深而加剧 → 可逐层级联剪枝;
2. **head 冗余**:部分 head 对输出影响甚微(与 head pruning 文献一致)→ 可剪;
3. **softmax 对量化误差的收缩性**:对 score 加误差 Δs₀,输出概率总误差 = Δs₀·2p₀(1−p₀) < Δs₀(2p(1−p)≤0.5)→ softmax 天然压缩量化误差,允许比 CNN/RNN 更激进的量化;且**概率分布越尖(有 dominator)误差越小**——尖分布行只需 MSB,平坦分布行才需要 LSB。

### 2.2 可复现细节:级联剪枝算法

**重要性分数(全部在线累计、input-dependent,无可训练剪枝参数——这是与 weight pruning 的本质区别)**:

```
token 重要性: s_t[token] += Σ_heads Σ_queries attention_prob[head][q][token]
              # attention 概率矩阵按列求和累计 (Fig.5: 很多 token attend 'fun' → 'fun' 重要)
head  重要性: s_h[head]  += Σ_queries Σ_dims |attention_out[head][q][dim]|
              # head 输出幅度累计 (幅度大 → 后续 FC 输出受它影响大)

remained_token = top-k(s_t, L1×(1−p_t))   # 剪掉后 QKV 全删, 且所有后续层永不再见
remained_head  = top-k(s_h, h×(1−p_h))    # 级联: 一次剪除, 层间继承
```

- **cascade 语义**:token/head 一旦被剪,其 QKV/对应 chunk 在**所有后续层和 head** 中都不再出现,每层只处理上层幸存集合,层越深剪得越多(BERT-Base SST-2 例:11 token → 层 2 剩 5 → 层 3 剩 2,计算 100%→38%→12%);
- **跨生成迭代累计**:GPT-2 生成阶段的 token 重要性跨解码步累计(对当前 token 不重要的,对后续 token 大概率也不重要);
- **local value pruning**:softmax 之后按当前 head 概率剪最小的一批 V,不取它们参与 prob×V(只影响当前 head,非级联);
- **progressive quantization 流程**:先只取 MSB 算概率 → 若 max(prob) < 阈值(默认 0.1,即分布平坦)→ 取 LSB 拼回重算一次;MSB+LSB 组合 {4+4, 6+4, 8+4, 10+4, 12+4},DRAM 中 MSB/LSB 连续分开存放以便独立取;**平均只有 5.9% 的样本需要 LSB**。BERT(计算受限)只用静态量化,GPT-2(访存受限)才用 progressive——按瓶颈选技术;
- **硬件要点**:O(n) 高并行 top-k 引擎(比全排序单元吞吐 1.4×、功耗 1/3.5);32×16 地址 crossbar + 16×32 数据 crossbar 服务 16 通道 HBM2(每通道 32GB/s @2GHz),把剪枝导致的随机访问摊平到所有通道;片上 196KB Key SRAM + 196KB Value SRAM;逐 head、逐 query 流水,Q×K 与 prob×V 各 512 个 12-bit 乘法器 + 可重构 adder tree;SpinalHDL 写 RTL,Verilator + Ramulator(HBM2)+ CACTI 评估,TSMC 40nm 综合。

### 2.3 数据对比

- **稀疏收益(30 benchmarks:BERT-Base/Large × GLUE+SQuAD,GPT-2-Small/Medium × Wikitext-2/103+PTB+1BW)**:token+value 剪枝平均 1.9×(GPT-2 上 3.8×),head 剪枝 1.1×;**计算 2.1×、DRAM 访问 10.0× 削减,无精度损失**(每任务微调 ~2h 恢复;仅 BERT-large SQuAD 掉 2%)。句越长冗余越多、可剪越狠;
- **芯片**:1.61 TFLOPS(BERT 类计算受限)/0.43 TFLOPS(GPT-2 类访存受限),8.30W(计算 1.36W + SRAM 1.24W + **DRAM 5.71W=69%**),18.71mm²;top-k 引擎只占功耗 1.0%/面积 2.7%;
- **对基线加速/能效**:vs A3 加速器 1.6×/1.4×;vs MNNFast 3.0×/3.2×;**vs TITAN Xp GPU 162×/1193×;vs Xeon CPU 347×/4059×**;vs Jetson Nano 1095×/406×;vs 树莓派 5071×/1910×;
- **剪枝比率配置**(实践可抄):前 15% 层不剪 token(前 30% 层不剪 head),其余层从 r_start 到 r_end 线性插值,r_start+r_end=2r_avg;同一总剪枝率下层间分配不敏感。

### 2.4 实践要点(SpAtten)

- **剪 token/head 不是剪权重**:没有可训练剪枝参数,选择完全在线(on the fly)、输入自适应 → 无法离线静态化,必须配 top-k 硬件/高效 kernel 才有实际收益;
- **DRAM 功耗占 69% 决定了优化优先级**:访存削减(剪枝+progressive quant)优先于计算削减;这也解释了为何 BERT(计算受限)不该用 progressive quant;
- **量化误差先过 softmax 再评估**:p(1−p) 收缩因子意味着 attention 概率的量化可以比 CNN 激活激进;
- **对今天的意义**:SpAtten 是"稀疏 attention 加速器"路线的代表(2021);通用 GPU 上的同思想后代是各类动态稀疏 attention kernel,而"attention 是访存受限"这一诊断被 FlashAttention 用 IO-aware tiling 在通用硬件上解决(见 [flash-attention-implementation](../implementation/flash-attention-implementation.md))——**同一个 profiling 结论,专用硬件与通用内核两条路线**。

## 3. 两条线索的交汇点

| | LASER | SpAtten |
|---|---|---|
| 目标阶段 | 训练(兼推理) | 推理 |
| 改动对象 | V 的激活空间(exp/log 包裹) | 计算图结构(token/head/V 剪枝+量化) |
| softmax 概率分布尖锐是 | **病因**(梯度∝a,尖→梯度消失) | **资源**(尖→剪枝安全、MSB 够用) |
| 对 attention kernel | 零修改(FlashAttention 兼容) | 全新硬件流水线 |
| 增益形态 | ~1-2% loss、10-15% 算力/参数 | 10× DRAM、百倍级加速/能效(专用芯片) |
| 会议 | ICML 2025 | HPCA 2021 |

共同的底层事实:**softmax 后的注意力概率在真实 LLM 中极度长尾**(LASER:80% <10⁻³;SpAtten:少数 dominator 主导行分布;StreamingLLM:和为 1 约束制造 sink)。三篇论文分别是这个事实的训练侧、硬件侧、KV cache 侧推论——读任何一个"attention 改进"工作,先问它对长尾概率分布做了什么假设。

## 溯源

- 主源:两篇本地 PDF,公式/表格数字直接核对原文(LASER Table 1-3、Fig.4 幂律;SpAtten Alg.2、Table I/II、Fig.13/14)。
- 交叉源:PMLR v267 官方页(确认 ICML 2025 / PMLR 267:14902-14917)、mit-han-lab/spatten 仓库(SpinalHDL RTL + PyTorch 算法参考)。URL 均已 curl 验证存活(2026-09-21);SpAtten 的 IEEE Xplore 页(9407256)对 curl 返回 202 反爬,不列为源,以 arXiv 版为准。
- 关联条目:[attention-is-all-you-need](attention-is-all-you-need.md)(√d_k 缩放与 QK-Norm 等"防尖锐"补丁的谱系)、[attention-sinks-streaming](attention-sinks-streaming.md)(长尾分布的 KV cache 侧推论)、[flash-attention-implementation](../implementation/flash-attention-implementation.md)(访存受限诊断的通用内核解)、[`quantization/`](../../quantization/index.md)(量化侧对照)。
