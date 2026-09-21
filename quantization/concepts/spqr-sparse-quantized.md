---
title: SpQR — 稀疏+量化混合表示,近无损 LLM 权重压缩
type: concept
tags: [quantization, spqr, llm-compression, outlier, gptq, ptq, csr, near-lossless]
source:
  - "pdf: SpQR 论文全文(Dettmers/Svirschevski/Egiazarian et al., ↑primary) | arXiv:2306.03078 (ICLR 2024) | ~/zhanbin/library/papers/SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression.pdf"
  - "web: SpQR 官方代码 | https://github.com/Vahe1994/SpQR (2026-09 实测 200;论文脚注称将并入 bitsandbytes)"
  - "web: bitsandbytes(作者 Tim Dettmers 的量化库,SpQR 拟集成) | https://github.com/TimDettmers/bitsandbytes (2026-09 实测 200)"
  - "web: ICLR 2024 poster 页(发表状态核对) | https://iclr.cc/virtual/2024/poster/18686 (2026-09 实测 200)"
  - "web: 相关技术脉络 LLM.int8() outlier features | arXiv:2208.07339;GPTQ | arXiv:2210.17323(交叉核对引用关系)"
date: 2026-09
---

# SpQR:稀疏+量化混合表示,近无损 LLM 权重压缩

## TL;DR

SpQR(Dettmers 等,arXiv:2306.03078,ICLR 2024)是**第一个把 LLM 权重压到 3-4 bit/参数仍近无损(perplexity 相对损失 <1%)**的后训练量化(PTQ)方法。核心两招:①**离群值隔离**——用 GPTQ 求解器动态算每个权重的量化敏感度,把 ~1% 的高敏感 outlier 抽出来存 16-bit(CSR 稀疏格式),其余压到 3-4 bit;②**两级量化(bilevel)**——group size 小到 16 的分组量化统计量(scale/zero)本身再量化成 3-bit,把"小组带来的元数据开销"压回去。结果:33B 模型塞进单张 24GB 消费级 GPU,LLaMA/Falcon 各尺寸 WikiText2 perplexity 距 FP16 <1%,推理还比 FP16 快 20-30%。

一句话心法:**量化误差不是均匀分布的——1% 的权重可占 >75% 误差;把这 1% 单独放高精度、其余狠压,是比"全模型统一位宽"更聪明的预算分配**。

---

## 1. 机制原理(为什么这样设计)

### 1.1 出发点:权重敏感度有结构,不是随机的

"某个权重量化后伤不伤模型"不能只看它离量化点多远——LLM 权重高度相关,`w_a` 向上舍入的误差可被 `w_b` 向下舍入补偿。SpQR 因此用**可补偿的最小输出误差**定义敏感度(式 1/2,Optimal Brain Surgeon 框架):

```
s_ij = min_{W'} ||WX - W'X||²   s.t. w'_ij = quant(w_ij)   (其余权重任意取值补偿)
     = (w_ij - quant(w_ij))² / [2·(XXᵀ)⁻¹]_jj              (闭式解,逆 Hessian 对角)
```

GPTQ 逐列量化时本来就要算这个逆 Hessian 子块,所以敏感度是**动态、零额外开销**得到的(基于权重当前值,自动吸收前面列已量化造成的调整)。

对 LLaMA-65B 做敏感度热图分析,发现四类**结构化**模式:
- **行 outlier**(某些输出单元整行/半行高敏感;attention 层里对应部分 head);
- **列 outlier**(某些输入维全列高敏感——即 LLM.int8() 报过的 outlier feature 现象,但**权重侧的行 outlier 是 SpQR 首次系统刻画**,且与激活 outlier 只部分相关,不是同一列);
- **敏感 attention head**(宽度 128 的规则条纹,head 级整块敏感);
- **RoPE 模式**(周期 64 的竖条纹:每 head 128 维拆成 cos/sin 各 64,低频维更敏感)。

因果链:**敏感度有结构 → 统一位宽的量化格式(全 4-bit 或全 8-bit)必然浪费预算 → 需要一种能同时表达"个体 outlier + 小组统计"的混合格式**。

### 1.2 两个设计决策

**① outlier 抽 16-bit + CSR 存储。** outlier 无结构(散点+局部小段),用行优先的 CSR(16-bit 值 + 16-bit 列索引 + 每行 32-bit 累计偏移,平均 ~32 bit/outlier)。检测准则:把某权重保留 16-bit 能让式(2)误差降 ≥τ 即选为 outlier,全模型二分搜 τ 控制 outlier 比例 <1%。关键细节:**outlier 的 16-bit 值不是原始权重,而是 GPTQ 调整后、用于补偿其他权重误差的值**——SpQR 是"隔离并利用 outlier",超越单纯检测(与 LLM.int8() 的列级检测不同)。

**② bilevel 量化让"超小 group"重新划算。** 直觉上 group size 越小精度越好,但 scale/zero 元数据开销爆炸(Yao et al. 2023 明确反对小组)。SpQR 的解法:一级用 β1=16 的组、scale/zero 只给 3-bit;再把 16 个组的统计量打包成二级组(β2=16~64),用 min-max 量化到 3-bit,只对二级统计量留 16-bit。min-max 的特性让"统计量最大/最小的组被精确表示",atypical 组自动占更多量化预算。**净效果:group=16 的精度,group≈512 的元数据开销**。

### 1.3 平均位宽公式(预算怎么算)

```
b ≈ b_w + (b_s + b_z)/β1 + 64/(β1·β2) + 32·r_o
```
例:b_w=b_s=b_z=3、β1=16、β2=32、r_o=0.4% → 3 + 6/16 + 64/512 + 0.128 ≈ **3.63 bit/参数**。

## 2. 可复现细节

### 2.1 算法骨架(Algorithm 1 提炼)

```
SPQR-QUANTIZE(W, X, b, β1, β2, τ, λ):
  H = 2XXᵀ; Hic = Cholesky((H+λI)⁻¹)         # GPTQ 式逆 Hessian
  for 每 β1 列块 i:
    OUTLIERS(W[:,i:i+β1]):                     # leave-one-out 敏感度
      E_base = error(整块); 对块内每个权重 j:
        E_ol = error(去掉 j 后重拟合)          # j 保留 16-bit 的情形
        若 E_base - E_ol > τ → j 入 outlier 集 O
    FIT_STATISTICS: 剔除 O 后拟合 min-max scale/zero(一级),
      再对 β2 个组的统计量拟合二级 scale/zero   # bilevel
    逐列 quantize + GPTQ 误差补偿回写(未量化列)
  输出: 3-bit 权重 Q + 两级统计量(Sq,Zq,Ss,Zs,Sz,Zz) + CSR outlier
```

LLaMA-30B 近无损配置(论文附录 B 原文命令行):
```
python main.py $MODEL custom --custom_data_path=$DATA \
  --wbits 4 --groupsize 16 --perchannel --qq_scale_bits 3 \
  --qq_zero_bits 3 --qq_groupsize 16 --outlier_threshold 0.1 \
  --fit_quantizer_without_outliers --permutation_order act_order
```

### 2.2 存储布局(单个权重张量)

- 主体:3/4-bit 权重,每 β1=16 个连续权重一对 3-bit scale+zero;每 β1×β2 块四个 16-bit 二级标量;
- outlier 的量化副本**原位保留**(方便 GPU 访存),16-bit 值做差值补偿;移动端可删量化副本再省空间;
- CSR:按行排序、行内按列排序,outlier 同行连续 → 推理 kernel 访存大体连续。

### 2.3 推理 kernel(为什么量化模型还能更快)

自回归生成是 memory-bound → 压缩本身就能藏住解码开销。dense 部分:group 统计量+3-bit 权重载入 SRAM,反量化成 16-bit 再乘。sparse 部分:矩阵分块 → 每 thread block 把一大段 outlier 载入 SRAM → 每线程判断本段是否含自己的 outlier → 命中才去主存加载 → 与 dense matmul 合并。实测**自定义 kernel 比 PyTorch(cuSPARSE)基线快近 2×,比 FP16 快 20-30%**;稀疏与稠密 matmul 未融合是作者自认的剩余优化空间。

## 3. 数据对比

**LLaMA perplexity(WikiText2,FP16 基线见括号)**:

| 模型 | FP16 | RTN-4bit | GPTQ-4bit | SpQR-3.9bit | SpQR-4.63bit |
|---|---|---|---|---|---|
| 7B | 5.68 | 6.43 | 6.13 | **5.87** | **5.73** |
| 13B | 5.09 | 5.55 | 5.40 | **5.22** | **5.13** |
| 65B | 3.53 | 3.87 | 3.83 | **3.68** | **3.57** |

- **同 4-bit 预算下 SpQR 相对 16-bit 的误差约为 GPTQ 的一半**;SpQR−GPTQ 的差距 ≈ GPTQ−RTN 的差距(误差谱系上再进一档);
- 近无损点(<1% 相对误差,MLCommons 定义):LLaMA 全系 ≤4.71 bit/参数,Falcon ≤4.5,OPT ≤4.3;最低可压到 **3.36 bit** 仍保持可用质量;
- **模型越小越吃亏,SpQR 收益越大**:7B 上 RTN-4bit 已 +13% perplexity(边缘部署最痛的 1-10B 区间正是 SpQR 设计目标);
- outlier 消融(LLaMA-65B, WikiText2):同 outlier 率下**非结构化(SpQR)显著优于整行 MSE 选择和 LLM.int8() 列准则**;
- bilevel 消融:同为 ~3.63 bit,"3-bit 统计量+group16" (3.74) 优于 "16-bit 统计量+group48" (3.84)——**小组+量化统计量 > 大组+精确统计量**;
- 推理速度(A100, tokens/s, batch=1):LLaMA-7B FP16 47 → SpQR 优化版 57;65B FP16 OOM → SpQR 12 tok/s(单卡跑通);
- 量化耗时:65B 在 A100 约 4.5 小时。

## 4. 实践要点

1. **适用场景**:单消费级 GPU(24GB)跑 30-65B 模型、边缘/笔记本部署 7-13B 模型;自回归生成 memory-bound,压缩=加速。不适配:训练/微调(纯 PTQ)、prefill 主导的长上下文批处理(compute-bound 时反量化有开销)。
2. **调参旋钮**:目标位宽由 (b_w, β1, β2, r_o) 共同决定,用 §1.3 公式先算预算再选;τ 在 [0.1,1.0] 二分搜到目标 outlier 率;**最优 outlier 率经验值 0.2-0.5%**,更多 outlier 不如把位宽加回权重本身(附录 E:高预算下加 b_w 优于减 group/加 outlier)。
3. **保留 act_order**(GPTQ 的按逆 Hessian 对角重排输入维):小幅但免费的收益。
4. **别混淆两类 outlier**:激活 outlier(LLM.int8()/SmoothQuant,列结构、量化激活时处理)vs 权重敏感度 outlier(SpQR,半结构化、量化权重时处理)——两者相关但不重合,方案可叠加。
5. **与后续方法的关系**:SpQR 的"outlier 高精度+主体低比特"思路被 AQLM/QuIP#/SqueezeLLM 等 3-bit 以下方法继承改造;engine 侧落地(GPTQ/AWQ 的注册分派)见 vllm 域 quantization 篇——本库 quantization 域管算法,vllm 域管集成。
6. **已知局限(作者自述)**:只测了 perplexity/zero-shot,未系统测生成质量;稀疏+稠密 matmul 未融合;outlier 32 bit/个的存储可用分组进一步压缩。

## 5. 溯源与多源综合说明

- 主源 ↑primary:本地 PDF(arXiv:2306.03078v1, 2023-06-05),正文/附录 A-H 全部数据出自该文;
- 发表状态:arXiv v1 为预印本,后收录 **ICLR 2024**(iclr.cc poster 页 + ICLR proceedings PDF,2026-09 实测);
- 代码:github.com/Vahe1994/SpQR(2026-09 实测存活);论文脚注承诺并入 TimDettmers/bitsandbytes(实测存活,SpQR 集成状态以仓库当前 README 为准);
- 交叉核对:GPTQ(arXiv:2210.17323)、LLM.int8()(arXiv:2208.07339)、Optimal Brain Compression(arXiv:2208.11580)的引用关系与本文 Related Work 一致;"4-bit 是 RTN 类方法最优点"出自 Dettmers & Zettlemoyer(arXiv:2212.09720)。
