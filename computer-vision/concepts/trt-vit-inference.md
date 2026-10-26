---
title: TRT-ViT — 以 TensorRT 延迟为准绳的 ViT/CNN 混合架构设计
type: concept
tags: [vit, cnn, tensorrt, inference, architecture-design, hardware-aware-nas]
source:
  - "pdf: TRT-ViT 论文全文 | https://arxiv.org/abs/2205.09579 | v2,ByteDance,长期 preprint 未见正式发表 ↑primary"
  - "web: TRT-ViT arXiv abs 页 | https://arxiv.org/abs/2205.09579 (抓取 2026-09-21)"
  - "web: NVIDIA TensorRT 官方文档 | https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (抓取 2026-09-21)"
  - "web: NVIDIA TensorRT GitHub | https://github.com/NVIDIA/TensorRT (抓取 2026-09-21)"
  - "code: 社区复现实现(非官方,论文作者未开源) | github.com/Phoenix8215/TRT-ViT (抓取 2026-09-21)"
date: 2026-09
---

# TRT-ViT:把「部署延迟」直接当架构设计的一等公民

> 论文阅读笔记(arXiv:2205.09579v2,ByteDance)。**注意:这是长期 preprint(2022 挂出未见正式发表),官方代码未开源**(GitHub bytedance/TRT-ViT 不存在,curl 实测 404);引用的数字以论文表格为准,别当成已发表定论。

## TL;DR

FLOPs/参数量是**间接、片面、硬件不敏感**的效率指标——很多「高效 ViT」在真实部署里连 ResNet 都跑不过。TRT-ViT 的做法:**直接把目标硬件上 TensorRT 的延迟当效率度量**(它同时反映算力、访存、带宽),用 TeraFLOPS=FLOPs/Latency 与 TeraParams=Params/Latency 两个「密度」指标做受控实验,得出 4 条设计准则:①stage 级 Transformer 放后期;②stage 级浅前深后;③block 级 Transformer+BottleNeck 混合比纯 Transformer 高效;④block 级先全局后局部(global-then-local)。据此造出 MixBlockC(Transformer 分支与 7×7 BottleNeck 分支按 shrinking ratio R 串接)与五 stage 的 TRT-ViT-A/B/C/D。结果:82.7% ImageNet top-1 时比 CSWin 快 2.7×、比 Twins 快 2.0×;COCO 检测与 Twins 精度相当而推理快 2.8×;ADE20K 分割快 4.4×。核心洞见:**Transformer 的计算效率(TeraFLOPS)随分辨率变小迅速逼近 BottleNeck,所以它该出现在低分辨率的后期 stage**。

## 机制原理(为什么)

### 1. 为什么 FLOPs 不够格当效率指标

FLOPs 只数乘加次数,不含:访存/IO 成本、算子在特定硬件上的实际利用率、kernel 启动与融合情况。ViT 的 attention 是访存密集的小算子序列,TensorRT 上跑出来的延迟远差于同 FLOPs 的卷积。论文用两个密度指标把「硬件感受」量化(T4 GPU,TensorRT-8.0.3,batch=16):

```
TeraFLOPS  = FLOPs / Latency      (计算密度,越高说明硬件把 FLOPs 吃得越满)
TeraParams = Params / Latency     (参数密度,近似「单位时间能过多少容量」)
```

### 2. 四条准则的因果链(这是全文骨架)

**准则 1(Transformer 放后期 stage)**:Table 1 的受控数据显示,56×56 高分辨率时 Transformer 的 TeraFLOPS 只有 81,不到 BottleNeck(378)的 1/4;而到 7×7 时两者几乎持平(599 vs 670)。因果:attention 的 token 数随分辨率平方增长,高分辨率下访存与二次复杂度主导、算力吃不满;低分辨率下矩阵变小、GEMM 占比升高,硬件利用率反而上来。同时 Transformer 的性能优势(全局建模)在后期语义 stage 才最值钱 → **把 Transformer 放低分辨率后期是最优 trade-off**。

**准则 2(浅前深后)**:TeraParams 在后期 stage 显著更高(BottleNeck 从 120 → 13723,Transformer 从 7.5 → 11988),说明同样延迟预算下,后期 stage 每毫秒能塞进的参数量大一个数量级。于是把 ResNet50 的 stage depth 从 3-4-6-3 改成 2-3-6-5(Refined-ResNet50):参数量 25.6M→34.1M、延迟反而 4.6→4.4ms、精度 79.0→79.3%。同样的改法在 PVT-Small(20.7→17.9ms,79.8→80.2%)和 MixNetV 上都成立 → **容量应该堆在「参数密度高」的后期**。

**准则 3(混合 block 比纯 Transformer 高效)**:MixBlockA(Transformer/BottleNeck 双分支 channel split + concat)与 MixBlockB(串行,R=0.5)的 TeraFLOPS 在 56×56 时是 117/122,远高于纯 Transformer 的 81;TeraParams 也更高。因果:把一部分通道交给计算密集的 BottleNeck,整体延迟被拉回卷积量级,同时保留 Transformer 的全局建模 → **混合 = 用卷积分支「补」Transformer 的效率短板**。

**准则 4(global-then-local)**:把 MixBlockB 的两分支顺序调换(先 Transformer 全局、后 BottleNeck 局部 refine)得 MixBlockC,TeraFLOPS 与 B 相当但精度更高(MixNetC 81.2% vs MixNetB 80.3%),且不牺牲效率。因果:Transformer 的大感受野适合先抽全局信息,卷积再局部细化;反过来「先局部再全局」信息流更别扭。附带好处:B 的 R 被锁死在 0.5,C 的 R 可任意取 → **最终选 MixBlockC 作基本单元**。

### 3. TRT-ViT 整体结构

ResNet 式特征金字塔(分辨率逐 stage 减半、通道加倍),五个 stage:stem+stage1-4 用卷积(3×3 stem、BottleNeck),**只在最后 stage5(及大模型的 stage4)用 MixBlockC**;stage depth 浅前深后(A/B/C/D = 2-4-5-4 / 3-4-7-4 / 3-4-9-6 / 4-5-9-5)。MixBlockC 内参:R=0.5(shrinking ratio)、S=空间缩减比、K=7(BottleNeck 核大小)、MLP expansion=3、head dim=32;BottleNeck 用 BN+ReLU,Transformer 用 LN+GeLU(与 ConvNeXt/Swin 一致的现代配方)。

## 可复现细节

```
MixBlockC(最终采用的 block,对应论文 Fig.3(c)):
  input(C_in)
    ├─ 1×1 Conv 投影
    ├─ Transformer 分支(输出 C_out×R,S=空间缩减,含 LN+GeLU MLP,expansion=3,head_dim=32)
    ├─ BottleNeck 分支(输出 C_out×(1-R),K=7 大核,BN+ReLU)
    └─ Concat → 1×1 Conv → C_out     (global→local 顺序:Transformer 在前)
  stride=2 时用 Avg-Pool 下采样

stage 配置(TRT-ViT-C 为例):
  Stem   H/2×W/2   C=32   [3×3,3×3]
  Stage1 H/2×W/2   C=64   [3×3,3×3]×1
  Stage2 H/4       C=192  BottleNeck(1×1,3×3,1×1)×3
  Stage3 H/8       C=384  BottleNeck×4
  Stage4 H/16      C=768  BottleNeck×7 + MixBlockC(R=0.5,S=2,K=7)×2
  Stage5 H/32      C=1536 MixBlockC(R=0.5,S=1,K=7)×6

训练:ImageNet-1k 300 epoch,8×V100,total batch 1024,AdamW(wd=0.05),
      lr 1e-3 cosine + 30 epoch linear warmup,stochastic depth drop-path 0.1
延迟测量:TensorRT-8.0.3,T4 GPU,batch=8(下游任务输入 512×2048 / 800×1280)
```

> ⚠ **数据可复现限定**:上表与下文全部加速比仅在论文实验硬件与批处理设置(如 T4 / batch=8 / TRT-8.0.3;下游任务 batch=8)下成立;论文作者未开源代码(见 §实践要点),官方无可复现代码,数字以论文表格为准、无法独立复现。

## 数据对比

ImageNet-1K 分类(T4,TensorRT,batch=8;延迟单位 ms):

| 模型 | Param(M) | FLOPs(G) | Latency | Top-1(%) |
|---|---|---|---|---|
| ResNet50 | 25.6 | 4.1 | 4.6 | 79.0 |
| PVT-Small | 24.5 | 3.8 | 20.7 | 79.8 |
| Twins-PCPVT-S | 24.1 | 3.8 | 20.2 | 81.7 |
| ConvNeXt-T | 29.0 | 4.5 | 18.1 | 82.1 |
| CSWin-T | 23.0 | 4.3 | 30.2 | 82.7 |
| **TRT-ViT-A** | 29.3 | 2.7 | **4.1** | 81.2 |
| **TRT-ViT-B** | 43.1 | 4.0 | **5.6** | 82.1 |
| **TRT-ViT-C** | 67.3 | 5.9 | **9.2** | 82.7 |
| **TRT-ViT-D** | 103.0 | 9.7 | **15.1** | 83.4 |

读法:TRT-ViT-B 精度≈ConvNeXt-T 而快 2.3×(18.1→5.6ms);TRT-ViT-C 82.7% 同精度下比 CSWin-T 快 3.3×、比 PVT-Medium(30.0ms,81.2%)精度更高且快 3.3×;TRT-ViT-D 比 CSWin-S(52.0ms,83.6%)省约 70% 时间。

下游任务(骨干延迟,T4):
- **ADE20K 分割(Semantic FPN 80k)**:TRT-ViT-A 82ms/44.2 mIoU,比 ResNet101(114ms/38.8)又快又高 5.4;TRT-ViT-B 104ms/45.4,超 Twins-SVT-S(636ms/43.2)约 83% 延迟。
- **COCO 检测(Mask R-CNN 1×)**:TRT-ViT-B 110ms/43.6 AP^b,与 Twins-SVT-S(628ms/43.4)精度相当而快 4.7×。
- **迁移时 ViT 的速度塌方**:分类→检测,Twins-SVT-B 延迟涨 32×(38.9→1255ms),ResNet101 只涨 15×(7.4→113ms);TRT-ViT 因早期 stage 是卷积,迁移后仍保持 ResNet 级效率(这是「准则 1」在下游的延伸验证)。

消融:stage type C-C-C-M(81.2%,4.1ms)→ C-C-M-M(81.6%,9.1ms)→ C-M-M-M(81.8%,15.2ms):Transformer 越往前放,精度只微涨、延迟暴涨(准则 1 的直接证据)。shrinking ratio R:1.0(6.8ms,80.8%)→0.5(4.1ms,81.2%)→0.25(3.9ms,80.6%):R=0.5 延迟压 40% 精度还略升。

## 实践要点

- [ ] **部署导向设计先测延迟,别看 FLOPs**:在目标硬件+目标推理框架(如 TensorRT)上实测 TeraFLOPS/TeraParams,用它排 stage/block 布局。
- [ ] **Transformer 只放低分辨率后期 stage**(本文 stage5,大模型 stage4):高分辨率下 attention 访存/二次复杂度主导,TeraFLOPS 差卷积 4 倍以上。
- [ ] **容量堆后期**(浅前深后):后期 stage 的 TeraParams 高一个数量级,同延迟下能塞更多参数。
- [ ] **混合 block 用 global-then-local 顺序**(Transformer 先、卷积后),R≈0.5 是效率/精度甜点。
- [ ] 大核(K=7)BottleNeck + BN/ReLU 与 Transformer + LN/GeLU 并存,归一化/激活按分支类型选,别混用。
- [ ] 下游任务(检测/分割,大输入)会放大 ViT 的速度劣势,**选型务必在下游输入尺寸上重测延迟**。
- [ ] ⚠️ 论文是 preprint、**官方未开源**;复现只能参考社区实现(github.com/Phoenix8215/TRT-ViT,非官方),数字以论文表格为准。
- [ ] 跨域:TensorRT 的 kernel 选择/融合机制属推理引擎层,与 [CUTLASS 域](../../cutlass/index.md) 的 GEMM 模板库同源——TensorRT 内部大量用 CUTLASS 生成 GEMM kernel,理解 attention/GEMM 在硬件上的真实效率(本准则 1 的根因)可参 [cutlass/concepts/gemm-tiling.md](../../cutlass/concepts/gemm-tiling.md)。

## 溯源

- 主源:论文论文原文(链接见 frontmatter)。所有表格数字(TeraFLOPS/TeraParams、ImageNet/ADE20K/COCO 延迟与精度、消融)直接取自论文 Table 1-9。
- 网络综合:arXiv abs 页(2205.09579v2)核对标题/作者/版本;NVIDIA TensorRT 官方文档与 GitHub(核对 TensorRT 定位:high-performance DL inference SDK);社区复现 Phoenix8215/TRT-ViT(确认论文作者未放官方代码,bytedance/TRT-ViT curl 404)。全部 URL 于 2026-09-21 curl 实测存活(200);paperswithcode 对应页为软 404(重定向 HF trending),已弃用不登记。
- 跨域互链:CUTLASS(TensorRT 的 GEMM kernel 底座)见 [`../../cutlass/index.md`](../../cutlass/index.md);本域索引「相关条目」亦登记。
