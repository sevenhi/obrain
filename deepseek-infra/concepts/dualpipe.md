---
title: DualPipe — 双向流水线并行(计算-通信重叠)
type: concept
tags: [dualpipe, pipeline-parallelism, v3, communication-overlap, dualpipev, microbatch, bubble, training]
source:
  - "code: deepseek-ai/DualPipe | https://github.com/deepseek-ai/DualPipe ↑primary"
  - "web: DeepSeek-V3 Technical Report(arXiv:2412.19437)| https://arxiv.org/abs/2412.19437"
date: 2026-10
---

# DualPipe — 双向流水线并行(计算-通信重叠)

## TL;DR

**DualPipe 是 DeepSeek-V3 技术报告引入的「双向流水线并行」算法**,核心目标是在 PP(流水线并行)里**完全重叠前向与反向的计算-通信阶段**,同时**减少流水线气泡(pipeline bubble)**。它把传统的单向微批次流改成**正反两个方向同时流动**,让一段微批次的通信恰好填进另一段微批次的计算时间里,从而把 PP 的通信开销「藏起来」。

**关键结论**(对比同 PP stage 数):

| 方法 | 气泡 | 每设备参数 | 每设备激活 | 设备数 |
|---|---|---|---|---|
| 1F1B | (PP−1)(F+B) | 1× | PP | PP |
| ZB1P | (PP−1)(F+B−2W) | 1× | PP | PP |
| **DualPipe** | **(PP/2−1)(F&B+B−3W)** | **2×** | **PP+1** | PP |
| **DualPipeV** | (PP/2−1)(F&B+B−3W) | 2× | PP+1 | **PP/2** |

**代价**:DualPipe 的气泡减半靠「双向流动 + 保存两份参数」,每设备参数 **2×**、激活 **PP+1**(1F1B 是 1×/PP)。即用显存换吞吐。DualPipeV 是 Sea AI Lab 的「cut-in-half」派生的 V 形调度,用同样气泡换**设备数减半**(PP/2)。

---

## 1. 机制:为什么双向能重叠

### 1.1 传统 1F1B 的气泡与通信暴露

1F1B(one-forward-one-backward)每个 PP stage 交错跑一个前向、一个反向,但**前向和反向的通信是显式占用时间的**:前向要把激活发给下一个 stage、反向要把梯度发回上一个 stage,这些通信与计算串行,加上 warm-up/cooldown 造成的气泡,PP 利用率上不去。

### 1.2 DualPipe 的双向调度

DualPipe 让微批次**沿两个方向同时流动**(正向 + 反向对称),这样某个 stage 上「正向微批次的计算」与「反向微批次的通信」正好落在同一时段,互相填满:

```
PP stage k 时间轴(示意):
[ F_m forward 计算 ][ B_n backward 计算 ][ F_{m+1} forward ][ ... ]
     ↘ 与反向的通信重叠 ↑            ↘ 与正向的通信重叠
```

- **F&B**:两个互相重叠的前向/反向 chunk 的执行时间;
- **B−3W**:反向 chunk 减去「backward for weights」的时间(权重梯度那部分被单独切出重叠)。

结果是气泡从 1F1B 的 `(PP−1)(F+B)` 降到 `(PP/2−1)(F&B+B−3W)`——**气泡近似减半**(且 `F&B` 项本身就比 `F+B` 小,因为两个 chunk 重叠了)。

### 1.3 代价与 DualPipeV

- **参数 2×**:双向流动需要为两个方向各维护一份模型参数(或等效的状态),显存翻倍;
- **激活 PP+1**:比 1F1B 的 PP 多 1(双向流动的边界多一份)。

**DualPipeV**:Sea AI Lab 用「cut-in-half」把 DualPipe 切成 V 形调度,在**相同气泡**下把设备数从 PP 降到 **PP/2**(8 PP stage → 4 卡)。代价仍是参数 2×、激活 PP+1。

---

## 2. 可复现细节

### 2.1 接口

```bash
python examples/example_dualpipe.py    # DualPipe 调度示例
python examples/example_dualpipev.py   # DualPipeV 调度示例
```

- 依赖仅 **PyTorch 2.0+**;
- 真实应用需自己实现 `overlapped_forward_backward` 方法(把「重叠的前向+反向」包成一个可调度单元,贴合你的具体 module)——**仓库给的是调度框架,不是即插即用训练器**。

### 2.2 气泡公式符号

- `PP`:PP stage 数(偶数);
- `F`:一个前向 chunk 的执行时间;
- `B`:一个完整反向 chunk 的执行时间;
- `W`:一个「backward for weights」chunk 的执行时间;
- `F&B`:两个互相重叠的前向/反向 chunk 的执行时间。

---

## 3. 数据对比(同 PP stage 数,README 表)

| 方法 | Bubble | 参数/设备 | 激活/设备 | 设备数 |
|---|---|---|---|---|
| 1F1B | (PP−1)(F+B) | 1× | PP | PP |
| ZB1P | (PP−1)(F+B−2W) | 1× | PP | PP |
| DualPipe | (PP/2−1)(F&B+B−3W) | 2× | PP+1 | PP |
| DualPipeV | (PP/2−1)(F&B+B−3W) | 2× | PP+1 | PP/2 |

> V3 训练实践:V3/R1 用 DualPipe 做训练时的计算-通信重叠,配套 profile-data 仓库公开了各算子的通信-计算重叠测量数据。

---

## 4. 实践要点 / 坑

- **用显存换吞吐**:DualPipe 参数 2×、激活 PP+1,选它前先确认显存扛得住(或与 EP/ZeRO 组合分摊)。别只看「气泡减半」不看代价。
- **PP 必须是偶数**(公式里 `PP/2`),调度图按 8 PP rank / 20 微批次、两方向对称画。
- **仓库是调度算法示例**:`overlapped_forward_backward` 要自己实现;生产训练栈里 DualPipe 是「训练框架要集成的一种 PP 策略」,不是独立可跑的训练器。
- **DualPipeV 是社区派生**(Sea AI Lab cut-in-half),不是 DeepSeek 原版;两者气泡公式相同、设备数不同,别混为一谈。
- **与 DeepEP 的分工**:DualPipe 管「PP 层的调度与重叠」,DeepEP 的 `PPBuffer` 管「PP 通信原语 send/recv」(见 [deepep-nvidia-vs-ascend.md](deepep-nvidia-vs-ascend.md))——一个算账、一个搬数据。
- **论文溯源**:DualPipe 出自 DeepSeek-V3 技术报告(arXiv:2412.19437),无独立论文。

---

## 溯源

- **DualPipe 仓库**(主源:调度图、气泡/显存对比表、DualPipeV、接口):https://github.com/deepseek-ai/DualPipe
- **DeepSeek-V3 Technical Report**(DualPipe 出处,arXiv:2412.19437):https://arxiv.org/abs/2412.19437
- **profile-data**(计算-通信重叠测量):https://github.com/deepseek-ai/profile-data
- **Sea AI Lab "Cut-in-half" 博客**(DualPipeV 出处):https://hackmd.io/@ufotalent/r1lVXsa9Jg
- **库内交叉**:[deepep-nvidia-vs-ascend.md](deepep-nvidia-vs-ascend.md)(PP 通信原语)、[lplb-epb-load-balancing.md](lplb-epb-load-balancing.md)(EP 侧的负载均衡,与 PP 正交)、[../../model-architecture/concepts/deepseek-arch-v32-r1.md](../../model-architecture/concepts/deepseek-arch-v32-r1.md)(V3/R1 训练背景)。
