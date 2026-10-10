---
title: open-infra-index — DeepSeek 开源基础设施总索引(开源周)
type: concept
tags: [open-infra-index, deepseek, open-source, flashmla, deepep, deepgemm, dualpipe, eplb, 3fs, smallpond, production]
source:
  - "code: deepseek-ai/open-infra-index | https://github.com/deepseek-ai/open-infra-index ↑primary"
  - "web: Insights into DeepSeek-V3(ISCA25,arXiv:2505.09343)| https://arxiv.org/abs/2505.09343"
date: 2026-10
---

# open-infra-index — DeepSeek 开源基础设施总索引(开源周)

## TL;DR

**open-infra-index 是 DeepSeek 基础设施开源的「目录/门户」仓库**,把 DeepSeek 线上服务的「生产级基础设施」按时间线登记成索引:2025-02 的「Open-Source Week」(一天放一个 repo)、2024-2025 的论文、以及各子系统的 README 入口。它本身**不含实现代码**,价值在于**给了这套基建的官方定位、性能口径、以及「生产验证」的出处**。

**一句话**:它是本域(deepep/deepgemm/dualpipe/eplb/3fs/smallpond/flashmla 等)的**官方总账**,写这些仓库的「开源背景 + 一手性能口径」都以它为准(如 FlashMLA 3000 GB/s / 580 TFLOPS、3FS 6.6 TiB/s、V3/R1 线上 73.7k/14.8k tok/s)。

---

## 1. 内容结构

### 1.1 2025-02 Open-Source Week(一天一 repo)

| Day | 仓库 | 定位(官方口径) |
|---|---|---|
| 1 | FlashMLA | MLA 高效解码内核,Hopper,可变长序列;BF16 3000 GB/s(访存界)/ 580 TFLOPS(算力界,H800),分页 KV cache(block 64) |
| 2 | DeepEP | 首个开源 EP 通信库;NVLink/RDMA 节点内+节点间;FP8 dispatch;训练+推理 prefill 高吞吐 / 解码低延迟 |
| 3 | DeepGEMM | FP8 GEMM 库,dense+MoE;Hopper 1350+ TFLOPS;~300 行核心逻辑;纯 JIT 编译 |
| 4 | DualPipe + EPLB + profile-data | 双向流水线并行 + 专家并行负载均衡 + 计算-通信重叠测量 |
| 5 | 3FS + smallpond | Fire-Flyer 分布式文件系统 + 数据加工框架;6.6 TiB/s 聚合读、GraySort 3.66 TiB/min |
| 6 | V3/R1 推理系统概览 | 跨节点 EP 批扩展 + 计算-通信重叠 + 负载均衡;线上 **73.7k/14.8k** tok/s(输入/输出,H800 节点),成本利润率 **545%** |

### 1.2 论文索引

- **ISCA25 Industry Track**(2025):《Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures》,arXiv:2505.09343;
- **SC24 AI Infrastructure**:《Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning》,arXiv:2408.14158;
- **202504**:《The Path to Open-Sourcing the DeepSeek Inference Engine》(推理引擎开源路线)。

---

## 2. 为什么它值得单独入库

1. **它是本域「生产验证」的锚**:库内多篇仓库的「战生产」声明(FlashMLA battle-tested、DeepEP 线上、3FS V3/R1 数据访问)都以本仓库的公告口径为准;
2. **它给出统一性能口径**:开源周各 Day 的「3000 GB/s / 580 TFLOPS / 1350 TFLOPS / 6.6 TiB/s / 73.7k tok/s」是本域原始一手数字,各专项篇引用时以此为源头(注:flashmla.md 的 660 TFLOPS 是 2025-04 新内核口径,580 TFLOPS 是开源周首发口径,两者时间不同不矛盾);
3. **它标记了 DeepSeek 基建的开源节奏**:2025-02 集中放「训练/推理/存储」三件套,2025-09/2026-09 补「昇腾适配 + 稀疏/并行新方向」——理解这套栈的演进时间线要靠它。

---

## 3. 实践要点 / 坑

- **它不含代码,是索引**:不要试图在这里「装」或「跑」任何东西;查实现去各子仓库,查定位/口径/时间线来这里。
- **性能数字是「当时」的一手口径**:开源周是 2025-02 的数据(H800/Hopper 时代),已落后于各仓库当前 README(如 DeepGEMM 现已 1550 TFLOPS、V2 EP 已重构)。引用时标时间,别当最新。
- **Day 4 的 eplb 链接是小写 `deepseek-ai/eplb`**(公告原文),与现行仓库 `deepseek-ai/EPLB` 同库;Day 5 的 3FS 是 `deepseek-ai/3fs`(小写),现行 `deepseek-ai/3FS`——GitHub 大小写不敏感,同一仓库。
- **V3/R1 线上数据是成本核算口径**:73.7k/14.8k tok/s 是「输入/输出 per H800 node」,545% 是「成本利润率」,不是纯吞吐 benchmark,别与算子 microbenchmark 混。

---

## 溯源

- **open-infra-index 仓库**(主源:开源周各 Day 定位与口径、论文索引):https://github.com/deepseek-ai/open-infra-index
- **ISCA25 论文**(DeepSeek-V3 硬件洞察,arXiv:2505.09343):https://arxiv.org/abs/2505.09343
- **SC24 Fire-Flyer AI-HPC 论文**(arXiv:2408.14158):https://arxiv.org/abs/2408.14158
- **库内交叉**:本域所有专项篇(deepgemm/deepep/dualpipe/3fs/smallpond/flashmla/deepselect)的「开源背景 + 一手口径」都溯源到本仓库;flashmla 见 [flashmla.md](flashmla.md)、deepselect 见 [deepselect.md](deepselect.md)。
