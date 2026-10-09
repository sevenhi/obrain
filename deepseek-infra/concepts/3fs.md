---
title: 3FS — Fire-Flyer 分布式文件系统(存储层)
type: concept
tags: [3fs, fire-flyer, distributed-filesystem, storage, rdma, craq, foundationdb, disaggregated, kvcache, checkpoint, graysort]
source:
  - "code: deepseek-ai/3FS | https://github.com/deepseek-ai/3FS ↑primary"
  - "code: deepseek-ai/open-infra-index(开源周 Day 5 口径)| https://github.com/deepseek-ai/open-infra-index"
  - "web: Fire-Flyer AI-HPC(SC24,arXiv:2408.14158)| https://arxiv.org/abs/2408.14158"
date: 2026-10
---

# 3FS — Fire-Flyer 分布式文件系统(存储层)

## TL;DR

**3FS(Fire-Flyer File System)** 是 DeepSeek 的**高性能分布式文件系统**,针对 AI 训练/推理负载设计,把**数千块 SSD 的吞吐**与**数百个存储节点的网络带宽**聚合成一个「位置无关」的共享存储层。核心卖点:**disaggregated(存算分离)架构 + 强一致语义 + 标准文件接口**。

**关键数据(README 一手)**:180 节点集群(每节点 2×200Gbps IB + 16×14TiB NVMe)聚合读吞吐 **6.6 TiB/s**;GraySort 25 存储节点 + 50 计算节点,110.5 TiB / 8192 分区 **30 分 14 秒**排序完,平均 **3.66 TiB/min**;KVCache 单客户端读峰值 **40+ GiB/s**。

**支撑场景**:数据预处理、dataloader(免预取/免 shuffle,随机访问训练样本)、并行 checkpoint、推理 KVCache(替代 DRAM 缓存的低成本大容量方案)。

---

## 1. 机制:存算分离 + 强一致 + 文件接口

### 1.1 三大设计支柱

| 支柱 | 做法 | 收益 |
|---|---|---|
| **Disaggregated 架构** | 存储与计算分离,应用以「locality-oblivious」方式访问任意存储 | 数千 SSD 吞吐 + 数百节点带宽聚合,扩容独立 |
| **强一致性** | 用 **CRAQ(Chain Replication with Apportioned Queries)** 实现强一致 | 应用代码简单、易推理,不必自己处理一致性 |
| **文件接口** | 无状态元数据服务,后端是**事务型 KV store(FoundationDB)** | 复用熟悉的文件 API,不用学新存储原语 |

### 1.2 为什么不用现成存储

AI 训练/推理的访问模式(海量随机读训练样本、突发大块读 checkpoint、KVCache 的高 IOPS + 大容量)与传统 HPC/云存储都不完全匹配。3FS 用「SSD + RDMA 全带宽」直连聚合,配合 CRAQ 强一致 + FoundationDB 元数据,专为这类负载做了取舍。

### 1.3 支撑的四类负载

1. **数据准备**:把分析流水线的输出组织成层级目录,高效管理海量中间结果;
2. **dataloader**:免 prefetch/shuffle,计算节点随机访问训练样本;
3. **checkpointing**:大规模训练的高吞吐并行 checkpoint;
4. **KVCache(推理)**:缓存 KV 向量,提供比 DRAM 缓存更便宜、更大容量的替代。

---

## 2. 可复现细节

### 2.1 构建(依赖繁,按发行版)

```bash
git clone https://github.com/deepseek-ai/3fs
cd 3fs && git submodule update --init --recursive && ./patches/apply.sh
# 依赖(Ubuntu 22.04 为例):
# apt install cmake libuv1-dev liblz4-dev ... libboost-all-dev build-essential
# 另需:libfuse 3.16.1+、FoundationDB 7.1+、Rust 1.75+(推荐 1.85+)
cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++-14 -DCMAKE_C_COMPILER=clang-14 \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSHUFFLE_METHOD=g++10
cmake --build build -j 32
```

> ⚠ **`-DSHUFFLE_METHOD` 必显式指定**:3FS 历史用了 `std::shuffle`,不同编译器版本(g++10 vs g++11+)产出的二进制**不兼容**。已有集群用「当初部署时的编译器对应值」,新集群二选一但部署后必须保持一致。

### 2.2 测试集群

`deploy/README.md` 给 setup guide;benchmark 用 `benchmarks/fio_usrbio/` 的 fio 引擎。

---

## 3. 数据(README 一手)

| 场景 | 数值 | 集群口径 |
|---|---|---|
| 聚合读吞吐 | **6.6 TiB/s** | 180 存储节点(2×200Gbps IB + 16×14TiB NVMe/节点),500+ 客户端,带训练背景流量 |
| GraySort | **3.66 TiB/min** | 25 存储节点 + 50 计算节点,110.5 TiB / 8192 分区,30 分 14 秒 |
| KVCache 读 | **40+ GiB/s** | 单客户端(1×400Gbps NIC)峰值 |

GraySort 分两阶段:(1)按 key 前缀 shuffle 分区、(2)分区内排序,两阶段都读写 3FS。

---

## 4. 实践要点 / 坑

- **构建重**:libfuse + FoundationDB + Rust + 一堆 C++ 依赖,按发行版(20.04/22.04/openEuler/OpenCloudOS/TencentOS)选对应安装命令;Docker 有现成 build 镜像(TencentOS-4 / OpenCloudOS-9)。
- **`SHUFFLE_METHOD` 兼容性(P0)**:编译时锁定 shuffle 算法,集群内所有构建必须同值,否则二进制不兼容——这是 3FS 最易踩的部署坑。
- **强一致 ≠ 快**:CRAQ 给强一致,代价是写路径有链式复制开销;读写混合负载要按需权衡(读多写少的 AI 负载正合适)。
- **定位边界**:3FS 是**存储层**,数据加工层是 smallpond(见 [smallpond.md](smallpond.md)),两者配套(GraySort 就是 smallpond 跑在 3FS 上)。
- **KVCache 是它的推理杀手级应用**:用 SSD 大容量替代 DRAM 缓存 KV,省成本;对 latency 敏感的场景需注意与 DRAM 方案的时延差异。

---

## 溯源

- **3FS 仓库**(主源:三大支柱/四类负载/构建/性能三表):https://github.com/deepseek-ai/3FS
- **open-infra-index(开源周 Day 5)**(6.6 TiB/s / 3.66 TiB/min 官方口径):https://github.com/deepseek-ai/open-infra-index
- **SC24 Fire-Flyer AI-HPC 论文**(软件-硬件协同,arXiv:2408.14158):https://arxiv.org/abs/2408.14158
- **库内交叉**:[smallpond.md](smallpond.md)(数据加工层)、[open-infra-index.md](open-infra-index.md)(开源周背景)、[../../vllm/index.md](../../vllm/index.md)(KVCache 推理侧消费)。
