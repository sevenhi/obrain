---
title: smallpond — 基于 DuckDB + 3FS 的轻量数据处理框架
type: concept
tags: [smallpond, duckdb, 3fs, data-processing, dataframe, sql, graysort, pb-scale, deepseek]
source:
  - "code: deepseek-ai/smallpond | https://github.com/deepseek-ai/smallpond ↑primary"
  - "code: deepseek-ai/3FS(存储底座)| https://github.com/deepseek-ai/3FS"
date: 2026-10
---

# smallpond — 基于 DuckDB + 3FS 的轻量数据处理框架

## TL;DR

**smallpond 是 DeepSeek 的轻量数据处理框架,构建在 DuckDB + 3FS 之上**。定位:高性能(PB 级可扩展)、易用(无长驻服务)、数据管道友好。它是 3FS 的「加工层」——3FS 存数据,smallpond 算数据(GraySort 就是 smallpond 跑在 3FS 上,110.5 TiB 30 分 14 秒排序完)。

**关键特点**:用 DuckDB 当执行引擎(PB 级 SQL 处理)、以 3FS 当分布式存储、**无长驻服务**(不像 Spark 要常驻集群),`pip install smallpond` 即用。

---

## 1. 机制:DuckDB 执行 + 3FS 存储 + 无服务

### 1.1 三件套分工

| 组件 | 角色 |
|---|---|
| **DuckDB** | 列式 SQL 执行引擎,单机高性能、PB 级可扩展 |
| **3FS** | 分布式存储,数据/中间结果读写 |
| **smallpond** | 编排层:session、分区/重分区、partial SQL、写出 |

smallpond 把 DuckDB 的列式处理能力**分布式化**:`repartition` 按 hash 分桶、`partial_sql` 在各分区上执行 SQL、结果写回 3FS。

### 1.2 无长驻服务的取舍

与 Spark/Presto 不同,smallpond **不要求常驻服务进程**,数据管道按需起算——这对「AI 数据预处理」这类**批式、一次性的重数据加工**场景更轻。

### 1.3 数据流(Quick Start)

```python
import smallpond
sp = smallpond.init()                       # 初始化 session
df = sp.read_parquet("prices.parquet")      # 读数据(3FS/本地)
df = df.repartition(3, hash_by="ticker")    # 按 ticker 哈希分 3 桶
df = sp.partial_sql("SELECT ticker, min(price), max(price) FROM {0} GROUP BY ticker", df)
df.write_parquet("output/")                 # 写回
print(df.to_pandas())                       # 收结果
```

---

## 2. 可复现细节

- **安装**:`pip install smallpond`(Python 3.8–3.12);
- **API**:`smallpond.init()` 起 session → `read_parquet` / `repartition` / `partial_sql` / `write_parquet` / `to_pandas`;
- **文档**:`docs/source/getstarted.rst`、`docs/source/api.rst`;官方站 https://deepseek-ai.github.io/smallpond/;
- **开发**:`pip install .[dev]` + `pytest -v tests/test*.py`。

---

## 3. 数据(GraySort,与 3FS 同一口径)

| 场景 | 数值 | 集群 |
|---|---|---|
| GraySort | **3.66 TiB/min**(110.5 TiB,30 分 14 秒) | 50 计算节点 + 25 存储节点(3FS) |

> 该数字与 3FS README 的 GraySort 同源(脚本 `benchmarks/gray_sort_benchmark.py`),分两阶段:shuffle(按 key 前缀分区)+ 分区内排序,均读写 3FS。详见 [3fs.md](3fs.md)。

---

## 4. 实践要点 / 坑

- **定位是「数据加工层」**:smallpond 算、3FS 存,两者配套;单用 smallpond(无 3FS)也可跑本地文件,但 PB 级要靠 3FS 的分布式存储。
- **无长驻服务 = 按需起算**:适合批式预处理/ETL,不适合需要毫秒级交互查询的服务场景(那该用常驻引擎)。
- **partial_sql 的 `{0}` 占位**:表名用 `{0}` 占位传入 DataFrame,别写成字面表名。
- **repartition 的 hash_by**:控制分区键(如 ticker),决定数据如何打散到各桶——桶数与后续并行度/文件数直接相关。
- **与 3FS 的构建成本分离**:smallpond 本身 `pip install` 轻量;重成本在 3FS 的部署(见 [3fs.md](3fs.md))。先跑通本地 `pip install smallpond` 验证逻辑,再接 3FS。

---

## 溯源

- **smallpond 仓库**(主源:特性/Quick Start/API/GraySort 口径):https://github.com/deepseek-ai/smallpond
- **3FS 仓库**(GraySort 同源 + 存储底座):https://github.com/deepseek-ai/3FS
- **DuckDB**(执行引擎):https://duckdb.org/
- **官方文档站**:https://deepseek-ai.github.io/smallpond/
- **库内交叉**:[3fs.md](3fs.md)(存储底座 + GraySort 同口径)、[open-infra-index.md](open-infra-index.md)(开源周 Day 5 背景)。
