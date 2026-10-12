---
title: vLLM 生产部署与观测清单
type: checklist
tags: [vllm, llm-inference, serving, observability, deployment, performance]
source:
  - "web: vLLM 官方文档首页 | https://docs.vllm.ai/en/latest/ | 阅读日期 2026-09-27"
  - "web: vLLM Architecture Overview | https://docs.vllm.ai/en/latest/design/arch_overview/ | 阅读日期 2026-09-27"
  - "web: vLLM Production Metrics | https://docs.vllm.ai/en/latest/usage/metrics.html | 阅读日期 2026-09-27"
  - "web: vLLM Model Runner V2 Design Document | https://docs.vllm.ai/en/latest/design/model_runner_v2/ | 阅读日期 2026-09-27"
date: 2026-09
---

# vLLM 生产部署与观测清单

> **TL;DR**：上线 vLLM 不能只验证“接口返回 200”。先固定模型、GPU、版本和启动参数，再验证显存边界、健康检查、并发退化、延迟分位数、日志和升级回滚。API Server、Engine Core、GPU Worker 是不同进程角色，CPU、共享内存和进程数也要纳入容量规划。

## 1. 上线前固定基线

记录以下信息并放进部署仓库或运行记录：

- vLLM 版本、Git commit、Python/PyTorch/CUDA 版本
- GPU 型号、显存、驱动、计算能力和 GPU 数量
- 模型 revision、tokenizer revision、量化格式和 dtype
- `tensor_parallel_size`、`pipeline_parallel_size`、`data_parallel_size`
- `max_model_len`、`gpu_memory_utilization`、`max_num_batched_tokens`
- 是否启用 prefix caching、chunked prefill、speculative decoding、LoRA
- 容器镜像 digest，而不是只记录 `latest`

## 2. 显存与容量规划

按四类分别观察：

1. 模型权重：受参数量、dtype、量化格式和分片影响。
2. KV cache：受层数、KV heads、head dimension、上下文长度和并发影响。
3. 激活与临时 workspace：受 batch、prefill 长度、kernel 和 CUDA graph 影响。
4. 框架和通信开销：包括 NCCL、进程间通信、allocator 保留空间。

不要把 `gpu_memory_utilization` 当成容量证明。应使用真实请求分布逐步增加并发，找到 OOM 前的安全水位，并为升级、突发长请求和碎片保留余量。

## 3. API 和网络边界

- API 只绑定到需要的网卡；家庭局域网服务不要直接暴露到公网。
- 在反向代理或网关层处理 TLS、认证、限流和请求大小限制。
- 明确 OpenAI 兼容接口的模型名、流式响应、超时和错误码约定。
- 对长请求设置最大输入 token、最大输出 token 和请求超时。
- 将健康检查和推理请求分开：存活检查不应触发模型生成。
- 管理端口、指标端口和业务端口分开控制。

## 4. 性能验证

至少准备四组基准：

| 场景 | 关注指标 |
|---|---|
| 单请求短 prompt | TTFT、单请求延迟、显存基线 |
| 单请求长 prompt | prefill 吞吐、TTFT、峰值显存 |
| 多请求并发 | request throughput、输出 token 吞吐、p50/p95/p99 ITL |
| 长上下文混合请求 | 调度公平性、KV cache 使用率、OOM 和排队时间 |

每次只改变一个变量。不要把不同模型、量化、batch 和上下文长度的结果混在同一张表里。

## 5. 观测指标

优先采集：

- 请求数、成功/失败数、排队时间
- TTFT、ITL、端到端延迟的 p50/p95/p99
- prompt tokens、generation tokens、tokens/s
- KV cache 使用率和请求等待情况
- GPU 利用率、显存、功耗和温度
- CPU、RAM、共享内存和进程数
- OOM、CUDA/NCCL 错误、模型加载失败和重启次数

指标命名和版本随 vLLM 演进，使用当前版本官方 metrics 文档核对，不要永久依赖旧 dashboard 的字段名。指标名示例(以当前版本实测为准,勿硬编码进告警):`vllm:num_requests_waiting` / `vllm:gpu_cache_usage_perc` / `vllm:time_to_first_token_seconds` / `vllm:e2e_request_latency_seconds`(Histogram);基线与告警阈值的设定方法见 [引擎选型篇](inference-engine-selection.md) 的统一评测一节,启动参数与健康检查项见本文 §2-§4,三者口径保持联动。

## 6. 日志、升级和回滚

- 日志中记录模型 revision、镜像 digest、启动参数和 GPU 信息。
- 保留一份已验证的旧镜像和启动配置。
- 升级前先在单独端口做 smoke test：加载模型、完成短请求、完成流式请求、检查指标。
- 升级后比较 TTFT、ITL、吞吐、显存和错误率，不只看容器是否 running。
- 发生回归时先回滚镜像和配置，再定位版本差异。
- 不要让 `latest` 标签自动替换生产镜像。

## 7. 家庭服务器的最低安全基线

- 仅允许家庭 LAN 或 VPN 访问 API 和管理面板。
- 使用强认证；不要把 API key 写入镜像或 compose 文件并提交到 Git。
- Docker socket、模型目录和宿主机文件系统按最小权限挂载。
- 为模型缓存、日志和临时目录设置磁盘配额或监控。
- 将模型文件和配置纳入独立备份，但不要把运行时缓存当作唯一备份。

## 相关条目

- [Model Runner V2 设计文档](https://docs.vllm.ai/en/latest/design/model_runner_v2/)
- [生产指标](https://docs.vllm.ai/en/latest/usage/metrics.html)
- [vLLM V1 引擎架构](../concepts/v1-architecture.md)
- [API Server 与 OpenAI 兼容层](../concepts/api-server-openai-compat.md)
- [连续批处理与统一 Token 预算](../concepts/continuous-batching.md)
- [PagedAttention 与分页 KV Cache](../concepts/paged-attention.md)
- [KV Cache 量化](../../quantization/concepts/kv-cache-quantization.md)

## 溯源

- 主源:vLLM 官方文档——Architecture Overview(https://docs.vllm.ai/en/latest/design/arch_overview/)、Production Metrics(https://docs.vllm.ai/en/latest/usage/metrics.html)、Model Runner V2 设计文档(https://docs.vllm.ai/en/latest/design/model_runner_v2/),阅读日期 2026-09-27。
