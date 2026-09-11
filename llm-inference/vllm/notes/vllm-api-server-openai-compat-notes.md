# vLLM API server 层与 OpenAI 兼容接口实现研究笔记

> **代码版本**：commit `b74ac0bc61d8b8f74848664b1a3b4e878f047861`（`git describe`: `v0.28.1rc0-40-gb74ac0bc6`，main 分支）
> 本笔记基于该版本源码实读，所有路径相对仓库根目录 `vllm/`（Python 包目录）。

## 0. 导读：与经典结构的差异

网上多数资料（含旧版官方文档）描述的 `vllm/entrypoints/openai/api_server.py`（数千行巨石文件）+ `serving_chat.py` / `serving_completion.py` 结构在该版本已完成重构：

- `vllm/entrypoints/openai/api_server.py` 现在只是 **59 行的 deprecation shim**，转发到 `vllm.entrypoints.launchers`，并警告改用 `vllm server` 命令（`vllm/entrypoints/openai/api_server.py:24`）。
- 每个协议端点拆成独立子包，固定四件套：`api_router.py`（路由）+ `serving.py`（业务）+ `protocol.py`（pydantic 模型）+ 可选 `batch_serving.py`。例如 chat：`vllm/entrypoints/openai/chat_completion/`。
- 引擎启动逻辑移入 `vllm/entrypoints/launchers/`，FastAPI 构建移入 `launchers/app.py`，通用 serve 路由（health/metrics/version/tokenize 等）移入 `vllm/entrypoints/serve/`。

## 1. CLI 入口与启动流程

### 1.1 从 `vllm serve` 到进程拓扑

- CLI 入口点定义：`pyproject.toml:44` → `vllm = "vllm.entrypoints.cli.main:main"`。
- `vllm/entrypoints/cli/main.py:17` `main()`：惰性导入全部子命令模块（注释明确要求 lazy import 防止 eager import 破坏），每个模块通过 `cmd_init()` 返回 `CLISubcommand`（`vllm/entrypoints/cli/types.py`）列表注册进 argparse subparsers，最后 `args.dispatch_function(args)` 分发。
- `vllm/entrypoints/cli/serve.py:45` `ServeSubcommand.cmd()` 是服务启动的编排核心，按参数选择四种拓扑：
  - `--grpc` → `serve_grpc`；
  - 多端口 DP（`--data-parallel-multi-port-external-lb`）→ `run_dp_supervisor`（`vllm/entrypoints/launchers/dp_supervisor.py`）；
  - `--headless` 或 `api_server_count < 1` → `run_headless`（serve.py:178）：只起引擎进程（`CoreEngineProcManager`），不起 HTTP 层；
  - `api_server_count > 1` 或 Rust frontend → `run_multi_api_server`（serve.py:262）；
  - 默认单进程：`uvloop.run(run_server(args))`。

**细节 1：多 API server 进程共享同一端口**。`run_multi_api_server`（serve.py:291）先 `setup_server(args, reuse_port=True)` 建好 socket，再由 `CoreEngineProcManager` 启动引擎子进程、`APIServerProcessManager`（`vllm/v1/utils.py`）把 socket **经 fd 传递**给 N 个 API server 子进程，靠 `SO_REUSEPORT` 内核级负载均衡——多个 uvicorn worker 监听同一 `host:port`。子进程真正 bind 后回传实际 ZMQ 地址给引擎握手（serve.py:376 `gather_actual_addresses`），避免 TOCTOU 端口竞争。

### 1.2 单进程启动链（`run_server`）

`vllm/entrypoints/launchers/api_server/entry.py`（本文件即新版 api_server 主体）：

```
run_server(args)                          # entry.py:163，绑 socket
  └─ run_server_worker(...)               # entry.py:179
       ├─ build_async_engine_client(args) # entry.py:35，async context manager
       │    └─ build_async_engine_client_from_engine_args  # entry.py:67
       │         ├─ engine_args.create_engine_config()
       │         └─ AsyncLLM.from_vllm_config(...) → yield 引擎，finally shutdown
       ├─ build_and_serve(engine_client, ...)  # entry.py:115
       │    ├─ engine_client.get_supported_tasks()
       │    ├─ build_app(args, supported_tasks, model_config)   # FastAPI 构建
       │    ├─ init_app_state(engine_client, app.state, ...)    # 状态注入
       │    └─ serve_http(app, sock=sock, ...)                  # uvicorn 启动
       └─ await shutdown_task；finally sock.close()
```

注意 `run_server_worker` 的收尾顺序（entry.py:197）：**先退出引擎 async with 上下文，再 await 服务 shutdown task**，保证 HTTP 层完全停止后引擎才清理。

## 2. FastAPI 应用构建与路由注册

### 2.1 build_app（`vllm/entrypoints/launchers/app.py:19`）

- `FastAPI(lifespan=lifespan)`（lifespan 来自 `launchers/utils/server_utils.py`）；`--disable-fastapi-docs` 时关闭 openapi/docs/redoc。
- `app.state.args = args`；`app.root_path = args.root_path`（支持反代挂载）。
- 依次执行：`register_api_routers()`（核心路由）→ `attach_endpoint_plugins()`（插件路由最后注册）→ `init_exception_handler()`（`entrypoints/serve/exception_handling/`）→ `init_entrypoints_middleware()`（`entrypoints/serve/middleware/register.py`）→ `sagemaker_standards_bootstrap()`。

**细节 2：按任务条件注册路由**。`launchers/api_server/routers.py:12` `register_api_routers()` 拿到模型实际支持的 `supported_tasks` 后决定暴露哪些 API：`"generate"` in tasks 才注册 chat/completions/responses/anthropic/cohere（经 `vllm/entrypoints/generate/api_router.py:21` `register_generate_api_routers`）；pooling 类任务才注册 embeddings 等（`vllm/entrypoints/pooling/factories.py`）；transcription/realtime 走 `speech_to_text/factories.py`。即**一个 embedding 模型的服务器上根本不存在 /v1/chat/completions 路由**，而不是运行时报错。

### 2.2 中间件（`vllm/entrypoints/serve/middleware/register.py`）

`init_entrypoints_middleware` 注册：`CORSMiddleware`（`--allowed-origins` 等）→ `AuthenticationMiddleware`（`--api-key`，优先于 `VLLM_API_KEY` 环境变量）→ 可选 `XRequestIdMiddleware` → generate 任务下的 `ScalingMiddleware`（弹性 EP 扩缩容状态拦截）。

## 3. app.state 注入与 serving 对象构造

`launchers/api_server/app_state.py:22` `init_app_state()` 把所有处理器挂到 `app.state`，路由函数运行时按需取用：

- `state.engine_client`（EngineClient 协议实现，即 AsyncLLM）；
- `OpenAIServingModels`（+ `init_static_loras()`，管理 base model 与 LoRA 的暴露名，`vllm/entrypoints/openai/models/serving.py`）；
- `OnlineRenderer` / `OnlineDerenderer`（`vllm/renderers/online_renderer.py`，统一负责 chat template 渲染与请求/响应双向转换，构造后立即 `warmup()`）；
- `ServingTokenization`（/tokenize 端点）；
- `init_generate_state()`（`vllm/entrypoints/generate/api_router.py:57`）构造并挂载：`state.openai_serving_chat = OpenAIServingChat(**_chat_kwargs)`、`openai_serving_chat_batch`、`openai_serving_completion`、`openai_serving_responses`、`anthropic_serving_messages`、cohere（受 `VLLM_ENABLE_COHERE_API` 门控，SDK 缺失时降级为 None）等。

设计模式：**路由层极薄，全部业务在 serving 类**。例如 `chat_completion/api_router.py:33` `def chat(request) -> OpenAIServingChat: return request.app.state.openai_serving_chat`，handler 为 None（任务不支持）时抛 `NotImplementedError`。

## 4. /v1/chat/completions 完整调用链

一条请求从 HTTP 进入到交给引擎的完整链路（本节为核心）：

### 4.1 路由层（`vllm/entrypoints/openai/chat_completion/api_router.py:41`）

```python
@router.post("/v1/chat/completions",
             dependencies=[Depends(validate_json_request)], ...)
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    handler = chat(raw_request)               # app.state.openai_serving_chat
    generator = await handler.create_chat_completion(request, raw_request)
    # 三种返回形态分流：
    #   ErrorResponse        → JSONResponse(error.code)
    #   ChatCompletionResponse → JSONResponse（非流式一次返回）
    #   AsyncGenerator       → StreamingResponse(media_type="text/event-stream")
    return StreamingResponse(
        content=with_sse_keep_alive(generator, keep_alive_interval), ...)
```

三个装饰器各司其职（`vllm/entrypoints/serve/utils/api_utils.py`）：
- `validate_json_request`（api_utils.py:350）：强制 `Content-Type: application/json`；
- `with_cancellation`（api_utils.py:51）：不用 `request.is_disconnected()`（与中间件不兼容），而是 starlette StreamingResponse 同款双任务模式——同时 await 断连消息与业务任务，客户端断开即取消生成；
- `load_aware_call`（api_utils.py:100）：`--enable-server-load-tracking` 时对 `app.state.server_load_metrics` 计数并在响应头回带（Orca 指标）。

### 4.2 业务层（`vllm/entrypoints/openai/chat_completion/serving.py`）

`OpenAIServingChat.create_chat_completion()`（serving.py:244）先用 `_with_kv_transfer_rejection_cleanup`（基类 `GenerateBaseServing`，`vllm/entrypoints/generate/base/serving.py:278`）包住核心逻辑，再进入 `_create_chat_completion()`（serving.py:260）：

1. **解析器构造**：`self.parser_cls(tokenizer, request.tools, ...)`（`vllm/parser/` 的 Parser，统一工具调用 + reasoning 增量解析）。
2. **渲染**：`render_chat_request(request)`（serving.py:217）→ `_check_model()`（模型名/LoRA 校验）→ 引擎健康检查 `engine_client.errored` → `self.online_renderer.render_chat(request)` 返回 `(conversation, engine_inputs)`。
3. **request_id**：`f"chatcmpl-{...}"`，多 prompt 时子请求编号 `{request_id}_{i}`；`RequestResponseMetadata` 挂到 `raw_request.state`。
4. **逐 engine_input 调度**：
   - `get_max_tokens(max_model_len, request.max_completion_tokens or max_tokens, prompt_len, default_sampling_params, ...)`（`serve/utils/api_utils.py`）解析最终 max_tokens；
   - `request.to_sampling_params(max_tokens, default_sampling_params)` 或 `to_beam_search_params`；
   - `self._log_inputs(...)`（`--enable-log-requests` 时记日志）；
   - 关键调用（serving.py:370）：
     ```python
     generator = self.engine_client.generate(
         engine_input, sampling_params, sub_request_id,
         lora_request=..., trace_headers=..., priority=...,
         data_parallel_rank=..., session_id=...,
         reasoning_ended=..., reasoning_parser_kwargs=...)
     ```
5. **分流**：`request.stream=True` → 返回 `chat_completion_stream_generator(...)`（serving.py:452，async generator，逐 RequestOutput 转 SSE chunk）；否则 `await chat_completion_full_generator(...)`（serving.py:914，聚合为单个 ChatCompletionResponse）。

### 4.3 SSE keep-alive（`vllm/entrypoints/openai/sse_keep_alive.py:19`）

**细节 3**：`with_sse_keep_alive(generator, interval)` 解决代理/负载均衡（Cloudflare Tunnel、NGINX `proxy_read_timeout`、AWS ALB）因流空闲而掐连接的问题：空闲超过 `interval` 秒就吐一行 SSE 注释 `": keep-alive\n\n"`（SSE 客户端忽略注释，但代理视为字节活动）。实现上用 `asyncio.wait({next_task}, timeout=interval)` 而非 `wait_for`——后者超时会**取消**正在推进的 `anext()` 任务，而 `asyncio.wait` 只观察不取消；单保留的 next_task 避免并发 `__anext__`。`interval <= 0` 时直接返回原 generator，零开销。

## 5. 协议定义与参数校验（`vllm/entrypoints/openai/chat_completion/protocol.py`）

### 5.1 ChatCompletionRequest（protocol.py:213）

继承 `OpenAIBaseModel`（`vllm/entrypoints/openai/engine/protocol.py`，`extra="allow"` 逻辑所在），字段按 OpenAI 官方文档顺序组织，源码中用 `--8<-- [start:...]` 标记块划分三段：

1. **OpenAI 标准字段**：`messages`、`model`、`frequency_penalty`、`logit_bias`、`logprobs`/`top_logprobs`、`max_tokens`（pydantic `deprecated` 标注，让位于 `max_completion_tokens`）、`n`、`presence_penalty`、`response_format`、`seed`（ge/le 限定 int64）、`stop`、`stream`/`stream_options`、`temperature`、`top_p`、`tools`/`tool_choice`、`reasoning_effort`、`parallel_tool_calls`、`user`（注释明示 vLLM 忽略）。
2. **vLLM 扩展采样参数块**：`use_beam_search`、`top_k`、`min_p`、`repetition_penalty`、`length_penalty`、`stop_token_ids`、`ignore_eos`、`min_tokens`、`skip_special_tokens`、`allowed_token_ids`、`bad_words` 等。
3. **vLLM 扩展服务参数块**：`echo`、`add_generation_prompt`、`continue_final_message`、`chat_template`（请求级覆盖 Jinja 模板）、`chat_template_kwargs`、`mm_processor_kwargs`、`structured_outputs`、`priority`、`request_id`（默认 `random_uuid`）、`session_id`、`return_token_ids` 等。

### 5.2 校验器（体现工程细节）

**细节 4**：pydantic 校验分两趟修复消息体（protocol.py:522）：
- `_normalize_messages_before`（`model_validator(mode="before")`）：把 `tool_calls` 的生成器/迭代物转成 list（防止一次性生成器在 union 类型匹配中被消费掉）；把废弃字段 `reasoning_content` 改名为 `reasoning`，下游只需处理一个字段。
- `_materialize_tool_calls_after`（`mode="after"`）：即便 before 已转 list，pydantic 对 `Iterable[...]` 类型校验时会**重新包成 ValidatorIterator**，此趟再物化回 list，保证下游 tokenizer / `model_dump_json` 永远拿到 list。
- `PrivateAttr _grammar_from_parser`（protocol.py:567）：仅供 parser-engine adapter 的 `adjust_request` 设置，控制 Mistral grammar 场景下 reasoning 结束判定。

### 5.3 to_sampling_params（protocol.py:661）

请求参数 → 引擎 `SamplingParams` 的映射规则：
- `temperature/top_p/top_k/min_p/repetition_penalty` 为 None 时依次回退：请求值 → `default_sampling_params`（模型 `generation_config.json`）→ 类常量 `_DEFAULT_SAMPLING_PARAMS`；
- `stop_token_ids` 与服务端默认（如 gpt-oss 的 `</call>`）用 `dict.fromkeys` 去重合并；
- `echo=True` 时自动开启 `prompt_logprobs=top_logprobs`；
- `kv_transfer_params`/`ec_transfer_params` 塞进 `extra_args` 透传；
- `output_kind=RequestOutputKind.DELTA if self.stream else FINAL_ONLY`——流式与非流式在引擎侧就是不同的输出聚合模式；
- `structured_outputs=self.extract_structured_outputs()`：统一归一 `response_format` 与 vLLM 私有 `structured_outputs` 两种来源。

`build_chat_params()`（protocol.py:570）则负责渲染侧参数：`reasoning_effort != None` 且用户未显式给 `enable_thinking` 时自动注入（Gemma4 类模板需要显式 opt-in）；无 `tools` 的请求把默认 `tool_choice="none"` 折叠为 None，避免模板渲染出模型可见的工具选择指令。

## 6. 与引擎对接：AsyncLLM（`vllm/v1/engine/async_llm.py`）

### 6.1 接口契约

- `vllm/engine/protocol.py:41` `EngineClient` 是抽象基类（Protocol class），定义 `generate/encode/abort/check_health/do_log_stats/...` 完整客户端契约，API 层只依赖此协议。
- `vllm/v1/engine/async_llm.py:72` `AsyncLLM(EngineClient)` 是 V1 引擎的异步门面。入口侧经 `AsyncLLM.from_vllm_config(...)`（async_llm.py:206）创建。

### 6.2 generate() 异步流（async_llm.py:551）

docstring 概括四步：**1) 为请求建 AsyncStream（实现类 `RequestOutputCollector`）；2) 处理输入；3) 加入 Detokenizer；4) 加入 EngineCore（独立进程）**。后台 `_run_output_handler` 任务（async_llm.py:666）持续从 EngineCore 输出队列拉取结果放入各请求的 collector。`generate()` 本体则循环：

```python
out = q.get_nowait() or await q.get()   # 先无 await 排空，负载下减少任务切换
finished = out.finished
if out is not STREAM_FINISHED: yield out
```

异常分类处理（体现工程化程度）：`CancelledError/GeneratorExit`（客户端断连/生成器被 GC）→ `await self.abort(request_id, internal=True)`；`EngineDeadError` → 直接上抛不 abort；`VLLMClientError`（请求校验错）→ 上抛；`InputStreamError` → abort 并解包 cause。

### 6.3 add_request（async_llm.py:283）

入口校验：`self.errored` → 抛 `EngineDeadError`；`kv_sharing_fast_prefill` 与 `prompt_logprobs` 组合冲突 → 抛 `VLLMValidationError`；prompt 是 `AsyncGenerator` 时走 `_add_streaming_input_request`（流式输入，如逐 token 到达的场景，且此路径不支持 reasoning 参数）。

## 7. 生命周期管理

- **uvicorn 不接管信号**：`launchers/launcher.py:38` `NoSignalServer(uvicorn.Server)` 重写 `capture_signals()` 为空 contextmanager——uvicorn 自带的 SIGINT/SIGTERM handler 会与上层竞争（源码注释引 #49668），改由 CLI 层（serve.py:201/281 的 `signal_handler`）统一实现优雅退出。
- **watchdog**：`serve_http`（launcher.py:50）为每个 server 起 `watchdog_loop(server, app.state.engine_client)` 后台任务监控引擎健康。
- **优雅停机**：`run_multi_api_server` 收到信号后按 `vllm_config.shutdown_timeout` 给 API server、引擎管理器、协调器依次限时停机（serve.py:394-410）。
- **forkserver 预热**：`entry.py:42` 当 `VLLM_WORKER_MULTIPROC_METHOD=forkserver` 时，把 `vllm.v1.engine.async_llm` 预导入 forkserver，避免每个 worker 子进程重复 import 的冷启动开销。

## 8. 架构小结

- **入口分层**：CLI（`cli/`）→ 启动编排（`launchers/`）→ 协议子包（`openai/chat_completion|completion|responses/`、`anthropic/`、`cohere/`）→ 通用 serve 设施（`serve/`：中间件、异常、tokenize 等）。
- **薄路由 + 状态注入**：路由函数只从 `app.state` 取 serving 对象并做返回形态分流（JSON / SSE 流），业务全在 per-protocol 的 serving 类中，便于 Anthropic/Cohere 等兼容层复用 `OnlineRenderer`/`GenerateBaseServing` 基类。
- **条件化路由注册**（supported_tasks 驱动）+ **多进程拓扑**（headless / 多 API server + SO_REUSEPORT / DP supervisor / Rust frontend）是这一版本服务层最显著的演进。
