---
title: DeepJIT — xPU kernel 运行时 JIT 编译
type: concept
tags: [deepjit, jit, cuda, ascend, nvcc, bisheng, cache, kernel-compilation, header-only, xpu]
source:
  - "code: deepseek-ai/DeepJIT | https://github.com/deepseek-ai/DeepJIT ↑primary"
  - "code: deepseek-ai/DeepGEMM(DeepJIT 消费方)| https://github.com/deepseek-ai/DeepGEMM"
  - "code: deepseek-ai/DeepEP(DeepJIT 消费方)| https://github.com/deepseek-ai/DeepEP"
date: 2026-10-19
---
> 更新记录:2026-10-19 增量 — PR #29 修复异常 backtrace 的线程安全(catch 多线程并发抛错时符号化/反解不崩溃);此前 2026-10-09 增量已覆盖 SGID 共享缓存权限。最新 commit `18fe520`(2026-10-19 合并)。

# DeepJIT — xPU kernel 运行时 JIT 编译

## TL;DR

**DeepJIT 是一个轻量、header-only 的 C++20 JIT 运行时**,同时支持 **NVIDIA CUDA GPU** 与 **华为昇腾 NPU** 两种后端,给 C++/Python 扩展作者一个统一接口:运行时编译 kernel 源码 → 缓存编译产物(内存 + 磁盘)→ 加载到设备 → 按后端特定选项 launch。DeepSeek 的 DeepGEMM、DeepEP、DeepGEMM-Ascend、DeepEP-Ascend、TileKernels 等库都用它做运行时编译,从而**安装阶段零 CUDA/CANN 编译**。

**核心设计(为什么这样设计)**:
1. **kernel 库专注设备代码,JIT 基建交给 DeepJIT**——编译、缓存、加载、launch 四件事是每个 xPU 库的公共重复劳动,抽出来省每家各写一套;
2. **缓存键穷尽一切影响产物的输入**——source + tracked include 树 + 编译器版本 + 生效 flag + 应用提供的依赖签名(如 CUTLASS 版本),命中即复用,避免「改了头文件却还在用旧二进制」;
3. **支持跨节点共享缓存**(本地 + 分布式文件系统,原子 rename 发布),多用户/多进程/多节点复用同一份编译产物,省集群级编译时间。

---

## 1. 机制:两后端一套接口

### 1.1 统一 Runtime 抽象

```cpp
// CUDA 后端
using JIT = deep_jit::Runtime<deep_jit::CUDA>;
// 昇腾后端
using JIT = deep_jit::Runtime<deep_jit::Ascend>;
```

两个 `Runtime` 共享同一套**配置、源/include 哈希、内存/磁盘缓存、lazy 初始化**;kernel 源码与编译/launch 选项则各后端独立。后端对照:

| 维度 | CUDA | Ascend |
|---|---|---|
| 编译 | NVCC → CUBIN | bisheng + ld.lld → 链接 .o,解析 `.ascend.meta.*` kernel 名 |
| 加载/launch | CUDA Driver API | ACL(adv_api) |
| 设备查询 | `get_num_sms` / `get_num_l2_cache_bytes` / `get_arch` | `get_npu_arch` / `get_num_aicore_cores` / UB/L2 size |
| 依赖 | CUDA headers 12.4+、NVCC 12.9+ | CANN 含 `bin/bisheng`+`bin/ld.lld`、`adv_api` headers、torch_npu |
| 默认流 | 当前 PyTorch CUDA stream | 当前 torch_npu stream |

### 1.2 编译与 launch

```cpp
const auto kernel = jit->compile("scale", R"(
extern "C" __global__ void scale(float* output, const float* input, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) output[index] = input[index] * 2.0f;
}
)");
jit->launch(kernel, {.grid_dim = dim3((count+255)/256,1,1), .block_dim = dim3(256,1,1)},
            output, input, count);
```

`compile()` 在 cache miss 时编译、加载 CUBIN、返回进程内缓存的 `shared_ptr<Kernel>`;`compile_without_load()` 只产 artifact 不加载。`CompilerOptions`(优化级别、fast_math、`check_no_spills`、arch、`extra_nvcc_flags`、post_hook)未设字段继承运行时默认值,实现「默认 + 逐 kernel 覆盖」。

### 1.3 缓存键与磁盘布局(缓存正确性的关键)

CUDA 缓存 digest 按序由以下五部分 FNV-1a 拼接(**每段前缀定长字节,避免二进制串歧义**):

1. `Config::extra_signature`(库方声明的影响产物的外部依赖,如 CUTLASS 版本号);
2. `nvcc --version` 完整输出 hash;
3. 生效编译 flag(`include_dirs` 里的路径**刻意不进 digest**,直接写在 `nvcc_flags` 里的 `-I` 才进);
4. 选中的 `post_hook` 路径 + 文件内容 hash;
5. 源文件与 tracked include 树的 parser digest。

磁盘条目:`<cache-root>/cache/<tag>.<digest>/`,含 `kernel.cu`、`kernel.cubin`、`meta.json`、`.committed` 标记(有该标记即视为完成产物可直接加载)。

> ⚠ digest 是**快速 cache 校验和,不是加密哈希**;缓存根必须是可信目录。

### 1.4 include parser(易踩的边界)

递归跟踪 include 需**同时**满足两个条件:指令是 `#include <...>`(字面尖括号)+ 文件名以 `Config::include_prefixes` 之一开头。parser 是**行扫描器,不是 C 预处理器**:不解释注释里的 token、宏展开 include、反斜杠续行、条件编译(`#if 0` 里也扫)。`#include "kernel.cuh"`(引号)**被拒绝**。典型配置:

```cpp
deep_jit::Config("/opt/my_library", "MYLIB",
    "cutlass-40000",                          // extra_signature
    {"/opt/my_library/include"},              // include_dirs
    {"my_library/"});                         // include_prefixes
// #include <my_library/kernel.cuh> 被跟踪;#include <cutlass/cutlass.h> 不跟踪(靠 extra_signature 表示)
```

### 1.5 共享缓存

同一缓存目录可被多用户/进程/节点共享(`DJ_JIT_CACHE_DIR=/shared/deep_jit`)。构建用**独立临时目录 + 内容同步 + 原子 rename 发布**完成条目,并发编译同一 entry 安全。多用户可设 SGID + `chmod 2775`(DeepJIT 读一次 SGID,置目录 0775 / 文件 0664 忽略 umask)。也可 `DJ_JIT_CACHE_DIR="$HOME/.dj:/shared/deep_jit"` 组合个人可写根 + 只读共享查找根(miss 只写进第一个根)。

### 1.6 环境变量优先级

`MYLIB_<SUFFIX> > DJ_<SUFFIX> > 内建默认`(如 `MYLIB_JIT_CACHE_DIR > DJ_JIT_CACHE_DIR > $HOME/.dj`)。DeepGEMM 的 `DG_JIT_*`、DeepEP 的 `EP_JIT_*` 正是各自的 `MYLIB_` 前缀。布尔值接受 true/false、yes/no、任意整数(大小写不敏感)。须在**首次 Runtime 构造前**设,否则缓存根/编译器选择被快照。

---

## 2. 数据 / 能力(无公开基准,列能力面)

- DeepJIT 本身无公开 benchmark,它的价值在**下游库的部署收益**:DeepGEMM/DeepEP 靠它做到「安装不编译 CUDA、kernel 按当前设备形状/精度即时生成 + 磁盘缓存复用」。
- WIP:历史缓存预热(用历史条目预判未来需要的 kernel 提前编译)、Python 直编译 API——均未发布。

---

## 3. 实践要点 / 坑

- **header-only 嵌入**:把 `DeepJIT/include` 加进宿主 target 的 include path,选一个后端头 `#include <deep_jit/backend/cuda/backend.hpp>`(或 `ascend`);pybind11 下可 `deep_jit::register_python_api(module, jit)` 一键暴露 `get_jit()`。
- **GIL**:默认启用 GIL 管理并引入 pybind11;宿主自己管 GIL(如 torch.ops 调用)则编译加 `-DDJ_DISABLE_GIL=1`。
- **路径别带空白/元字符**:编译/缓存/include/hook 路径经简单 shell 命令传递,不能有空格或 shell 元字符。
- **cache digest 不发现的环境变量**:`NVCC_PREPEND_FLAGS`/`NVCC_APPEND_FLAGS`/`CPATH`/host 编译器选择/传递第三方头变更**不自动进 digest**——要么避免这类隐式输入,要么写进 `extra_signature`。
- **post_hook 确定性**:hook 必须对给定输入 CUBIN 确定性;导入的 Python 模块/环境/时间/网络不自动发现,须塞进 `extra_signature`。hook 路径/内容进 digest,改 hook 后要重启进程(哈希按线程缓存)。
- **CUDA Graph 兼容**:在捕获前先 compile+load kernel;已加载 kernel 的 launch 可捕获(含继承当前 PyTorch stream 时)。
- **每设备一个 Runtime**:Runtime 构造时快照当前设备/架构,别在设备间搬动已加载 kernel。
- **昇腾 launch**:`num_blocks` 必填,`ASCEND_LAUNCH_BLOCKING` 开启后每次 launch 后同步设备。
- **异常 backtrace 非线程安全(2026-10 已修)**:`get_backtrace()` 内部用 `execinfo`/`libdw.so.1`(dlopen 动态符号解析)做符号化与反解,同一进程多线程并发抛异常时静态 `DwflApi` 会被多路并发访问。PR #29 在 `get_backtrace()` 里加了一把**全局静态 `std::mutex` + `lock_guard`** 串行化符号化,保证多线程 JIT 编译/launch 中任一线程抛错都能安全打出带函数名、文件:行的 backtrace。启示:**别在并发路径里裸调线程不安全的符号化/demangle**,要符号化就串行化。

---

## 溯源

- **DeepJIT 仓库**(主源:两后端/Cache/Config/include parser/环境变量/共享缓存):https://github.com/deepseek-ai/DeepJIT
- **DeepGEMM**(`DG_JIT_*` 消费方):https://github.com/deepseek-ai/DeepGEMM
- **DeepEP**(`EP_JIT_*` 消费方):https://github.com/deepseek-ai/DeepEP
- **库内交叉**:[deepgemm-nvidia-vs-ascend.md](deepgemm-nvidia-vs-ascend.md)、[deepep-nvidia-vs-ascend.md](deepep-nvidia-vs-ascend.md)、[deepgemm-ascend.md](deepgemm-ascend.md)、[deepep-ascend.md](deepep-ascend.md)(四库都用 DeepJIT 运行时编译)、[tilekernels.md](tilekernels.md)。
