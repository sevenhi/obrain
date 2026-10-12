---
title: dsh-node-addon-require-builtin — 不亮 --expose-internals 拿到 Node 私有 requireBuiltin 的原生插件
type: howto
tags: [deepseek, harness, node, node-api, napi, v8, addon, native, electron, internal-module]
date: 2026-10-12
source:
  - "code: dsh-node-addon-require-builtin 源码 | https://github.com/deepseek-ai/dsh-node-addon-require-builtin | master@a58d3599(0.1.9), 2026-10-12 | ↑primary"
  - "web: node_addon_require_builtin 主 README | 同上 | 2026-10-12"
  - "web: docs/architecture.md | 同上 | 2026-10-12"
---

# dsh-node-addon-require-builtin — 不亮 `--expose-internals` 拿到 Node 私有 `requireBuiltin` 的原生插件

> **TL;DR**:DeepSeek 为 [deepseek-harness](./deepseek-harness.md)(`dsh`)开的本机 Node 插件(DeepSeek Harness 的内部组件,2026-09-30 建仓)。它让普通进程**不需要 `node --expose-internals` 启动**就能拿到 Node 引导用的私有 `requireBuiltin()` 函数,从而加载 `internal/modules/cjs|esm/loader` 这类内部模块。做法不是纯公开 N-API:插件通过一个稳定的 N-API v9 入口加载,却要**探测私有 Node/V8 运行时态**——动态解析 `node::PrincipalRealm::builtin_module_require()` 符号 → 解析该 getter 的机器码拿到 Realm 字段偏移 → 读 V8 embedder-data 里的 Realm 指针 → 按平台 C++ 调用约定回调 getter → 用公开 N-API 校验返回的确是函数 → 再冒烟自检。**核心设计哲学是 fail-closed**:任何一步不满足运行时不变量就拒绝加载,绝不猜偏移、绝不加载未检验的内部模块。

## 1. 是什么 / 解决什么问题

- 一个 **Node 原生插件(node native addon)**,C++ 写成,通过 `node-addon-api`(N-API v9)提供对 JS 的入口。
- 解决的核心痛点:**Node 内部模块**(`internal/...`)默认只在 `node --expose-internals` 启动时才可通过 `require` 拿到;但生产进程改启动参数不现实(别人启动、改 flag 影响面大)。这个插件在**不改 Node 启动参数**的前提下,把引导阶段 Node 自己用的那个私有 `requireBuiltin` 函数「偷」出来。
- 归属:**enhancement of deepseek-harness**——README 官方描述 "An internal component used by DeepSeek Harness",homepage 指向 https://deepseek.com/harness/。它和 dsh 同属 DeepSeek 的「非模型」工具侧。

### 两个产品族(Product)

| 产品 | npm 包 | 行为 | 用途 |
|---|---|---|---|
| `internal-loader`(默认) | `node-addon-internal-loader` | **白名单**只放 `internal/modules/cjs/loader`、`internal/modules/esm/loader` 两个 id,白名单在 C++ 层、调用私有 `requireBuiltin` **之前**强制 | 只想要 CJS/ESM loader 内部(拿 cascaded loader 等) |
| `require-builtin` | `node-addon-require-builtin` | **不限 id**:`isAllowedInternalId()` 恒 true,任何字符串 id 转发给 Node | 需要无限制访问内部模块 |

两族共享一个加载器包 `node-addon-native-custom-loader`。JS 侧暴露三个小 API:`requireBuiltin(moduleId)`、`isAllowedInternalId(moduleId)`、`getBindingInfo()`(惰性返回 mode/product/backend/abi/bindingSource/bindingPath 诊断信息)。

## 2. 机制:如何不靠 `--expose-internals` 拿到私有 `requireBuiltin`

### 2.1 全链路图

```
JS entry package / shared optional-package loader
  -> Node-API addon 入口 (node_api_addon.cc)
  -> require_builtin_probe.cc 产品策略 + 决策流
  -> 后端 runtime context 适配器 (napi | nodeabi)
        -> 共享私有 runtime probe (runtime_probe/)
             -> Realm 指针校验
             -> PrincipalRealm::builtin_module_require 符号查找
             -> 平台 getter 机器码解析器 -> 拿到 Realm 字段偏移
             -> 用公开 N-API 校验 handle
  -> 冒烟测试 + 目标导出加载检查
```

### 2.2 关键洞察:为什么能拿

Node 启动引导时,`node::PrincipalRealm`(Realm = 一个环境的核心上下文对象)上有一个私有成员函数 `builtin_module_require()`,返回 Node 引导用的 `requireBuiltin` 函数。`--expose-internals` 只是把这个函数**挂到全局**让 JS 能直接 `require` 内部模块;而它本身**始终存在**于 Realm 上。这个插件就是把「拿到并调用这个函数」这步在 C++ 原生层做掉,JS 侧就能像 `--expose-internals` 一样用。

难点不在于"调用",而在于 **Realm 指针和函数偏移都是运行时事实,跨 Node major / Electron / 发行版编译差异会变**。所以不能编译期写死。

### 2.3 probe 决策流(require_builtin_probe.cc)

`RequireBuiltinProbe::RunStatus()` 的顺序(每步失败即 fail-closed):

1. `ResolveRequireBuiltin()`:问后端要一个候选 `requireBuiltin` 值;
2. `ValidateRequireBuiltinName()`:校验 JS 函数 `name === "requireBuiltin"`(防混淆);
3. `SmokeTest()`:先 `requireBuiltin('internal/bootstrap/realm')`,校验返回对象,再查 realm 导出里是否存在严格等于该函数自身的 `require` / `requireBuiltin`(**自引用冒烟**,证明拿到了真正的引导函数而非假货);
4. 对请求的 id 应用产品策略(internal-loader 白名单强制);
5. `LoadTargetModule()`:加载目标内部模块,记录是否返回 exports;目标加载异常转成带诊断的明确 unsupported 错误。

### 2.4 后端与私有探测(runtime_probe/)

后端实现同一个签名 `ProbeRuntimeRequireBuiltin(napi_env)`:

- **`napi` 后端**(默认发布后端):**不 include `node.h` 也不 include `v8.h`**,编译期不绑定 Node/V8 C++ 头,运行时动态找 V8 符号,立刻用 N-API 校验候选 handle。每个平台一个 `napi-v9` 二进制。
  - 先建 runtime profile(Node 版本来自 `napi_get_node_version()`,V8 版本来自 `v8::V8::GetVersion()`)决定 embedder-data ABI 与 tag:Node 20/22/24 用 untagged overload;Node 26+ 用 tagged overload(tag=2);Electron 用 tagged overload(tag=0)。Electron 靠 V8 编译的 `-electron.` embedder 后缀识别,再按精确 Node/V8 对匹配。
  - `NapiEmbedderDataReader` 按 profile 选择入口:tagged 优先导出的 `Context::GetAlignedPointerFromEmbedderData(int, uint16_t)`,否则旧 `SlowGetAlignedPointerFromEmbedderData`。V8 15.5 把前者移出 inline、slow helper 改收 `CppHeapPointerTag`(与 embedder-data tag 不可互换,从不调用)。
  - **Realm 字段偏移全靠运行时解析 getter 机器码**,实现里没有「版本→偏移」查表:Electron 43 当前得 0x1b8,44/45 得 0x1c0。
- **`nodeabi` 后端**:用官方 node headers(`node-vX.Y.Z-headers.tar.gz`)源码构建校验用,不发布为 optional prebuild。

共享私有探测(`runtime_probe/helper.cc`):
- 动态解析 `node::PrincipalRealm::builtin_module_require() const`;
- Windows 上私有 getter 未从 PE 导出时,回退扫描 `PrincipalRealm` vtable **按结构识别**(最长严格升序字段偏移链 + 精确 `requireBuiltin` N-API 身份),不用固定 slot/offset 步长;
- 解析一小段 getter 机器码拿到运行时字段偏移 → 读 `Realm + offset` 的字段 → 按平台 C++ 成员调用约定回调 getter(MSVC 走隐藏 struct-return 指针:x64 `rdx`、arm64 `x1`、x86 `__thiscall` 栈槽,以解码出的 getter 形状选调用模式)→ 要求 getter 结果与字段读一致 → 用公开 N-API 校验是函数。

### 2.5 平台解析器与发行版硬化差异

支持的平台解析器族:macOS arm64/x64、Linux arm64/x64(glibc+musl)、Windows arm64/x64/x86。Linux musl 用 Alpine 3.23。

**一个实测发现(发行版编译差异)**:Fedora 的 Node 是链接共享 `libnode.so` 的薄 `/usr/bin/node`,且编译带 `-mbranch-protection=standard` + `-mno-omit-leaf-frame-pointer` 同时硬化所有叶函数(74839+ `paciasp`),导致 leaf getter 有 28 字节函数体(官方构建只有 8-12 字节)。因此 Linux 两平台解码 getter 分两段:先标准 16 字节窗口,再 32 字节宽窗口。测下来 Fedora 是唯一需要宽窗口的;Debian/Ubuntu26.04/RHEL9 重建/Amazon Linux 用 `pac-ret`(不带 `+leaf`)只硬化非叶函数,leaf getter 仍 `bti c; ldr; ret`;Ubuntu 24.04 不硬化但 Node 低于本项目下限。**静态链接 + 隐藏私有符号**不可支持(Amazon Linux 默认 `nodejs`18 就是),但其带版本号 `nodejs20/22` 用共享布局可工作。

## 3. 数据:支持矩阵

| Node major | NODE_MODULE_VERSION | 产物 | 内部 loader 支持范围 |
|---|---|---|---|
| 20 | 115 | `napi-v9` | `>=20 <21` |
| 22 | 127 | `napi-v9` | `>=22 <23` |
| 24 | 137 | `napi-v9` | `>=24 <25` |
| 26 | 147 | `napi-v9` | `>=26 <27` |

- 平台 × Node:darwin-arm64/x64、linux-arm64/x64-gnu/musl、win32-x64/arm64-msvc 在 Node 20/22/24/26 全支持;`win32-ia32-msvc` 只到 Node 20/22(Node 24 起不再发 32 位 Windows 运行时)。
- Linux GNU prebuild 要求 glibc ≥2.28、C++ 运行时兼容 `GLIBCXX_3.4.25`,显式链 `libdl.so.2`,共享进程 C++ 运行时(不内嵌 libstdc++/libgcc)。
- Electron 43.0.0 / 44.0.0 / 44.7.0 / 45.0.0-alpha.6 在真实 main 进程测试(各平台),覆盖 macOS arm64/x64、Linux glibc arm64/x64、Windows arm64/x64(Electron 43 额外覆盖 win ia32)。
- Node 27 nightly:仅前向兼容覆盖(macOS/Linux glibc/Windows arm64/x64 复用同一 N-API prebuild),**不扩展稳定支持/发布包矩阵**;musl 与 nodeabi 源构建不覆盖。
- optional prebuild(按当前平台)加载:glibc/musl 用 `process.report.getReport()` 作为**最后手段**(它会枚举每个 sysfs 暴露的 CPU 做实时频率查询,多核机上要数秒),优先用廉价信号——node 二进制的 ELF interpreter + `/proc/self/maps` 里的 libc。

## 4. 安装与用法(实践)

```sh
npm install node-addon-require-builtin      # 无限制产品
# 或
npm install node-addon-internal-loader      # 白名单产品
```

```js
// internal-loader:只要 CJS/ESM loader 内部
const internalLoader = require('node-addon-internal-loader');
const esmLoader = internalLoader.requireBuiltin('internal/modules/esm/loader');
const cascadedLoader = esmLoader.getOrInitializeCascadedLoader();

// require-builtin:无限制加载内部模块
const addon = require('node-addon-require-builtin');
const realm = addon.requireBuiltin('internal/bootstrap/realm');

// 诊断:mode/product/backend/abi/bindingSource/bindingPath
console.log(addon.getBindingInfo());
```

## 5. 关键点 / 坑 / 决策准则

- **fail-closed 是生命线**:这是一个探测私有运行时态的插件,「猜偏移」风险 > 「装不上」风险。任何不变量不满足(函数名不对、自引用冒烟失败、找不到兼容符号、null 结果)都**绝不加载未经检验的内部模块**,返回带诊断的 unsupported。绝不为了兼容去绕 V8 的 tag 转换(如直换 slow getter 符号名绕过 `CppHeapPointerTag` 转换)。
- **不把内部模块变成公开 API**:README 明确「这些包不把 Node 内部变为公开 API」,返回的内部模块导出形状/对象身份/行为都是 Node 实现细节,跨版本会变。使用者要把返回值当不稳定实现细节。
- **发布模型**:发布包**不含原生源码**,安装时不做 fallback 编译(本地构建只用于仓库开发/CI)。当前平台没有 CI 验证过的 optional prebuild 就 fail-closed,而不是现场编一个没验证的二进制。共享 loader 先把选中的 `.node` 复制到确定性缓存路径(namespaced by 包名/版本/平台,短于 Windows MAX_PATH)再加载,内容哈希仅用于校验缓存文件完整性。
- **Node 27 兼容教训**:V8 15.5 把入口移出 inline,不能只按 Node major 切入口(nightly 的日期/平台/V8 版本不一致);用**已知导出符号 > 版本阈值**来判断入口,新入口只在有源码依据时加,配套自测。
- **环境变量**:`NARB_BACKEND=napi|nodeabi|auto`(auto 默认,解析到当前平台 napi-v9 optional prebuild)、`NARB_PRODUCT=internal-loader|require-builtin`(源构建选产品,默认 internal-loader)。

## 溯源

- 仓库:https://github.com/deepseek-ai/dsh-node-addon-require-builtin(master@a58d3599,release 0.1.9,2026-10-12 核 200)
- 主 README / docs/architecture.md / docs/support-matrix.md / docs/node-27-compatibility.md(同仓)
- npm 包名:`node-addon-native-custom-loader` / `node-addon-internal-loader` / `node-addon-require-builtin`
- 上游消费方:https://deepseek.com/harness/ (DeepSeek Harness)

## 相关条目(库内互链)

- 与 [deepseek-harness](./deepseek-harness.md) 同族:本篇是 dsh 的内部组件(底层 Node 插件),harness 是 agent 框架本体。
- 与 [deepseek-infra](../../deepseek-infra/index.md) 同属 DeepSeek 的「非模型」工具侧,但按简报归本域工具节(harness 在这里)。
