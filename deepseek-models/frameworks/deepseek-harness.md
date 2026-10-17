---
title: DeepSeek Harness(dsh)— everything-is-a-plugin 的 agent 框架
type: howto
tags: [deepseek, agent, harness, plugin, cordis, typescript, tool]
date: 2026-10
source:
  - "code: deepseek-harness | https://github.com/deepseek-ai/deepseek-harness | 2026-10-16 | ↑primary"
  - "web: DeepSeek Harness 文档 | https://deepseek-harness.github.io/deepseek-harness/ | 2026-10-09"
  - "pdf: A Programming Paradigm for Spatiotemporal Composability | arXiv:2608.25512 | https://arxiv.org/abs/2608.25512"
---

# DeepSeek Harness(dsh)— everything-is-a-plugin 的 agent 框架

> **TL;DR**:DeepSeek Harness(`dsh`)是 DeepSeek 开源的 **agent 框架**,TypeScript 写成,基于 **everything-is-a-plugin** 架构、底层跑在 **Cordis**(一个「时空可组合性」编程范式)之上。当前 **developer preview**,快速迭代、**会有破坏性变更**。一条命令 `npx @deepseek-ai/dsh web` 起 Web UI(默认 127.0.0.1:3080)。定位是工具类,与数学/编码谱系无关,归本域工具节。

## 1. 是什么

- 一个**开源 agent harness**(agent 运行框架),不是模型。
- 架构口号:**everything is a plugin**——框架的一切能力(工具、模型、UI、存储)都以 plugin 形式挂载。
- 底层依赖 **Cordis**(cordiverse/cordis),设计论文《A Programming Paradigm for Spatiotemporal Composability》(arXiv:2608.25512)。

## 2. 上手(howto)

### 从 npm 跑

```sh
# 需先装 Node.js
npx @deepseek-ai/dsh web
# 启动 Web UI 于 http://127.0.0.1:3080,默认开浏览器;加 --no-open 不自动开
# SSH 启动只打印 host URL(本地端口转发归 SSH 客户端/编辑器管)
```

### 从源码跑

```sh
git clone https://github.com/deepseek-ai/deepseek-harness.git
cd deepseek-harness
pnpm install
pnpm run build
pnpm dsh web
```

开发热更:`pnpm run dev:web`(单终端 build+serve+rebuild);`make help` 列出 Web/Desktop 的 Make target。

## 3b. 增量:0.2.1-alpha.3(2026-10-14)

> 版本演进:`0.2.0-rc.2`(09-29)→ `0.2.1-alpha.1/2/3/4`(10-03/10-09/10-14/10-15)。仍为 alpha,迭代很快,破坏性变更常态。本节收录 v0.2.1-alpha.3 相对既有笔记的新观察点。

**新增能力(本次值得记的):**

- **账号登录与密钥脱敏**:Web 端支持本机 HTTP 访问时登录 DeepSeek 账号;自动识别已发送消息中的密钥/令牌并**默认脱敏、按需查看**。API Key 读取优先级调整:`Web/桌面端`优先用已保存凭据,**headless 默认改用环境变量与 .env、不读已保存密钥**。
- **/compact 手动压缩上下文**:极简模式可读图 + 自动压缩长对话;Web 与桌面端随时 `/compact` 手动压缩,缓解长上下文成本。
- **子智能体 Agent Preset**:子智能体可配置独立 Agent Preset 统一角色/工具;delegation 达到层数上限会提前提示、等待子任务期间不再消耗新的目标执行轮次(委派/等待指引细化)。
- **PTC Mode in-memory 执行**:Node 执行环境新增可选 in-memory 模式,多次运行代码可复用变量/函数/已导入模块(类比 REPL 持久状态)。
- **SDK resume**:TypeScript 与 Python SDK 支持 `resume` 恢复已保存会话,沿用原会话 ID 与工作目录。
- **热重载插件生效**:插件更新/重装后不再跑旧代码——可替换插件启用热重载即时生效,否则提示重启。

**需注意的破坏性/迁移点:**

- **Azure 提供方名称统一为 `azure`**;旧名 `azure-openai-responses` 的配置需迁移。
- `--from-default-profile` 重建已有自定义配置时会**替换该配置及其插件与设置**(不是合并)。
- Python SDK 的 Office 引擎与完整创作环境改为**按需下载**。

**文档 URL 实测存活**(2026-10-14 复核):仓库 `https://github.com/deepseek-ai/deepseek-harness`、文档站 `https://deepseek-harness.github.io/deepseek-harness/` 均可访问;release tag 走 `dsh-v*`(如 `dsh-v0.2.1-alpha.3`)。

### 增量溯源
- 版本/功能点以仓库 release body(2026-10-14)为准;本次仅增量更新已有篇,机制/上手不变。

## 3c. 增量:0.2.1-alpha.4(2026-10-15)

> 版本演进延续:`0.2.1-alpha.4`(10-15)。仍是 alpha,迭代快、破坏性变更常态。本节收录 alpha.4 相对 3b 的新观察点。相对 alpha.3 共 89 个 commit,新增功能收敛为两类,**无架构级变化**。

**新增能力(本次值得记的):**

- **自定义模型 API 推理等级 UI 化**:此前改各模型 reasoning level 要手改配置文件,alpha.4 起可在 **UI 中直接为每个自定义模型 API 配置推理等级**(读 pi-ai schema、keep reasoning edits with their model、mirror pi-ai refusals),无需碰配置文件。
- **自定义插件索引源**:支持从**自定义插件市场源(plugin index source)发现与安装插件**;一并引入 `engines.dsh` 版本兼容声明——第三方插件**优先按 `engines.dsh` 判版本兼容,未声明时沿用旧规则**(插件作者需检查声明覆盖的 DSH 版本)。

**问题修复(略记):**

- 命令与参数交界处输入异常(支持整体选中/删除、撤销/刷新后恢复);
- Office 文件交付改进(避免引用已删除预览图、减少 PDF 导出后多余工具检索);
- Linux 启动命令前置重复检查减少、命令启动开销降低、兼容较旧 systemd;失败命令现在呈现有效状态信息。

**其他**:卸载/重装插件实例清理、host release 变化时提示 reload 等非破坏性收尾。

### 增量溯源
- release body tag `dsh-v0.2.1-alpha.4`(2026-10-15,prerelease,作者 LegGasai);feature/commit 明细见仓库 `master` 分支 `dsh-v0.2.1-alpha.3..dsh-v0.2.1-alpha.4` 区间。


## 3d. 增量:0.2.1-alpha.5(2026-10-16)

> 版本演进延续:`0.2.1-alpha.5`(10-16,共 29 commit)。仍为 alpha,迭代快。相对 alpha.4 收敛为**无架构变化**、以问题修复为主的一版:1 个新 Slot + 4 个修复。

**新增能力(唯一值得记的):**

- **`conversation.input.model.label` Slot**:新增插件 Slot,允许第三方插件**按模型渠道自定义输入框(composer)模型选择器中的名称展示**。习惯用 `conversation.input.model.*` 组合钩子定制模型 UI 的插件作者可直接挂这个新 Slot。@LegGasai

**问题修复(略记):**

- 插件页启停插件后,已打开会话的 `/` 菜单技能列表未同步更新的问题(修复需重新拉取技能);
- 新建/恢复多个终端时减少重复查询,并避免失效终端在切换面板后反复重试;
- PTC Mode 补全缺失指引,统一 SDK/Web/headless 三种 profile 下的模型身份说明;
- 修复 Windows/Linux 下右侧栏收起时,在文本输入框按 Page Up 导致整个界面左移的问题。

### 增量溯源
- release body tag `dsh-v0.2.1-alpha.5`(2026-10-16,prerelease,`published_at 2026-10-16T12:23:28Z`);commit 区间 `dsh-v0.2.1-alpha.4..dsh-v0.2.1-alpha.5` 共 29 个 commit,新增功能收敛为此 1 个 Slot,故仅小幅补记。

## 3e. 增量:0.2.1-rc.1 → rc.2(2026-10-17,进入 release candidate)

> 版本演进再进一步:`0.2.1-alpha.5`(10-16)→ **`0.2.1-rc.1`**(10-16 深夜)→ **`0.2.1-rc.2`**(10-17,HEAD=rc.2)。**从 alpha 阶段正式进入 release candidate(rc)阶段**,标志 0.2.1 接近正式发布。rc.1 仅 vendor 升级、无功能变化;rc.2 收敛为 2 个新功能,仍无架构级变化。

**新增能力(rc.2 值得记的):**

- **DeepSeek Search & Fetch API(可选 Web Search API,`#6103`)**:此前"搜索"只走 Messages/内置搜索;rc.2 起新增一个**可选的「Use the new Web Search API」开关**,走 DeepSeek 官方的可检索 **Search / Fetch API**(默认 base `https://api.deepseek.com/v1/tools`),Messages 搜索仍是默认行为。要点:
  - **鉴权只用 API key**:DeepSeek **Account token 在 `/v1/tools/*` 上会返回空 HTTP 404**(2026-10-16 实测),所以该 API 只用 API key 鉴权。
  - **`web-search-deepseek.count` 结果数**:1–50,默认 10(服务端默认);请求里更小的 `maxResults` 仍优先。
  - **发送触发问题 `ori_question`**:把发起该会话的最近一条人类 prompt 压缩空白、截到 128 字符上限后作为 `ori_question` 传给提供商;可 `sendOriginalQuestion:false` 关掉。搜索去重改按请求的 `query` 字段(因 ori_question 现在带了每条 query)。
  - **auxiliary 搜索请求关闭 thinking**:providers 丢弃辅助 Messages turn 的生成文本,thinking 没消费者,开启只会加延迟和 output token(修 #6099)。
  - **行为实测记录**:总结 1000–2300 字符;Fetch 对无法解析的目标返回 **HTTP 500**;`WebFetchResult.statusCode` 保持必填(DeepSeek Fetch 报告 HTTP 200 并丢弃页标题)。
- **DeepSeek Flash "1017" 临时徽章(`#6406`/`#6417`)**:给内置 DeepSeek Flash 显示名临时追加 "1017" 徽章(先加、revert、再加回:badge 显示的临时改动)。

**vendor 升级(rc.1,无功能面变化但值得记版本锚点):** `cordis 4.0.5`、`cosmokit 1.8.6`、`group 1.0.5`、`hmr 1.0.20`、`include 1.0.10`、`loader 1.0.6`、`logger-console 1.0.5`、`schemastery 3.18.5`、`timer 1.1.7`。

### 增量溯源
- rc.1 tag `dsh-v0.2.1-rc.1`(2026-10-16)、rc.2 tag `dsh-v0.2.1-rc.2`(2026-10-17,当前 HEAD);功能/commit 区间 `dsh-v0.2.1-alpha.5..dsh-v0.2.1-rc.2`。

## 3. 关键点/坑

- **developer preview,破坏性变更**是常态,生产使用前读 SAFETY.md 与 AGENTS.md。
- **文档入口**:https://deepseek-harness.github.io/deepseek-harness/ (用户指南、开发指南、架构文档都在那)。
- **插件生态**:给插件仓库加 `dsh-plugin` topic 便于发现;反馈走 GitHub Discussions 或官方 Discord。
- 许可证 MIT。

## 溯源

- 仓库:https://github.com/deepseek-ai/deepseek-harness
- 文档:https://deepseek-harness.github.io/deepseek-harness/
- Cordis:https://github.com/cordiverse/cordis
- Cordis 设计论文:https://arxiv.org/abs/2608.25512

## 相关条目(库内互链)

- 本域其余条目均为「领域模型谱系」(数学/证明/编码/MoE),本篇是唯一的工具类条目,单独归 frameworks/ 以区分「模型」与「工具」。
- 与 [deepseek-infra](../../deepseek-infra/index.md) 同为 DeepSeek 的「非模型」开源(那边是算子/昇腾栈,这边是 agent 框架)。
