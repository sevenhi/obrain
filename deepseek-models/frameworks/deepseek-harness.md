---
title: DeepSeek Harness(dsh)— everything-is-a-plugin 的 agent 框架
type: howto
tags: [deepseek, agent, harness, plugin, cordis, typescript, tool]
date: 2026-10-09
source:
  - "code: deepseek-harness | https://github.com/deepseek-ai/deepseek-harness | 2026-10-09 | ↑primary"
  - "web: DeepSeek Harness 文档 | https://deepseek-harness.github.io/deepseek-harness/ | 2026-10-09"
  - "pdf: A Programming Paradigm for Spatiotemporal Composability | arXiv:2608.25512"
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
