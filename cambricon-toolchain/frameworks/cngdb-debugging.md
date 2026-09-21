---
title: CNGDB 上手:MLU/BANG C 程序调试器
type: howto
tags: [cambricon, mlu, cngdb, gdb, bang-c, coredump, debugging, cncc]
source:
  - "ppt: CNGDB培训 v0.3 | ~/zhanbin/library/training/CNGDB培训-v0.3.pptx (primary)"
  - "web: 寒武纪 CNGDB 用户手册 3.7.2(黄区,版本较老) | https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cngdb_3.7.2/index.html"
  - "web: GNU GDB 手册(CNGDB 基于 GDB 二次开发,原生命令兼容) | https://sourceware.org/gdb/current/onlinedocs/gdb.html/Symbols.html"
date: 2026-09
---

# CNGDB 上手:MLU/BANG C 调试器

## TL;DR

CNGDB(Cambricon GNU Debugger)是基于 **GNU gdb 7.11.1** 二次开发的寒武纪软件调试工具,跑在 Linux 上。**全面兼容 GDB 原生命令**,扩展命令统一以 `cngdb` 前缀。四个关键能力:① 一次运行**同时调 Host(C/C++)端和 MLU(BANG C)端**;② 调**单核和多核** MLU 应用;③ 设 MLU **硬件断点**、单步、查改 core 上的变量/内存;④ 解析 MLU **coredump**(CNCORE `.cndump` 文件)。

上手三步:**编译带调试信息**(`cncc -g -O0`,或只带行数 `-gline-directives-only` 配 `-O3`)→ `cngdb <程序>`(带参数用 `--args`)→ 多核用 **focus 机制**(`info cngdb all` 看全部 core 状态,`cngdb focus` 切到某个 core 拿控制权)。命令都支持 TAB 补全、`help`、`apropos`。

**心智模型**:CNGDB 不是"GDB + MLU 补丁",而是把 GDB 的单线程 focus 概念推广到 MLU 多核——命令只作用在 focus 的那一个 core 上,靠手动/自动切 core 完成多核程序执行。这是与 GDB 调 CPU 时最大的思维差异。

---

## 1. 环境依赖与内外版本

**依赖链**(缺一个都跑不起来):
- CNGDB 本体基于 gdb 7.11.1,系统库依赖与 gdb 7.11.1 相同;
- Cambricon 库:`libcngdb_backend.so`(后端交互接口)+ 对应版本 **CNDrv**(驱动)+ **CNBin**;
- 被调程序**编译**要 g++ + **CNCC**(Cambricon Compiler Collection,BANG C 编译器);
- 被调程序**运行**要 **CNRT**(Cambricon Runtime Library);
- **`/tmp` 目录读写权限**:CNGDB 用它存临时文件,权限不足会失败。

**内外版本区别**(重要,排错能力不同):
- 外部版 = cntoolkit 发布包里对应的 CNGDB;内部版 = 红区 `/tools/neuware` 里的 CNGDB。
- **解析 CNCORE 时内部版显示更详细的错误信息**(红框部分外部版不显示);
- **内部版支持反汇编**,外部版不支持。
- 实践:定位疑难 coredump 尽量用内部版(红区),能看到更多现场与反汇编。

## 2. 编译被调试程序(CNCC)

| 目的 | 命令 | 说明 |
|---|---|---|
| 完整调试信息 | `cncc kernel.mlu -o a.out -g -O0 --bang-mlu-arch=mtp_592` | `-g` 在 `-O0` 生成含调试信息的二进制;**只有 `-g -O0` 才支持打印出错现场的变量** |
| 仅行数信息 | `cncc kernel.mlu -o a.out -gline-directives-only -O3 --bang-mlu-arch=mtp_592` | 不影响生成指令,只带行号;适合 `-O3` release 版也能定位到源码行(CNNL 红区 master 的 release 编译默认带此选项) |

选型:要**看变量值**必须 `-g -O0`;`-O3` 下优化会让变量不可见,此时退而求其次用 `-gline-directives-only` 至少定位行。`--bang-mlu-arch` 指定目标架构(mtp_592 等),编错架构会在加载/运行时出问题。

## 3. 启动 CNGDB

```bash
cngdb drv_test                      # 直接传程序
cngdb --args drv_test arg1 arg2     # 程序需要额外参数(如读入数据)用 --args
```

帮助:`help <cmd>` 内置帮助;所有命令 **TAB 自动补全**;`apropos` 查询命令。原生命令与 GDB 一致,扩展命令带 `cngdb` 前缀。

## 4. Kernel focus 机制(多核调试的核心)

MLU 硬件支持一个应用在**多个 core 上并行运行**。CNGDB 用 **focus 机制**简化:输入命令**只作用在当前 focus 的一个 core 上**,通过手动/自动切 core 完成多核程序执行。

```
info cngdb all      # 查看当前 focus 的 core + 所有 core(s) 的状态
cngdb focus <core>  # 切换 focus 到指定 core,获得该 core 的控制权
```

**单步的作用范围随模式不同**(易错):
- **单核调试模式**:单步只对**当前 focus 的 core** 有效;
- **多核调试模式**:单步对**当前运行任务的所有 core** 都有效。
- 支持 BANG C 源码级单步。

**中断**:程序挂起/卡在无限循环时按 **Ctrl+C** 手动中断,CNGDB 收到信号后停止硬件运行、拿回控制权,此时可查硬件状态 / 重跑 / 退出。

## 5. 断点(对所有 core 生效)

**关键**:断点设置对**所有 core** 生效;任一 core 命中断点就挂起等待进一步命令。四种方式与 GDB 调 CPU 一致:

```
break <函数名>                       # 1. 函数断点(函数入口)
break <文件名>:<行号>                # 2. 行断点
break <位置> if <条件表达式>          # 3. 条件断点:可引用任意变量 + 内建变量(coreId、taskIdX 等)
watch <变量名>                       # 4. watchpoint:变量被改时中断
```

条件断点能引用 **MLU 内建变量**(`coreId`、`taskIdX`)是 CNGDB 相对 GDB 的扩展点——可以只在某个 core / 某个 task 命中时停,多核调试利器。

## 6. 查看/设置运行状态

```
print <变量>        # 查看变量;可直接打印 BANG C 内建变量(即使程序里没定义/没用到)
x <地址>            # 按地址读内存数据
```

CNGDB 能**直接打印 BANG C 内建变量**(如 coreId/taskId),不必在源码里显式使用它们——排查"这个 task 跑在哪个 core、taskId 是多少"时很省事。还支持查看状态、设置内存(改运行中数据)。

## 7. MLU coredump(CNCORE)解析

MLU 程序 core dump 后生成 **`.cndump` 后缀的 CNCORE 文件**,记录崩溃现场。用 CNGDB 解析可看崩溃原因、打印内存值。

流程:
1. **加载 CNCORE 文件**;
2. 加载后 CNGDB 展示**异常中断的核心(core)及其相关信息**;
3. 若程序编译时带 `-g` 或 `-gline-directives-only`,可进一步显示**异常中断时的源码行**;
4. **`-g -O0` 才支持打印出错现场的变量**(优化级下变量不可见)。

内部版还能:切换 CNCORE 核心(CNCORE 核心切换)、打印参数空间、看更详细错误信息与反汇编。案例场景:coredump 解析、仅有行数信息的 coredump 解析、在线查看异常现场(用 print/x 看现场)、算子基础调试、CNStudio 图形化调试算子。

## 8. 排错顺序(踩坑速查)

1. **起不来/找不到符号**:查依赖链(CNDrv/CNBin/CNRT/libcngdb_backend 版本对齐)、`/tmp` 读写权限。
2. **能看到行但看不到变量**:编译不是 `-g -O0`(用了 `-O3` 或只带 `-gline-directives-only`)→ 变量被优化掉,只能定位行。
3. **多核程序行为困惑**:先 `info cngdb all` 看全 core 状态,确认当前 focus 的是哪个 core;记住命令只作用于 focus core,单步范围随单核/多核模式变。
4. **coredump 信息太少**:多半用了外部版 → 换内部版(红区)看详细错误 + 反汇编。
5. **只想在特定核停**:`break ... if coreId==N`(或 taskIdX)用条件断点。

## 溯源

- 主源:`~/zhanbin/library/training/CNGDB培训-v0.3.pptx`(内部培训,含命令示例截图与多个案例;图片内容无法从文本抽取,本篇按文字骨架 + 命令还原为可操作 howto)。PPT 正文 ~8.7K 字符,较薄,故聚焦"命令 + 心智模型 + 排错",不做无据扩写。
- 交叉源:寒武纪公开 CNGDB 用户手册 3.7.2(黄区)目录结构与本篇一致(简介/快速入门/CNGDB 扩展/focus 机制/程序执行/断点/获取设置运行状态/CNCORE 支持/常用命令/FAQ/附录源码示例);URL 2026-09 实测存活。红区用户手册/完全使用指南为内部地址,不登记明文链接。
- CNGDB 基于 GNU gdb 7.11.1,原生命令语义参见 GDB 手册 https://sourceware.org/gdb/current/onlinedocs/gdb.html/ 。
- 库内互链:[mlu-mim-partitioning.md](mlu-mim-partitioning.md)(同域:MLU 资源切分)、[cpp-language](../../cpp-language/index.md)(CNCC 编译/符号与链接是 BANG C 调试的语言基础)。
