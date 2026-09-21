---
title: C/C++ 语言工程
type: domain
tags: [cpp, c, template, linking, symbol, howto]
source:
  - "ppt: C++模板编程技巧 | ~/zhanbin/library/training/C++模板编程.pptx"
  - "ppt: C++编程进阶技巧(2022-07-13) | ~/zhanbin/library/training/C++编程进阶技巧.pptx"
  - "ppt: C/C++链接简介 | ~/zhanbin/library/training/C语言符号与链接.pptx"
date: 2026-09
---

# C/C++ 语言工程 (cpp-language)

> 领域索引。主题:C/C++ 在算子库/异构编译器(CNCC/BANG C)工程里的语言机制——模板与元编程、宏、函数指针、符号与链接。
> 定位:**howto/practices 为主**(三份内部 PPT 正文薄,刻意不做教科书注水),重"报错对照、判据、可复现排错"。

## 一句话核心

> 模板的收益 = 把分支从运行时挪到编译期(逻辑同型才用模板,特化优先于主模板);链接的唯一标准是符号名字(声明与定义不一致链接器不查,弱符号静默取第一个)。

## 内容索引(2 篇)

| 专题 | 文件 | 定位 | 状态 |
|---|---|---|---|
| C++ 模板与元编程实战 | [frameworks/cpp-templates-metaprogramming.md](frameworks/cpp-templates-metaprogramming.md) | 用:模板 vs 重载判据、变量→常量分支消除、特化/SFINAE/enable_if、宏四用法、函数指针 vs lambda、高频报错对照表 | ✅ |
| C 符号与链接 | [frameworks/c-symbols-and-linking.md](frameworks/c-symbols-and-linking.md) | 用:TEXT/DATA/BSS/COMMON、符号表四类项、链接四动作、四类经典错误(undefined/multiple/二进制不兼容/弱符号)、nm 字母表、排错顺序 | ✅ |
| 参考资料索引 | [references.md](references.md) | 溯源:cppreference/binutils/gcc 文档链接 | ✅ |

## 速记

1. **函数模板不是函数,类模板不是类** → `undefined reference to f<int,float>()` 先查实例化(定义 include / 显式实例化),再查 mangling
2. 特化优先级:全特化 > 偏特化 > 主模板;函数模板**只能全特化**,切子集用 enable_if 重载且条件无交集
3. 依赖名里的成员模板调用要 `obj.template method<T>()`,否则 `<` 被当小于号
4. `#ifdef A` 为真 ≠ `#if A` 为真;宏参数逐个加括号
5. 头文件放**函数定义** = 每个 include 它的 .o 一个 `T` → multiple definition;加 static 转内部链接
6. 声明与定义类型不符(变量当函数链)→ 链接**成功**、运行时炸;cnnl 内部真实案例

## 相关条目(跨域)

- [cambricon-toolchain](../cambricon-toolchain/index.md):CNCC 编译(-g/-O0、BANG C)与 CNGDB 调试是这些语言机制在 MLU 工具链上的落地
- [cutlass](../cutlass/index.md):CUTLASS 是 C++ 模板元编程的极端工业案例(TiledMMA/Layout 全是编译期代数)
