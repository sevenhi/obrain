---
title: C 符号与链接:目标文件、符号表、强弱符号、name mangling
type: howto
tags: [c, cpp, linking, symbol-table, elf, weak-symbol, name-mangling, gcc, nm, objdump]
source:
  - "ppt: C/C++链接简介(C语言符号与链接) | ~/zhanbin/library/training/C语言符号与链接.pptx (primary)"
  - "web: GNU ld 文档 — Symbols | https://sourceware.org/binutils/docs/ld/Symbols.html"
  - "web: GCC 文档 — Common Function Attributes(weak 等) | https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html"
  - "web: cppreference — Class template(模板实例为弱符号/COMDAT 语义背景) | https://en.cppreference.com/w/cpp/language/class_template"
date: 2026-09
---

# C 符号与链接

## TL;DR

**链接的唯一标准是符号名字**——这句话是全部坑的根源。四段心智模型:

1. **每个 .c 独立编译**成 .o(`gcc -c`),.o 里有 TEXT(代码)/ DATA(编译期已初始化的数据,完整存于文件内)/ BSS(未初始化数据,**只记大小和 offset**,加载时由 OS 分配并清零)三段;`objdump` 看段,`nm` 看符号。
2. **符号表只登记四类东西**:定义的全局变量、定义的函数、声明未定义但**被调用过至少一次**的函数、声明未定义但**被用过至少一次**的全局变量。加 `static` = 内部链接,否则默认外部链接。
3. **链接做四件事**:合并符号表 → 把 U(undefined)解析到别的链接单元的 T/B/D → 合并同名段 → 地址重定向。
4. `nm` 字母:**小写 = 内部链接,大写 = 外部链接**;`t/d/b` = text/data/bss,`U` = undefined,`w/v` = 弱函数/弱变量。弱符号可被强符号覆盖,多个同名弱符号**不报 multiple definition**,统一链到第一个。

C++ 侧两条推论:**name mangling** 让同名重载/模板实例生成不同符号(要 C 风格符号名用 `extern "C"`);**函数模板不是函数**,每个实例生成一个**弱符号**,n 个编译单元同时实例化同一实例 → 编译时间 ×n,但最终可执行文件里**不会**有 n 份代码(链接期合并)。

---

## 1. 编译与目标文件

```bash
gcc -c 1.c        # 只编译,产出 1.o
objdump -t 1.o    # 看符号表 / 段(注:cnobjdump 看不了反汇编)
nm 1.o            # 只看符号表,更常用
```

| 段 | 存什么 | 目标文件里的形态 |
|---|---|---|
| TEXT | 代码编译出的二进制指令 | 完整存储 |
| DATA | 编译期已初始化的数据 | **完整存储**在 .o 内 |
| BSS | 未初始化的数据 | **只记 size + offset**,加载时 OS 分配并清零 → .o 更小 |

两个 gcc 实现细节(标准情况的偏离):
- gcc 默认开一项优化:**手动初始化为 0 的数据也放 BSS** 而不是 DATA;
- **COMMON 段**:unix 目标文件里未初始化的全局变量先进 COMMON,链接完成后并入 BSS。

## 2. 符号表与链接属性

符号表项四类(培训原文口径):定义的全局变量、定义的函数、**声明但未定义且被调用过至少一次的函数**、**声明但未定义且被用过至少一次的全局变量**。

- **链接属性**:加 `static` → 内部(internal);默认 → 外部(external)。
- `nm` 字母对照:

| 字母 | 含义 | | 字母 | 含义 |
|---|---|---|---|---|
| `T` | 外部 text(函数) | | `t` | 内部 text |
| `D` | 外部 data(已初始化) | | `d` | 内部 data |
| `B` | 外部 bss(未初始化) | | `b` | 内部 bss |
| `U` | undefined(本单元没定义,等链接期解析) | | `w`/`v` | 弱函数 / 弱变量 |

## 3. 链接过程 + 三类经典错误(可复现)

链接输入:各 .o + 外部库 → 输出 a.out/库;动作:**合并符号表 → U 解析到其它链接单元的 T/B/D → 合并相同段 → 地址重定向**。

**示例 A:正常流程**
```c
// 1.h            // 1.c                 // 2.c
int func1();      int func1(){return 0;} #include "1.h"
                                         int main(){ return func1(); }
```
`gcc 1.c 2.c` → 1.o: `T func1`;2.o: `T main`, `U func1`;链接把 `U func1` 解析到 1.o 的 `T func1`。✅

**示例 B:漏链接单元 → `undefined reference`**
```bash
gcc 2.c     # 少了 1.c:2.o 里 U func1 无处解析
```
内部对应场景:一个算子依赖另一个算子的部分 api,但 build 的 filter 里没写被依赖的那个算子 → 链接期 undefined reference。**症状在链接期,病根在构建配置**。

**示例 C:multiple definition**
```c
// 1.h 里写了函数定义(不是声明!):
int func1();
int func2() { return 0; }   // ← 定义放头文件
```
1.c 与 2.c 都 include 1.h → 两个 .o 各有 `T func2` → 链接冲突。
修法:头文件只放声明,定义放一个 .c;或给 `func2` 加 `static`(T → t,内部链接,各编译单元独立一份,不冲突)。

**示例 D:二进制不兼容(最阴)**
```c
// 1.c                    // 1.h            // 2.c
int func1 = 9;            int func1();     #include "1.h"
   ↓ 1.o: D func1                            int main(){ return func1(); }
```
链接只看名字:2.o 的 `U func1` 被解析到 1.o 的 **`D func1`(一个变量!)**,链接**成功**,运行时把数据当代码跳 → 崩溃/诡异行为。内部典型症状:cnnl_extra 与 cnnl 之间 core 目录下的依赖对不上。
教训:**声明与实际定义必须一致**,链接器不会帮你检查类型。

## 4. 强符号与弱符号

- 弱符号**可在链接期被强符号覆盖**;
- **多个同名弱符号不产生 multiple definition 错误**,统一链接到第一个;
- `nm` 中弱函数 = `w`,弱变量 = `v`;
- 用途:让用户覆盖部分库函数(经典例子是 `malloc`)。
- 工程含义:**多个弱符号静默取第一个**意味着"顺序决定行为",这类 bug 不报错,只在行为异常时才被发现——链接顺序变化后突然换实现,查这里。

## 5. C++ 侧:name mangling 与模板弱符号

- **name mangling**:C++ 允许同名函数(重载 + 模板),mangling 使同名函数生成**不同符号名**;`extern "C"` 生成 C 风格符号名(= 函数名本身),这是 C++ 库暴露 C ABI 的标准做法(算子库对外 API、dlsym 查符号都依赖它)。
- **函数模板不是函数**:每个独特实例生成一个**弱符号**函数。n 个编译单元同时实例化同一实例 → **编译时间 ×n,但可执行文件里不会有 n 倍代码**(弱符号在链接期合并为一个)。这解释了模板的两个账单:编译慢、binary 膨胀,但重复实例化不会线性放大最终体积。
- 与模板篇的接口:`undefined reference to func<int,float>()` 是"模板实例这个弱符号在所有链接单元里都不存在"(定义没 include / 没显式实例化),不是 mangling 错;而 `extern "C"` 用错导致的 undefined reference 可以用 `nm -C`(demangle)对比两边符号名快速定位。

## 6. 排错速查

```bash
nm -C libfoo.a | grep -i func      # 看符号是否存在、T/U/w/v、是否 mangling 对不上
objdump -t x.o                     # 看段归属(DATA vs BSS vs COMMON)
gcc -c a.c && nm a.o               # 单编译单元先自证清白
ld -r / ar t                       # 确认库里的 .o 是否真的进了链接
```

判定顺序:① 符号在不在(U 还是 T)?② 在,但类型对不上(示例 D 的 D-vs-T)?③ 多个同名(T 冲突 → 加 static/挪定义;w 冲突 → 顺序决定,查链接顺序)?④ C/C++ 混编看 mangling(`nm -C` demangle 后对比,该加 `extern "C"` 没加?)。

## 溯源

- 主源:`~/zhanbin/library/training/C语言符号与链接.pptx`(内部培训《C/C++链接简介》,含 gcc 四段示例与 cnnl 内部案例备注)。PPT 正文薄(~4K 字符),本篇按 howto 组织:段/符号表/链接四动作/四类错误可复现示例/nm 字母表/排错顺序,不做教科书扩写。
- 交叉核对:GNU ld 符号解析规则(强弱符号、U 的定义)https://sourceware.org/binutils/docs/ld/Symbols.html ;gcc weak 属性文档 https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html ;cppreference class_template(模板实例链接语义)。URL 均 2026-09 实测存活。
- 库内互链:[cpp-templates-metaprogramming.md](cpp-templates-metaprogramming.md)(模板实例化/undefined reference/`template` 消歧)——链接篇讲"符号为什么在不在",模板篇讲"怎么让编译器生成它"。
