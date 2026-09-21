---
title: C++ 模板与元编程实战(泛型/特化/SFINAE/宏/函数指针)
type: howto
tags: [cpp, template, metaprogramming, sfinae, macro, function-pointer, mlu]
source:
  - "ppt: C++模板编程技巧 | ~/zhanbin/library/training/C++模板编程.pptx (primary)"
  - "ppt: C++编程进阶技巧(基础软件部 于希文, 2022-07-13) | ~/zhanbin/library/training/C++编程进阶技巧.pptx (primary)"
  - "web: cppreference — Partial template specialization | https://en.cppreference.com/w/cpp/language/partial_specialization"
  - "web: cppreference — SFINAE | https://en.cppreference.com/w/cpp/language/sfinae"
  - "web: cppreference — Template parameters(非类型模板参数/C++20 放宽) | https://en.cppreference.com/w/cpp/language/template_parameters"
  - "web: cppreference — Function template | https://en.cppreference.com/w/cpp/language/function_template"
date: 2026-09
---

# C++ 模板与元编程实战

## TL;DR

两份内部培训(C++模板编程 + 编程进阶技巧)合并提炼,场景是 MLU 算子库(cnnl/CNCC)开发。最值钱的判断:

1. **模板适用判据**:逻辑相同只有类型不同 → 模板;类型不同逻辑也不同 → 重载(模板无意义)。
2. **模板的本质收益是"把分支从运行时挪到编译期"**:`if (std::is_same<T,float>::value)` 编译期折叠成常量,消灭分支跳转代价(MLU core 分支要同步所有队列,CPU 要清流水线 revert 乱序)——模板参数不减少代码里 if-else 的数量,但减少**运行时执行**的 if-else 次数,还能把分支选择从 MLU 提前到 CPU。
3. **特化三件套**:全特化 > 偏特化 > 主模板;**函数模板只能全特化不能偏特化**(要切子集用 `std::enable_if` 重载,条件间不许有交集)。
4. **函数模板不是函数,类模板不是类**:`undefined reference to func<int,float>()` 九成是实例化问题(定义没被 include / 缺显式实例化)。
5. 宏是编译**前**的文本替换:既不是运行时的东西也不是编译期的类型系统成员;`#if`/`#ifdef` 语义不同、宏参数必须加括号。

---

## 1. 什么时候用模板、什么时候用重载

- **函数模板适用**:`conv` 这类——不管 TO/TW/TI 是什么类型,调的接口写法相同,模板把重复代码收掉。
- **应该用重载**:`quantify` 这类——每种 TO/TI 组合内部代码完全不同,套模板没有意义,直接重载。
- **性能动机(变量 vs 常量)**:
  ```cpp
  template <typename T> void func(void* input, cnnlDataType_t dtype);
  if (std::is_same<T, float>::value)   // 好:constexpr,编译期折叠
  if (dtype == CNNL_DTYPE_FLOAT)       // 差:运行时分支
  ```
  `std::is_same<...>::value` 是 constexpr,编译器实例化 `Kernel<float>` 时它就是 `if (false)`,直接删掉。运行时分支的代价:MLU core 分支 = 同步所有队列;CPU 分支预测失败 = 清空指令流水线(深度可达 20 级)+ revert 乱序执行。
- **非类型模板参数**:整型/枚举/指针(C++20 起放宽到浮点数与满足条件的类对象);实参必须编译期常量。经典手法——把运行时 bool 变模板参数:
  ```cpp
  matmul(a, b, trans_b) {          // 运行时版
    if (trans_b) {...} else {...}
  }
  // → host 侧(CPU)分派,device 侧零分支:
  if (trans_b) matmul<true>(a,b); else matmul<false>(a,b);
  ```
  坑:`kernel<desc.act_pref><<<>>>()` 编不过——`act_pref` 是**变量**,模板实参必须是常量。

## 2. 特化与 SFINAE

**集合视角**(培训的核心心智模型):主模板 = 全集;偏特化 = 子集(如"A 是 float"的 `Something<float,B>`);全特化 = 单个元素,**全特化出来的不是模板,是实在的类/函数**。匹配优先级:全特化 > 偏特化 > 主模板,"越特殊越优先";只有所有能匹配的模板全部失败才报错。

- **"更特殊"的判定**:比较两个偏特化能表示的类型集合。`const U*` 是 `X*` 的**真子集**(能被 `const U*` 表示的都能被 `X*` 表示,反之不成立)→ `const U*` 更特殊。多个偏特化同时满足时选更特殊的;带默认值的参数在比较时被忽略(这也导致重定义检查的坑,见 §5)。
- **SFINAE**(Substitution Failure Is Not An Error):替换失败不算错,只是把这个候选踢出重载集。`std::enable_if<true,T>::type = T`;`enable_if<false,...>` 没有 `type` → 替换失败 → 该偏特化/重载静默出局。这就实现了字面意义的"满足某种条件的类型集合"。
- **函数模板没有偏特化**,用 enable_if 重载达到类似效果,**条件之间不能有交集**:
  ```cpp
  // is_integer / is_floating_point 两组互斥 → OK
  // 但若一组是 "sizeof(T)==8",double 同时落进两组 → 二义性
  template<class T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
  void func(T);   // 浮点版
  ```
- **只特化部分逻辑的两种写法**:① if-else 留在函数体里——注意**永不执行的分支也必须能编译**(`if constexpr` 才能真正剪掉,培训口径是 C++17 前的老写法);② 把要特化的逻辑抽成独立模板再特化它(激活/softmax/升降位宽等算子实例都走这条路)。
- **类型 ↔ 值互转**(元编程基本功):类型→常量:类型作模板参数,常量放类模板 `static constexpr` 字段;常量→类型:常量作模板参数,类模板里 `using type = ...`。元函数的"return 类型" = 嵌套类型别名,"return 数值" = `static constexpr`。典型应用:类型 ↔ `cnnlDataType_t` 互转;更高级是量纲分析。
- `<type_traits>` 常用件:`std::is_same<T1,T2>::value`、`std::is_pointer<T>::value`、`std::conditional<cond,T1,T2>::type`。
- **手写 is_pointer 的原理**:让满足条件的模板更特殊——`int*` 同时匹配 `template<class T> struct P{false}` 与 `template<class T> struct P<T*>{true}`,后者更特殊,胜出。更复杂的条件 = 更多的代码,这就是模板元编程的全部秘密。

## 3. 宏:预编译文本替换

**发生时机**:编译前置阶段——转码 → 反斜杠续行拼接 → 去注释 → 处理预编译指令(include 复制/宏展开/pragma)→ 删预处理指令 → 处理转义字符 → 拼相邻字符串 → 才开始语法分析编译。所以宏**既不是运行时的东西**(误当运行时值)也不是编译器类型系统的东西(误当编译期常量/类型),两类误解都是常见错误来源。

**易错点**:
- `#if A` vs `#ifdef A`:A 未定义时两者都 false;A 定义为 0 时 `#if A` 为 false 而 `#ifdef A` 为 **true**——"定义了"≠"为真"。
- 优先级陷阱:`#define ADD(x,y) x+y` → `2*ADD(2,2)` = `2*2+2` = 6。**宏体与每个参数都要加括号**:`((x)+(y))`。

**正当用法**:
- `#` 取参数字符串:`#define PRINT(x) cout << #x ": " << x << endl`;
- `__FILE__`/`__LINE__` 做调试打印;
- `##` 拼接标识符:`#define DESC(x) x##_desc` → `DESC(input)` = `input_desc`;
- 可变参数 `...` / `__VA_ARGS__`:`#define PRINT(...) __bang_printf(__VA_ARGS__)`;
- 生成重复代码、代码转字符串、处理**模板覆盖不了的场景(如 inline 汇编)**。

## 4. 函数指针

**为什么函数是"指针"**:调用 = 按 ABI 放参数和返回地址 → 跳到函数起始地址执行 → 完毕跳回。函数指针指向函数代码段地址;函数签名描述 ABI/参数/返回值(MSVC 有 `__cdecl/__stdcall/__fastcall` 之分)。

```cpp
void func(int arg);            // 普通函数声明
void (*fp)(int) = func;        // 名字前加 * 表指针,加括号定结合优先级
typedef void (*func_t)(int);   // typedef:在声明变量的语句前加 typedef
```

**用途**:回调(sort 会排不会比大小;pthread_create 会起线程不知道执行什么)与**表驱动**(cnnlMatmul host 端按 dtype/布局查表选 kernel)。
**与 lambda 的区别**:函数指针是指针;lambda 是**类**——每个 lambda 对应一个匿名类,捕获的变量成为类字段(所以带捕获的 lambda 不能直接转函数指针)。

## 5. 高频报错对照表(培训原例)

| 症状 | 根因 | 解法 |
|---|---|---|
| `error: expected primary-expression before 'xxx'` | 模板可特化,依赖名里的 `ClassTemplate::method_template` 在不同特化中可能是别的东西(甚至字段),`<` 被解释成小于号 | 调用处加 `template` 消歧:`obj.template method_template<float>();` |
| `undefined reference to void func<int,float>()` | 模板用之前必须实例化;"函数模板不是函数" | ① 显式实例化;② 把模板**定义**也 include 进来触发隐式实例化 |
| 三个"不同"的函数模板报重定义 | 编译器查重定义时**忽略有默认值的模板参数**,`template<typename T> void test(T)` 三种写法是同一个 | 改名或用 enable_if 区分 |
| `kernel<desc.act_pref><<<>>>()` 编不过 | 模板实参是运行时变量 | 提升为编译期常量,或 host 侧 if 分派 `kernel<true>/<false>` |
| 模板用爽了但… | 编译时间超级加倍、binary size 膨胀、debug 困难 | 权衡:热路径用模板,冷路径别炫技 |

## 溯源

- 主源:`~/zhanbin/library/training/C++模板编程.pptx`、`~/zhanbin/library/training/C++编程进阶技巧.pptx`(寒武纪基础软件部内训,含 cnnl/CNCC 语境实例与备注页问答)。两份 PPT 正文薄(合计 ~11K 字符),本篇以实践要点为主、不做教科书式注水;代码示例按 PPT 骨架还原为可编译形态。
- 语言规则交叉核对:cppreference 的 partial_specialization / sfinae / template_parameters / function_template 页(URL 2026-09 实测存活),确认"函数模板只能全特化""偏特化按更特殊优先""C++20 放宽 NTTP"等口径。
- 库内互链:[c-symbols-and-linking.md](c-symbols-and-linking.md)——`template` 消歧的原理(依赖名解析)与函数模板弱符号行为在链接篇展开。
