---
title: MLU MIM 算力切分上手(对标 NVIDIA MIG)
type: howto
tags: [cambricon, mlu, mim, mig, sriov, container, virtualization, mlu370, partitioning]
source:
  - "ppt: MLU系列MIM介绍材料 v0.1 (Product Intro, 2.18.2022) | ~/zhanbin/library/training/MLU系列MIM介绍材料_v0.1.pptx (primary)"
  - "web: NVIDIA Multi-Instance GPU User Guide(MIG 概念/profile 对照口径) | https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/index.html"
  - "code: Cambricon k8s device plugin(SMLU/虚拟化部署的驱动与 cndev 版本门槛) | https://github.com/Cambricon/cambricon-k8s-device-plugin"
date: 2026-09
---

# MLU MIM 算力切分上手(对标 MIG)

## TL;DR

MIM 是寒武纪 MLU 的**硬件级资源切分**(Multi-Instance MLU),工作模式类似 NVIDIA MIG:把一张卡(MLU370-X8 为基准)**最大 4 拆分**,每个实例(MI)拥有**独立、隔离的物理资源**——IPU(推理处理单元)、VPU(视频)、JPU(JPEG)与内存,实例之间并行执行互不影响,实现**内存带宽与算力的硬隔离**。

两级抽象(与 MIG 的 GI/CI 同构):**PI = 按内存容量和带宽隔离出的资源抽象;CI = 在切好的 PI 内重新分配计算资源(IPU/VPU/JPU)**。两种落地场景:**容器化**(每个 MI 挂进容器)与**虚拟化 SRIOV**(每个 VF 一个隔离实例,配 pf/vf mdev 驱动 + qemu-kvm)。

**与软切分的本质区别**(这是选型关键):CUDA/MLU 世界里 streams 是单进程软队列、MPS 是逻辑共享——都**没有错误隔离**;MIG/MIM 是**物理分区**,有 SM/IPU 性能隔离、内存保护、带宽 QoS、错误隔离。代价是切分数目有限(MIG 最多 7、MIM 最多 4)、重配置要求空闲、图形 API 与 P2P 不支持。

---

## 1. 参照系:NVIDIA MIG 怎么切(理解 MIM 的最短路径)

MIG(Multi-Instance GPU)是硬件支持的资源切分,**按内存为单位 + 按算力为单位**两条线切;能创建的 GPU 实例(GI)片数不是任意的,NVIDIA 用 **profile** 描述每种卡的合法切法。

**A100-40GB 的物理形态**:8 个 5GB 内存片 + 7 个 SM 片,内存片与算力片组合成 GI,GI 内再切 CI。管理走 NVML/`nvidia-smi`:创建 GI → 在 GI 内创建 CI → CUDA 应用把一个 CI 看作一个计算设备。实例可动态创建/销毁不影响其它实例;未使用的空闲分区不会分给活跃分区。

**三种并发机制对比**(MIM 介绍材料引用的核心表,判断"要不要硬切分"就用它):

| | Streams | MPS | MIG(≈MIM) |
|---|---|---|---|
| 分区类型 | 单进程 | 逻辑 | **物理** |
| 最大分区数 | 无限 | 48 | **7** |
| SM/IPU 性能隔离 | 无 | 有(按百分比限,非分区) | **有** |
| 内存保护 | 无 | 有 | **有** |
| 内存带宽 QoS | 无 | **无** | **有** |
| 错误隔离 | 无 | **无** | **有** |
| 跨分区互操作 | 总是 | IPC | 有限 IPC |
| 重配置时机 | 动态 | 进程启动时 | **空闲时** |

**MIG 的系统/应用限制**(MIM 场景做方案时对照检查):支持 Linux + CUDA 11;bare-metal/container/hypervisor(SRIOV、MDEV)三种模式;配置模式驻留(同 ECC,除非强制退出);**busy 时不可重配置**;触发 MIG 模式需 CAP_SYS_ADMIN,创建/销毁只需 sudo/root;不支持图形 API;**不支持 GPU P2P**(PCIe/NVLink);GI 之间不支持 CUDA IPC(CI 之间支持);cuda-gdb/cuda-memcheck 可用;支持 MPS;GI 支持 GPUDirect RDMA。云端运维:DCGM 健康管理 + MIG-parted 容器化批量管理。

## 2. MIM:寒武纪的对应物

**资源单位**:按配置,每个 MI 可设为不同配置的整张 MLU370-X8 卡资源的一部分;可切分的板卡资源包括 **IPU(推理处理单元)、VPU(视频处理单元)、JPU(JPEG 处理单元)和内存**。

**术语两级**(对照 MIG 的 GI/CI):

| MIM | 含义 | MIG 对应 |
|---|---|---|
| **PI** | 按照内存容量和带宽隔离出的资源抽象 | GI(GPU Instance) |
| **CI** | 按照切分后的 PI,重新分配 PI 内的计算资源(IPU/VPU/JPU) | CI(Compute Instance) |

**硬指标**:整张 MLU370-X8 **最大 4 拆分**;每个 MI/VF 拥有各自单独、隔离的物理资源,可彼此互不影响并行执行任务,实现设备内存带宽、算力的**硬隔离**。MLU370 与 MLU590 各有硬件支持的切分方式表(材料中以图呈现,合法 profile 以 cnmon 实测输出为准)。

**两种落地场景**:
1. **容器化**:基于硬件的容器化方案,MI 挂进容器,提供高性能、可伸缩的容器化隔离;
2. **虚拟化(SRIOV)**:基于硬件的虚拟化方案,每个 VF 可设为不同配置的卡资源切片;配套 pf mdev 驱动、vf mdev 驱动、type2 qemu-kvm 与 type1 方案。

**版本路线**(v0.1 材料的规划表,2022 口径):3M'22 起 bare-metal(主机侧部署,cnmon 支持 mim 特性、cnToolkit 支持、cndev 管理库支持)→ 9M'22 容器化(容器化环境/运行时管理库)+ 虚拟化(sriov 管理工具、mdev 驱动、qemu-kvm/type1)→ 12M'22 运维监控(DCGM 支持 MIM 设备监控)+ 性能测试报告与典型配置规范。**注意该表是 2022 年规划,现状以所装 Neuware 版本的 cnmon/cndev 实际能力为准。**

## 3. 上手路径(bare-metal → 容器 → 虚拟化)

1. **确认软硬件门槛**:MLU370 系列;驱动与 cndev 版本满足特性要求(参考:Cambricon k8s device plugin 要求 mtp_372 驱动 ≥4.20.9、mtp_592 驱动 ≥5.0.0,SMLU 需驱动 ≥5.10.27 + cndev ≥3.9.0——版本门槛随特性收紧,部署前对表)。
2. **主机侧用 cnmon 管理切分**:cnmon 是 MIM 特性的管理入口(对标 `nvidia-smi mig` 工作流:开启模式 → 创建 PI/GI → 创建 CI → 查看实例与详细信息)。切分模式驻留、空闲时才可重配置,与 MIG 行为一致。
3. **应用视角**:一个 CI 实例 = 一个计算设备;CNToolkit/CNRT 程序无需改代码即可跑在实例上(CUDA 编程模型对 MIG 透明的同款设计)。
4. **容器化部署**:切好的 MI 经容器化管理库挂进容器;K8s 场景走 Cambricon device plugin(上报 MLU 数量、健康监控、MLU-enabled 容器调度)。
5. **虚拟化部署**:SRIOV 模式下每个 VF 一个隔离实例,配 mdev 驱动与 qemu-kvm;云端监控接 DCGM 类健康管理。

## 4. 实践要点与坑

1. **要硬隔离才用 MIM**:只是想在一张卡上跑多个互信任务、要动态弹性 → streams/多进程就够;要**故障隔离、带宽 QoS、多租户安全**(云推理多模型服务、按 SLA 分资源)→ MIM。MPS 类逻辑共享**没有错误隔离**,一个客户端崩全场崩。
2. **重配置必须空闲**:任务跑着不能改切分(bare-metal 不支持 busy 配置);编排系统里把"重切分"做成排水(drain)后动作。
3. **跨实例通信受限**:实例间 P2P 不支持、跨 GI/PI 的 IPC 受限——切分粒度要按"一个完整业务放一个实例"规划,别把一个模型的张量并行拆到两个 MI 上。
4. **切分数目少是特性不是缺陷**:物理分区最多 4(MIM)/7(MIG),空闲分区不回收给活跃分区;profile 化(合法组合有限)换来了确定性 QoS。
5. **权限分层**:触发切分模式是高权限操作(对标 CAP_SYS_ADMIN),日常创建/销毁只需 root/sudo——运维系统按此分权。
6. **与显存生命周期问题的关系**:MIM 切分改变的是**设备资源的物理边界**,不改变实例内部 host-vs-device 的生命周期规则——实例内的 allocator/queue 异步语义与整卡完全一致,见 [gpu-memory-lifecycle](../../gpu-memory-lifecycle/index.md)。

## 溯源

- 主源:`~/zhanbin/library/training/MLU系列MIM介绍材料_v0.1.pptx`(2022-02-18 Product Intro)。注意:材料约一半篇幅是 **MIG 介绍**(作参照系),MIM 本体信息量为:术语(PI/CI)、MLU370-X8 最大 4 拆分、IPU/VPU/JPU+内存可切、容器化/SRIOV 两场景、版本规划表;切分方式明细在材料中是图片,无法文本还原——本篇明确标注"以 cnmon 实测输出为准",不做无据编造。
- 交叉源:NVIDIA MIG User Guide(GI/CI 模型、profile、限制条款口径)https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/index.html (2026-09 实测存活);Cambricon k8s device plugin README(驱动/cndev 版本门槛)https://github.com/Cambricon/cambricon-k8s-device-plugin (2026-09 实测存活)。内部红区 cnmon 用户指南未登记明文链接。
- 跨域互链:[gpu-memory-lifecycle](../../gpu-memory-lifecycle/index.md)(实例内显存生命周期规则不因切分改变);同域 [cngdb-debugging.md](cngdb-debugging.md)。
