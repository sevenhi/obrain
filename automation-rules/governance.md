---
date: 2026-08-16
source: miloco-rule-governance
tags: [automation, governance, rules, best-practices]
---

# 自动化规则治理文档

## 核心治理原则

### 1. 只关不开
- **原则**: Miloco 规则**只负责关设备**，不负责开设备
- **理由**: "人来灯亮"由 HA 自动化/小爱语音/手动控制负责，避免竞态条件
- **执行**: 所有 Miloco rule 的 `on_enter_actions` **必须为空**

### 2. 无人关设备标准参数
```yaml
duration_seconds: 900        # 15 分钟无人才触发
exit_debounce_seconds: 30    # 30 秒防抖窗口 (HA 后置兜底)
```

### 3. 设备例外清单
#### 3.1 永不自动关闭
| 设备类型 | 理由 |
|----------|------|
| 摄像头 | 安防监控不能中断 |
| 网关/路由器 | 网络基础设施 |
| 服务器/NAS | 服务持续运行 |
| 冰箱/冰柜 | 食品安全 |
| 热水器 | 生活热水需求 |

#### 3.2 分时段关闭
| 设备类型 | 理由 |
|----------|------|
| 空气净化器 | 夜间睡眠时关闭，减少噪音干扰，白天正常运行 |
| 除湿机 | 夜间睡眠时关闭，减少噪音干扰，白天根据湿度情况自动运行 |

---

## 区域传感器映射表

### Occupancy 传感器 (有人/无人状态)
| 区域 | 传感器类型 | 备注 |
|------|------------|------|
| 客卫 | occupancy | |
| 主卫 | occupancy | |
| 次卫 | occupancy | |
| 厨房 | occupancy | |
| 餐厅 | occupancy | |
| 阳台 | occupancy | |
| 偏厅 | occupancy | |

### Motion 传感器 (运动触发)
| 区域 | 传感器类型 | 备注 |
|------|------------|------|
| 楼梯 | motion | 过道照明触发 |
| 一楼玄关 | motion | 进出感应 |
| 负一玄关 | motion | 进出感应 |
| 储藏室 | motion | 偶尔进入 |

### 无传感器区域
- 客厅（除餐厅外） — 依赖相邻区域或摄像头感知
- 主卧            — 无
- 次卧            — 无
- 客卧            — 无

---

## 摄像头部署与感知范围

| 位置 | 数量 | 品牌 | 感知用途 |
|------|------|------|----------|
| 偏厅 | 1 | 小米 | 办公监护、孩子活动、异常检测 |
| 客厅 | 1 | 小米 | 主活动区、安防、跌倒检测 |
| 阳台 | 1 | 创米 | 孩子玩耍、晾晒监控、入侵检测 |

### 感知范围配置 (miloco-miot-scope)
- 每台摄像头可独立配置：是否参与视觉感知、是否参与音频感知
- 偏厅双摄：视觉+音频全开
- 客厅/阳台：视觉+音频全开

---

## 规则变更流程

### 变更前
1. 在 `automation-rules/changes/` 创建变更提案 `.md`
2. 标明：影响范围、预期效果、回滚方案
3. 经用户确认后执行

### 变更执行
```bash
# 1. 备份当前规则
miloco-cli rule list --format yaml > /data/cold/obrain/automation-rules/backup/rules-$(date +%F).yaml

# 2. 应用变更
miloco-cli rule apply <new-rules.yaml>

# 3. 验证
miloco-cli rule list --format json | jq '.[] | select(.enabled==true)'
```

### 变更后
1. 记录变更日志到 `automation-rules/changelog.md`
2. 更新治理文档 (如参数调整)
3. 观察 24h 无异常才标记完成

---

## 规则审计清单 (每周巡检)

- [ ] 所有规则 `on_enter_actions` 为空
- [ ] 无人关规则 `duration_seconds=900`, `exit_debounce_seconds=30`
- [ ] 例外设备未被纳入关设备规则
- [ ] 区域传感器映射与实际部署一致
- [ ] 摄像头感知范围配置正确
- [ ] 无孤立/僵尸规则 (未关联有效传感器/设备)
- [ ] 规则命名规范: `<区域>-<设备类型>-<动作>-<触发条件>`

---

## 命名规范示例

```
livingroom-light-off-occupancy-900s
kitchen-fan-off-occupancy-900s
balcony-light-off-occupancy-900s
study-ac-off-occupancy-900s
bathroom-exhaust-off-occupancy-900s
```

---

## 2026-08-16 批量治理记录

| 动作 | 规则数量 | 说明 |
|------|----------|------|
| 删除 "人来灯亮" 类规则 | 12 条 | 违反只关不开原则 |
| 修正偏厅 on_enter 错配 | 1 条 | 错绑定开灯动作 |
| 统一 duration 30s → 900s | 6 条 | 避免误关 |
| 移除偏厅智能屏麦克风/音量控制 | 2 项 | 设备属性不可控 |
| **清理前总计** | **17 条** | |
| **清理后总计** | **6 条** | 仅保留无人关设备规则 |## Test Knowledge Mon 17 Aug 2026 10:38:01 PM CST
