#!/usr/bin/env bash
# /data/cold/obrain/scripts/collect-memory.sh
# 采集 Hermes/HA/Miloco 当日增量数据，写入对应分类目录

set -euo pipefail

OBRAIN_DIR="/data/cold/obrain"
TODAY=$(date +%F)
TIMESTAMP=$(date '+%F %T')

# ========== Hermes 记忆导出 ==========
collect_hermes_memory() {
    local out_dir="$OBRAIN_DIR/hermes-memory"
    mkdir -p "$out_dir"

    # 导出 Hermes 记忆（需要 hermes CLI 支持）
    if command -v hermes >/dev/null 2>&1; then
        # 尝试导出记忆
        hermes memory export --format markdown --output "$out_dir/export.md" 2>/dev/null || true

        # 导出偏好/配置
        hermes config show --format yaml > "$out_dir/config.yaml" 2>/dev/null || true

        # 导出技能列表
        hermes skills list --format json > "$out_dir/skills.json" 2>/dev/null || true
    fi

    # 记录采集时间
    echo "---" >> "$out_dir/export.md"
    echo "collected_at: $TIMESTAMP" >> "$out_dir/export.md"
    echo "source: hermes-cli" >> "$out_dir/export.md"
}

# ========== Home Assistant 状态快照 ==========
collect_ha_snapshot() {
    local out_dir="$OBRAIN_DIR/device-logs"
    mkdir -p "$out_dir"
    
    # 从环境变量读取 HA 配置
    HA_URL="${HA_URL:-http://192.168.8.248:8123}"
    HA_TOKEN="${HA_TOKEN:-}"
    
    if [ -n "$HA_TOKEN" ]; then
        # 获取所有实体状态
        curl -s -H "Authorization: Bearer $HA_TOKEN" \
            "$HA_URL/api/states" > "$out_dir/ha-states-$TODAY.json" 2>/dev/null || true
        
        # 获取自动化列表
        curl -s -H "Authorization: Bearer $HA_TOKEN" \
            "$HA_URL/api/config/automation/config" > "$out_dir/ha-automations-$TODAY.json" 2>/dev/null || true
        
        # 生成 Markdown 摘要
        if command -v jq >/dev/null && [ -f "$out_dir/ha-states-$TODAY.json" ]; then
            jq -r '.[] | select(.state != "unavailable") | "\(.entity_id): \(.state) \(.attributes.friendly_name // "")"' \
                "$out_dir/ha-states-$TODAY.json" > "$out_dir/ha-summary-$TODAY.md" 2>/dev/null || true
        fi
    fi
    
    echo "---" >> "$out_dir/ha-summary-$TODAY.md"
    echo "collected_at: $TIMESTAMP" >> "$out_dir/ha-summary-$TODAY.md"
    echo "source: ha-rest-api" >> "$out_dir/ha-summary-$TODAY.md"
}

# ========== Miloco 状态采集 ==========
collect_miloco_state() {
    local out_dir="$OBRAIN_DIR/device-logs"
    mkdir -p "$out_dir"
    
    if command -v miloco-cli >/dev/null 2>&1; then
        # 设备列表
        miloco-cli device list --format json > "$out_dir/miloco-devices-$TODAY.json" 2>/dev/null || true
        
        # 规则列表
        miloco-cli rule list --format json > "$out_dir/miloco-rules-$TODAY.json" 2>/dev/null || true
        
        # 感知状态
        miloco-cli perception status --format json > "$out_dir/miloco-perception-$TODAY.json" 2>/dev/null || true
        
        # 生成 Markdown 摘要
        if command -v jq >/dev/null; then
            echo "# Miloco 设备状态 - $TODAY" > "$out_dir/miloco-summary-$TODAY.md"
            echo "" >> "$out_dir/miloco-summary-$TODAY.md"
            if [ -f "$out_dir/miloco-devices-$TODAY.json" ]; then
                jq -r '.[] | "\(.name) (\(.did)): \(.state // "unknown")"' "$out_dir/miloco-devices-$TODAY.json" >> "$out_dir/miloco-summary-$TODAY.md" 2>/dev/null || true
            fi
        fi
    fi
    
    echo "---" >> "$out_dir/miloco-summary-$TODAY.md"
    echo "collected_at: $TIMESTAMP" >> "$out_dir/miloco-summary-$TODAY.md"
    echo "source: miloco-cli" >> "$out_dir/miloco-summary-$TODAY.md"
}

# ========== 家庭档案同步 ==========
sync_home_profile() {
    local out_dir="$OBRAIN_DIR/home-profile"
    mkdir -p "$out_dir"

    # 从 Miloco 导出 home-profile（如果有该技能） - 单一文件
    if command -v miloco-cli >/dev/null 2>&1; then
        miloco-cli home-profile export --format markdown --output "$out_dir/members.md" 2>/dev/null || true
    fi

    # 也同步 ~/workspace/docs/ 到 family-docs（如果存在） - 单一文件夹
    if [ -d "$HOME/workspace/docs" ]; then
        rsync -av --delete "$HOME/workspace/docs/" "$OBRAIN_DIR/family-docs/" 2>/dev/null || true
    fi
}

# ========== 自动化规则导出 ==========
export_automation_rules() {
    local out_dir="$OBRAIN_DIR/automation-rules"
    mkdir -p "$out_dir"

    # Miloco 规则 - 单一文件
    if command -v miloco-cli >/dev/null 2>&1; then
        miloco-cli rule list --format yaml > "$out_dir/miloco-rules.yaml" 2>/dev/null || true

        # 规则治理文档 - 单一文件 governance.md
        cat > "$out_dir/governance.md" <<'GOV_EOF'
# 自动化规则治理原则

## 核心原则
1. **只关设备，不开设备** — "人来灯亮"由 HA/小爱负责
2. **所有 Miloco rule 的 on_enter_actions 必须为空**
3. **无人关设备 rule**：duration_seconds=900（15min），exit_debounce_seconds=30

## 设备例外清单（不在无人自动关规则中）
- 除湿机
- 空气净化器
- 摄像头
- 网关/路由器
- 服务器/NAS

## 区域传感器映射
- occupancy: 客卫/主卫/次卫/厨房/餐厅/阳台/偏厅
- motion: 楼梯/玄关/储藏室
- 无传感器: 负一玄关

## 摄像头分布
- 偏厅: 2个
- 客厅: 1个
- 阳台: 1个
GOV_EOF
    fi

    # HA 自动化（已在 collect_ha_snapshot 中获取）
    if [ -f "$OBRAIN_DIR/device-logs/ha-automations-$TODAY.json" ]; then
        cp "$OBRAIN_DIR/device-logs/ha-automations-$TODAY.json" "$out_dir/ha-automations-$TODAY.json" 2>/dev/null || true
    fi
}

# ========== 每日简报生成 ==========
generate_daily_brief() {
    local out_dir="$OBRAIN_DIR/daily-notes"
    mkdir -p "$out_dir"
    
    cat > "$out_dir/$TODAY.md" <<EOF
# 每日简报 - $TODAY

## 📊 系统健康
- Hermes: $(systemctl is-active hermes-gateway 2>/dev/null || echo "unknown")
- Miloco: $(systemctl is-active miloco-supervisord 2>/dev/null || echo "unknown")
- HA: $(curl -s -o /dev/null -w "%{http_code}" http://192.168.8.248:8123/api/ 2>/dev/null || echo "unreachable")

## 🔄 同步状态
- Git 状态: $(cd /data/cold/obrain && git status --porcelain | wc -l) 个待提交文件
- 远端同步: $(cd /data/cold/obrain && git log --oneline -1 --format="%h %s" 2>/dev/null || echo "no commits")

## 📝 待办/关注
- [ ] 检查冲突日志
- [ ] 审查新增自动化规则
- [ ] 验证设备状态一致性

---
*自动生成于 $TIMESTAMP*
EOF
}

# ========== 主流程 ==========
main() {
    echo "[$TIMESTAMP] 开始采集增量数据..."
    
    collect_hermes_memory
    echo "  ✅ Hermes 记忆"
    
    collect_ha_snapshot
    echo "  ✅ HA 快照"
    
    collect_miloco_state
    echo "  ✅ Miloco 状态"
    
    sync_home_profile
    echo "  ✅ 家庭档案"
    
    export_automation_rules
    echo "  ✅ 自动化规则"
    
    generate_daily_brief
    echo "  ✅ 每日简报"
    
    echo "[$TIMESTAMP] 采集完成"
}

main