#!/usr/bin/env bash
# /data/cold/obrain/scripts/load-obrain-memory.sh
# Hermes 启动时加载 obrain 知识库记忆
# 将关键 Markdown 文件内容注入到 Hermes 记忆/上下文中

set -euo pipefail

OBRAIN_DIR="/data/cold/obrain"
HERMES_MEM_DIR="$OBRAIN_DIR/hermes-memory"
HOME_PROFILE_DIR="$OBRAIN_DIR/home-profile"
AUTO_RULES_DIR="$OBRAIN_DIR/automation-rules"

# 要加载的核心文件 (按优先级排序)
CORE_FILES=(
    "$HERMES_MEM_DIR/preferences.md"
    "$HERMES_MEM_DIR/corrections.md"
    "$HOME_PROFILE_DIR/members.md"
    "$AUTO_RULES_DIR/governance.md"
)

# 可选文件 (最新的单一文件版本)
LATEST_EXPORT="$HERMES_MEM_DIR/export.md"
LATEST_HA_SUMMARY=$(ls -t "$OBRAIN_DIR/device-logs"/ha-summary-*.md 2>/dev/null | head -1 || true)
LATEST_MILOCO_SUMMARY=$(ls -t "$OBRAIN_DIR/device-logs"/miloco-summary-*.md 2>/dev/null | head -1 || true)
LATEST_DAILY=$(ls -t "$OBRAIN_DIR/daily-notes"/*.md 2>/dev/null | grep -v "sync-\|conflicts-" | head -1 || true)

OPTIONAL_FILES=()
[ -n "$LATEST_EXPORT" ] && OPTIONAL_FILES+=("$LATEST_EXPORT")
[ -n "$LATEST_HA_SUMMARY" ] && OPTIONAL_FILES+=("$LATEST_HA_SUMMARY")
[ -n "$LATEST_MILOCO_SUMMARY" ] && OPTIONAL_FILES+=("$LATEST_MILOCO_SUMMARY")
[ -n "$LATEST_DAILY" ] && OPTIONAL_FILES+=("$LATEST_DAILY")

echo "=== Loading obrain memory into Hermes ==="
echo "Core files:"
for f in "${CORE_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ⚠️  $f (not found)"
    fi
done

echo "Optional files:"
for f in "${OPTIONAL_FILES[@]}"; do
    echo "  📄 $f"
done

# 生成合并的上下文文件，供 Hermes 读取
CONTEXT_FILE="$OBRAIN_DIR/hermes-memory/merged-context-$(date +%F).md"
{
    echo "# obrain 知识库合并上下文"
    echo ""
    echo "生成时间: $(date '+%F %T')"
    echo "来源: /data/cold/obrain"
    echo ""
    echo "---"
    echo ""
    
    # 核心文件
    for f in "${CORE_FILES[@]}"; do
        if [ -f "$f" ]; then
            echo "## $(basename "$f" .md)"
            echo ""
            cat "$f"
            echo ""
            echo "---"
            echo ""
        fi
    done
    
    # 可选文件
    for f in "${OPTIONAL_FILES[@]}"; do
        if [ -f "$f" ]; then
            echo "## $(basename "$f" .md) (latest)"
            echo ""
            cat "$f"
            echo ""
            echo "---"
            echo ""
        fi
    done
} > "$CONTEXT_FILE"

echo ""
echo "✅ 合并上下文已生成: $CONTEXT_FILE"
echo ""
echo "=== 加载完成 ==="
echo ""
echo "Hermes 可通过以下方式读取:"
echo "  1. 直接读取 $CONTEXT_FILE"
echo "  2. 通过 skill 系统加载 (obrain-knowledge-sync 技能)"
echo "  3. 启动脚本中 source 此文件的输出"