#!/usr/bin/env bash
# /data/cold/obrain/scripts/sync-to-hermes-memory.sh
# 将 obrain 知识库同步到 Hermes 内置记忆 (MEMORY.md / USER.md)

set -euo pipefail

OBRAIN_DIR="/data/cold/obrain"
HERMES_MEM_DIR="$HOME/.hermes/memories"
HERMES_MEMORY_FILE="$HERMES_MEM_DIR/MEMORY.md"
HERMES_USER_FILE="$HERMES_MEM_DIR/USER.md"

# 要同步的 obrain 核心文件
CORE_FILES=(
    "$OBRAIN_DIR/hermes-memory/preferences.md"
    "$OBRAIN_DIR/hermes-memory/corrections.md"
    "$OBRAIN_DIR/home-profile/members.md"
    "$OBRAIN_DIR/automation-rules/governance.md"
)

# 创建备份
TIMESTAMP=$(date +%F_%H%M%S)
cp "$HERMES_MEMORY_FILE" "$HERMES_MEMORY_FILE.bak.$TIMESTAMP" 2>/dev/null || true
cp "$HERMES_USER_FILE" "$HERMES_USER_FILE.bak.$TIMESTAMP" 2>/dev/null || true

echo "=== Syncing obrain to Hermes built-in memory ==="

# 追加到 MEMORY.md (技术/系统知识)
{
    echo ""
    echo "§"
    echo "# obrain Knowledge Base - $(date '+%F %T')"
    echo ""
    for f in "${CORE_FILES[@]}"; do
        if [ -f "$f" ]; then
            echo "## $(basename "$f" .md | tr '[:lower:]' '[:upper:]')"
            echo ""
            cat "$f"
            echo ""
            echo "§"
            echo ""
        fi
    done
} >> "$HERMES_MEMORY_FILE"

echo "✅ Appended to MEMORY.md"

# 追加到 USER.md (用户偏好/档案) - 只追加 preferences 和 members
{
    echo ""
    echo "§"
    echo "# obrain User Profile - $(date '+%F %T')"
    echo ""
    if [ -f "$OBRAIN_DIR/hermes-memory/preferences.md" ]; then
        echo "## PREFERENCES"
        echo ""
        cat "$OBRAIN_DIR/hermes-memory/preferences.md"
        echo ""
        echo "§"
        echo ""
    fi
    if [ -f "$OBRAIN_DIR/home-profile/members.md" ]; then
        echo "## FAMILY_MEMBERS"
        echo ""
        cat "$OBRAIN_DIR/home-profile/members.md"
        echo ""
        echo "§"
        echo ""
    fi
} >> "$HERMES_USER_FILE"

echo "✅ Appended to USER.md"

# 尝试通过 Honcho CLI 注入 (如果可用)
if command -v hermes >/dev/null 2>&1; then
    echo "=== Trying Honcho injection via hermes CLI ==="
    # 这里可以尝试 honcho_conclude，但需要 hermes 运行时环境
    # 暂时跳过，因为需要运行中的 Hermes 实例
    echo "ℹ️  Honcho injection requires running Hermes instance, skipping in standalone mode"
fi

echo ""
echo "=== Sync complete ==="
echo "MEMORY.md: $HERMES_MEMORY_FILE"
echo "USER.md: $HERMES_USER_FILE"