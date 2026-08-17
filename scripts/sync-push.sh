#!/usr/bin/env bash
# /data/cold/obrain/scripts/sync-push.sh
# 推送脚本：检测本地变更并推送到远端

set -euo pipefail

OBRAIN_DIR="/data/cold/obrain"
LOG_FILE="$OBRAIN_DIR/daily-notes/sync-push-$(date +%F).log"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"
}

cd "$OBRAIN_DIR"

# 检查是否有本地变更（包括未跟踪文件）
if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    log "检测到本地变更，准备推送..."
    
    git add -A
    COMMIT_MSG="auto: $(date +%F-%H%M) - local changes sync"
    git commit -m "$COMMIT_MSG" 2>&1 | tee -a "$LOG_FILE"
    
    log "推送到远端..."
    if git push origin main 2>&1 | tee -a "$LOG_FILE"; then
        log "✅ 推送完成"
    else
        log "❌ 推送失败，可能需要先拉取"
        # 尝试拉取后再推
        log "尝试拉取后重新推送..."
        git pull --rebase origin main 2>&1 | tee -a "$LOG_FILE"
        git push origin main 2>&1 | tee -a "$LOG_FILE"
        log "✅ 重新推送完成"
    fi
else
    log "无本地变更，跳过推送"
fi

log "=== 推送同步完成 ==="