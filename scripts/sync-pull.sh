#!/usr/bin/env bash
# /data/cold/obrain/scripts/sync-pull.sh
# 高频拉取脚本：每 15 分钟从远端拉取更新，自动解决冲突

set -euo pipefail

OBRAIN_DIR="/data/cold/obrain"
LOG_FILE="$OBRAIN_DIR/daily-notes/sync-pull-$(date +%F).log"
CONFLICT_LOG="$OBRAIN_DIR/daily-notes/conflicts-$(date +%F).md"

cd "$OBRAIN_DIR"

# 先检查是否有本地未提交变更（包括未跟踪文件），必须在任何日志写入之前
# 避免日志文件本身被检测为变更导致死循环
HAS_CHANGES=0
if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    HAS_CHANGES=1
fi

# 现在可以安全地创建日志目录和定义 log 函数
mkdir -p "$(dirname "$LOG_FILE")"
log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"
}

if [ "$HAS_CHANGES" -eq 1 ]; then
    log "检测到本地未提交变更，暂存..."
    git stash push -m "auto-stash-before-pull-$(date +%s)" --include-untracked
    STASHED=1
else
    STASHED=0
fi

# 拉取远端（rebase 保持历史整洁）
log "拉取远端 main..."
if ! git pull --rebase origin main 2>&1 | tee -a "$LOG_FILE"; then
    log "⚠️ rebase 冲突，尝试自动解决..."
    
    CONFLICT_FILES=$(git diff --name-only --diff-filter=U)
    echo "# 冲突记录 - $(date '+%F %T')" > "$CONFLICT_LOG"
    echo "" >> "$CONFLICT_LOG"
    
    for file in $CONFLICT_FILES; do
        log "处理冲突: $file"
        echo "## $file" >> "$CONFLICT_LOG"
        echo '```diff' >> "$CONFLICT_LOG"
        git diff "$file" >> "$CONFLICT_LOG" 2>/dev/null || true
        echo '```' >> "$CONFLICT_LOG"
        echo "" >> "$CONFLICT_LOG"
        
        # 自动解决策略：文本文件尝试三方合并，保留双方新增
        if file "$file" | grep -q text; then
            BASE=$(git merge-base HEAD origin/main)
            git show "$BASE:$file" > "$file.base" 2>/dev/null || true
            git show "HEAD:$file" > "$file.local" 2>/dev/null || true
            git show "origin/main:$file" > "$file.remote" 2>/dev/null || true
            
            if command -v git-merge-file >/dev/null && [ -f "$file.base" ] && [ -f "$file.local" ] && [ -f "$file.remote" ]; then
                if git merge-file -p "$file.local" "$file.base" "$file.remote" > "$file.merged" 2>/dev/null; then
                    mv "$file.merged" "$file"
                    log "  ✅ 三方合并成功"
                    echo "✅ 自动三方合并成功" >> "$CONFLICT_LOG"
                else
                    log "  ⚠️ 三方合并失败，保留本地版本"
                    mv "$file.local" "$file"
                    echo "⚠️ 合并失败，保留本地版本" >> "$CONFLICT_LOG"
                fi
            else
                log "  ⚠️ 无法三方合并，保留本地版本"
                mv "$file.local" "$file" 2>/dev/null || true
                echo "⚠️ 无法三方合并，保留本地版本" >> "$CONFLICT_LOG"
            fi
            rm -f "$file.base" "$file.local" "$file.remote" "$file.merged"
        else
            log "  📦 二进制文件冲突，备份本地并采用远端"
            mv "$file" "$file.local.$(date +%s)" 2>/dev/null || true
            git checkout origin/main -- "$file"
            echo "📦 二进制文件：采用远端版本，本地备份" >> "$CONFLICT_LOG"
        fi
        git add "$file"
    done
    
    # 继续 rebase
    git rebase --continue 2>&1 | tee -a "$LOG_FILE"
    log "✅ 冲突解决完成，rebase 继续"
fi

# 恢复暂存的本地变更
if [ "$STASHED" -eq 1 ]; then
    log "恢复本地暂存变更..."
    git stash pop 2>&1 | tee -a "$LOG_FILE" || log "⚠️ stash pop 有冲突，需手动处理"
fi

log "=== 拉取同步完成 ==="