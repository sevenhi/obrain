#!/usr/bin/env bash
# /data/cold/obrain/scripts/sync-pull.sh
# 高频拉取脚本：每 15 分钟从远端拉取更新，自动解决冲突，并同步新知识到记忆系统

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

# 记录拉取前的 commit hash
PRE_PULL_HASH=$(git rev-parse HEAD 2>/dev/null || echo "")

if [ "$HAS_CHANGES" -eq 1 ]; then
    log "检测到本地未提交变更，暂存..."
    git stash push -m "auto-stash-before-pull-$(date +%s)" --include-untracked
    STASHED=1
else
    STASHED=0
fi

# 拉取远端（rebase 保持历史整洁）
log "拉取远端 main..."
PULL_OUTPUT=""
if ! PULL_OUTPUT=$(git pull --rebase origin main 2>&1); then
    log "⚠️ rebase 冲突，尝试自动解决..."
    echo "$PULL_OUTPUT" | tee -a "$LOG_FILE"
    
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
else
    echo "$PULL_OUTPUT" | tee -a "$LOG_FILE"
fi

# 恢复暂存的本地变更
if [ "$STASHED" -eq 1 ]; then
    log "恢复本地暂存变更..."
    git stash pop 2>&1 | tee -a "$LOG_FILE" || log "⚠️ stash pop 有冲突，需手动处理"
fi

# 记录拉取后的 commit hash
POST_PULL_HASH=$(git rev-parse HEAD 2>/dev/null || echo "")

# 检查是否有远端更新（commit hash 变化）
if [ -n "$PRE_PULL_HASH" ] && [ -n "$POST_PULL_HASH" ] && [ "$PRE_PULL_HASH" != "$POST_PULL_HASH" ]; then
    log "🔄 检测到远端更新 (${PRE_PULL_HASH:0:8} -> ${POST_PULL_HASH:0:8})"
    
    # 显示变更的文件
    CHANGED_FILES=$(git diff --name-only "$PRE_PULL_HASH" "$POST_PULL_HASH" 2>/dev/null || true)
    if [ -n "$CHANGED_FILES" ]; then
        log "变更文件:"
        echo "$CHANGED_FILES" | while read -r f; do
            log "  📝 $f"
        done
        
        # 筛选核心知识文件（需要同步到记忆系统的文件）
        KNOWLEDGE_FILES=$(echo "$CHANGED_FILES" | grep -E "(hermes-memory|home-profile|automation-rules)/.*\.(md|yaml|yml)$" | grep -vE "(sync-|conflicts-|governance-\d)" || true)
        
        if [ -n "$KNOWLEDGE_FILES" ]; then
            log "📚 核心知识文件有更新，触发记忆同步..."
            echo "$KNOWLEDGE_FILES" | while read -r f; do
                log "  🔄 $f"
            done
            
            # 1. 重新加载 obrain 记忆到合并上下文
            if [ -f "$OBRAIN_DIR/scripts/load-obrain-memory.sh" ]; then
                log "📖 重新生成合并上下文..."
                bash "$OBRAIN_DIR/scripts/load-obrain-memory.sh" 2>&1 | tee -a "$LOG_FILE" || log "⚠️ load-obrain-memory.sh 执行异常"
            fi
            
            # 2. 同步到 Hermes 内置记忆
            if [ -f "$OBRAIN_DIR/scripts/sync-to-hermes-memory.sh" ]; then
                log "🔄 同步到 Hermes 内置记忆 (MEMORY.md/USER.md)..."
                bash "$OBRAIN_DIR/scripts/sync-to-hermes-memory.sh" 2>&1 | tee -a "$LOG_FILE" || log "⚠️ sync-to-hermes-memory.sh 执行异常"
            fi
            
            # 3. 注入到 Honcho (如果可用)
            if [ -f "$OBRAIN_DIR/scripts/inject-to-honcho.py" ]; then
                log "🧠 注入到 Honcho Conclusions API..."
                python3 "$OBRAIN_DIR/scripts/inject-to-honcho.py" 2>&1 | tee -a "$LOG_FILE" || log "⚠️ inject-to-honcho.py 执行异常"
            fi
            
            log "✅ 知识同步完成"
        else
            log "ℹ️ 无核心知识文件变更，跳过记忆同步"
        fi
    else
        log "ℹ️ 无文件变更"
    fi
else
    log "✅ 远端无新更新"
fi

log "=== 拉取同步完成 ==="