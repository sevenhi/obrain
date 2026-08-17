#!/usr/bin/env bash
# /data/cold/obrain/scripts/cleanup-structure.sh
# 清理 obrain 目录结构，保留单一文件版本，移除按日期的重复文件

set -euo pipefail

OBRAIN_DIR="/data/cold/obrain"

echo "=== 清理 obrain 目录结构 ==="

# 1. automation-rules: 保留 governance.md, 删除 governance-YYYY-MM-DD.md 和空的 miloco-rules-*.yaml
echo "1. 清理 automation-rules..."
cd "$OBRAIN_DIR/automation-rules"
# governance.md 已是最新版本（包含最新内容），保留它
rm -f governance-2026-08-16.md governance-2026-08-17.md
rm -f miloco-rules-2026-08-16.yaml miloco-rules-2026-08-17.yaml
# 确保 governance.md 是最新的（从 governance-2026-08-17.md 同步过来）
# 实际上 governance.md 已经是最新的，因为它在同步时被更新了
echo "   ✅ 保留: governance.md"

# 2. hermes-memory: 保留 preferences.md, corrections.md, export.md, skills.json, config.yaml
# 删除按日期的版本
echo "2. 清理 hermes-memory..."
cd "$OBRAIN_DIR/hermes-memory"
rm -f config-2026-08-16.yaml config-2026-08-17.yaml
rm -f export-2026-08-16.md export-2026-08-17.md
rm -f skills-2026-08-16.json skills-2026-08-17.json
rm -f merged-context-2026-08-16.md  # merged-context 每天重新生成，保留最新的
# 保留: preferences.md, corrections.md, merged-context-2026-08-17.md, config.yaml, skills.json, export.md
echo "   ✅ 保留: preferences.md, corrections.md, merged-context-*.md (最新), config.yaml, skills.json, export.md"

# 3. home-profile: 保留 members.md (单一文件)
echo "3. 清理 home-profile..."
cd "$OBRAIN_DIR/home-profile"
# members.md 已是单一文件，无需清理
echo "   ✅ 保留: members.md"

# 4. device-logs: 保留按日期的文件 (daily-notes 和 device-logs 按日期区分)
echo "4. device-logs 保留按日期结构..."
# ha-summary-YYYY-MM-DD.md 保留
# miloco-*-YYYY-MM-DD.json 保留
# miloco-summary-YYYY-MM-DD.md 保留
echo "   ✅ 保留所有按日期文件"

# 5. daily-notes: 保留按日期的文件
echo "5. daily-notes 保留按日期结构..."
# 2026-08-16.md, 2026-08-17.md 保留
# sync-*.log, sync-pull-*.log, sync-push-*.log, conflicts-*.md 保留 (已在 .gitignore)
# test-watcher.md, watcher-test-*.md 这些测试文件可以清理
cd "$OBRAIN_DIR/daily-notes"
rm -f test.log test-watcher.md watcher-test-221009.md watcher-test-221444.md
echo "   ✅ 清理测试文件，保留日期文件和日志"

# 6. family-docs: 保留单一文件
echo "6. family-docs 保留单一文件..."
# README.md 保留
echo "   ✅ 保留: README.md"

# 7. 删除 test-knowledge-update.md
echo "7. 清理测试文件..."
cd "$OBRAIN_DIR/hermes-memory"
rm -f test-knowledge-update.md

echo ""
echo "=== 清理完成 ==="
echo ""
echo "最终结构:"
find "$OBRAIN_DIR" -type f -name "*.md" -o -name "*.yaml" -o -name "*.json" | grep -v ".git" | sort