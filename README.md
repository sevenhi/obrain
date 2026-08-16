# obrain - Hermes 知识库 & 家庭智能中枢

> 本地路径：`/data/cold/obrain`  
> 远端仓库：GitHub（每日自动同步）  
> 核心原则：**只追加、分类存储、Markdown 格式、Git 版本控制、双向同步**

---

## 📁 目录结构

| 目录 | 用途 | 更新频率 | 说明 |
|------|------|----------|------|
| `hermes-memory/` | Hermes 长期记忆、偏好、纠正记录 | 持续/每日 | 供 Hermes 启动时读取、矫正自身 |
| `home-profile/` | 家庭成员档案、偏好、习惯、作息 | 持续 | Miloco home-profile 技能同步 |
| `device-logs/` | 设备状态快照、能耗、异常记录 | 每日/事件触发 | 供能源管理、故障复盘 |
| `automation-rules/` | Miloco/Hass 规则导出、变更记录、治理文档 | 变更时 | 规则审计、回滚依据 |
| `skills/` | 自研技能文档、最佳实践、迁移指南 | 持续 | 便于其他设备/实例复用 |
| `projects/` | 项目级上下文、决策记录、进度 | 项目周期 | 多会话跨设备协作 |
| `daily-notes/` | 每日简报、巡检报告、异常复盘 | 每日 | 可按 `YYYY/MM/DD.md` 归档 |
| `family-docs/` | 说明书、装修图纸、保修、合同扫描件 | 低频 | `~/workspace/docs/` 定期同步至此 |

---

## 🔄 自动化流程

### 每日同步（Cron：0 3 * * *）
1. `git pull --rebase origin main` — 拉取远端，自动解决冲突（优先保留本地新增）
2. 运行 `scripts/collect-memory.sh` — 采集 Hermes/HA/Miloco 当日增量
3. `git add -A && git commit -m "daily: $(date +%F)"`
4. `git push origin main`

### 冲突解决策略
- **同一文件双端修改**：`git merge-file -p` 三方合并，保留双方新增行
- **二进制/大文件**：以远端为准，本地备份至 `.conflicts/`
- **冲突日志**：记录至 `daily-notes/conflicts-$(date +%F).md`

---

## 🤖 Hermes 读取协议

Hermes 启动/周期性读取：
```python
# 伪代码
for category in ["hermes-memory", "home-profile", "automation-rules"]:
    for file in sorted(glob(f"/data/cold/obrain/{category}/*.md")):
        load_context(file)
```

### 关键文件命名约定
- `hermes-memory/preferences.md` — 用户偏好（语言、界面、自动化原则）
- `hermes-memory/corrections.md` — 用户纠正记录（日期、原行为、纠正后行为）
- `home-profile/members.md` — 成员档案（姓名、角色、偏好设备/场景）
- `automation-rules/governance.md` — 规则治理原则（只关不开、duration=900s 等）

---

## 🛠 维护命令

```bash
# 手动触发同步
/data/cold/obrain/scripts/sync.sh

# 查看冲突历史
cat /data/cold/obrain/daily-notes/conflicts-*.md

# 导出当前 Hermes 记忆到库
hermes memory export --format markdown --output /data/cold/obrain/hermes-memory/export-$(date +%F).md
```

---

## 📌 约定

1. **只追加不覆盖** — 修正用新文件/新段落，保留历史
2. **前置元数据** — 每个 `.md` 以 YAML frontmatter 开头（date, source, tags）
3. **软链接友好** — `~/workspace/obrain -> /data/cold/obrain` 便于 Hermes 技能访问