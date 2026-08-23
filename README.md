# obrain - 个人知识库与智能家居中枢

> 运行后端：Basic Memory  
> 知识库规则中心：`Rules/`

## 📁 目录结构

| 目录 | 用途 | 说明 |
| :--- | :--- | :--- |
| `Tasks/` | 待办事项与任务追踪 | 遵循 Task Schema (`status`, `due`, `priority`) |
| `Projects/` | 核心项目与领域 | 包含 `Home/` (米家自动化及设备治理) 与 `Research/` (课题研究) |
| `Notes/` | 日志、偏好与经验记录 | 包含每日日志 (`YYYY-MM-DD.md`)、用户偏好、系统修正等 |
| `Rules/` | 规则与自动化指令 | 包含 Startup Router、各领域指令、Schema 定义及模版 |
| `hermes-sessions/`| 历史会话归档 | 系统交互痕迹 |

## 🛠 规则与规范
详细规范及 AI 启动路由请参阅：
- [Startup Router](Rules/Startup%20Router.md)
- [Schemas 规范定义](Rules/Schemas.md)
- [Templates 模版中心](Rules/Templates.md)
