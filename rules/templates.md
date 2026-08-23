---
title: Templates
type: note
permalink: obrain/rules/templates
---

# Templates

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 23, 2026 | 詹彬 |

## 1. Task Template
```markdown
# [Task Title]

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | [Date] | 詹彬 |

[Task description...]

## Observations
- [status] open
- [priority] medium
- [due] [YYYY-MM-DD]

## Relations
- project [[Project Name]]
```

## 2. Device Template
```markdown
# [Device Name]

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | [Date] | 詹彬 |

[Device description, placement, notes...]

## Observations
- [location] [Room]
- [status] online
- [model] [Model Name]

## Relations
- part_of [[Home Automation]]
- controlled_by [[Automation Rule Name]]
```

## 3. Automation Template
```markdown
# [Automation Rule Name]

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | [Date] | 詹彬 |

[Rule description and logic details...]

## Observations
- [status] active
- [trigger] [Trigger condition]
- [action] [Action executed]

## Relations
- target_device [[Device Name]]
```
