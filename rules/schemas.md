---
title: Schemas
type: note
permalink: obrain/rules/schemas
---

# Schemas

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 23, 2026 | 詹彬 |

## Task Schema
- **Entity**: `Task`
- **Fields**:
  - `status(enum)`: `[open, in-progress, blocked, done]` (required)
  - `priority?(enum)`: `[low, medium, high]` (optional)
  - `due?`: `string` (optional)
  - `project?`: `Project` (optional relation)

## Device Schema
- **Entity**: `Device`
- **Fields**:
  - `location`: `string` (required)
  - `status(enum)`: `[online, offline, warning]` (required)
  - `model?`: `string` (optional)
  - `control_type?`: `string` (optional)

## Automation Schema
- **Entity**: `Automation`
- **Fields**:
  - `status(enum)`: `[active, testing, disabled]` (required)
  - `trigger`: `string` (required)
  - `action`: `string` (required)
  - `target_device?`: `Device` (optional relation)

## Note Schema
- **Entity**: `Note`
- **Fields**:
  - `category`: `string` (required observation)
