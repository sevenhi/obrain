---
title: Startup Router
type: note
permalink: obrain/rules/startup-router
---

# Startup Router

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 23, 2026 | 詹彬 |

> **IMPORTANT**: READ THIS NOTE AT THE START OF EVERY SESSION BEFORE ANY KNOWLEDGE-BASE WORK.

## Non-Negotiable Rules
1. **Search Before Create**: Always search (`bm_search`) before writing a new note to prevent duplicates.
2. **Exact Casing & Paths**: Always use exact directory casing (`Tasks/`, `Projects/Home/`, `Notes/`, `Rules/`).
3. **Observations & Schemas**: Every note MUST carry at least one categorized observation (`- [category] fact`) in its `## Observations` section.
4. **Bidirectional Linking**: Always link related entities bidirectionally (e.g. Device ↔ Automation).
5. **Rules Center**: All schemas, instructions, and templates live in `Rules/`.

## Dispatch Table

| Task / Domain | Notes to Load (in order) |
|:--|:--|
| Tasks & Projects | 1. `rules/task instructions` · 2. `tasks/` |
| Mi Home / Devices / Automations | 1. `rules/home instructions` · 2. `projects/home/` |
| Notes / Journal / Freeform | 1. `rules/note instructions` · 2. `notes/` |
| Ambiguous / Everything Else | Load all instructions in `rules/` |

## Known Failure Modes
- Creating duplicate device/automation notes without title search.
- Writing scalar values only in frontmatter instead of `## Observations`.
- Overwriting notes without reading existing content first.
