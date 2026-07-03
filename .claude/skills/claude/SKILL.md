---
name: claude
description: Write or update CLAUDE.md following project conventions. Use when adding gotchas, restructuring CLAUDE.md, or updating project documentation.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# CLAUDE.md Authoring & Maintenance

Apply these principles when writing or modifying CLAUDE.md.

## Authoring Principles

| Principle | Description |
|-----------|-------------|
| **Timeliness** | Update as code evolves — not write-once-and-forget |
| **Specificity** | Write `set connection timeout to 30s` not `configure reasonable timeout` |
| **High-frequency first** | Sort by usage frequency — most frequently referenced at top |
| **Single source** | Architecture hard constraints are authoritatively defined in `specs/modules/index.md`; CLAUDE.md references them |
| **Length control** | Target 200–400 lines; if exceeding 400, consider moving content down to specs |

## Recommended Structure

```markdown
# CLAUDE.md

## 1. Project Overview
## 2. Specifications — point to specs/, declare spec authority
## 3. Architecture — module dependency diagram, hard constraints, data flow pipeline
## 4. Critical Implementation Details — most error-prone areas (sorted by frequency)
## 5. Development Commands — install, test, lint, manual verification
## 6. Known Gotchas — pitfall checklist (sorted by frequency)
## 7. Test Structure
```

## Known Gotchas Writing Guidelines

### Entry Format

```
N. **Keyword**: problem description + correct approach.

Elements: scenario (in what operation) → wrong (common mistake) → correct (how to do it right) → reason (why)
```

Counter-example: `"Watch the encoding"` → unclear what to watch for.
Good example: `"DB connections: Use connection pool (min=5, max=20), never create per-request."` → scenario + mistake + fix + reason.

### Entry Sources

| Source | Trigger |
|--------|---------|
| Bug fixes | After each bug fix, ask: could a Gotcha have prevented this? Yes → add one |
| Onboarding | Every friction a newcomer hits is a potential Gotcha |
| Code review | Issue types repeatedly flagged in reviews |
| Architecture decisions | Constraints whose violation has severe consequences |

### Maintenance

- Sort by frequency — most common pitfalls at the top
- Each Gotcha keeps bidirectional references with related specs / Critical Details
- If a Gotcha's corresponding bug has been eliminated by an architecture refactor, delete it
- If exceeding 20 entries, consider upgrading some to Critical Implementation Details
