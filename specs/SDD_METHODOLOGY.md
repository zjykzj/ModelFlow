# SDD Methodology

> **This document defines the universal SDD (Spec-Driven Development) methodology.**
>
> Target audience: AI Agents (such as Claude Code) and human developers.
>
> This document can be copied to any new project adopting the SDD methodology.
> It was originally developed in the [DataFlow-CV](https://github.com/zjykzj/DataFlow-CV) project
> and adapted for ModelFlow.

---

## 1. Core Philosophy

```
Specs (what is correct) → Dev Context (how the code works) → Make a plan → Implementation
     ↑         ← (write back when behavior changes)               |
     └─────────── Test verification + doc sync ←──────────────────┘
```

**The three-layer SDD system:**

| Layer | Role | Change Frequency |
|-------|------|------------------|
| **Specs** | Behavioral contract — defines "what is correct" | Rarely (only when requirements change) |
| **Dev Context** (CLAUDE.md) | Architecture knowledge — describes "how the code is written" | Evolves with the codebase |
| **Code** | Implementation — the actual running code | Daily |

**Iron rules:**
- Specs are the highest authority. If code behavior conflicts with specs, the specs take precedence — fix the code.
- Specs are living documents. When requirements change or specs are insufficient, update the specs first, then proceed.

---

## 2. Development Workflow

### 2.1 When Receiving a New Task

**Step 1: Determine the scope of impact**

Ask yourself three questions:
1. Which module does the change affect?
2. Which external interface/format does it involve?
3. Is it cross-module? (Cross-module changes carry the highest risk.)

**Step 2: Read specs and evaluate sufficiency**

> ⚠️ **Specs are living documents.** Read specs with a critical eye:
> - Do the specs cover the current scenario? Are the definitions clear and unambiguous?
> - Are the behavioral definitions in the specs reasonable and internally consistent?
> - **If insufficient or unreasonable → prioritize updating the specs before proceeding.** Do not build on an unstable foundation.

**Step 3: Cross-reference the dev context document**

Read the relevant sections of CLAUDE.md (or equivalent), focusing on:
- **Known Gotchas** — common traps
- **Critical Implementation Details** — key correctness constraints

**Step 4: Create a development plan**

After reading specs (to understand "what is correct") and the dev context doc (to understand "how the code is written"), **explicitly create a development plan** before writing code:

1. **List all files involved**: which ones to create, modify, or delete
2. **Determine the change order**: prioritize by dependency relationships (modify base interfaces first, then upper-level callers)
3. **Identify risk points**: which existing features might be affected? Where is it most error-prone?
4. **Use Plan mode for complex tasks**: for cross-module changes or new capabilities, use `EnterPlanMode` to generate a complete plan

> Plan first, then implement — avoid discovering architecture conflicts mid-implementation and starting over.

### 2.2 Before Submitting

1. **Run tests** (must pass)
2. **Format check** (lint / format)
3. **Documentation sync check** (must do for every change):

| Priority | Document | Check Condition | Action |
|----------|----------|----------------|--------|
| **P0** | Specs | Behavior changes (interface, contract, data flow) | **Must** sync update |
| **P1** | Dev Context (CLAUDE.md) | New architecture details, new gotchas, new hard constraints, new key implementations | Sync update |
| **P1** | README | API changes, new feature entry points, installation step changes | Sync update user documentation |
| **P2** | Sample code | User API changes, new task types, calling convention changes | Update examples |

**Git commit format (Conventional Commits):**

```
<type>(<scope>): <subject>

<body if needed>
```

Types: `feat` / `fix` / `docs` / `refactor` / `test` / `style` / `perf` / `chore` / `build` / `ci`

---

## 3. Code Review Checklist (Universal)

Self-check after every change:

- [ ] Architecture hard constraints are not violated (module dependency rules, layering constraints)
- [ ] New functions/classes have corresponding tests
- [ ] All tests pass
- [ ] Behavior changes have been synced to specs (P0)
- [ ] New architecture details/gotchas have been synced to dev context document (P1)
- [ ] API / feature entry point changes have been synced to README (P1)
- [ ] User interface changes have been synced to sample code (P2)

---

## 4. Specs Directory Structure

### 4.1 Core Principle: WHAT vs HOW Separation

Specs are organized into two layers — **WHAT** (external contracts) and **HOW** (internal architecture). This is the most important structural design of SDD.

```
specs/
├── <contract-layer-1>/    # WHAT — external data/interface contracts
│   └── index.md
│
├── <contract-layer-2>/    # WHAT — other contract layers (optional)
│   └── index.md
│
└── modules/               # HOW — internal module architecture
    ├── index.md
    ├── spec_<module-1>.md
    ├── spec_<module-2>.md
    └── ...
```

**HOW to name the WHAT layer depends on the project domain:**

| Project Type | Suggested Name | Example Content |
|-------------|---------------|-----------------|
| Data processing / format conversion | `formats/` | Data format definitions, conversion rules |
| Web API | `api/` | REST/GraphQL interface contracts |
| Library / SDK | `interfaces/` | Public API signatures, type definitions |
| Evaluation / benchmarking | `evaluate/` | Metric definitions, baselines |
| Protocol / communication | `protocols/` | Message formats, state machines |

A project can have multiple WHAT layers (e.g., `formats/` + `evaluate/`), each defining one class of external contracts. There is only one HOW layer (`modules/`), with one spec per code module.

### 4.2 Index.md Template Per Layer

Each spec subdirectory needs an `index.md` as the entry point. Recommended structure:

```markdown
# <Layer Name> — Specification Index

> **Status:** Canonical — these documents define the authoritative
> <contract-type> for <project-name>.

## What This Layer Covers

Brief description of what this layer defines and who it is "ground truth" for.

## Documents

| # | Document | Purpose |
|---|----------|---------|
| 1 | `spec_xxx.md` | One-line description |
| 2 | `spec_yyy.md` | One-line description |

## Relationship to Other Layers

ASCII diagram showing how this layer relates to other spec layers:
- This layer (WHAT) maps to which module(s) in modules/ (HOW)
- Which layers this layer is independent of

## Reading Order

Per-role/task recommended reading order:
- Newcomer → read what first
- Specific task → read what
```

### 4.3 The WHERE Principle

Each spec file answers exactly one question:

| File | Answers the Question |
|------|---------------------|
| `spec_<format>.md` | What does this data format look like? |
| `spec_conversion.md` | How is format A converted to format B? |
| `spec_<module>.md` | What is the interface and behavior of this module? |

If a spec file simultaneously answers "what the data looks like" and "how the code implements it", it should be split.

---

## 5. Dev Context Document Template

### 5.1 Overview

The dev context document (e.g., `CLAUDE.md`) is the middle layer of the SDD three-layer system — connecting specs (what is correct) to code (how it's implemented). Its audience is AI Agents and developers, and its goal is to **reduce onboarding cost and prevent mistakes**.

### 5.2 Recommended Structure

```markdown
# CLAUDE.md (or CONTRIBUTING.md / DEVELOPER_GUIDE.md)

## 1. Project Overview
One sentence: what the project is, what problem it solves.

## 2. Specifications
Point to specs/ directory, declare specs' authority, explain specs vs this doc.

## 3. Architecture
- Module dependency diagram (ASCII art)
- Architecture hard constraints table (# | Constraint | Violation Consequence)
- Data flow pipeline (Input → Processing → Output)

## 4. Critical Implementation Details
Centralize "the most error-prone areas":
- Coordinate systems / encoding rules / state management / concurrency models
- Each detail with code snippet + ❌/✅ contrast

## 5. Development Commands
- Installation
- Testing (including single-file/single-case examples)
- Lint/Format
- Manual verification commands

## 6. Known Gotchas
Common trap checklist, each with one-line title + one-line explanation.
Sorted by impact frequency (most-stepped-on first).

## 7. Test Structure
Test directory layout so developers know where tests go and where test data goes.
```

### 5.3 Writing Principles

| Principle | Description |
|-----------|-------------|
| **Timeliness** | Sync with code evolution; not a write-once-and-forget document |
| **Concreteness** | Write "RLE must use latin1 encoding", not "pay attention to encoding" |
| **Verifiability** | Each Critical Detail should map to a test or use case |
| **Layer clarity** | This document describes "how to do it", not repeating specs' "what it is" |
| **Length control** | Target 200-400 lines. If exceeding 400 lines, consider whether content should sink to specs or rise to a general document |

---

## 6. Known Gotchas Writing Guide

### 6.1 Why

Known Gotchas are the highest-ROI section in a dev context document — one line of warning can save hours of debugging. They record traps that **theoretically shouldn't happen but repeatedly trip people up**.

### 6.2 Entry Format

```
N. **Keyword**: symptom + correct approach. Specific details.

Anti-pattern:
- "Pay attention to encoding" → useless, don't know what to watch for

Good pattern:
- "RLE encoding: Always latin1, never utf-8, for byte↔string round-trips."
  → One sentence covers: scenario, wrong way, right way, reason
```

### 6.3 Entry Sources

| Source | Trigger |
|--------|---------|
| Bug fixes | After each bug fix, check: could this have been prevented? Yes → add Gotcha |
| Newcomer onboarding | Every obstacle a newcomer hits is a potential Gotcha |
| Code review | Issues repeatedly flagged in review |
| Architecture decisions | Constraints with severe violation consequences |

### 6.4 Maintenance

- Sort by **frequency**, most common at the top
- Each Gotcha should cross-reference related specs / Critical Details
- If a Gotcha's corresponding bug has been eliminated by an architecture refactor, delete it (prevent rot)
- No hard limit on Gotcha count, but if exceeding 20, consider whether some should be upgraded to Critical Details

---

## 7. Document Language Strategy

### 7.1 Core Principle

SDD documents use a mixed-language strategy: **terms must match code, explanations can be in the developer's native language.** This applies to all non-English-native developers — Chinese, Korean, Japanese, French... same logic.

| Content Type | Language | Reason |
|-------------|----------|--------|
| Terms, field names, API signatures | English | Must match symbols in code precisely, otherwise not grepable |
| Formulas, algorithm descriptions | English | Math/code symbols are not translatable |
| Code references (class names, function names, paths) | English | `BoundingBox` not `边界框`, otherwise unsearchable |
| Explanations, background, "why" | Free | Prioritize comprehension; use the most familiar language for the most complex logic |
| Notes, warnings, trap alerts | Free | Clarity first; cost of errors far outweighs language uniformity |
| Pure architecture descriptions | Free | modules/ can be all English (structure is clear, no complex domain knowledge required) |

### 7.2 Anti-Pattern vs Good Pattern

```
❌ All-native-language terms:
   "边界框使用左上角原点，坐标归一化到零一之间。"
   → BoundingBox or Bbox? What field represents the origin? What can I grep to find this code?

✅ Terms in English, explanations in native language:
   "BoundingBox uses (x_tl, y_tl) as top-left origin, coordinate space normalized [0, 1]."
   → Every term is directly grepable, explanation is clear and unambiguous.
```

---

## 8. Adapting to a New Project

When applying this methodology to a new project, follow this initialization order:

```
1. Create specs/ directory structure (WHAT layers + HOW layer)
   └── Reference §4: Specs Directory Structure

2. Write index.md for each layer
   └── Reference §4.2: Index.md Template

3. Write the first spec (usually the core data format or API contract)
   └── Reference §4.3: WHERE Principle

4. Create dev context document (CLAUDE.md)
   └── Reference §5: Dev Context Document Template

5. Determine document language strategy
   └── Reference §7: Document Language Strategy

6. As development progresses, continuously add:
   ├── Critical Implementation Details
   ├── Known Gotchas (reference §6)
   └── Project-specific development scenarios and checklists
```

It is recommended to create a project-specific development guide (e.g., `SDD_GUIDE.md`) in the `specs/` directory, referencing this methodology and supplementing with project-specific content:

```markdown
# SDD Development Guide

> Universal methodology: see SDD_METHODOLOGY.md. This document focuses on <project-name>-specific content.

## 1. Project Architecture
- Module system (concrete module names + dependency diagram)
- Architecture hard constraints
- External interface/format inventory

## 2. Development Workflow (Project-Specific Supplements)
- Step 1 template (concrete module names / format names)
- Spec mapping table (change type → corresponding spec)
- Project-specific implementation details (coordinate systems, encoding rules, etc.)

## 3. Specs Navigation Map
- "Where do I find what?" → spec quick index
- Complete file inventory

## 4. Common Development Scenarios

## 5. Project-Specific Code Review Checklist

## 6. References
```
