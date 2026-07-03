---
name: spec
description: Create or modify spec files following project methodology. Use when writing, editing, or reviewing specs/ files.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Spec Maintenance

Apply this methodology when creating or modifying spec files.

## Specs Serve Two Readers

| Reader | Needs from specs |
|--------|-----------------|
| **Agent** (Claude Code) | Behavioral contracts — "what is correct" to verify compliance |
| **Human developer** | Understanding — "why" a contract exists and "what are the boundaries" |

Both readers matter. Content that explains a contract (not just defines it) should be kept.

## Classification Principle

For each piece of content, ask: **"When would I read this while writing code?"**

| Answer | Layer | Typical Content |
|--------|-------|-----------------|
| "Every change" | **CLAUDE.md** | Architecture hard constraints, global conventions (ordering/naming), high-frequency gotchas (encoding rules, state cleanup patterns), critical implementation details that affect every edit |
| "Specific task" | **Specs — WHAT layer** | External contract definitions — what the outside world expects from this system. Data formats, API/protocol specs, metric definitions, interface contracts with external systems. Includes explanations that clarify contract semantics (e.g., "why this field uses center coordinates, not top-left"). |
| "Specific task" | **Specs — HOW layer** | Internal module contracts — public API signatures, design constraints, option/parameter definitions, dependency rules, exception types and exit codes. How modules relate to each other, not how they are implemented internally. |
| "Neither" | Delete | — |

WHAT layer naming depends on project domain (see §Spec Directory Structure for the project-type mapping). HOW has exactly one `modules/` layer.

## What Does NOT Belong in Specs

These are not "delete on sight" rules — each requires case-by-case judgment. The key question is always: **does this define or help understand a behavioral contract?**

### 1. Change History / Version Changelogs → Delete

Version history belongs in `git log` / `CHANGELOG.md`. Specs define current contracts, not how they evolved. Example: "Key change from v1: Internal Model removed" → delete; the spec should just describe current behavior.

### 2. Implementation Pseudocode / Code Examples → Judge

**Keep** if the code **defines or clarifies a behavioral contract** — it specifies what the code must do more precisely than prose could. Examples that ARE contracts:

- Greedy matching algorithm pseudocode — without it, the matching contract is ambiguous
- Coordinate transformation formulas (`cx_abs = cx * image_width`) — these ARE the mathematical contract
- A `try/finally` code block showing which variable must be cleaned up and how — prose "must clean up" is ambiguous; the code says exactly what
- Constructor calls showing parameter values like `strict_mode=False` — these define the behavioral contract ("visualizers never reject data")

**Delete** only if the code is **truly redundant** — the prose already defines the same requirement with equal precision, and the code adds nothing. Example: `data = json.load(open(path))` when the prose says "read the file as JSON".

**Key distinction:** Is the code the *clearest*, *most precise* expression of the requirement? If yes → keep. The fact that something is "implementation-like" (variable names, function calls) does NOT make it a violation — those can be the contract when they specify required behavior.

**Tiebreaker:** "If I replaced this code with prose, would the behavioral requirement become less precise or ambiguous?" If yes → keep.

### 3. CLI Command Signatures → Keep (with nuance)

**Keep** command signatures — they ARE the module's public API contract. Example:
```
dataflow-cv convert yolo2coco [OPTIONS] IMAGE_DIR LABEL_DIR CLASS_FILE OUTPUT_FILE
```
This defines: subcommand name, positional argument order, argument count. Code must implement this exact signature.

**Don't copy** full `--help` output verbatim — the executable is the authority for option descriptions.

### 4. Migration Guides / Legacy API Tables → Delete

One-time transition docs. Ship with the release, delete after migration is complete.

### 5. Directory Tree File Listings → Judge by Level

**Keep** module-level file listings — they ARE architecture documentation. Example:
```
dataflow/convert/
├── base.py                # BaseConverter + shared pipelines
├── yolo_and_coco.py       # YOLO ↔ COCO converter
└── utils.py               # Shared coordinate transforms
```
This tells the reader: what files exist in this module, what each is responsible for. Essential for both agent and human to navigate the codebase.

**Simplify** project-level recursive trees that just mirror `ls` output. Example: a full `specs/` tree in an index.md → redundant; the Documents table already serves as the index.

### 6. Tutorials / How-To Guides → Keep, Label as Usage Guide

**Keep** if it helps readers understand or apply the contract. Example: "Metric Selection Guide" telling users which metric to pick for small-object detection — it's not a contract, but it's context the reader needs when using evaluate specs.

**Label** these sections clearly (e.g., "## Usage Guide") so readers know they're guidance, not behavioral requirements.

**Delete** only truly standalone workflow recipes that don't reference any contract (e.g., "How to set up your first ML project").

**Comparison tables** are a special case of this rule: keep if the table defines differentiated behavioral requirements between entities. If it just helps the user choose between options (e.g., "Metric Selection Guide"), keep it but label as `## Usage Guide`. Tiebreaker: "does this table help the reader apply a contract defined in this file?"

## Pre-Deletion Checks

Before deleting any content, apply two checks:

**1. Contract check:** "Does this content help explain a behavioral contract — even if it reads like education or FAQ?" If yes, keep it. Contract-defining clarifications (e.g., "coordinates must be in [0, 1]", "TN is not applicable because there is no negative class") define the contract by explaining its boundaries.

**2. Framing check:** Before deleting a section that seems "about the old architecture", check whether the *substance* is still correct but the *framing* is stale. Example: coordinate transformation formulas labeled "for Internal Model" are still mathematically correct — fix by renaming the section and adding a note about current architecture, not by deleting the formulas.

## Extra Check for `modules/` Specs — Interface Contract or Implementation Description?

The classification table below is a **starting point**, not a hard rule. Apply the "behavioral contract" test before deleting:

| Interface contract → keep | Implementation description → likely delete |
|---------------------------|-------------------------------------|
| Public API signatures, return types, command signatures | Function-internal variable assignments |
| Design constraints and rules (e.g., "must not import X from Y") | Step-by-step internal plumbing (e.g., "then call `ctx.obj['verbose']`") |
| Option/parameter definition tables | Internal helper function docstrings that mirror code comments |
| Exception types and exit codes | Directory tree of nested subdirectories |

**Nuance for common borderline cases:**

- **Step-by-step pipeline flow**: Keep if it defines the **contractual sequence** of steps (what must happen in what order). Delete if it describes **how** each step is implemented internally.
- **Code snippets showing constructor calls**: Judge — a constructor call with specific parameter values can define behavioral requirements (e.g., `strict_mode=False` means "visualizers never reject data"; `logger=self.logger` means "visualizer passes its logger to the handler"). Delete only if the call merely repeats the signature without adding contract-relevant information.
- **Internal utility function tables**: Delete if they duplicate what's in the module file listing. Keep if they define behavioral differences between utilities that the file listing alone doesn't convey.

## Spec Directory Structure

### WHAT vs HOW Separation

```
specs/
├── <contract-layer-1>/    # WHAT — external data/interface/protocol contracts
│   ├── index.md
│   └── spec_<topic>.md
│
├── <contract-layer-2>/    # WHAT — other contract layers (optional)
│   ├── index.md
│   └── spec_<topic>.md
│
└── modules/               # HOW — internal module architecture
    ├── index.md           # Architecture diagram + hard constraints (single source of truth for module dependencies)
    ├── spec_<module-1>.md
    ├── spec_<module-2>.md
    └── ...
```

WHAT layer naming depends on project domain:

| Project Type | Suggested Name | Example Content |
|-------------|---------------|-----------------|
| Data processing / format conversion | `formats/` | Data format definitions, conversion rules |
| Web API | `api/` | REST/GraphQL interface contracts |
| Library / SDK | `interfaces/` | Public API signatures, type definitions |
| Evaluation / benchmarking | `evaluate/` | Metric definitions, baselines |
| Protocol / communication | `protocols/` | Message formats, state machines |

WHAT layers may have multiple; HOW has exactly one `modules/`, mirroring code modules.

### Index Template

When creating a new layer index, use this template:

```markdown
# <Layer Name> — Specification Index

> **Status:** Canonical — these documents define the authoritative
> <contract-type> for <project-name>.

## What This Layer Covers

Briefly describe what this layer defines and what it is the ground truth for.

## Documents

| # | Document | Purpose |
|---|----------|---------|
| 1 | `spec_xxx.md` | One-line description |

## Relationship to Other Layers

- This layer (WHAT) maps to which modules in `modules/` (HOW)
- This layer is independent of which other layers

## Reading Order

Recommended reading order by task:
- Newcomer → what to read first
- Specific task → what to read
```

### What Each Spec Should Answer

| Spec Type | Questions It Answers |
|-----------|---------------------|
| Data format / protocol spec | What does this external contract look like? What required fields are defined? |
| Conversion / adapter spec | How does A become B? How are edge cases handled? |
| Module spec | What are this module's public interfaces, design constraints, and behavioral contracts? |

A spec file should not answer both "what does the data look like" and "how does the code implement it" simultaneously. If both appear, split them.

### Version Management

Each spec file starts with a version and last-updated date:

```markdown
> **Version:** vX.Y | **Last Updated:** YYYY-MM-DD
```

| Scenario | Version Change |
|----------|---------------|
| New definitions / extending existing contracts | Minor increment (v1.0 → v1.1) |
| Behavioral change (breaking change) | Major increment (v1.2 → v2.0) |
| Clarification / wording fix (no behavior change) | Update date only, keep version |
