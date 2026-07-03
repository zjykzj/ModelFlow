---
name: commit
description: Create git commits following project conventions
allowed-tools: Bash
---

# Git Commit Skill

Use this skill whenever committing code to the repository.

## Commit Message Format

All commits must use the following heredoc format:

```bash
git commit -m "$(cat <<'EOF'
<type>(<scope>): <subject>

<body if needed>

Co-Authored-By: {{AI_MODEL_NAME}} <{{AI_MODEL_EMAIL}}>
EOF
)"
```

The `Co-Authored-By` line is **mandatory** for all commits. `{{AI_MODEL_NAME}}` and `{{AI_MODEL_EMAIL}}` are configured per-project in CLAUDE.md.

## Conventional Commit Types

| Type | Description |
|------|------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `build` | Build system or external dependencies |
| `test` | Adding missing tests or correcting existing tests |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `style` | Changes that do not affect the meaning of the code (white-space, formatting, etc.) |
| `perf` | Code change that improves performance |
| `ci` | Changes to CI configuration files and scripts |
| `chore` | Other changes that don't modify src or test files |

## Procedure

1. Review the diff to determine the appropriate type and scope
2. Write a concise subject line (imperative mood, ≤50 chars)
3. Add body if the change needs explanation
4. Append the `Co-Authored-By` line
5. Execute the commit command
