# Agent Team

This project uses a 3-agent development team. Use the agents in order for any non-trivial task:

## Team

| Agent | Trigger | Role |
|-------|---------|------|
| `@planner` | New task / feature request | Reads the codebase and produces a numbered implementation plan |
| `@coder` | After plan is confirmed | Implements the plan step by step, runs tests |
| `@reviewer` | After coder finishes | Reviews changes for correctness, quality, and security |

## Workflow

```
User task → @planner → (user confirms plan) → @coder → @reviewer → done
```

## When to skip agents

- Simple one-liner fixes: skip @planner, go straight to @coder + @reviewer.
- Read-only questions: answer directly without delegating.

## General rules

- Never commit secrets or credentials.
- Keep changes minimal — solve the stated task, nothing more.
- If the plan and the implementation diverge, the reviewer will catch it.
