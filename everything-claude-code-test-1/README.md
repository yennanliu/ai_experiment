# everything-claude-code-test-1

- https://github.com/affaan-m/everything-claude-code


## Install

```bash
# Add this repo as a marketplace
/plugin marketplace add affaan-m/everything-claude-code

# Install the plugin
/plugin install everything-claude-code@everything-claude-code
```

---

## Slash cmd 

Inside Claude Code, run commands to trigger workflows:

| Command           | What it does                         |               |
| ----------------- | ------------------------------------ | ------------- |
| `/plan`           | Plan implementation steps            |               |
| `/tdd`            | Run test-driven development workflow |               |
| `/e2e`            | Generate end-to-end tests            |               |
| `/code-review`    | Review code for bugs/quality         |               |
| `/build-fix`      | Fix build errors                     |               |
| `/refactor-clean` | Clean up dead code                   |               |
| `/learn`          | Extract patterns mid-session         |               |
| `/verify`         | Run verification loops               | ([GitHub][2]) |

---

## How agents get used (3 ways)

```bash

# V1
Use the planner agent to break this feature into steps.

# V2
Act as @planner and produce an implementation plan for:
- Feature: OAuth login
- Stack: Next.js + Supabase
```
