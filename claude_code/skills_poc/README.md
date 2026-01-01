# Skills

## What are Skills?

**Skills** are reusable, model-invoked capabilities that teach Claude how to do something specific. They're markdown files that extend Claude's knowledge without requiring explicit invocation. When you ask Claude something that matches a Skill's purpose, Claude automatically applies it.

Think of Skills as specialized instructions that Claude learns about at startup:
- Review pull requests using your team's specific standards
- Generate commit messages in your preferred format
- Explain code with visual diagrams

**Key difference from slash commands**: Claude chooses when to use Skills automatically based on your request, while slash commands require you to type `/command` explicitly.

## Setup

### File Structure

Skills are organized by location:

| Location   | Path                    | Applies to                        |
|:-----------|:------------------------|:----------------------------------|
| Personal   | `~/.claude/skills/`     | You, across all projects          |
| Project    | `.claude/skills/`       | Anyone working in this repository |
| Plugin     | Bundled with plugins    | Anyone with the plugin installed  |
| Enterprise | Organization-wide       | All users in your organization    |

### Basic Steps

1. **Create a directory** for your Skill (e.g., `~/.claude/skills/my-skill/`)
2. **Write `SKILL.md`** with YAML frontmatter and Markdown instructions
3. **Restart Claude Code** to load the new Skill
4. **Test it** by asking something that matches the description

## Quick Demo: Code Review Helper

**Directory structure:**
```
~/.claude/skills/code-review-helper/
└── SKILL.md
```

**`SKILL.md` content:**
```yaml
---
name: code-review-helper
description: Provides structured code review feedback with specific sections for improvements, strengths, and suggestions. Use when reviewing code, pull requests, or asking for feedback on code quality.
---

# Code Review Helper

## Instructions

When reviewing code, always follow this format:

1. **Strengths**: What's working well in this code? Highlight good patterns or decisions.
2. **Opportunities for improvement**: Point out specific issues (readability, performance, security).
3. **Suggested changes**: Provide concrete code examples or specific recommendations.
4. **Questions**: Ask clarifying questions about design decisions if needed.

## Best practices

- Be constructive and specific
- Reference line numbers or code sections
- Provide examples of better patterns
- Consider the context and constraints
```

### Create the Skill:

```bash
# Create directory
mkdir -p ~/.claude/skills/code-review-helper

# Create SKILL.md
cat > ~/.claude/skills/code-review-helper/SKILL.md << 'EOF'
---
name: code-review-helper
description: Provides structured code review feedback with specific sections for improvements, strengths, and suggestions. Use when reviewing code, pull requests, or asking for feedback on code quality.
---

# Code Review Helper

## Instructions

When reviewing code, always follow this format:

1. **Strengths**: What's working well in this code? Highlight good patterns or decisions.
2. **Opportunities for improvement**: Point out specific issues (readability, performance, security).
3. **Suggested changes**: Provide concrete code examples or specific recommendations.
4. **Questions**: Ask clarifying questions about design decisions if needed.

## Best practices

- Be constructive and specific
- Reference line numbers or code sections
- Provide examples of better patterns
- Consider the context and constraints
EOF

# Restart Claude Code to load the new Skill
```

### Test it:

Ask Claude: "Review this function for quality" or "Give me feedback on my code"

Claude will detect that your request matches the Skill's description, ask permission to use it, then follow the instructions.

## How Skills Work

1. **Discovery**: At startup, Claude loads all available Skill names and descriptions (not full content)
2. **Activation**: When your request matches a Skill's description, Claude asks permission to use it
3. **Execution**: After confirmation, Claude loads the full `SKILL.md` and follows its instructions

### Example Requests

With the code-review-helper Skill:

```
"Review this function for quality"      → Triggers the Skill
"How does this code work?"              → Won't trigger (description says "review")
"Give me feedback on my code changes"   → Triggers the Skill
```

## Best Practices

1. **Write clear descriptions**: Include specific keywords users would mention
2. **Keep SKILL.md focused**: Aim for under 500 lines
3. **Use progressive disclosure**: Link to detailed reference files instead of including everything in SKILL.md
4. **Restrict tools if needed**: Use `allowed-tools` in frontmatter to limit tool access:
   ```yaml
   allowed-tools: Read, Grep, Glob
   ```

## Advanced Features

### Multi-file Skills

```
~/.claude/skills/my-skill/
├── SKILL.md          # Main instructions
├── examples.md       # Reference examples
└── config.json       # Configuration data
```

Reference other files in SKILL.md:
```markdown
See examples.md for code samples
```

### Tool Restrictions

Limit which tools a Skill can use:

```yaml
---
name: read-only-skill
description: A skill that can only read files
allowed-tools: Read, Grep, Glob
---
```

## Skills vs Slash Commands

| Feature          | Skills                           | Slash Commands              |
|:-----------------|:---------------------------------|:----------------------------|
| Invocation       | Automatic (Claude decides)       | Manual (you type `/cmd`)    |
| Discovery        | Based on description match       | Explicit command name       |
| Use case         | Context-aware automation         | Explicit user-triggered     |
| Permission       | Asked first time per session     | Implicit (you invoked it)   |

## Cmd

View available Skills:
```
What Skills are available?
```

## Ref

- https://youtu.be/IoqpBKrNaZI?si=oWKM0tu6IeGpZDQm
- https://www.aiposthub.com/vibe-coding-claude-skills-guide/
- https://code.claude.com/docs/en/skills.md
