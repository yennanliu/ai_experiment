# Claude Code Advanced Commands & Features

A comprehensive guide for advanced Claude Code usage including MCP, agents, hooks, and best practices.

## Table of Contents
- [MCP (Model Context Protocol)](#mcp-model-context-protocol)
- [Subagents](#subagents)
- [Hooks](#hooks)
- [Best Practices](#best-practices)
- [Advanced Workflows](#advanced-workflows)

---

## MCP (Model Context Protocol)

MCP enables Claude Code to connect to external tools and services through standardized servers.

### What is MCP?

MCP allows Claude Code to:
- Access external data sources (databases, APIs, file systems)
- Integrate with third-party tools
- Extend functionality beyond built-in capabilities
- Use community-created servers

### Installing MCP Servers

MCP servers are configured in your settings file:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "<your-token>"
      }
    }
  }
}
```

### Popular MCP Servers

- **Filesystem**: Access local files and directories
- **GitHub**: Interact with GitHub repositories
- **PostgreSQL**: Query databases
- **Slack**: Send messages and read channels
- **Google Drive**: Access and manage documents

### Using MCP Tools

Once configured, MCP tools appear automatically:
```
Use the mcp__github__create-issue tool to create a new issue
```

---

## Subagents

Subagents are specialized AI assistants with focused expertise and separate context windows.

### Creating Subagents

#### Interactive Creation
```bash
/agents
```
Follow the prompts to create a new subagent.

#### Manual Creation

Create a file in `.claude/agents/code-reviewer.md`:

```markdown
---
name: code-reviewer
description: Expert code review specialist focusing on best practices
tools: Read, Grep, Glob, Bash
model: inherit
---

You are an expert code reviewer. Focus on:
- Code quality and maintainability
- Security vulnerabilities
- Performance issues
- Best practices adherence
- Test coverage

Provide actionable feedback with specific examples.
```

### Subagent Configuration Options

```yaml
---
name: string               # Unique identifier
description: string        # What the agent does
tools: comma,separated     # Allowed tools (or "all")
model: inherit|specific    # Model to use
---
System prompt goes here
```

### Common Subagent Types

**Code Reviewer**
```yaml
---
name: code-reviewer
description: Reviews code for quality and best practices
tools: Read, Grep, Glob
---
```

**Test Generator**
```yaml
---
name: test-generator
description: Generates comprehensive test cases
tools: Read, Write, Bash, Grep
---
```

**Debugger**
```yaml
---
name: debugger
description: Analyzes and fixes bugs systematically
tools: Read, Edit, Bash, Grep, Glob
---
```

**Documentation Writer**
```yaml
---
name: doc-writer
description: Creates clear technical documentation
tools: Read, Write, Grep
---
```

### Using Subagents

**Automatic Delegation**: Claude delegates to subagents automatically based on task.

**Explicit Invocation**:
```
Use the code-reviewer subagent to review app.js
```

### Subagent Best Practices

1. **Single Purpose**: Each subagent should have one clear responsibility
2. **Detailed Prompts**: Write comprehensive system prompts
3. **Limit Tools**: Only grant necessary tool access
4. **Version Control**: Store project subagents in `.claude/agents/`
5. **Test Iterations**: Refine prompts based on performance

---

## Hooks

Hooks allow you to intercept and modify Claude Code's behavior with custom scripts.

### Hook Types

1. **PreToolUse**: Before any tool execution
2. **PostToolUse**: After tool completion
3. **UserPromptSubmit**: When user submits a prompt
4. **Notification**: For system notifications
5. **Stop/SubagentStop**: When Claude finishes responding
6. **SessionStart/SessionEnd**: Session lifecycle events

### Configuration

Edit `~/.claude/settings.json` or `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/safety-check.sh"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/logger.sh"
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/prompt-enhancer.sh"
          }
        ]
      }
    ]
  }
}
```

### Hook Examples

#### Safety Check Hook
```bash
#!/bin/bash
# safety-check.sh - Prevent destructive commands

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.input.command // empty')

# Block dangerous commands
if echo "$COMMAND" | grep -qE "(rm -rf|mkfs|dd if=)"; then
    echo '{"block": true, "message": "Dangerous command blocked"}'
    exit 0
fi

# Allow command
exit 0
```

#### Logging Hook
```bash
#!/bin/bash
# logger.sh - Log all tool usage

INPUT=$(cat)
echo "$INPUT" >> ~/.claude/tool-log.json
exit 0
```

#### Git Auto-Stage Hook
```bash
#!/bin/bash
# auto-stage.sh - Stage modified files before commits

INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool // empty')

if [ "$TOOL" = "Bash" ]; then
    COMMAND=$(echo "$INPUT" | jq -r '.input.command // empty')
    if echo "$COMMAND" | grep -q "git commit"; then
        git add -u
    fi
fi

exit 0
```

### Hook Security Warning

⚠️ **USE AT YOUR OWN RISK**: Hooks execute arbitrary shell commands with your permissions. Always:
- Validate all inputs
- Use absolute paths
- Quote shell variables properly
- Avoid accessing sensitive files
- Test hooks thoroughly

---

## Best Practices

### Project Setup

1. **Initialize New Projects**
```bash
/init
```
Generates CLAUDE.md with project context.

2. **Create Custom Commands**
Store frequently used workflows in `.claude/commands/`:
```bash
.claude/commands/deploy.md
.claude/commands/test-all.md
.claude/commands/review-pr.md
```

3. **Define Subagents Early**
Create specialized subagents for your project's needs.

### Effective Communication

1. **Reference Files**: Use `@filename` for context
2. **Be Specific**: Provide clear, actionable requests
3. **Use Planning Mode**: `shift + tab + tab` for complex tasks
4. **Add Memory**: Use `#` to save important context

### Code Quality

1. **Incremental Changes**: Make small, testable changes
2. **Review Before Commit**: Always review diffs
3. **Test Coverage**: Request tests for new features
4. **Documentation**: Keep docs updated alongside code

### Performance Optimization

1. **Compact Long Sessions**: Use `/compact` to summarize
2. **Clear When Needed**: Use `/clear` to reset context
3. **Focused Subagents**: Create task-specific agents
4. **Selective File Access**: Only reference needed files

### Security

1. **Review Permissions**: Check tool access before granting
2. **Validate Hooks**: Test hooks in safe environments
3. **Protect Secrets**: Never commit sensitive data
4. **Audit Commands**: Review generated commands before execution

---

## Advanced Workflows

### Multi-Feature Development with Git Worktrees

```bash
# Setup
mkdir .trees
git worktree add .trees/feature-a
git worktree add .trees/feature-b

# Work in parallel
cd .trees/feature-a
claude  # Work on feature A

cd .trees/feature-b
claude  # Work on feature B simultaneously
```

### Automated Code Review Pipeline

1. Create `.claude/agents/reviewer.md`
2. Add pre-commit hook:
```bash
#!/bin/bash
# .git/hooks/pre-commit
claude --dangerously-skip-permissions << EOF
Use the reviewer subagent to review staged changes
EOF
```

### Custom Testing Workflow

Create `.claude/commands/test-and-fix.md`:
```markdown
---
description: Run tests and automatically fix failures
---

Run the test suite. For each failure:
1. Analyze the error
2. Identify the root cause
3. Propose a fix
4. Apply the fix
5. Re-run the test
6. Continue until all tests pass
```

Usage: `/test-and-fix`

### Documentation Automation

Create `.claude/agents/doc-generator.md`:
```markdown
---
name: doc-generator
description: Generates API documentation from code
tools: Read, Write, Grep, Glob
---

Generate comprehensive documentation by:
1. Scanning all source files
2. Extracting public APIs
3. Creating markdown docs
4. Adding usage examples
5. Updating README
```

### Debugging Session

```bash
# Start with context
claude --resume

# In Claude Code:
Use the debugger subagent to investigate the authentication bug
@ src/auth/login.js
@ tests/auth.test.js
```

---

## Configuration Files Reference

### Settings Hierarchy

1. **Global**: `~/.claude/settings.json` (user-wide)
2. **Project**: `.claude/settings.json` (version controlled)
3. **Local**: `.claude/settings.local.json` (gitignored)

### Complete Settings Example

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "/usr/local/bin/safety-check.sh"
          }
        ]
      }
    ]
  },
  "dangerouslySkipPermissions": false,
  "model": "claude-sonnet-4-5"
}
```

---

## Tips & Tricks

1. **Rapid Prototyping**: Use planning mode + subagents for complex features
2. **Code Migration**: Create specialized subagents for language conversion
3. **Batch Operations**: Use custom commands for repetitive tasks
4. **Integration Testing**: Combine MCP servers for end-to-end workflows
5. **Knowledge Base**: Use memory (`#`) to build session-specific context
6. **Parallel Work**: Git worktrees + multiple Claude sessions
7. **Resume Sessions**: `claude --resume` to continue previous work
8. **Skip Confirmations**: `claude --dangerously-skip-permissions` for trusted environments

---

## Troubleshooting

### Subagent Not Found
```bash
# List available subagents
ls -la .claude/agents/
ls -la ~/.claude/agents/

# Verify YAML frontmatter is correct
```

### MCP Server Connection Failed
```bash
# Test server manually
npx -y @modelcontextprotocol/server-filesystem /tmp

# Check logs
cat ~/.claude/logs/mcp-*.log
```

### Hook Not Executing
```bash
# Verify hook is executable
chmod +x /path/to/hook.sh

# Test hook manually
echo '{}' | /path/to/hook.sh

# Check hook output
cat ~/.claude/logs/hooks.log
```

---

## Resources

- **Official Docs**: https://docs.claude.com/claude-code
- **MCP Servers**: https://github.com/modelcontextprotocol/servers
- **Community Examples**: https://github.com/anthropics/claude-code/examples
- **Issue Tracker**: https://github.com/anthropics/claude-code/issues
