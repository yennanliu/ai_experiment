# Claude Code Basic Commands

A quick reference guide for essential Claude Code commands.

## Initialization

### Init Project
```bash
/init
```
Generate a new CLAUDE.md based on project scan.

## Session Management

### Resume Previous Sessions
```bash
claude --resume
```
Access and resume previous conversation sessions.

### Compact Session
```bash
/compact
```
Summary the current session as a single message.

### Clear Previous Messages
```bash
/clear
```
Remove previous message history.

## Interaction Controls

### Stop/Interrupt
```
esc
```
Interrupt current action.

### Revert to Previous Message
```
esc + esc
```
Go back to previous message.

### Planning Mode
```
shift + tab shift + tab
```
Enable planning mode for complex tasks.

## File & Context Management

### Mention a File
```
@ <file_name>
```
Reference a specific file in conversation.

### Paste Image
```
ctrl + v
```
Paste copied image into Claude Code CLI.

### Add Memory
```
# <things to ask claude code to remember>
```
Save specific context or instructions for the session.

## Custom Commands

### Create Custom Commands
Location: `.claude/commands`

Example:
1. Create a file: `.claude/commands/audit.md`
2. Run with: `/audit`

Custom commands allow you to define reusable prompts and workflows.

## Advanced Options

### Skip Minor Confirmations
```bash
claude --dangerously-skip-permissions
```
Bypass minor permission checks (use with caution).

## Git Worktree Integration

Create isolated code environments for parallel work:

```bash
# Create worktree directory
mkdir .trees

# Add worktrees for different features
git worktree add .trees/ui_feature
git worktree add .trees/testing_feature
```

This allows working on multiple features simultaneously in separate directories.

## Tips

- Use `/init` at the start of a new project to help Claude understand your codebase
- Create custom commands in `.claude/commands/` for frequently used workflows
- Use `@` to reference files for better context
- Use `/compact` to summarize long conversations
- Use planning mode (shift + tab twice) for complex multi-step tasks
