# Claude Code Custom Commands Guide

A comprehensive guide to creating and using custom commands in Claude Code, from basic to advanced examples.

## Table of Contents
- [What are Custom Commands?](#what-are-custom-commands)
- [Setup and Structure](#setup-and-structure)
- [Basic Examples](#basic-examples)
- [Intermediate Examples](#intermediate-examples)
- [Advanced Examples](#advanced-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## What are Custom Commands?

Custom commands (slash commands) allow you to create reusable prompts and workflows that can be invoked with a simple `/command-name` syntax.

### Benefits
- **Reusability**: Define once, use many times
- **Consistency**: Same prompt every time
- **Efficiency**: Faster than typing full prompts
- **Shareability**: Project commands can be version controlled
- **Flexibility**: Support arguments and file references

### Types of Custom Commands

1. **Project Commands**: `.claude/commands/` (version controlled, shared with team)
2. **Personal Commands**: `~/.claude/commands/` (local to your machine)

---

## Setup and Structure

### Directory Setup

```bash
# Project-level commands (shared with team)
mkdir -p .claude/commands

# Personal commands (local only)
mkdir -p ~/.claude/commands
```

### Command File Format

Commands are Markdown files with optional YAML frontmatter:

```markdown
---
description: Brief description of what this command does
tools: Read, Write, Bash  # Optional: Restrict available tools
---

Your prompt goes here.
You can use $ARGUMENTS or $1, $2, etc. for parameters.
Reference files with @filename
```

### Using Commands

```bash
# Basic usage
/command-name

# With arguments
/command-name arg1 arg2

# With file references
/command-name @src/app.js
```

---

## Basic Examples

### Example 1: Simple Code Review

**File**: `.claude/commands/review.md`

```markdown
---
description: Review code for bugs and improvements
---

Please review the code carefully and provide feedback on:
1. Potential bugs or errors
2. Code quality and readability
3. Performance concerns
4. Security issues
5. Suggestions for improvement
```

**Usage**:
```bash
/review @src/utils.js
```

**Step-by-step Setup**:
```bash
# 1. Create the commands directory
mkdir -p .claude/commands

# 2. Create the review command file
cat > .claude/commands/review.md << 'EOF'
---
description: Review code for bugs and improvements
---

Please review the code carefully and provide feedback on:
1. Potential bugs or errors
2. Code quality and readability
3. Performance concerns
4. Security issues
5. Suggestions for improvement
EOF

# 3. Use the command in Claude Code
# Type: /review @src/utils.js
```

---

### Example 2: Add Documentation

**File**: `.claude/commands/docs.md`

```markdown
---
description: Add JSDoc comments to functions
---

Add comprehensive JSDoc documentation to all functions in the specified file.

Include:
- Function description
- @param tags with types
- @returns tag with type
- @throws if applicable
- @example with usage
```

**Usage**:
```bash
/docs @src/api.js
```

**Step-by-step Setup**:
```bash
# 1. Create the command
cat > .claude/commands/docs.md << 'EOF'
---
description: Add JSDoc comments to functions
---

Add comprehensive JSDoc documentation to all functions in the specified file.

Include:
- Function description
- @param tags with types
- @returns tag with type
- @throws if applicable
- @example with usage
EOF

# 2. Use it
# In Claude Code: /docs @src/api.js
```

---

### Example 3: Explain Code

**File**: `.claude/commands/explain.md`

```markdown
---
description: Explain code in simple terms
---

Explain the following code in clear, simple language:
- What it does
- How it works
- Key concepts used
- Any potential gotchas

Use analogies where helpful. Assume the reader is familiar with basic programming but not necessarily with this specific codebase.
```

**Usage**:
```bash
/explain @src/algorithm.js
```

---

## Intermediate Examples

### Example 4: Test Generator with Arguments

**File**: `.claude/commands/test.md`

```markdown
---
description: Generate tests for a file or function
---

Generate comprehensive unit tests for $ARGUMENTS

Include:
1. Test setup and teardown
2. Happy path tests
3. Edge cases
4. Error handling tests
5. Mock external dependencies

Use the testing framework already in the project.
Follow existing test patterns and naming conventions.
```

**Usage**:
```bash
/test @src/auth.js
/test the login function in @src/auth.js
```

**Step-by-step Setup**:
```bash
# 1. Create the test command with argument support
cat > .claude/commands/test.md << 'EOF'
---
description: Generate tests for a file or function
---

Generate comprehensive unit tests for $ARGUMENTS

Include:
1. Test setup and teardown
2. Happy path tests
3. Edge cases
4. Error handling tests
5. Mock external dependencies

Use the testing framework already in the project.
Follow existing test patterns and naming conventions.
EOF

# 2. Use with various arguments
# /test @src/auth.js
# /test the login function in @src/auth.js
```

---

### Example 5: Refactor with Specific Patterns

**File**: `.claude/commands/refactor.md`

```markdown
---
description: Refactor code following best practices
tools: Read, Edit, Grep
---

Refactor $ARGUMENTS according to these principles:

1. **DRY**: Remove code duplication
2. **SOLID**: Apply SOLID principles where applicable
3. **Clean Code**: Improve naming and structure
4. **Performance**: Optimize where beneficial
5. **Testability**: Make code easier to test

Provide:
- List of changes made
- Explanation of why each change improves the code
- Any trade-offs considered
```

**Usage**:
```bash
/refactor @src/services/payment.js
```

---

### Example 6: Security Audit

**File**: `.claude/commands/security-audit.md`

```markdown
---
description: Perform security audit on code
---

Perform a security audit of $ARGUMENTS

Check for:
1. **Injection vulnerabilities**: SQL, NoSQL, Command, XSS
2. **Authentication issues**: Weak passwords, session management
3. **Authorization flaws**: Missing access controls
4. **Data exposure**: Sensitive data in logs/errors
5. **Cryptography**: Weak algorithms, hardcoded secrets
6. **Dependencies**: Known vulnerabilities

For each issue found:
- Severity level (Critical/High/Medium/Low)
- Detailed explanation
- Proof of concept if applicable
- Remediation steps
```

**Usage**:
```bash
/security-audit @src/api/routes.js
/security-audit the entire @src/api/ directory
```

**Step-by-step Setup**:
```bash
# 1. Create comprehensive security audit command
cat > .claude/commands/security-audit.md << 'EOF'
---
description: Perform security audit on code
---

Perform a security audit of $ARGUMENTS

Check for:
1. **Injection vulnerabilities**: SQL, NoSQL, Command, XSS
2. **Authentication issues**: Weak passwords, session management
3. **Authorization flaws**: Missing access controls
4. **Data exposure**: Sensitive data in logs/errors
5. **Cryptography**: Weak algorithms, hardcoded secrets
6. **Dependencies**: Known vulnerabilities

For each issue found:
- Severity level (Critical/High/Medium/Low)
- Detailed explanation
- Proof of concept if applicable
- Remediation steps
EOF

# 2. Run the audit
# /security-audit @src/api/routes.js
```

---

## Advanced Examples

### Example 7: Multi-Step Workflow with Conditional Logic

**File**: `.claude/commands/deploy-check.md`

```markdown
---
description: Complete pre-deployment checklist
tools: Bash, Read, Grep
---

Run a comprehensive pre-deployment check:

## 1. Code Quality
- Run linter and report any errors
- Check for console.log statements
- Verify no TODO comments in critical paths

## 2. Testing
- Run full test suite
- Ensure 100% of tests pass
- Check test coverage (minimum 80%)

## 3. Build
- Run production build
- Verify build succeeds with no errors
- Check bundle size (warn if > 500KB)

## 4. Security
- Scan for known vulnerabilities
- Check for exposed secrets/API keys
- Verify environment variables are properly configured

## 5. Git
- Ensure all changes are committed
- Check branch is up to date with main
- Verify no merge conflicts

## 6. Documentation
- Ensure CHANGELOG is updated
- Check README reflects latest changes

Provide:
- ✅ for passing checks
- ❌ for failing checks with details
- ⚠️ for warnings

Stop and list all issues if any critical checks fail.
```

**Usage**:
```bash
/deploy-check
```

**Step-by-step Setup**:
```bash
# 1. Create comprehensive deployment check
cat > .claude/commands/deploy-check.md << 'EOF'
---
description: Complete pre-deployment checklist
tools: Bash, Read, Grep
---

Run a comprehensive pre-deployment check:

## 1. Code Quality
- Run linter and report any errors
- Check for console.log statements
- Verify no TODO comments in critical paths

## 2. Testing
- Run full test suite
- Ensure 100% of tests pass
- Check test coverage (minimum 80%)

## 3. Build
- Run production build
- Verify build succeeds with no errors
- Check bundle size (warn if > 500KB)

## 4. Security
- Scan for known vulnerabilities
- Check for exposed secrets/API keys
- Verify environment variables are properly configured

## 5. Git
- Ensure all changes are committed
- Check branch is up to date with main
- Verify no merge conflicts

## 6. Documentation
- Ensure CHANGELOG is updated
- Check README reflects latest changes

Provide:
- ✅ for passing checks
- ❌ for failing checks with details
- ⚠️ for warnings

Stop and list all issues if any critical checks fail.
EOF

# 2. Run before deployment
# /deploy-check
```

---

### Example 8: Code Migration Between Languages

**File**: `.claude/commands/migrate.md`

```markdown
---
description: Migrate code from one language to another
tools: Read, Write, Grep, Glob
---

Migrate $ARGUMENTS

## Process:
1. **Analyze source code**: Understand logic and dependencies
2. **Map language features**: Identify equivalent constructs
3. **Generate target code**: Write idiomatic code in target language
4. **Preserve functionality**: Ensure same behavior
5. **Add tests**: Create equivalent test suite
6. **Document differences**: Note any behavioral changes

## Requirements:
- Use idiomatic patterns for target language
- Maintain same API/interface
- Preserve comments and documentation
- Follow target language best practices
- Add migration notes at top of file

Specify source and target languages in your message.
```

**Usage**:
```bash
/migrate @src/utils.py from Python to TypeScript
```

---

### Example 9: Performance Profiling and Optimization

**File**: `.claude/commands/optimize.md`

```markdown
---
description: Analyze and optimize code performance
tools: Read, Edit, Bash, Grep
---

Perform performance analysis and optimization for $ARGUMENTS

## Phase 1: Analysis
1. Identify performance bottlenecks
2. Analyze algorithmic complexity (Big O)
3. Find unnecessary computations
4. Detect memory leaks or excessive allocations
5. Check I/O operations

## Phase 2: Benchmark (if applicable)
- Run performance tests
- Measure baseline metrics
- Profile hot paths

## Phase 3: Optimization
Apply optimizations:
- Algorithm improvements
- Caching strategies
- Lazy evaluation
- Batch operations
- Parallel processing where safe

## Phase 4: Validation
- Re-run benchmarks
- Compare before/after metrics
- Verify correctness maintained

## Output:
Provide:
- Performance issues found (prioritized)
- Optimization suggestions with rationale
- Estimated improvement for each
- Code changes made
- Before/after comparison
```

**Usage**:
```bash
/optimize @src/data-processor.js
```

**Step-by-step Setup**:
```bash
# 1. Create optimization command
cat > .claude/commands/optimize.md << 'EOF'
---
description: Analyze and optimize code performance
tools: Read, Edit, Bash, Grep
---

Perform performance analysis and optimization for $ARGUMENTS

## Phase 1: Analysis
1. Identify performance bottlenecks
2. Analyze algorithmic complexity (Big O)
3. Find unnecessary computations
4. Detect memory leaks or excessive allocations
5. Check I/O operations

## Phase 2: Benchmark (if applicable)
- Run performance tests
- Measure baseline metrics
- Profile hot paths

## Phase 3: Optimization
Apply optimizations:
- Algorithm improvements
- Caching strategies
- Lazy evaluation
- Batch operations
- Parallel processing where safe

## Phase 4: Validation
- Re-run benchmarks
- Compare before/after metrics
- Verify correctness maintained

## Output:
Provide:
- Performance issues found (prioritized)
- Optimization suggestions with rationale
- Estimated improvement for each
- Code changes made
- Before/after comparison
EOF

# 2. Use for performance work
# /optimize @src/data-processor.js
```

---

### Example 10: API Design Review

**File**: `.claude/commands/api-review.md`

```markdown
---
description: Review API design for consistency and best practices
---

Review the API design in $ARGUMENTS

## Evaluation Criteria:

### 1. REST Principles (if REST API)
- Proper HTTP methods (GET, POST, PUT, DELETE, PATCH)
- Appropriate status codes
- Resource naming conventions
- Stateless communication

### 2. Consistency
- Naming patterns across endpoints
- Response format consistency
- Error handling patterns
- Authentication/authorization approach

### 3. Usability
- Intuitive endpoint structure
- Clear input/output contracts
- Helpful error messages
- API documentation quality

### 4. Versioning
- Version strategy (URL, header, etc.)
- Backward compatibility
- Deprecation plan

### 5. Performance
- Pagination for large datasets
- Filtering and sorting options
- Caching headers
- Rate limiting

### 6. Security
- Input validation
- Authentication requirements
- Authorization checks
- Sensitive data protection

## Output Format:
For each endpoint:
- ✅ Follows best practices
- ⚠️ Minor improvements suggested
- ❌ Significant issues found

Provide:
1. Summary of findings
2. Specific recommendations
3. Example improvements
4. Priority ranking
```

**Usage**:
```bash
/api-review @src/api/routes/
/api-review @docs/api-spec.yaml
```

---

### Example 11: Database Schema Review

**File**: `.claude/commands/schema-review.md`

```markdown
---
description: Review database schema design
---

Analyze database schema in $ARGUMENTS

## Review Areas:

### 1. Normalization
- Check for proper normalization (1NF, 2NF, 3NF)
- Identify data redundancy
- Recommend denormalization where beneficial

### 2. Indexing
- Required indexes for queries
- Missing composite indexes
- Over-indexing concerns

### 3. Relationships
- Foreign key constraints
- Cardinality (one-to-one, one-to-many, many-to-many)
- Cascading deletes/updates

### 4. Data Types
- Appropriate column types
- Size constraints
- Nullability

### 5. Performance
- Query performance implications
- Join complexity
- Scalability concerns

### 6. Naming
- Consistent naming conventions
- Clear, descriptive names
- Standard prefixes/suffixes

### 7. Constraints
- Primary keys
- Unique constraints
- Check constraints
- Default values

## Output:
- Schema diagram (if helpful)
- Issues found with severity
- Recommended changes
- Migration strategy
```

**Usage**:
```bash
/schema-review @database/schema.sql
/schema-review @models/
```

---

## Best Practices

### 1. Command Organization

```bash
# Group related commands in subdirectories
.claude/commands/
├── review/
│   ├── code-review.md
│   ├── security-review.md
│   └── api-review.md
├── generate/
│   ├── tests.md
│   ├── docs.md
│   └── migration.md
└── fix/
    ├── bugs.md
    ├── lint.md
    └── types.md
```

### 2. Use Descriptive Names

```bash
# Good
/security-audit
/generate-tests
/optimize-performance

# Bad
/sa
/gt
/opt
```

### 3. Document Expected Arguments

```markdown
---
description: Generate tests for FILE or FUNCTION
---

Usage: /test @path/to/file.js
   or: /test the functionName in @file.js

Generate comprehensive unit tests for $ARGUMENTS
```

### 4. Leverage Tool Restrictions

```markdown
---
description: Review code (read-only)
tools: Read, Grep, Glob
---
# This prevents accidental modifications
```

### 5. Version Control Project Commands

```bash
# .gitignore
.claude/settings.local.json

# Include project commands
# .claude/commands/ - committed to repo
```

### 6. Keep Personal Commands Separate

```bash
# Personal preferences go in ~/.claude/commands/
# Examples:
~/.claude/commands/explain-eli5.md  # Your personal explanation style
~/.claude/commands/my-review.md     # Your review preferences
```

### 7. Use Clear Output Format Specifications

```markdown
---
description: Generate changelog
---

Generate changelog for $ARGUMENTS

Output format:
## [Version] - YYYY-MM-DD
### Added
- Feature 1
### Changed
- Change 1
### Fixed
- Bug fix 1
```

---

## Argument Usage Patterns

### Pattern 1: All Arguments as Single String

```markdown
Process $ARGUMENTS
```
**Usage**: `/command this is all one argument string`

### Pattern 2: Positional Arguments

```markdown
Convert $1 to $2
```
**Usage**: `/command Python TypeScript`

### Pattern 3: File References

```markdown
Analyze @filename mentioned in $ARGUMENTS
```
**Usage**: `/command @src/app.js for security issues`

### Pattern 4: Mixed Pattern

```markdown
Generate $1 tests for $2 covering $3
```
**Usage**: `/command unit @src/auth.js edge cases`

---

## Testing Your Commands

### 1. Create Test Command

```bash
# Quick test
echo "---
description: Test command
---
Echo back: $ARGUMENTS" > .claude/commands/echo-test.md

# Use: /echo-test hello world
```

### 2. Validate Syntax

```bash
# Check YAML frontmatter
head -n 5 .claude/commands/your-command.md
```

### 3. Test with Different Arguments

```bash
# In Claude Code:
/your-command arg1
/your-command arg1 arg2
/your-command @file.txt
/your-command @file.txt with extra context
```

---

## Troubleshooting

### Command Not Found

```bash
# Check file exists
ls -la .claude/commands/

# Check filename matches command
# File: review.md → Command: /review

# Check for typos in filename
```

### Arguments Not Working

```markdown
# Make sure you're using $ARGUMENTS or $1, $2, etc.
# Correct:
Process $ARGUMENTS

# Incorrect:
Process {arguments}
```

### Command Not Executing

```bash
# Check YAML frontmatter syntax
---
description: Valid description
tools: Read, Write
---

# Make sure there are exactly three dashes
```

### Tool Restrictions Not Working

```markdown
---
tools: Read, Grep  # Comma-separated, no quotes
---
```

---

## Quick Reference Card

```bash
# Create command directory
mkdir -p .claude/commands

# Create basic command
cat > .claude/commands/name.md << 'EOF'
---
description: What it does
---
Your prompt here using $ARGUMENTS
EOF

# Use command
/name @file.txt with arguments

# List commands
ls .claude/commands/

# Edit command
vim .claude/commands/name.md

# Test command
/name test arguments
```

---

## Real-World Example: Complete Setup

Let's create a complete custom command workflow from scratch:

```bash
# 1. Initialize project commands directory
mkdir -p .claude/commands

# 2. Create a code review command
cat > .claude/commands/review.md << 'EOF'
---
description: Comprehensive code review
tools: Read, Grep, Glob
---

Perform a thorough code review of $ARGUMENTS

Focus on:
1. Code quality and maintainability
2. Potential bugs
3. Performance issues
4. Security concerns
5. Best practices adherence

Provide specific line-by-line feedback where needed.
EOF

# 3. Create a test generation command
cat > .claude/commands/gen-tests.md << 'EOF'
---
description: Generate comprehensive tests
tools: Read, Write, Grep
---

Generate unit tests for $ARGUMENTS

Include:
- Happy path scenarios
- Edge cases
- Error handling
- Mock external dependencies

Follow existing test patterns in the project.
EOF

# 4. Create a refactor command
cat > .claude/commands/refactor.md << 'EOF'
---
description: Refactor code for better quality
tools: Read, Edit, Grep
---

Refactor $ARGUMENTS to improve:
1. Readability
2. Maintainability
3. Performance
4. Testability

Explain each change made.
EOF

# 5. Add to version control
git add .claude/commands/
git commit -m "Add custom Claude Code commands"

# 6. Use the commands in Claude Code
# /review @src/app.js
# /gen-tests @src/utils.js
# /refactor @src/legacy/old-code.js
```

---

## Summary

Custom commands are powerful tools for:
- ✅ Standardizing workflows
- ✅ Improving consistency
- ✅ Saving time on repetitive tasks
- ✅ Sharing best practices with team
- ✅ Creating specialized AI behaviors

Start simple, iterate based on usage, and build up a library of commands that match your workflow.
