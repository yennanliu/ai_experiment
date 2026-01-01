---
name: python-code-reviewer
description: Provides comprehensive Python code review with focus on quality, bugs, security, and best practices. Use when reviewing Python code, functions, or modules for quality assessment.
---

# Python Code Reviewer

## Instructions

When reviewing Python code, follow this comprehensive review format:

### 1. Strengths Section (✅)
Identify and highlight what's working well:
- Good code organization and structure
- Proper use of Python idioms and patterns
- Clear documentation (docstrings, comments)
- Appropriate error handling
- Good naming conventions
- Proper use of language features

### 2. Issues & Concerns Section (⚠️)
Categorize issues by severity:

**Critical Bugs:**
- Runtime errors (ZeroDivisionError, IndexError, etc.)
- Logic errors that break functionality
- Security vulnerabilities (injection, XSS, etc.)
- Reference specific line numbers using format: `filename:line_number`

**Recommendations:**
- Code quality improvements
- Better error handling
- Edge case handling
- Performance optimizations
- Code maintainability issues

### 3. Code Examples
For each issue or recommendation:
- Provide concrete code examples showing the fix
- Use proper Python formatting
- Show both the problem and solution
- Explain why the change improves the code

### 4. Additional Considerations
Review for:
- **Edge cases**: Empty inputs, boundary conditions, null/None values
- **Type safety**: Consider suggesting type hints
- **Code style**: PEP 8 compliance, consistent formatting
- **Testing**: Are there testable concerns or missing validations?
- **Documentation**: Are docstrings clear and complete?
- **Performance**: Any obvious performance bottlenecks?
- **Security**: Input validation, SQL injection, command injection, etc.

### 5. Overall Rating
Provide a score out of 10 with brief justification:
- 9-10: Production-ready, minimal issues
- 7-8: Good quality, minor improvements needed
- 5-6: Functional but needs refactoring
- 3-4: Significant issues, requires major work
- 1-2: Critical problems, needs rewrite

### 6. Summary
End with a concise 1-2 sentence summary of the code quality and main concerns.

## Review Checklist

Always check for:
- [ ] Division by zero or similar runtime errors
- [ ] Empty collection handling (lists, dicts, etc.)
- [ ] Input validation and sanitization
- [ ] Exception handling completeness
- [ ] Resource management (file handles, connections)
- [ ] Security vulnerabilities (OWASP Top 10)
- [ ] Type correctness and potential type errors
- [ ] Function side effects and purity
- [ ] Code duplication and DRY principle
- [ ] Naming clarity and consistency

## Best Practices

1. **Be specific**: Always reference line numbers using `filename:line_number` format
2. **Be constructive**: Frame issues as opportunities for improvement
3. **Provide context**: Explain WHY something is an issue, not just WHAT
4. **Show examples**: Demonstrate better approaches with code snippets
5. **Prioritize**: Critical bugs first, then recommendations
6. **Consider scope**: Don't over-engineer simple scripts, don't under-engineer production code

## Response Format Template

```
## Code Review: `path/to/file.py`

[Brief description of what the code does]

### ✅ Strengths

1. [Strength 1]
2. [Strength 2]
...

### ⚠️ Issues & Concerns

**Critical Bug (filename:line_number):**
[Description and code reference]

**Recommendations:**
1. [Recommendation with code example]
2. [Recommendation with code example]
...

### 📊 Overall Rating: X/10

**Summary:** [1-2 sentence summary]
```

## Tool Usage

- Use `Read` to examine the code file
- Use `Grep` if you need to search for patterns across multiple files
- Use `Glob` to find related files if reviewing a module
- Do NOT use `Edit` or `Write` unless explicitly asked to fix issues
