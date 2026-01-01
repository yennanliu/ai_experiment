# Orchestration Agents - Simple POC Guide

> A minimal, elegant approach to building multi-agent orchestration with Claude

## Overview

This guide presents the **simplest possible POC** for orchestration agents, synthesizing insights from GPT and Gemini research with practical Claude Code implementation.

---

## Core Concept

**Agent Orchestration** = Multiple specialized Claude instances working together through:
- **Clear roles** - Each agent has specific responsibilities
- **Shared state** - Agents communicate through files, not direct chat
- **Central coordinator** - Orchestrator manages execution flow
- **Parallel execution** - Independent tasks run simultaneously

---

## Architecture

```
┌─────────────────────┐
│   Orchestrator      │  ← Coordinates workflow
│   (You or Script)   │
└──────────┬──────────┘
           │
    ┌──────┼──────┬──────────┐
    │      │      │          │
┌───▼───┐ ┌▼────┐ ┌▼──────┐ ┌▼───────┐
│  PM   │ │Design│ │  BE   │ │  FE    │
│ Agent │ │Agent │ │ Agent │ │ Agent  │
└───┬───┘ └┬─────┘ └┬──────┘ └┬───────┘
    │      │        │         │
    └──────┴────────┴─────────┘
           │
    ┌──────▼──────────────┐
    │  Shared Workspace   │
    │  (Files/Memory)     │
    └─────────────────────┘
```

---

## Simplest POC: 3-Agent System

Start with just **3 agents** to prove the concept:

1. **PM Agent** - Breaks down requirements
2. **Backend Agent** - Creates API contract
3. **Frontend Agent** - Builds UI using API contract

---

## Implementation Steps

### Step 1: Project Structure

```
orchestration_agents/
├── .claude/
│   └── agents/
│       ├── pm.md
│       ├── backend.md
│       └── frontend.md
├── workspace/
│   ├── requirements.md
│   ├── api-contract.json
│   └── frontend-spec.md
└── orchestrator.sh
```

### Step 2: Define Agent Prompts

#### `pm.md` - Product Manager Agent

```markdown
# PM Agent

## Role
You are a Senior Product Manager. Your job is to translate user requests into clear, structured requirements.

## Responsibilities
- Break down features into user stories
- Define acceptance criteria
- Create task breakdown

## Rules
- Do NOT write code
- Output in markdown format
- Be concise and structured
- Focus on WHAT, not HOW

## Output Format
Save your output to `workspace/requirements.md` with:
- Feature overview
- User stories
- Acceptance criteria
- Success metrics
```

#### `backend.md` - Backend Agent

```markdown
# Backend Agent

## Role
You are a Senior Backend Engineer. You create API contracts and data models.

## Responsibilities
- Design API endpoints
- Define data schemas
- Specify authentication/authorization
- Consider scalability and security

## Rules
- Follow RESTful principles
- Output API contract as JSON
- Include request/response examples
- No frontend concerns

## Input
Read from: `workspace/requirements.md`

## Output Format
Save to `workspace/api-contract.json`:
- Endpoints (URL, method, params)
- Request/response schemas
- Error codes
- Authentication requirements
```

#### `frontend.md` - Frontend Agent

```markdown
# Frontend Agent

## Role
You are a Senior Frontend Engineer. You implement UI based on requirements and API contracts.

## Responsibilities
- Design component architecture
- Implement UI logic
- Integrate with backend APIs
- Handle state management

## Rules
- Use modern best practices (React/TypeScript)
- Follow API contract strictly
- Consider UX and accessibility
- Output component structure only (no full implementation unless requested)

## Input
Read from:
- `workspace/requirements.md`
- `workspace/api-contract.json`

## Output Format
Save to `workspace/frontend-spec.md`:
- Component hierarchy
- State management approach
- API integration points
- Key implementation notes
```

### Step 3: Create Orchestrator Script

#### `orchestrator.sh` - Simple Bash Orchestrator

```bash
#!/bin/bash

# Orchestration Agents POC
# Simple sequential execution with Claude Code

echo "🚀 Starting Agent Orchestration POC..."

# Step 1: PM Agent defines requirements
echo "\n📋 Step 1: PM Agent - Gathering Requirements"
echo "USER_REQUEST: $1" > workspace/input.txt
claude-code --agent .claude/agents/pm.md workspace/input.txt

# Step 2: Backend Agent creates API contract
echo "\n⚙️  Step 2: Backend Agent - Designing API"
claude-code --agent .claude/agents/backend.md workspace/requirements.md

# Step 3: Frontend Agent builds UI spec (parallel could happen here)
echo "\n💻 Step 3: Frontend Agent - Designing UI"
claude-code --agent .claude/agents/frontend.md workspace/requirements.md workspace/api-contract.json

echo "\n✅ Orchestration Complete!"
echo "📁 Check workspace/ for outputs"
```

### Step 4: Run the POC

```bash
chmod +x orchestrator.sh
./orchestrator.sh "Build a simple todo list app with user authentication"
```

---

## Key Principles for Success

### ✅ DO
- **Clear boundaries** - Each agent has ONE job
- **Shared state** - Agents write to files, not to each other
- **Sequential when needed** - FE waits for BE's API contract
- **Parallel when possible** - Designer + BE can work simultaneously
- **Structured output** - JSON/Markdown, not freeform text

### ❌ DON'T
- Let agents "chat" with each other
- Give overlapping responsibilities
- Skip validation between steps
- Forget to version control the workspace

---

## Example Execution Flow

**User Input:**
```
"Build a user login system"
```

**PM Agent Output** (`workspace/requirements.md`):
```markdown
# Login System Requirements

## User Story
As a user, I want to log in securely so I can access my account.

## Acceptance Criteria
- Email + password authentication
- JWT token-based session
- Password reset flow
- Remember me option

## Success Metrics
- Login success rate > 99%
- Average login time < 2s
```

**Backend Agent Output** (`workspace/api-contract.json`):
```json
{
  "endpoints": [
    {
      "path": "/api/auth/login",
      "method": "POST",
      "request": {
        "email": "string",
        "password": "string",
        "rememberMe": "boolean"
      },
      "response": {
        "token": "string",
        "user": { "id": "string", "email": "string" }
      },
      "errors": [
        { "code": 401, "message": "Invalid credentials" },
        { "code": 429, "message": "Too many attempts" }
      ]
    }
  ]
}
```

**Frontend Agent Output** (`workspace/frontend-spec.md`):
```markdown
# Login UI Component Spec

## Component Hierarchy
- LoginPage
  - LoginForm
    - EmailInput
    - PasswordInput
    - RememberMeCheckbox
    - SubmitButton
  - ErrorMessage
  - LoadingSpinner

## State Management
- `useAuth()` hook for authentication state
- Form state with validation
- Error handling for API failures

## API Integration
- POST /api/auth/login on form submit
- Store JWT in localStorage (if rememberMe) or sessionStorage
- Redirect to dashboard on success
```

---

## Advanced: Parallel Execution

For truly parallel execution, use Node.js instead of bash:

```javascript
// orchestrator.js
const { spawn } = require('child_process');

async function runAgent(agentFile, inputs) {
  return new Promise((resolve, reject) => {
    const agent = spawn('claude-code', ['--agent', agentFile, ...inputs]);
    // ... handle output
  });
}

async function orchestrate(userRequest) {
  // Step 1: PM (sequential)
  await runAgent('.claude/agents/pm.md', ['workspace/input.txt']);

  // Step 2: Backend + Designer (parallel)
  await Promise.all([
    runAgent('.claude/agents/backend.md', ['workspace/requirements.md']),
    runAgent('.claude/agents/designer.md', ['workspace/requirements.md'])
  ]);

  // Step 3: Frontend (sequential, waits for BE + Designer)
  await runAgent('.claude/agents/frontend.md', [
    'workspace/requirements.md',
    'workspace/api-contract.json',
    'workspace/design.md'
  ]);
}
```

---

## Extending the POC

Once the basic 3-agent system works, add:

1. **Designer Agent** - UI/UX specifications
2. **QA Agent** - Test generation and validation
3. **Reviewer Agent** - Code review and conflict resolution
4. **DevOps Agent** - Deployment and infrastructure

---

## Comparison with Frameworks

| Approach | Pros | Cons |
|----------|------|------|
| **Custom (This POC)** | Simple, full control, no dependencies | Manual orchestration, no built-in features |
| **LangGraph** | Visual flow, built-in state management | Heavier, learning curve |
| **CrewAI** | Pre-built agent roles, easy setup | Less flexible, opinionated |
| **AutoGen** | Multi-agent conversations, Microsoft support | Complex setup, more suited for chat |

**Recommendation:** Start with this custom POC, then migrate to LangGraph if you need complex branching logic.

---

## Cost Optimization

- **Use Haiku for PM/Designer** (cheaper, fast enough for text)
- **Use Sonnet for Backend/Frontend** (better code quality)
- **Cache system prompts** (if using Claude API directly)
- **Stream responses** (show progress in real-time)

---

## Testing the POC

Create test scenarios:

```bash
# Test 1: Simple feature
./orchestrator.sh "Add a search bar to the homepage"

# Test 2: Complex feature with dependencies
./orchestrator.sh "Build a real-time chat system with WebSocket support"

# Test 3: Conflict scenario (agents need revision)
./orchestrator.sh "Implement infinite scroll with pagination"
```

---

## Next Steps

1. **Run this POC** with a simple feature request
2. **Validate outputs** - Check if agents stay in their lanes
3. **Add conflict resolution** - Orchestrator checks for inconsistencies
4. **Introduce shared tools** - Git commits, file searching, testing
5. **Scale up** - Add more specialized agents

---

## Conclusion

This POC demonstrates the **minimum viable orchestration system**:
- 3 agents with clear roles
- Shared workspace for communication
- Simple coordinator script
- No external frameworks required

**Philosophy:** Start simple, prove the concept, then add complexity only when needed.

The key insight: **Orchestration is about coordination, not conversation.**

---

## Resources

- Existing guides: `gpt_idea.md`, `gemini_idea.md`
- Claude Code docs: `.claude/agents/` patterns
- Example skills: `../claude_code/skills_poc/`
