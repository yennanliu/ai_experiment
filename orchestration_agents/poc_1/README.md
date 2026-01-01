# Orchestration Agents POC

A minimal proof-of-concept for multi-agent orchestration using Claude Code.

---

## 🚀 Quick Access - Running Application

**Frontend:** http://localhost:5173
**Backend API:** http://localhost:3001

**Start the servers:**
```bash
# Terminal 1 - Backend
cd backend && npm start

# Terminal 2 - Frontend
cd frontend && npm run dev
```

**Features:**
- ✅ Create, edit, delete notes
- ✅ Real-time search
- ✅ Tag filtering and management
- ✅ Auto-save (1 second debounce)
- ✅ Responsive UI (Tailwind CSS)

---

## Overview

This POC demonstrates a **3-agent system** working together to transform a feature request into detailed technical specifications:

1. **PM Agent** - Translates requests into structured requirements
2. **Backend Agent** - Designs API contracts and data models
3. **Frontend Agent** - Creates UI architecture and component specifications

## Architecture

```
User Request
     ↓
  PM Agent → requirements.md
     ↓
Backend Agent → api-contract.json
     ↓
Frontend Agent → frontend-spec.md
```

All agents communicate through shared files in the `workspace/` directory.

## Directory Structure

```
poc_1/
├── .claude/
│   └── agents/              # Agent definitions
│       ├── pm.md            # Product Manager agent
│       ├── backend.md       # Backend Engineer agent
│       └── frontend.md      # Frontend Engineer agent
├── workspace/               # Shared workspace for agent outputs
│   ├── input.txt           # User's feature request
│   ├── requirements.md      # PM output
│   ├── api-contract.json   # Backend output
│   └── frontend-spec.md    # Frontend output
├── orchestrator.sh         # Bash orchestrator (manual)
├── orchestrator.js         # Node.js orchestrator (automated)
└── README.md               # This file
```

## Quick Start

### Method 1: Bash Orchestrator (Manual)

Run the orchestrator with your feature request:

```bash
./orchestrator.sh "Build a simple todo list app with user authentication"
```

The script will guide you through each step, prompting you to run Claude Code for each agent.

### Method 2: Node.js Orchestrator (Simulated)

```bash
node orchestrator.js "Build a simple todo list app"
```

This version simulates the flow and creates placeholder files. To make it work with real Claude API:

1. Install dependencies:
   ```bash
   npm install @anthropic-ai/sdk
   ```

2. Set your API key:
   ```bash
   export ANTHROPIC_API_KEY="your-api-key"
   ```

3. Uncomment the Claude API integration code in `orchestrator.js`

### Method 3: Manual Execution

Run each agent manually for full control:

#### Step 1: PM Agent
```bash
# Start Claude Code
claude-code chat

# In the chat, load the agent and input
/read .claude/agents/pm.md
/read workspace/input.txt

# Ask: "Generate requirements following the PM agent template"
```

Save the output to `workspace/requirements.md`.

#### Step 2: Backend Agent
```bash
/read .claude/agents/backend.md
/read workspace/requirements.md

# Ask: "Generate API contract following the backend agent template"
```

Save the output to `workspace/api-contract.json`.

#### Step 3: Frontend Agent
```bash
/read .claude/agents/frontend.md
/read workspace/requirements.md
/read workspace/api-contract.json

# Ask: "Generate frontend specification following the frontend agent template"
```

Save the output to `workspace/frontend-spec.md`.

## Agent Descriptions

### PM Agent (pm.md)

**Responsibilities:**
- Break down features into user stories
- Define acceptance criteria
- Identify technical considerations
- Set success metrics

**Output:** `workspace/requirements.md`

### Backend Agent (backend.md)

**Responsibilities:**
- Design RESTful API endpoints
- Define data models and schemas
- Specify authentication requirements
- Consider security and scalability

**Output:** `workspace/api-contract.json`

### Frontend Agent (frontend.md)

**Responsibilities:**
- Design component hierarchy
- Plan state management
- Define API integration strategy
- Consider UX and accessibility

**Output:** `workspace/frontend-spec.md`

## Example Execution

**Input:**
```bash
./orchestrator.sh "Build a real-time chat application"
```

**Outputs:**

1. `requirements.md` - User stories, acceptance criteria, technical requirements
2. `api-contract.json` - WebSocket endpoints, message schemas, authentication
3. `frontend-spec.md` - Component structure, real-time state management, UI patterns

## Key Principles

### ✅ DO
- Keep agents focused on their specific role
- Use structured output formats (Markdown, JSON)
- Validate outputs for consistency
- Iterate if agents produce conflicting specs

### ❌ DON'T
- Let agents communicate directly with each other
- Give agents overlapping responsibilities
- Skip validation between steps
- Forget to version control the workspace

## Customization

### Adding New Agents

Create a new agent file in `.claude/agents/`:

```markdown
# [Agent Name]

## Role
[Description]

## Responsibilities
- [Task 1]
- [Task 2]

## Rules
- [Rule 1]
- [Rule 2]

## Input
[What files to read]

## Output Format
[Structure and location]
```

Then update `orchestrator.sh` or `orchestrator.js` to include the new agent in the workflow.

### Modifying Execution Order

**Sequential:**
```
PM → Backend → Frontend
```

**Parallel:**
```
PM → [Backend + Designer] → Frontend
```

Edit the orchestrator scripts to change execution order.

## Integration with Claude Code

### Using Subagents

Claude Code supports subagents natively. To use this POC as subagents:

1. Place agent files in `.claude/agents/`
2. Reference them with: `--agent .claude/agents/pm.md`

### Using Skills

Alternatively, convert agents to skills in `.claude/skills/`:

```
.claude/skills/
└── pm-agent/
    └── SKILL.md
```

## Troubleshooting

**Problem:** Agent outputs don't match expectations

**Solution:**
- Review the agent's role and rules
- Provide more detailed input context
- Add examples to the agent definition

**Problem:** Agents produce conflicting specifications

**Solution:**
- Run a "Reviewer Agent" to check consistency
- Iterate with clearer requirements from PM
- Add validation steps between agents

**Problem:** Outputs are too generic

**Solution:**
- Provide more specific user input
- Add domain-specific examples to agent definitions
- Include constraints and preferences in requirements

## Next Steps

1. **Test with Real Projects**
   - Try different feature requests
   - Validate output quality
   - Refine agent prompts based on results

2. **Add More Agents**
   - Designer Agent (UI/UX specs)
   - QA Agent (test scenarios)
   - DevOps Agent (deployment specs)

3. **Implement Automation**
   - Integrate Claude API for fully automated execution
   - Add validation and consistency checks
   - Create feedback loops for agent refinement

4. **Scale Up**
   - Support parallel execution
   - Add dependency management
   - Implement task queuing

## Resources

- [Guide Documentation](../doc/claude_poc_guide.md)
- [GPT Research](../doc/gpt_idea.md)
- [Gemini Research](../doc/gemini_idea.md)
- [Claude Code Docs](https://docs.anthropic.com/claude-code)

## License

MIT - Feel free to use and modify for your projects!

---

**Philosophy:** Start simple, prove the concept, add complexity only when needed. Orchestration is about coordination, not conversation.
