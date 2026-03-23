# Agent Team

Specialized Agent Orchestration Framework - a minimal, elegant implementation of multi-agent coordination patterns.

## Features

- **Specialized Agents**: Analyst, Developer, Reviewer, Doc Writer with targeted prompts (~2500 tokens vs ~16000 for monolithic)
- **Smart Routing**: Keyword matching + LLM fallback for task classification
- **Orchestration Patterns**:
  - **Pipeline**: Sequential execution, each output feeds the next
  - **Hub-and-Spoke**: Central orchestrator coordinates all agents
  - **Parallel Merge**: Concurrent execution with result merging
- **Token Budget Management**: Track and control API costs
- **Minimal Context Passing**: Only summaries transfer between agents

## Installation

```bash
uv sync
```

## Usage

### CLI

```bash
# Simple task with auto-routing
uv run agent-team "Write a function to parse JSON"

# Specify pattern
uv run agent-team --pattern pipeline "Analyze and implement a cache"

# Development workflow
uv run agent-team --workflow dev "Build a REST API endpoint"

# Parallel review
uv run agent-team --workflow review "def foo(): return eval(input())"
```

### Python API

```python
from agent_team import Orchestrator, AgentRole, OrchestrationPattern

# Initialize
orchestrator = Orchestrator()

# Simple single-agent task
response = orchestrator.simple(
    "Write a function to validate emails",
    role=AgentRole.DEVELOPER,
)

# Full development workflow (Analyst -> Developer -> Reviewer)
state = orchestrator.analyze_and_implement(
    "Create a rate limiter using token bucket algorithm"
)

# Custom pipeline
state = orchestrator.run(
    "Document this API",
    pattern=OrchestrationPattern.PIPELINE,
    agents=[AgentRole.ANALYST, AgentRole.DOC_WRITER],
)

# Parallel review from multiple perspectives
state = orchestrator.review_from_all_angles(code)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Orchestrator                        │
│  ┌─────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Router  │  │ TokenBudget  │  │ Pattern Selector │   │
│  └─────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Pipeline │    │Hub-Spoke │    │ Parallel │
    └──────────┘    └──────────┘    └──────────┘
          │               │               │
          ▼               ▼               ▼
    ┌─────────────────────────────────────────┐
    │           Specialized Agents            │
    │  Analyst │ Developer │ Reviewer │ Docs  │
    └─────────────────────────────────────────┘
```

## Configuration

```python
from agent_team.orchestrator import Orchestrator, OrchestratorConfig

config = OrchestratorConfig(
    model="claude-sonnet-4-20250514",  # or claude-3-haiku, etc.
    token_budget=100_000,
    max_retries=2,
    auto_select_pattern=True,
)

orchestrator = Orchestrator(config)
```

## Token Savings

By using specialized agents with targeted prompts instead of a single monolithic agent:

| Approach | Tokens per Agent | Overhead |
|----------|-----------------|----------|
| Monolithic | ~16,000 | High |
| Specialized | ~2,500 | Low |
| **Savings** | **~80%** | |
