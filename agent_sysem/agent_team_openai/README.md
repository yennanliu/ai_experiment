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

### Prerequisites
- Python 3.11+
- OpenAI API key (set `OPENAI_API_KEY` in `.env`)

### Setup Steps

```bash
# Initial setup - installs all dependencies and creates virtual environment
uv sync

# After setup, run commands with:
uv run agent-team "Your task here"
uv run python main.py
```

### When to Use Which Command

| Command | When to Use | Purpose |
|---------|------------|---------|
| `uv sync` | First time setup, after updating `pyproject.toml`, or `.venv` is corrupted | Installs/updates all dependencies and syncs lock file |
| `uv run <cmd>` | Running scripts/CLI after setup | Runs command in virtual environment without activating it |
| `source .venv/bin/activate` | (Optional) Want to use Python interactively | Activates virtual environment manually |
| `uv pip install <pkg>` | Need to add a quick dependency | Direct pip install (updates lock file) |

### Configuration

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxx
```

The CLI automatically loads this file, or set manually:

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxx
uv run agent-team "Your task"
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

## Output & Logging

### Directory Structure

Each job creates organized output and logs:

```
output/
  └── YYYYMMDD_HHMMSS/           # Timestamp directory
      └── {job_id}/              # Unique job ID (8 chars)
          ├── metadata.json       # Execution metadata & stats
          └── output.txt          # Full execution output

log/
  └── YYYYMMDD_HHMMSS/
      └── {job_id}/
          └── output.log          # Timestamped log file (no colors)
```

### Output Files

**metadata.json** - Contains:
- Job ID and timestamp
- Task, pattern, agents used
- Token usage per agent
- Success status and errors

**output.txt** - Contains:
- Execution summary
- Full output from each agent
- Token usage breakdown

**output.log** - Contains:
- Timestamped log entries
- All INFO, WARNING, ERROR logs
- Clean format (no ANSI codes)

### Colored Logging

Logs are displayed in color in the terminal for easy reading:
- 🟢 **INFO** - Green
- 🟡 **WARNING** - Yellow
- 🔴 **ERROR** - Red
- 🔵 **DEBUG** - Cyan

Color codes are stripped from log files for clean output.

### Custom Output/Log Directories

```bash
# Save to custom locations
uv run agent-team --output-dir ./results --log-dir ./logs "Your task"

# Or use defaults (output/ and log/)
uv run agent-team "Your task"
```

## Token Savings

By using specialized agents with targeted prompts instead of a single monolithic agent:

| Approach | Tokens per Agent | Overhead |
|----------|-----------------|----------|
| Monolithic | ~16,000 | High |
| Specialized | ~2,500 | Low |
| **Savings** | **~80%** | |
