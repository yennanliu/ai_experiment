# Ticket Processing System

An AI-powered customer support ticket processor built with **LangGraph** and **OpenAI**.

## Architecture

```
START → classify → prioritize → route → generate_response → quality_check → END
                                              ↑                    |
                                              └─── retry if score < 0.8 (max 3x)
```

### Nodes

| Node | Description |
|------|-------------|
| `classify` | Tags ticket category (technical, billing, account, feature_request, bug_report, other) with confidence score |
| `prioritize` | Assigns priority (critical/high/medium/low) and SLA hours (1/4/12/24h) |
| `route` | Maps category to department (Engineering, Finance, Product, etc.) |
| `generate_response` | Writes a concise, professional customer reply |
| `quality_check` | Scores the response 0–1 across 5 dimensions; loops back if below 0.8 |

## Setup

```bash
# 1. Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# 2. Run
uv run main.py
```

## Dependencies

Managed by `uv`:
- `langgraph` — graph orchestration
- `openai` — LLM calls
- `python-dotenv` — `.env` loading
