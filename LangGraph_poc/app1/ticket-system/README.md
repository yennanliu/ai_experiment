# Ticket Processing System

An AI-powered customer support ticket processor built with **LangGraph** and **OpenAI**, served via **FastAPI**.

- [Reference blog](https://yennj12.js.org/yennj12_blog_V4/posts/langgraph-ai-customer-ticket-system-zh/)

---

## Architecture

```
START → classify → prioritize → route → generate_response → quality_check → END
                                              ↑                    |
                                              └─── retry if score < 0.8 (max 3x)
```

### Graph Nodes

| Node | Description |
|------|-------------|
| `classify` | Tags ticket category (technical, billing, account, feature_request, bug_report, other) with confidence score |
| `prioritize` | Assigns priority (critical/high/medium/low) and SLA hours (1/4/12/24h) |
| `route` | Maps category to department (Engineering, Finance, Product, etc.) |
| `generate_response` | Writes a concise, professional customer reply |
| `quality_check` | Scores 0–1 across 5 dimensions; loops back to generate if below 0.8 |

---

## Project Structure

```
ticket-system/
├── server.py               ← FastAPI app entry point
├── demo.py                 ← CLI runner (3 sample tickets)
├── graph/
│   ├── state.py            ← TicketState dataclass
│   ├── nodes.py            ← 5 LangGraph node functions
│   └── pipeline.py         ← build_graph() + process_ticket()
├── models/
│   └── ticket.py           ← Pydantic schemas (SubmitRequest, TicketRecord, TicketResult)
├── services/
│   └── ticket_service.py   ← In-memory store + async pipeline runner
├── routers/
│   └── tickets.py          ← FastAPI router (/tickets)
├── utils/
│   └── llm.py              ← OpenAI client + chat() helper
└── ui/
    └── index.html          ← Single-page dashboard UI
```

---

## Setup

```bash
# 1. Set your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# 2. Start the server
uv run server.py
# → http://localhost:8000

# 3. (Optional) CLI demo
uv run demo.py
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tickets` | Submit a ticket `{"message": "..."}` → returns 202 immediately |
| `GET` | `/tickets` | List all tickets |
| `GET` | `/tickets/stats` | Aggregated counts by status / category / priority |
| `GET` | `/tickets/{id}` | Get ticket status + result |
| `GET` | `/graph` | LangGraph node/edge definition (used by UI) |
| `GET` | `/` | Serve the UI |

Tickets are processed asynchronously. Poll `GET /tickets/{id}` until `status` is `done` or `error`.

---

## Seed Script

Submit random tickets in bulk for testing:

```bash
# 100 tickets (default)
uv run scripts/seed_tickets.py

# Custom count, concurrency, and server URL
uv run scripts/seed_tickets.py --count 50 --concurrency 5 --url http://localhost:8000
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--count` | `100` | Number of tickets to submit |
| `--concurrency` | `10` | Max parallel requests |
| `--url` | `http://localhost:8000` | Server base URL |

---

## Dependencies

Managed by `uv`:
- `langgraph` — graph orchestration
- `openai` — LLM calls (gpt-4o-mini)
- `fastapi` + `uvicorn` — HTTP server
- `python-dotenv` — `.env` loading
