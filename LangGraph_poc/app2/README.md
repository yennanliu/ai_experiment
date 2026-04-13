# AI Email Reply Assistant

A FastAPI + LangGraph service that classifies inbound business emails and generates template-driven reply drafts using OpenAI GPT-4o.

## Architecture

```
POST /generate-draft
        │
        ▼
   LangGraph Agent
   ┌─────────────────────┐
   │  1. Classify Email  │  → New PO / Change Order / Wrong Order / Other
   │  2. Select Template │  → in-memory template per type
   │  3. Generate Draft  │  → GPT-4o fills template from email context
   │  4. Build Checklist │  → flags missing fields + risk keywords
   └─────────────────────┘
        │
        ▼
   JSON response
```

## Project Structure

```
app2/
├── .env                  # OPENAI_API_KEY
├── main.py               # FastAPI app
├── pyproject.toml        # uv-managed dependencies
└── agent/
    ├── state.py          # AgentState TypedDict
    ├── templates.py      # reply templates + risk keywords per email type
    ├── nodes.py          # 4 LangGraph node functions
    └── graph.py          # StateGraph wiring
```

## Setup

**Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Set your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env
```

## Run

```bash
uv run uvicorn main:app --reload
```

Server starts at `http://localhost:8000`.

## API

### `POST /generate-draft`

**Request:**
```json
{
  "email": "Hi, we received the wrong item. We ordered SKU-123 but got SKU-456. Order #PO-789."
}
```

**Response:**
```json
{
  "email_type": "Wrong Order",
  "draft_reply": "Dear [Customer Name],\n\nWe sincerely apologize...",
  "checklist": [
    "Missing: correct item expected",
    "Risk flag: 'pricing' detected in email"
  ]
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## Email Types

| Type | Description |
|---|---|
| `New PO` | New purchase order received |
| `Change Order` | Request to modify an existing order |
| `Wrong Order` | Incorrect item shipped |
| `Other` | General enquiries |

## Tech Stack

| Layer | Choice |
|---|---|
| API | FastAPI |
| Orchestration | LangGraph |
| LLM | OpenAI GPT-4o |
| Package manager | uv |
| Config | python-dotenv |
