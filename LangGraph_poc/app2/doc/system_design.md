# AI Email Reply Assistant — System Design & Implementation Plan

## Context

Build a **simplified MVP** of the AI Email Reply Assistant — no RAG, no Notion, no DB. Just a clean FastAPI + LangGraph backend that accepts an email, classifies it, and generates a template-driven draft reply.

---

## Architecture

```
User (curl / simple UI)
        │
        ▼
   FastAPI  (/generate-draft)
        │
        ▼
   LangGraph Agent
   ┌────────────────────────────┐
   │  Node 1: Classify Email    │  → email type (New PO / Wrong Order / Change / Other)
   │  Node 2: Select Template   │  → load template + few-shot examples (in-memory)
   │  Node 3: Generate Draft    │  → call OpenAI with structured prompt
   │  Node 4: Build Checklist   │  → flag missing info / risk keywords
   └────────────────────────────┘
        │
        ▼
   JSON response
   {
     "email_type": "...",
     "draft_reply": "...",
     "checklist": ["...", ...]
   }
```

---

## Tech Stack

| Layer | Choice |
|---|---|
| API | FastAPI |
| Orchestration | LangGraph (StateGraph) |
| LLM | OpenAI GPT-4o via `openai` SDK |
| Config | `.env` → `python-dotenv` |
| Data | In-memory templates (Python dicts) |
| No DB, No RAG, No Notion | |

---

## LangGraph State

```
AgentState {
  inbound_email: str       # raw input
  email_type: str          # classified
  template: str            # selected template
  draft_reply: str         # generated output
  checklist: list[str]     # flagged items
}
```

---

## Node Design

### Node 1 — Classify
- Single OpenAI call
- System prompt: "Classify this email into one of: New PO / Change Order / Wrong Order / Other"
- Output: `email_type`

### Node 2 — Select Template
- Pure Python: dict lookup by `email_type`
- Templates stored as multi-line strings in `templates.py`
- Each template has: greeting, body placeholders, closing, required fields list

### Node 3 — Generate Draft
- OpenAI call with:
  - System: role + tone instructions + template
  - User: the inbound email
- Output: filled-in `draft_reply`

### Node 4 — Build Checklist
- Lightweight rule-based + LLM pass
- Check for: missing PO#, pickup date, quantity, bank account mentions, NDA keywords
- Output: `checklist` list

---

## File Structure

```
app2/
├── .env                    # OPENAI_API_KEY=...
├── main.py                 # FastAPI app + /generate-draft endpoint
├── agent/
│   ├── graph.py            # LangGraph StateGraph definition
│   ├── nodes.py            # 4 node functions
│   └── templates.py        # in-memory template library per email type
└── doc/
    └── req.md
```

---

## API Contract

**POST** `/generate-draft`

Request:
```json
{
  "email": "Dear team, we found the wrong item shipped..."
}
```

Response:
```json
{
  "email_type": "Wrong Order",
  "draft_reply": "Dear [Customer],\n\nThank you for contacting us...",
  "checklist": [
    "Missing: original order number",
    "Risk flag: pricing mentioned"
  ]
}
```

---

## Key Design Decisions

1. **Templates in code, not DB** — simplest starting point; easy to later move to Notion/DB
2. **LangGraph over plain function chain** — state is explicit, nodes are swappable, easy to add RAG node later
3. **Classification first** — routes to the right template before drafting; avoids generic output
4. **Checklist as separate node** — keeps generation clean; checklist is internal-only, not part of draft

---

## Upgrade Path (when ready)

| Phase | What to add |
|---|---|
| +RAG | Add a `Retrieve` node between Classify and Generate; connect to vector DB |
| +Notion | Trigger via Notion webhook; write draft back via Notion API |
| +Outlook | Microsoft Graph API to create Outlook draft |
