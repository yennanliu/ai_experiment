# Resume Tailor

A multi-agent LangGraph app that tailors your resume and generates a cover letter for a specific job description. Four AI agents simulate the real hiring pipeline — ATS, recruiter, and hiring manager — to score and improve your application before you submit.

---

## Architecture

```
                        ResumeState (TypedDict)
                               │
        ┌──────────────────────▼──────────────────────┐
        │              LangGraph Pipeline              │
        │                                             │
        │  parse_inputs → ats_simulate → rewrite_resume
        │      → write_cover_letter → recruiter_review
        │      → hiring_manager_review → score_output  │
        └──────────────────────────────────────────────┘
                               │
                          FastAPI (SSE)
                               │
                      Vanilla HTML UI
                               │
                          SQLite (history)
```

**Agent roles:**

| Agent | Persona | Output |
|---|---|---|
| ATS Simulator | Automated keyword scanner | Score, missing keywords, suggestions |
| Resume Writer | Professional resume coach | Tailored resume + cover letter |
| Recruiter | 30-second scan reviewer | Verdict + readability feedback |
| Hiring Manager | Technical fit evaluator | Score + interview decision |

**Final score** = ATS score × 40% + Hiring Manager score × 60%

---

## Features

- **Resume tailoring** — rewrites bullets to mirror JD language without fabricating experience
- **Cover letter generation** — 3-paragraph, role-specific, grounded in your actual resume
- **ATS simulation** — keyword gap analysis with missing terms highlighted
- **Multi-agent panel** — recruiter and hiring manager feedback with color-coded verdicts
- **Diff view** — side-by-side line comparison of original vs tailored resume
- **Live pipeline viz** — per-node SSE streaming shows progress in real time
- **History** — all runs saved to SQLite; track application status (Draft → Applied → Offer)

---

## Project Structure

```
app3/
├── main.py              # FastAPI app (sync + SSE endpoints, history CRUD)
├── db.py                # SQLite persistence layer
├── run.py               # CLI test script (no server needed)
├── pyproject.toml       # uv dependencies
├── .env                 # OPENAI_API_KEY
├── agent/
│   ├── state.py         # ResumeState TypedDict
│   ├── graph.py         # LangGraph StateGraph + compile()
│   ├── nodes.py         # 7 node functions
│   └── prompts.py       # System prompts per agent persona
├── static/
│   ├── index.html       # Main UI (tailor + pipeline viz + tabbed results)
│   └── history.html     # History table with status management
└── doc/
    └── system_design.md # Architecture and build phases
```

---

## How to Run

**Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

### 1. Install dependencies

```bash
uv sync
```

### 2. Set your API key

```bash
# .env
OPENAI_API_KEY=sk-...
```

### 3a. Run the web app

```bash
uv run uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000), paste your resume and a job description, and click **Tailor Resume**.

### 3b. Run as CLI (no server)

```bash
uv run python run.py
```

Prints the full output — tailored resume, cover letter, all agent feedback, and final score — to stdout.

---

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/tailor` | Sync — returns full result |
| `POST` | `/tailor-stream` | SSE — streams per-node events, then `done` |
| `GET` | `/history` | List all past runs |
| `PATCH` | `/history/{id}/status` | Update application status |
| `GET` | `/health` | Health check |

**Request body** (`/tailor` and `/tailor-stream`):
```json
{ "resume": "...", "job_description": "..." }
```
