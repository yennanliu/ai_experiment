# Resume App — System Design

## Overview

A multi-agent LangGraph application that tailors a candidate's resume and generates a cover letter for a specific job description (JD). Four specialized agents simulate real hiring pipeline perspectives — ATS system, recruiter, hiring manager, and resume writer — to iteratively improve the output before delivering a polished, JD-aligned resume and cover letter.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Runtime | Python 3.11+, managed by `uv` |
| Orchestration | LangGraph (`StateGraph`) |
| LLM | OpenAI GPT-4o |
| API | FastAPI |
| Config | `.env` → `OPENAI_API_KEY` |
| UI | Vanilla HTML (static/) |

---

## Agent Roles

| Agent | Role | Persona |
|---|---|---|
| **ATS Simulator** | Scans resume for keyword match vs JD, flags missing terms, estimates ATS pass rate | Automated screening system |
| **Resume Writer** | Rewrites bullets to mirror JD language, injects missing keywords naturally | Professional resume coach |
| **Recruiter** | Reviews tone, formatting, and first impression; flags red flags | Human recruiter (30-second scan) |
| **Hiring Manager** | Evaluates technical fit, scores candidate against JD requirements | Domain expert decision-maker |

---

## State Design

```python
class ResumeState(TypedDict):
    raw_resume: str               # Original resume text (markdown or plain)
    job_description: str          # Full JD text
    tailored_resume: str          # Rewritten resume from Resume Writer
    cover_letter: str             # Generated cover letter
    ats_report: dict              # {score: int, missing_keywords: list, suggestions: list}
    recruiter_feedback: str       # Tone/format feedback
    hiring_manager_feedback: str  # Fit score + rationale
    final_score: int              # 0–100 overall confidence
```

---

## LangGraph Pipeline

```
parse_inputs
    ↓
ats_simulate          ← ATS Simulator agent
    ↓
rewrite_resume        ← Resume Writer agent (uses ats_report)
    ↓
write_cover_letter    ← Resume Writer agent (uses tailored_resume + JD)
    ↓
recruiter_review      ← Recruiter agent
    ↓
hiring_manager_review ← Hiring Manager agent
    ↓
score_output          ← Aggregate final_score
    ↓
END
```

**Node signature pattern** (mirrors app2):
```python
def node_name(state: ResumeState) -> ResumeState:
    # ... call OpenAI or run logic ...
    return {**state, "field": updated_value}
```

**Graph wiring:**
```python
graph = StateGraph(ResumeState)
graph.add_node("parse_inputs", parse_inputs)
graph.add_node("ats_simulate", ats_simulate)
# ... add remaining nodes ...
graph.set_entry_point("parse_inputs")
graph.add_edge("parse_inputs", "ats_simulate")
# ... linear edges ...
graph.add_edge("score_output", END)
resume_agent = graph.compile()
```

---

## File Structure

```
app3/
├── main.py              # FastAPI app
├── pyproject.toml       # uv dependencies
├── .env                 # OPENAI_API_KEY
├── agent/
│   ├── __init__.py
│   ├── state.py         # ResumeState TypedDict
│   ├── graph.py         # StateGraph wiring + compile()
│   ├── nodes.py         # All 7 node functions
│   └── prompts.py       # System prompts per agent role
├── static/
│   └── index.html       # Upload resume + JD, view output
└── doc/
    └── system_design.md
```

---

## API

```
POST /tailor
  Body: { resume: str, job_description: str }
  Returns: { tailored_resume, cover_letter, ats_report, recruiter_feedback,
             hiring_manager_feedback, final_score }

POST /tailor-stream
  Body: { resume: str, job_description: str }
  Returns: SSE stream — one event per node completion
  Event format: { node: str, state: partial ResumeState }
```

---

## Environment

```env
# .env
OPENAI_API_KEY=sk-...
```

Loaded via:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Build Phases

### Phase 1 — Core Pipeline (MVP)
> Goal: Get a working end-to-end tailored resume from CLI

- [ ] Set up project with `uv init`, `pyproject.toml`, `.env`
- [ ] Define `ResumeState` in `agent/state.py`
- [ ] Implement `parse_inputs`, `ats_simulate`, `rewrite_resume` nodes
- [ ] Wire LangGraph in `agent/graph.py`, compile agent
- [ ] Write system prompts for ATS Simulator and Resume Writer in `agent/prompts.py`
- [ ] Test via Python script: `resume_agent.invoke({raw_resume, job_description})`

**Deliverable:** CLI invocation produces a tailored resume with ATS report.

---

### Phase 2 — Cover Letter + Full Agent Panel
> Goal: Add remaining agents and complete the pipeline

- [ ] Add `write_cover_letter` node (Resume Writer persona)
- [ ] Add `recruiter_review` node + prompt
- [ ] Add `hiring_manager_review` node + prompt
- [ ] Add `score_output` node to aggregate `final_score`
- [ ] Test full 7-node pipeline end-to-end

**Deliverable:** Full output — tailored resume, cover letter, all agent feedback, final score.

---

### Phase 3 — FastAPI + Basic UI
> Goal: Expose the agent via HTTP and add a simple web interface

- [x] Create `main.py` with `POST /tailor` (sync)
- [x] Add `POST /tailor-stream` SSE endpoint for per-node progress
- [x] Build `static/index.html` — paste resume + JD, display results
- [x] Run with `uvicorn main:app --reload`

**Deliverable:** Web UI where you can paste a resume + JD and see results in real time.

---

### Phase 4 — Polish & Enhancements
> Goal: Make it production-ready and more useful

- [ ] Accept resume as file upload (`.pdf`, `.docx`, `.md`)
- [x] Add history persistence (SQLite) to track past tailoring runs
- [x] Show diff between original and tailored resume
- [ ] Add retry logic if LLM returns malformed output
- [ ] Support multiple JD profiles saved and reused

**Deliverable:** A polished, reusable tool for job application prep.

---

## Node Prompts (prompts.py)

Each agent has a distinct system prompt persona:

- **ATS Simulator**: _"You are an ATS parser. Extract keywords from the JD and score the resume on keyword coverage. Be strict and mechanical."_
- **Resume Writer**: _"You are an expert resume writer. Rewrite the resume bullets to naturally incorporate missing keywords without fabricating experience."_
- **Recruiter**: _"You are a recruiter doing a 30-second scan. Assess readability, format, and whether this candidate warrants a call."_
- **Hiring Manager**: _"You are a hiring manager. Evaluate technical depth and culture fit against the JD. Score 0–100 with a brief rationale."_
