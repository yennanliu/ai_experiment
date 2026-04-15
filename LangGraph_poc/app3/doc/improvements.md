# app3 — Improvement Opportunities

## 1. Agent & AI Quality

### 1.1 Fragile string parsing for LLM outputs
`nodes.py` parses ATS/recruiter/hiring-manager responses with manual `line.startswith("SCORE:")` splits. Any minor deviation in model output silently produces zero scores or empty data.

**Fix:** Use OpenAI structured outputs (JSON mode / `response_format={"type":"json_object"}`) or Pydantic tool-calling so outputs are schema-validated at the source.

---

### 1.2 All nodes run sequentially — no parallelism
`recruiter_review` and `hiring_manager_review` are fully independent but run one after the other, wasting ~2–4 s per request.

**Fix:** Use LangGraph's `fan-out / fan-in` pattern — branch both review nodes from `write_cover_letter`, then merge back into `score_output`.

---

### 1.3 No retry or fallback on LLM errors
A transient OpenAI error aborts the entire graph with a 500. The user loses all progress.

**Fix:** Wrap each node call with a retry decorator (e.g., `tenacity`) or add a conditional edge that retries failed nodes up to N times before surfacing the error.

---

### 1.4 Model hardcoded to `gpt-4o` everywhere
No way to swap model without touching source code. Costs ~5× more than `gpt-4o-mini` for simpler tasks like the ATS parse.

**Fix:** Use a config constant or env var `OPENAI_MODEL`. Consider `gpt-4o-mini` for `ats_simulate` and `score_output` (deterministic, low-complexity), and `gpt-4o` only for writing nodes.

---

### 1.5 `lru_cache` on OpenAI client is not async-safe
The `@lru_cache(maxsize=1)` on `get_client()` works under sync usage, but FastAPI runs in an async context and multiple concurrent requests could hit threading edge-cases. OpenAI's client is thread-safe, but the pattern is fragile.

**Fix:** Instantiate the client once at module level or in FastAPI's `lifespan` startup handler.

---

## 2. Backend / API

### 2.1 Blocking sync LLM calls inside async FastAPI routes
`/tailor` and `/tailor-stream` call `resume_agent.invoke()` / `.stream()` synchronously inside async route handlers. This blocks the event loop for the entire duration (~15–30 s), preventing other requests from being served.

**Fix:** Wrap blocking calls with `asyncio.to_thread()` or switch to `ainvoke` / `astream` (LangGraph supports async graphs).

---

### 2.2 No request authentication or rate limiting
Any caller can hit `/tailor` and consume OpenAI credits. No API key, session, or per-IP limit exists.

**Fix:** Add an `X-API-Key` header check (even a simple shared secret from `.env`) and use a middleware like `slowapi` for rate limiting.

---

### 2.3 Input size limits are too loose / inconsistently enforced
`resume` allows 8000 chars and `job_description` 4000 chars via Pydantic, but these are character limits, not token limits. A dense 8000-char resume can exceed context limits and cause silent truncation.

**Fix:** Add a rough token estimate check (chars / 4) and return a `422` with a clear message before invoking the LLM.

---

### 2.4 SQLite connection created per-call with `check_same_thread=False`
`_connect()` opens a new connection on every DB call. Under concurrent load this is inefficient and the `check_same_thread=False` suppresses a safety guard without a real connection pool backing it.

**Fix:** Use a connection pool (e.g., `aiosqlite` + a single shared connection, or switch to SQLAlchemy with a pool size of 5).

---

### 2.5 PDF export lacks font/Unicode support
ReportLab defaults (`Helvetica`) do not support non-Latin characters. A resume with Chinese, Korean, or accented characters will produce garbled or missing text.

**Fix:** Register a Unicode-capable font (e.g., embed a Google Noto TTF) via `pdfmetrics.registerFont`.

---

### 2.6 `@app.on_event("startup")` is deprecated
FastAPI deprecated `on_event` in favour of the `lifespan` context manager.

**Fix:**
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)
```

---

## 3. Data & Persistence

### 3.1 Raw resume and JD stored in plain text
Sensitive personal data (name, address, employment history) stored as plain text in SQLite with no encryption.

**Fix:** Either document that this is a local-only dev tool, or encrypt sensitive columns using SQLite's `sqlcipher` extension or application-level AES encryption before storing.

---

### 3.2 No delete / archive endpoint
Users can update status but cannot delete records. Old entries accumulate indefinitely.

**Fix:** Add `DELETE /history/{id}` and a soft-delete `archived` boolean column.

---

### 3.3 No pagination on `/history`
`get_all_records()` fetches every row with `SELECT *`. With hundreds of records this becomes a slow, large JSON payload.

**Fix:** Add `limit` / `offset` query params and return `{ total, items }`.

---

## 4. Frontend

### 4.1 All CSS/JS is inline in two ~600–800-line HTML files
No separation of concerns. Changing the button colour requires editing both files. Impossible to cache styles separately.

**Fix:** Extract shared styles to `static/style.css` and shared logic to `static/app.js`.

---

### 4.2 No loading skeleton / partial render during streaming
The SSE stream emits node completion events, but the UI only updates a progress bar. Users see a blank result area for 15–30 s.

**Fix:** Render a skeleton card immediately on submit and fill it section-by-section as each SSE `node` event arrives (e.g., show ATS score when `ats_simulate` fires, show resume draft when `rewrite_resume` fires).

---

### 4.3 No client-side input validation before submit
Users can submit an empty resume and get a confusing LLM error rather than an immediate inline validation message.

**Fix:** Validate non-empty + minimum length (e.g., 100 chars) on the client before firing the request.

---

### 4.4 Export buttons always enabled, even before results exist
Clicking "Export PDF" before a run completes sends an empty string to the endpoint.

**Fix:** Disable export buttons until `done` SSE event is received; re-enable with the result payload.

---

## 5. Developer Experience & Ops

### 5.1 No tests
Zero test coverage across agents, DB helpers, or API routes.

**Fix (priority order):**
1. Unit-test `score_output` and `ats_simulate` parsing logic with mocked LLM responses.
2. Integration-test `/tailor` with a `pytest-asyncio` + `httpx` client and a recorded OpenAI fixture.
3. Test `insert_record` / `update_status` against an in-memory SQLite DB.

---

### 5.2 No environment variable validation at startup
If `OPENAI_API_KEY` is missing, the current check raises a `RuntimeError` inside the startup event. Missing keys for future additions (e.g., `DATABASE_URL`) would only fail at first use.

**Fix:** Use `pydantic-settings` to declare and validate all env vars in a `Settings` model loaded once at startup.

---

### 5.3 No structured logging
`print()` / exception tracebacks go to stdout with no timestamp, request ID, or level. Hard to trace issues in production.

**Fix:** Configure `logging` with a JSON formatter (e.g., `python-json-logger`) and attach a request-ID middleware.

---

### 5.4 `pyproject.toml` likely missing dev dependencies
No `pytest`, `httpx`, `ruff`, or `mypy` declared as dev dependencies, making onboarding slower.

**Fix:** Add a `[tool.uv.dev-dependencies]` section with linting/testing tools.

---

## Priority Summary

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | Structured LLM outputs (1.1) | Medium | High — eliminates silent data corruption |
| 2 | Async LLM calls (2.1) | Low | High — fixes event-loop blocking |
| 3 | Parallel review nodes (1.2) | Low | Medium — ~40% latency reduction |
| 4 | Rate limiting + auth (2.2) | Low | High — prevents credit abuse |
| 5 | Unit tests for parsing (5.1) | Medium | High — catches regressions |
| 6 | Pagination on /history (3.3) | Low | Medium |
| 7 | Client-side validation (4.3) | Low | Medium — UX improvement |
| 8 | Streaming partial render (4.2) | Medium | Medium — perceived performance |
| 9 | Unicode PDF fonts (2.5) | Low | Medium — correctness for non-Latin résumés |
| 10 | Delete endpoint (3.2) | Low | Low |
