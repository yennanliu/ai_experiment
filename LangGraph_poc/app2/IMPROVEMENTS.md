# Improvement Areas — AI Email Reply Assistant

## 1. Persistence: History Lost on Restart

**Problem:** `history` in `main.py` is a plain Python list in memory. Every server restart wipes all draft history.

**Fix:** Persist to SQLite (via SQLModel or raw sqlite3) or a simple JSON file. This is critical for any production use where operators need a record of processed emails.

---

## 2. Pipeline Visualization is Fake / Misleading

**Problem:** `pipelineStart()` in `index.html` uses hardcoded 2.6s timers to simulate node execution. The UI shows nodes "running" sequentially, but this is pure theater — the backend runs synchronously and returns only after all 6 nodes complete. The visual doesn't reflect actual execution state.

**Fix:** Either accept it as decoration and document it clearly, or switch to a streaming endpoint (FastAPI `StreamingResponse` + SSE) that emits real per-node events as the graph executes.

---

## 3. No Request Validation or Size Limits

**Problem:** `POST /generate-draft` accepts any string with no length cap. A very large email body will be sent verbatim to GPT-4o across all 3 prompts (classify, generate_draft, build_checklist), inflating cost and risk of hitting token limits.

**Fix:** Add a `max_length` validator on `EmailRequest.email` (e.g., 4000 chars) and return a 422 with a clear message.

---

## 4. Four GPT-4o Calls Per Request (Cost)

**Problem:** Each `POST /generate-draft` makes 4 serial GPT-4o calls: `classify`, `generate_draft`, `build_checklist`, `score_draft`. At current pricing this is expensive at any volume.

**Options:**
- Use `gpt-4o-mini` for `classify` and `build_checklist` — these are classification/extraction tasks that don't require full GPT-4o capability.
- Combine `build_checklist` and `score_draft` into one call since they both read the same state.

---

## 5. XSS Risk in History Table

**Problem:** In `history.html`, the `openModal()` function is called via an inline `onclick='openModal(${safe})'` where `safe = JSON.stringify(r).replace(/'/g, '&#39;')`. The `draft_reply` field is LLM-generated text and could contain characters (backticks, template literals, etc.) that break out of the attribute context in some browsers.

**Fix:** Store records by ID in a JS Map and pass only `r.id` to `openModal(id)`, then look up the record client-side. Eliminates the inline JSON embedding entirely.

---

## 6. No Error Handling for Missing `OPENAI_API_KEY`

**Problem:** If `.env` is missing or `OPENAI_API_KEY` is not set, the server starts fine but every request fails with an unhandled `openai.AuthenticationError` at runtime. There's no startup check.

**Fix:** Add a startup event in `main.py` that validates `os.environ.get("OPENAI_API_KEY")` and raises a clear error if absent.

---

## 7. `PATCH /history/{record_id}/status` Accepts Arbitrary Status Values

**Problem:** The endpoint accepts `body: dict` with no validation — any string can be set as status. The UI enforces `["Drafted", "Reviewed", "Sent", "Rejected"]` but the API does not.

**Fix:** Use a `StatusUpdate` Pydantic model with a `Literal` or `Enum` constraint.

---

## 8. RAG Collection Has No Deduplication Guard at Seed Time

**Problem:** `seed_rag.py` uses `upsert` (good), but if called without `--reset` on a collection that already has entries, it silently upserts. The `count()` print before/after can mislead (count won't change on re-upsert). This is minor but confusing during development.

**Fix:** Log whether each example was inserted or was already present, or add a `--dry-run` flag.

---

## 9. Pipeline Sublabel Reset Bug

**Problem:** In `pipelineReset()` in `index.html`, the `retrieve` node sublabel is reset via a dict lookup that is missing the `'retrieve'` key — so it gets `undefined` and the sublabel becomes the string `"undefined"` on reset.

```js
// 'retrieve' is missing from this dict in pipelineReset():
{
  classify:  'Detect email type',
  template:  'Select reply template',
  draft:     'GPT-4o fills template',
  checklist: 'Flag missing info & risks',
  score:     'Confidence assessment',
}[n.id]
```

**Fix:** Add `retrieve: 'ChromaDB RAG lookup'` to the dict.

---

## 10. CSS Duplication Across HTML Files

**Problem:** Nav styles, badge styles, and button styles are copy-pasted identically across `index.html`, `history.html`, and `templates.html`. Any style change must be made three times.

**Fix:** Extract shared styles into `static/style.css` and include it via `<link>`.

---

## 11. No Automated Tests

**Problem:** `test_mock.py` is a manual smoke test that requires the server to be running. There are no unit tests for the node functions (`classify`, `build_checklist`, `score_draft`) or integration tests that mock the OpenAI client.

**Fix:** Add `pytest` + `unittest.mock` tests for each node function. The nodes are pure functions over `AgentState` dicts, making them straightforward to test by mocking `get_client()`.
