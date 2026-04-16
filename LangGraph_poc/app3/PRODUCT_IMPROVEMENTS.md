# Product Improvement Review — Resume Tailor AI

*Reviewed: 2026-04-16*

---

## What's Solid

- Multi-agent pipeline with real-time SSE streaming is genuinely impressive
- History + status tracking (Draft → Offer) is the right idea
- PDF export with style/page options is a differentiator
- Rate limiting and structured logging show production awareness

---

## Highest-Impact Improvements

### 1. No Job Title / Company Stored in History
The biggest UX problem. The history table shows a truncated job description snippet but no **company name or job title** column. Users managing 10+ applications can't tell them apart at a glance.

**Fix:** Store a `company_name` + `job_title` field — extract it from the JD via the parse agent, or let users type it before submitting. This is the single most disorienting thing in the current UX.

---

### 2. No Way to Re-Tailor or Edit an Existing Record
Once a result is in history, it's frozen. Users can view it but can't go back and re-run with a tweak (e.g. change materials, re-submit to get a better score).

**Fix:** A **"Re-tailor"** button on the history drawer that pre-fills the inputs on the main page would close this loop.

---

### 3. Score Means Nothing Without Context
The ATS score (0–100) is shown everywhere but there's no explanation of what 70 means vs 85 vs 50. Is 70 good? Does it depend on the job level?

**Fix:** Add a benchmark label (e.g. "Good — above 70% of similar applications") or at minimum a tier label (Weak / Competitive / Strong) next to the number.

---

### 4. Cover Letter Is Completely Generic
The cover letter tab exists but users have no way to give it a **tone** (formal, conversational, concise), a **target length**, or to specify what to emphasize. These are the exact things people spend time manually editing.

**Fix:** A small "cover letter intent" field before tailoring — tone, length preference, what to emphasize — would dramatically improve output quality.

---

### 5. No Notes Field Per Application
The status dropdown (Draft → Offer) is useful but users need a free-text **notes** field per record — interview date, recruiter name, next step.

**Fix:** Without it, users will track the rest in a spreadsheet anyway, which defeats the purpose of the history page. Add a notes column to the DB and an editable textarea in the history drawer.

---

### 6. Materials Field Is Buried and Unclear
The "Additional Materials" accordion is a great idea but users don't know what to put there. In practice the most useful input is: *"What do you specifically want to highlight for this role?"*

**Fix:** Rename it to something more directive and add 2–3 example prompt chips (e.g. "Emphasize leadership", "Highlight cost savings", "Mention remote work experience") to guide users toward useful input.

---

### 7. No Empty-State Onboarding
First-time visitors see two blank textareas with no guidance on what a good input looks like.

**Fix:** A **"Load sample"** button for both resume and JD would let people try the product in 5 seconds without needing their own content ready.

---

## Smaller but Quick Wins

| Issue | Fix |
|---|---|
| History page has no search/filter | Add a search box filtering by JD text or status |
| "Upgrade to Pro" in sidebar goes nowhere | Remove it or wire it to something real |
| PDF download requires completing the full pipeline | Let users download PDFs directly from history records |
| ATS "missing keywords" shown as flat red tags | Sort by importance or add a "critical / nice-to-have" label |
| No mobile sidebar | The sidebar hides on mobile but there's no hamburger menu replacement |

---

## Suggested Priority Order

| Priority | Item | Effort |
|---|---|---|
| 🔴 P0 | Store company name + job title in history | Low |
| 🔴 P0 | Notes field per application | Low |
| 🟠 P1 | Re-tailor from history | Medium |
| 🟠 P1 | Score context / tier labels | Low |
| 🟡 P2 | Cover letter intent options | Medium |
| 🟡 P2 | Materials field UX / example chips | Low |
| 🟢 P3 | Sample data for onboarding | Low |
| 🟢 P3 | History search/filter | Medium |
| 🟢 P3 | Mobile sidebar hamburger | Medium |
