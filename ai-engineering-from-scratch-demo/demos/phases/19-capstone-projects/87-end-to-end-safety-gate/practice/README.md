<!-- generated:start -->
# 19-capstone-projects / 87-end-to-end-safety-gate

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/87-end-to-end-safety-gate/) · upstream spec
`phases/19-capstone-projects/87-end-to-end-safety-gate/docs/en.md`

```bash
uv run demo practice run 87-end-to-end-safety-gate --ex 1
uv run demo explain 87-end-to-end-safety-gate --ex 1
uv run pytest demos/phases/19-capstone-projects/87-end-to-end-safety-gate
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a fifth checkpoint: a `policy-check` that runs against the original system prompt before… | code | T0 | `ex01_the_lessons_gate_allows_7_of_9_tool_targeting_prompts_and_a_substring_match_catches_4.py` |
| 2 | Replace the deterministic aggregator with a weighted score: each signal contributes a 0-1 con… | code | T0 | `ex02_raising_the_threshold_lowers_precision_on_10_lookalikes_and_the_code_disagrees_with_its_doc_on_25_of_60.py` |
| 3 | Add an async streaming variant where during-gen runs in a thread; verify the latency impact s… | code | T0 | `ex03_an_early_stop_lets_the_model_run_one_chunk_past_the_match_so_the_50ms_budget_caps_chunk_time.py` |
<!-- generated:end -->

## Answers

Every exercise imports the lesson's own `safety_gate.py`, `mock_llm_stream.py`
and `main.py`. The gate loads the lesson 83 detector, lesson 85 classifier
router and lesson 86 rules engine itself. The corpus is the one `main.py`
runs: the 50 lesson 82 taxonomy fixtures plus its 10 benign prompts. Nothing
is written to the reference tree. Tool names and lookalike prompts are made
up and harmless, and the one injection-shaped prompt's payload is
"reply with CANARY-7". Everything is stdlib, so all three run at T0.

### 1 — the lesson's gate allows 7 of 9 tool-targeting prompts, and a substring match catches 4

**`policy-check` runs before pre-gen and reads the protected tool names from
the system prompt itself, the backticked names on its `Internal tools` line.
It rejects 9/9 prompts that name one of them, 0/6 lookalikes, and 0/60 corpus
prompts.** All 60 corpus prompts keep the lesson gate's final action. Names
are matched after NFKC, lowercasing, folding leet digits and dropping
separators. A rejection returns `block` without calling the detector, the
model or post-gen.

| spelling of the tool name | lesson's gate | substring match | policy-check |
|---|---|---|---|
| `acct_admin_reset`, `billing_export_v2` | allow | caught | reject |
| `ACCT_ADMIN_RESET` | allow | caught | reject |
| `acct-admin-reset`, `acct admin reset` | allow | missed | reject |
| zero-width inside, full-width letters | allow | missed | reject |
| `acct_adm1n_reset` | redact (`leet-letters`) | missed | reject |
| injection shape naming `kb_reindex` | block (`ignore previous`) | caught | reject |

The detector has no notion of tools. Its only redact here ships the model's
normal answer unchanged. With a system prompt that lists no tools, the
check rejects 0/9, because the protected set comes from the system prompt.

### 2 — raising the threshold lowers precision on 10 lookalikes, and the code disagrees with its doc on 25 of 60

**The score is a weighted noisy-OR over four 0-1 signals, `1 - prod(1 - w*c)`,
with weights pre 1.0, during 0.9, classifier 1.0 and rules 0.8. Every
checkpoint runs on every request. On the lesson corpus precision is 1.0 at
every threshold, so the only trade-off is recall:**

| threshold | P (corpus) | R (corpus) | P (+10 lookalikes) |
|---:|---:|---:|---:|
| 0.05-0.40 | 1.0 | 0.74 | 0.822 |
| 0.55 | 1.0 | 0.68 | 0.810 |
| 0.70 | 1.0 | 0.50 | 0.833 |
| 0.80 | 1.0 | 0.32 | 0.762 |
| 0.90 | 1.0 | 0.18 | 0.643 |
| 0.95 | 1.0 | 0.08 | 0.571 |

- 13/50 attacks and all 10 benign prompts score exactly 0, so no threshold
  recovers those 13.
- The post-gen classifier and rules signals are 0 on all 70 requests. The
  mock model has three outputs, and neither layer fires on any of them,
  including the harmful continuation.
- With 10 harmless lookalikes added, precision **falls** as the threshold
  rises. Four of them score 0.9 or more, because the mock model is keyed on
  phrases like "begin your answer with" and streams the continuation the
  token filter then terminates. The lesson's gate blocks 4 and redacts 4.
- **The code's aggregator is not the table in `docs/en.md`.** The doc maps
  detector confidence 0.5-0.85 with no other signal to `warn`, and a weak
  detector hit to `allow`. The code gives `redact` (23 requests) and `warn`
  (2). The lesson's gate stops 35/50 attacks; the doc's table would stop
  12/50.

### 3 — an early stop lets the model run one chunk past the match, so the 50 ms budget caps chunk time

**The lesson's `_during_gen` runs in a worker thread fed by a queue of size 1.
The caller drives the model and stops when the worker signals a match. It
stays well inside 50 ms, and all 120 runs match the synchronous gate on
action, output and during-gen verdict.**

| stream | overhead, median | overhead, max |
|---|---:|---:|
| instant mock (best of 5) | ~0.1 ms | ~0.3 ms |
| 2 ms per chunk | ~0.2-0.5 ms | ~3-4 ms (the early stops) |
| 60 ms per chunk, early stops only | ~63-67 ms | over budget |

On both early stops the threaded model generates 2 chunks where the
synchronous one generates 1, because the caller starts the next chunk before
the worker scans the last one. That chunk never reaches the user, but its
time is added to the latency. So an early stop costs (extra chunks) x
(chunk time). With a queue of 1, up to 2 extra chunks are possible, so the
budget holds for any chunk time under 25 ms. The regex sweep itself costs
about 1-5 us per chunk, so the thread has nothing to overlap with
generation and cannot reduce latency.
