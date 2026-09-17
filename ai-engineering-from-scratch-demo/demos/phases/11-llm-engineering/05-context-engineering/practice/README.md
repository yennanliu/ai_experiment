<!-- generated:start -->
# 11-llm-engineering / 05-context-engineering

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/05-context-engineering/) · upstream spec
`phases/11-llm-engineering/05-context-engineering/docs/en.md`

```bash
uv run demo practice run 05-context-engineering --ex 1
uv run demo explain 05-context-engineering --ex 1
uv run pytest demos/phases/11-llm-engineering/05-context-engineering
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a "token waste detector" to the ContextBudget class. It should flag components using more… | code | T0 | `ex01_the_detector_cannot_fire_because_the_counter_undercounts.py` |
| 2 | Implement semantic deduplication for retrieved context. If two retrieved documents are more t… | code | T0 | `ex02_nothing_is_duplicated_and_the_waste_is_the_word_the.py` |
| 3 | Build a "context replay" tool. Given a conversation transcript, replay it through the Context… | code | T0 | `ex03_the_budget_truncates_fourteen_turns_before_the_manager_compresses.py` |
| 4 | Implement a priority-based tool selector. Instead of binary include/exclude, assign each tool… | code | T0 | `ex04_twenty_and_fifty_tools_do_not_exist_and_ten_are_free.py` |
| 5 | Build a multi-strategy context compressor. Implement three compression strategies (truncation… | code | T0 | `ex05_the_lessons_summariser_is_a_character_truncation.py` |
<!-- generated:end -->

## Answers

The lesson is numpy plus stdlib, so all five exercises are **T0** on the `math`
group and run in CI. The shape they share: every number the lesson reports goes
through `count_tokens`, which is `int(len(text.split()) * 1.3)` — and the four
places that count tokens (the budget's caps, the budget's allocations, the
conversation manager's gate, the tool registry's declared cost) do not agree
with each other. Every exercise here ends up being about which of those four
numbers you happened to measure.

### 1 — the detector cannot fire, because the tool block is counted as six tokens

| query | system | tools | retrieved | query | util |
|---|---:|---:|---:|---:|---:|
| how do I fix the failing test | 41 | **6** | 80 | 9 | 0.106% |
| schedule a meeting for tuesday | 41 | **1** | 53 | 6 | 0.079% |
| what is the api rate limit | 41 | **1** | 94 | 7 | 0.112% |
| query the database for user stats | 41 | **2** | 96 | 7 | 0.114% |
| send an email to the team | 41 | **2** | 83 | 7 | 0.104% |

**ANSWER: it never fires.** The largest component anywhere is 96 tokens against
an available budget of **124,000**. Nothing is within two orders of magnitude of
30%.

**FINDING: read as a share of tokens *used*, it fires every time.**
`retrieved_context` is 52%–66% of used tokens on all five queries and nothing
else passes 30%. The detector's answer is decided by a denominator the exercise
does not name.

**MECHANISM: the tool block is priced as a comma-separated list of names.**

| query | booked by `count_tokens` | declared by `TOOL_REGISTRY` |
|---|---:|---:|
| code | 6 | **710** |
| calendar | 1 | **180** |
| research | 1 | **140** |
| data | 2 | **360** |
| email | 2 | **360** |

Up to a **180×** under-count, on exactly the component whose compression
strategy ("prune tools") the exercise names first. Price it at the registry's
own numbers and `tools` is flagged on all five queries, displacing
`retrieved_context` on four.

**CONTROL: run the detector on what `assemble` reserves, not on what it books.**
The per-component caps it passes to `allocate` are 1,000 / 2,000 / 3,000 /
5,000 / 500 = **11,500 tokens** — 9.3% of a 128k available budget, and **274%**
of an 8,192-token one, where they cannot all be honoured. Against that window
the detector flags `conversation_history`, `retrieved_context` and `tools` at
once. Those are numbers the code already believes.

### 2 — nothing is duplicated, and the recoverable waste is the word "the"

**ANSWER: 0 tokens recovered, on every query, under both measures.**

```text
highest Jaccard between any two retrieved documents   0.176
highest count-cosine                                  0.334
the exercise's threshold                              0.800
```

**FINDING: most of what is retrieved matches on a stopword.** For "how do I fix
the failing test", six documents are retrieved and five of them score *exactly*
0.143 — one shared word out of seven. The shared words:

```text
['the'] ['the'] ['the'] ['test'] ['the'] ['the']
```

**MECHANISM:** `score_relevance` is
`len(query_words & doc_words) / len(query_words)` — no stopword list, no
document-length normalisation — and the threshold is 0.05. One shared token out
of a twenty-word query clears it.

**FINDING: the recoverable budget is 378 tokens, not 0.**

| query | retrieved | after dropping stopword-only matches |
|---|---:|---:|
| how do I fix the failing test | 80 | **13** |
| schedule a meeting for tuesday | 53 | **0** |
| what is the api rate limit | 94 | **15** |
| query the database for user stats | 96 | **0** |
| send an email to the team | 83 | **0** |
| **total** | **406** | **28** |

93% of the retrieved context, against deduplication's 0%.

**CONTROL: the filter is correct; the corpus was clean.** Adding each retrieved
document back one word short — what an overlapping chunker produces — puts every
pair over the threshold, and the same dedup recovers exactly the added copies.

### 3 — the budget truncates 14 turns before the manager compresses

```text
turn 133   allocate("conversation_history", …, max_tokens=5000) starts truncating
           -> the component books 4,999 tokens, silently, every turn after
turn 147   _compress_if_needed fires for the first time
           -> 292 live turns, 1 summary
```

**ANSWER: 147 — and also 133.** The exercise's question has two answers 14 turns
apart, and `stats()` only shows the later one.

**MECHANISM: the manager and the budget count different strings.** Just after the
compression pass:

```text
sum(count_tokens(turn["content"]))   4,980   <- the gate; back under its 5,000 trigger
count_tokens(get_context())          5,561   <- what it hands the budget (112%)
```

The difference is `[Recent Conversation]` plus a `role: ` prefix per turn. The
manager believes it is finished and the budget truncates anyway.

**FINDING: one pass barely moves the number.** The history allocation over the
compression turn and the next three is `[4999, 4999, 4999, 4999]` — pinned at the
cap.

**CONTROL: gate on `token_count()`** and the first compression moves to turn
**133**, the same turn the silent truncation starts. One counter, one answer.

### 4 — 20 and 50 tools do not exist, and all 10 are free

**ANSWER: two of the four comparison points are unreachable.** `TOOL_REGISTRY`
holds **10** tools, and there is no larger registry anywhere in the lesson.

**FINDING: the ten-tool point is free.** All ten declare **1,580** tokens against
the **2,000**-token budget `assemble` passes, so the accumulator in
`select_tools` is dead code:

| query | intents | tools | declared tokens |
|---|---|---:|---:|
| how do I fix the failing test | `code` | 5 | 710 |
| schedule a meeting for tuesday | `calendar` | 1 | 180 |
| what is the api rate limit | `research` | 1 | 140 |
| query the database for user stats | `data` | 2 | 360 |
| send an email to the team | `email` | 2 | 360 |

**FINDING: the priority selector picks the same set at every budget that
matters.** Ordering by relevance instead of by insertion order gives the same
*set* on all five queries and a different *order* on two. The largest budget at
which the sets differ is **700** tokens.

**FINDING: `classify_intent` falls back to `["code"]`.** "Who won the 1998 World
Cup?" matches no keyword in any category, so the selector ships `read_file`,
`run_command`, `search_code`, `write_file` and `query_database` — 710 tokens of
file, search and shell access for a football question. The binary
include/exclude the exercise wants to replace is not the problem; the default is.

**ANSWER: task performance has no observable.** No task, no model. The only
thing either selector decides is the set.

### 5 — the lesson's summariser is a character truncation

20 documents built by pairing the knowledge-base sentences; retention measured
over the 20 of 100 document–query pairs whose document holds its answer.

| strategy, matched 20-token budget | ratio | retained | answer in first fragment | answer later |
|---|---:|---:|---:|---:|
| **extraction** | **0.518** | **18 / 20** | 10 / 10 | 8 / 10 |
| truncation | 0.717 | 12 / 20 | 10 / 10 | **2 / 10** |
| summarization | 0.891 | 16 / 20 | 10 / 10 | 6 / 10 |

**ANSWER: extraction wins on both axes.** There is no trade-off to plot at a
matched budget: the strategy that reads the query is cheaper *and* keeps more.

**FINDING: only extraction reads the query.** Documents whose compressed form
changes when the query changes, out of 20:

```text
truncation 0      summarization 0      extraction 8
```

For two of the three strategies, "does the compressed version still contain the
answer" is settled before the compressor is called.

**MECHANISM: `_summarize_turns` is `content[:100] + "..."`,** 16 words here, and
it prepends `Previous: doc: ` — three words that are not in the document. The
"summarization" row measures a change of unit plus invented text.

**CONTROL: at natural settings the trade-off appears, and it is the budget's.**
Truncating to half the words gives the **best** ratio of the three (0.479) and
the **worst** retention (10 / 20). Same technique, different budget, opposite
verdict — so what the exercise's plot shows is the budgets chosen. (The
documents split into 2, 3 or 4 fragments, because the splitter breaks on every
decimal: `Python 3.12`, `PostgreSQL 16`, `80%`.)
