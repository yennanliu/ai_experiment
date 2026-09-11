<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 29-dialogue-state-tracking

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/29-dialogue-state-tracking/) · upstream spec
`phases/05-nlp-foundations-to-advanced/29-dialogue-state-tracking/docs/en.md`

```bash
uv run demo practice run 29-dialogue-state-tracking --ex 1
uv run demo explain 29-dialogue-state-tracking --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/29-dialogue-state-tracking
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Build the rule-based state tracker in `code/main.py` for 3 slots (cuisine, area, price)… | code | T0 | `ex01_main_measures_dialogues_not_turns.py` |
| 2 | Medium. Same dataset with Instructor + Pydantic + a small LLM. Compare JGA. Inspect the harde… | code | T0 | `ex02_regeneration_reverts_the_corrections.py` |
| 3 | Hard. Implement both and route: rule-based primary, LLM fallback when rule-based emits <2 slo… | code | T1 | `ex03_there_is_no_confidence_to_route_on.py` |
<!-- generated:end -->

## Answers

The lesson's tracker scores 3 of 3 on the three dialogues it ships with. Every
finding here comes from asking what that number is, and what it is a number
*of*: the metric `main()` prints is not the metric the doc defines, the pipeline
the doc writes down is not the one it ships, and the two components the exercises
ask you to compare — an LLM extractor and a confidence-gated router — cannot be
built from anything in `code/main.py`.

Exercises 1 and 2 run at **T0** — `code/main.py` imports only `re`. Exercise 3 is
**T1** because it times the rule arm.

### 1 — main() measures dialogues, not turns

Ten invented dialogues, 32 turns, each turn hand-labelled with the slots it
changes:

| slot set | turn-level JGA | dialogue-final JGA |
|---|---:|---:|
| cuisine, area, price (the exercise's) | **29 / 32** (0.9062) | **8 / 10** |
| all four shipped slots | 28 / 32 (0.8750) | 8 / 10 |
| `main()`'s own three dialogues | — | **3 / 3** |

**MECHANISM: the exercise names three slots, the code ships four.**
`SLOT_EXTRACTORS` holds `cuisine, area, price, people`. JGA is all-or-nothing, so
dropping a slot can only raise it, and it does — by exactly one turn.

**FINDING: the shipped metric is not the one the doc defines.** The doc calls JGA
"the fraction of *turns* where every slot is correct" and Step 4 zips per-turn
state lists. `main()` compares once per dialogue against the final state only, on
gold carrying **3 labels for 11 turns** — per-turn JGA is not computable from the
data the lesson ships.

**FINDING: at ten dialogues the doc's own target is not on the grid.**
Dialogue-final JGA over 10 dialogues moves in steps of 0.10. The doc says to beat
MultiWOZ's ~0.83; the reachable values straddle it, and the measured **0.80 is
one dialogue below 0.90**. The whole comparison turns on one dialogue flipping.
At `main()`'s n=3 the step is 33.3 points.

**FINDING: the clear-on-negation invariant never runs in the lesson's demo.** It
fires **0 times across `main()`'s 11 turns** — in "Never mind the cuisine, any
food is fine" `extract_cuisine` already returns `any`, and `continue` skips the
branch — against 2 times across the 32 turns here.

**CONTROL: `is_correction` is defined and never called.** `update_state`'s source
never mentions it, though 4 of 32 turns match a cue. Corrections survive only
because extraction overwrites unconditionally — so the doc's "overwrite the
last-updated slot" rule has nothing to hook into.

### 2 — Regeneration reverts the corrections

`instructor`, `pydantic`, `openai`, `transformers` and `torch` are all absent, so
the LLM arm cannot be built. What can be built is the substitute the doc itself
prescribes — regenerate the whole state from history — driven by the lesson's own
extractors:

| arm | turn-level JGA | dialogue-final JGA |
|---|---:|---:|
| incremental `update_state` loop | **28 / 32** | **8 / 10** |
| regenerate from history (`AREA_WORDS` pinned ascending) | 20 / 32 | 4 / 10 |
| regenerate from history (pinned descending) | 22 / 32 | 5 / 10 |

**ANSWER: the doc's own production pattern is much worse, not better.** "Always
let the LLM regenerate the whole state from history rather than incrementally
updating — this naturally handles corrections" is backwards for this extractor.

**MECHANISM: the extractors return the first canonical key, not the last
mention.** `PRICE_WORDS` is a dict ordered `cheap, moderate, expensive`, so on the
concatenated history of "…moderate price. / On second thought, make it
expensive." `extract_price` returns **`moderate`**. The scan is over its own key
order; the user's second thought sits behind the first one in that order.

**FINDING: every dialogue the arms disagree on holds a correction or a negation.**
Dialogues `[2, 4, 5, 9]` diverge, all inside the cue-carrying set `[2, 4, 5, 7,
9]`. The seven the regeneration arm gets right are the ones where nothing was
ever taken back.

**FINDING: the regeneration arm's score is not reproducible.** `AREA_WORDS` is a
`set`, and `extract_area` returns its first matching member, so a history naming
two areas resolves by set-iteration order — a function of `PYTHONHASHSEED`.
**20 / 32 one way, 22 / 32 the other.** The incremental loop never sees two areas
in one utterance and is unaffected.

**CONTROL: the pipeline the doc writes down is not the one it ships.** Its Step 2,
3 and 4 blocks call `is_negated`, `NEGATION_CLEARS`, `joint_goal_accuracy` and
`render` — `hasattr` is False for all four. And `RestaurantState` declares **5
slots against `SLOT_EXTRACTORS`' 4**, adding `day`, which no extractor can fill,
so "compare JGA" would compare two different metrics.

### 3 — There is no confidence to route on

`extract_cuisine('…', confidence=True)` raises `TypeError: extract_cuisine() got
an unexpected keyword argument 'confidence'`; `update_state` returns a plain
`dict` of values; no name in the module contains `conf` or `score`. So "emits <2
slots with confidence" can only mean "emits <2 slots" — and that has two readings:

| reading of "<2 slots" | escalations | combined turn-JGA (oracle fallback) | rule-only |
|---|---:|---:|---:|
| slots **this turn's extractors** emit | **21 / 32** (0.6562) | **32 / 32** | 28 / 32 |
| slots **the state** holds after the turn | **0 / 32** (0.0000) | 28 / 32 | 28 / 32 |

The fallback is modelled as an oracle that writes the gold state — the upper
bound on any model, which is the honest way to price a router you cannot run.

**FINDING: under the state reading the combined JGA is the rule-only JGA,
exactly.** Every dialogue's opening utterance already fills two slots, so the
fallback is never reached. The cheap reading of the exercise's own sentence buys
precisely nothing, and the exercise never asks for the escalation rate that would
have shown it.

**FINDING: the trigger that does fire is nearly uninformative.** **18 of the 21**
escalated turns were already correct — a trigger precision of **0.1429**. Slot
count measures how much the user said, not whether the extractor understood it.

**CONTROL: the lowest-confidence bucket is the one that is mostly right.** Of the
**5 turns that emit nothing at all, only 1 is wrong**. "Sounds good.", "Yes, that
one." and the two negations are turns where emitting nothing is the correct
answer — and they are exactly what the trigger ranks as least confident.

**FINDING: the combined JGA measures the oracle, and only one arm has a price.**
All 32 turns are right once the fallback is perfect, and 1 of the repairs lands
on a turn the router never escalated, because a repaired state carries forward.
`update_state` runs at ~**4 µs/turn**, so cost per turn is that plus 0.6562 of an
LLM call: the rule-based primary sheds at most **34.4%** of the model traffic
under one reading, and 100% of it — along with all of the benefit — under the
other.
