<!-- generated:start -->
# 13-tools-and-protocols / 21-llm-routing-layer

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/21-llm-routing-layer/) · upstream spec
`phases/13-tools-and-protocols/21-llm-routing-layer/docs/en.md`

```bash
uv run demo practice run 21-llm-routing-layer --ex 1
uv run demo explain 21-llm-routing-layer --ex 1
uv run pytest demos/phases/13-tools-and-protocols/21-llm-routing-layer
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Trigger the outage scenario; confirm fallback lands on the second provide… | code | T0 | `ex01_the_failed_attempt_is_free_because_the_model_charges_the_winner.py` |
| 2 | Add semantic caching: SHA256 of the prompt is a lookup key; cache hits return instantly. Meas… | code | T0 | `ex02_a_sha256_cache_is_exact_match_and_the_key_has_to_choose_a_side.py` |
| 3 | Add a prompt classifier that routes "code ..." prompts to an alias favoring intelligence and… | code | T0 | `ex03_a_prefix_is_not_a_classifier_and_the_default_is_the_expensive_one.py` |
| 4 | Design per-team budgets: each team has a monthly spend cap; gateway refuses requests once cap… | code | T0 | `ex04_the_cost_is_known_after_the_call_so_a_hard_cap_cannot_exist.py` |
| 5 | Read LiteLLM, OpenRouter, and Portkey docs side by side. Name the one feature each ships that… | code | T0 | `ex05_the_three_differentiators_are_the_three_exercises_before_this_one.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five ship code — including exercise 5,
which asks you to compare three vendors' docs. That is not checkable from
inside this repo, so it is answered as the question the code *can* answer:
what a routing gateway needs that this one has no place for.

A gateway is a set of decisions taken around a provider call, and the
recurring finding is **the decision that the exercise does not mention**: a
classifier's default, a cache's key, a cap's granularity. Each is where the
money actually goes.

### 1 — the failed attempt is free because the model charges the winner

**ANSWER: the fallback lands on `anthropic/claude-sonnet`, priced at its own
rates.**

**FINDING: the failed attempt contributes zero, by construction.**
`provider_call` raises before returning usage, so there is nothing to price.
A provider that streamed tokens and then failed would bill for them, and
`Invocation` has no field in which that charge could appear.

**FINDING: a partial outage is invisible outside `attempts`.**
`except RuntimeError: continue` discards the exception; `error` is set only
when the chain is exhausted.

**FINDING: redaction happens once, before the chain**, so every provider sees
the same text — but `redacted` is a bool, so which pattern matched is lost.

### 2 — a SHA256 cache is exact-match, and the key has to choose a side

**ANSWER: the repeat costs 0; 3 hits and 2 misses over a 5-call workload.**

**FINDING: it is exact-match, not semantic.** Of **4** prompts a human would
call one question, the cache serves **0** and pays for **4**. SHA256 gives
byte equality with a nicer name.

**FINDING: the prompt alone is the wrong key.** Keyed on the prompt only, the
`fast` alias is served `gpt-4o`'s answer. The alias chooses the chain, so it
belongs in the key.

**FINDING: hashing before or after redaction is a real choice.** A
raw-prompt digest is invertible over a 1000-candidate SSN space; a
redacted-prompt digest makes two customers collide on `[REDACTED]`.

### 3 — a prefix is not a classifier, and the default is the expensive one

**ANSWER: `code` → `smart`, `summarize` → `fast`, unmatched → `fast`.**

**FINDING: defaulting to `smart` is a cost decision taken by omission.**
`5.0/15.0` against `0.15/0.60` — **33x** on input. Nothing in the exercise
says which way to fall, and it is the largest cost decision in the
classifier.

**FINDING: prefix matching misses the phrasings people use.**

| rule | summary prompts caught (of 5) |
|---|---:|
| `startswith("summarize")` | 1 |
| lowercased substring | 4 |

The British spelling needs its own entry either way.

**FINDING: a misroute is silent.** The `Invocation` records the alias and no
rule, so a deliberate `fast` and a defaulted one are identical.

### 4 — the cost is known after the call, so a hard cap cannot exist

**ANSWER: per-request admission against a running total — 2 admitted, 4
refused, overshooting by at most one call.**

**FINDING: the cap cannot be checked before the cost is known.** `route`
prices from `resp["usage"]`. Reserving an upper-bound estimate never
overshoots and refuses requests that would have fit.

**FINDING: the two granularities overshoot differently, and the window is
worse.** The exercise's "pick one" is a choice between a bounded error and
one the window size sets.

**FINDING: the refusal costs nothing and has nowhere to be recorded.** No
`Invocation` exists at all, so refused calls are absent from the only
structure that could count them.

### 5 — the three differentiators are the three exercises before this one

**ANSWER: caching, classification and budgeting — and `Invocation` has a
field for none of them.** Its **9** fields cannot say a response was cached,
which rule chose the alias, or which team is billed.

**FINDING: the module's whole surface is routing and pricing.** `PRICES`,
`OUTAGE`, `ROUTES`, `PII_PATTERNS` — cost, health, fallback, redaction. Each
exercise adds a component, not a setting.

**FINDING: the three land at three points before routing**, and redaction —
the one the lesson ships — sits *inside* it. One position of four is taken.

**FINDING: cost attribution is the field they all need.** The gateway has the
number and not the dimensions to group it by. Adding `cache_hit`, `rule` and
`team` makes all three reportable: the same change, three times.
