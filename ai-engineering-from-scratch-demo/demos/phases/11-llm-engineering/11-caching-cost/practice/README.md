<!-- generated:start -->
# 11-llm-engineering / 11-caching-cost

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/11-caching-cost/) · upstream spec
`phases/11-llm-engineering/11-caching-cost/docs/en.md`

```bash
uv run demo practice run 11-caching-cost --ex 1
uv run demo explain 11-caching-cost --ex 1
uv run pytest demos/phases/11-llm-engineering/11-caching-cost
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement LRU eviction for the semantic cache. Replace the oldest-first eviction with least-r… | code | T0 | `ex01_the_cache_holds_five_hundred_and_the_exercise_sends_a_hundred.py` |
| 2 | Build a cost projection tool. Given a log of API calls (the CostTracker logs), project the mo… | code | T0 | `ex02_every_log_entry_lands_in_the_same_microsecond.py` |
| 3 | Implement tiered semantic caching. Use two similarity thresholds: 0.98 for high-confidence hi… | code | T0 | `ex03_the_medium_band_is_empty_for_any_query_under_twenty_words.py` |
| 4 | Build a model routing classifier. Replace the keyword-based classifier with an embedding-base… | code | T0 | `ex04_the_length_test_fires_first_so_every_complex_query_is_simple.py` |
| 5 | Implement a circuit breaker with degradation levels. At 70% budget, log a warning. At 85%, au… | code | T0 | `ex05_the_degradation_removes_the_event_the_exercise_asks_you_to_verify.py` |
<!-- generated:end -->

## Answers

The lesson is pure stdlib, so all five exercises are **T0** and run in CI. One
thing runs through most of them: `classify_complexity` tests
`len(q.split()) <= 5` *before* it looks at `COMPLEX_KEYWORDS`, and its
`SIMPLE_KEYWORDS` list contains `"hi"` and `"no"`, which match as substrings.
Between them those two lines decide the routing, the cost and the breaker in
exercises 4 and 5. The other recurring theme is scale: the exercise texts name
sizes — 100 queries, 1,000 requests, a $1.00 budget — that are too small to
make the thing they are testing happen.

### 1 — the cache holds 500 and the exercise sends 100

| max_size | FIFO hit rate | LRU hit rate | evictions (FIFO / LRU) |
|---:|---:|---:|---:|
| **500** (the default) | 0.780 | 0.780 | 0 / 0 |
| 10 | 0.580 | **0.630** | 32 / 27 |

**ANSWER: over 100 queries neither policy evicts anything.** `SemanticCache`
defaults to `max_size=500` and the stream puts 22 distinct queries in it, so
the branch that distinguishes the two strategies never runs and both report
0.78. Shrink the cache until it binds and LRU wins by 5 points — which is the
comparison the exercise wants, at a cache size the exercise does not set.

**FINDING: `get` does not touch the timestamp.** The entry dict carries
`timestamp`, set once in `put`, and `access_count`, incremented on every hit
and read by nothing. "Track the last access time" is a field that has to be
added, not one that exists.

**FINDING: `access_count` starts at 1 in `put`.** A never-read entry and an
entry read once are indistinguishable, so a policy built on the counter the
lesson already tracks cannot tell a cold entry from a warm one.

**FINDING: the similarity threshold is worth exactly as much as the policy.**
Lowering `similarity_threshold` from 0.85 to 0.70 and leaving eviction alone
takes FIFO from 0.580 to 0.630 — a gain of **0.050**, the same as switching to
LRU. Two levers of equal size, and the exercise varies one of them.

### 2 — every log entry lands in the same microsecond

**ANSWER: the trailing 7-day average has one day in it.** `log_call` stamps
`time.time()` and takes no timestamp argument, so 1,000 calls made in a loop
span single-digit milliseconds and land on one calendar day. The projection the
exercise asks for cannot be computed from a log the lesson's API can produce.

**CONTROL: backdate the same 1,000 calls over 14 days and the tool works.**

```text
weekday daily total   $0.0118
weekend daily total   $0.0056      2.12x
trailing-7 projection  [0.3152, 0.3175, 0.3101]   <- by window end day
```

A **2.4%** swing from window alignment alone, on a 2.12× signal — the exercise
is right that the weekday split matters and right that it needs averaging.

**FINDING: the exercise's alert and the tracker's alerts are different
mechanisms.** `_check_budget` fires on *cumulative spend so far* at 70/85/95%
of the monthly budget. The exercise's alert fires on a *projection* exceeding
budget by 20%. A tracker at 5% of budget today and projected to 150% at month
end says nothing.

**FINDING: the 20% rule cannot fire at the tracker's default budget.** The
1,000-call run costs **$0.17** and projects to $0.32 a month against the
`monthly_budget=1000.0` default — 0.03%. The alert only fires at a budget of
$0.005, three orders of magnitude below the one the lesson ships.

### 3 — the medium band is empty for any query under twenty words

**ANSWER: `simple_embed` is a normalised bag of words, so for an n-word query
sharing k words the cosine is exactly k/n.** The similarity grid has n+1
points. For an 8-word query those are 0, 0.125, 0.25, …, 1.0 — and *none* of
them falls in the medium band [0.90, 0.98).

| query length n | values in [0.90, 0.98) |
|---:|---|
| 8 | none |
| 10 | 0.900 (on the edge) |
| 20 | 0.950 (strictly inside) |

**FINDING: the only hits are exact repeats.** Across 190 distinct pairs from a
catalogue of the lesson's own shape, **0** reach 0.98 and **0** land in the
medium band. The 10 hits the tiered lookup records are queries literally in the
cache, scoring 1.0. The disclaimer branch is reachable in principle and never
taken.

**FINDING: the default threshold is below both tiers.** `SemanticCache` ships
`similarity_threshold=0.85`, so the cache already accepts matches the
exercise's *medium* tier would reject — 0.875 for a 7-of-8 word overlap. Adding
the tiers makes the cache stricter, not more nuanced.

**FINDING: "measure user satisfaction differences" has nothing to measure.**
There is no user, no rating and no feedback channel in the lesson;
`SemanticCache.stats` reports hits, misses, hit rate and size. Which tier a hit
came from can be tracked. Whether anyone was happier cannot.

### 4 — the length test fires first, so every complex query is simple

| classifier | score on the 20-query test set | cost per 1,000 queries |
|---|---:|---:|
| `classify_complexity` (shipped) | 14 / 20 | $1.5976 |
| 1-NN over 50 labelled queries | **20 / 20** | $2.4357 |
| the same two tests, swapped | **20 / 20** | — |

**ANSWER: the keyword classifier scores 14 of 20, and 6 of the 8 complex
queries come back "simple".** The length test fires before `COMPLEX_KEYWORDS`,
so "analyze this architecture" and "debug the payment flow" are simple. The two
that escape are the two long enough to reach the keyword test.

**FINDING: `"hi"` and `"no"` are matched as substrings.** `"hi" in "analyze
this architecture"` is True — on "this", and again on "architecture". **5** of
the 8 complex test queries are caught that way, so length is not the only thing
routing them wrongly.

**ANSWER: nearest-neighbour scores 20 of 20** — perfect, because the labelled
set carries the test set's vocabulary. A bag of words over 50 examples is a
synonym table with extra steps.

**FINDING: the better classifier is the more expensive one.** Being right about
"complex" means paying for `gpt-4o`: $1.60 → $2.44 per 1,000 queries, **1.5×**.
The exercise measures accuracy and the lesson is about cost.

**CONTROL: swapping two lines matches the embedding classifier exactly.**
Testing the keywords before the length reaches the same 20 of 20, with no
embedding, no labelled set and no nearest-neighbour search.

### 5 — the degradation removes the event the exercise asks you to verify

1,000 requests over the lesson's own 10-query pipeline mix, `$1.00` budget:

| | total | utilisation | alerts |
|---|---:|---:|---|
| breaker off | $0.9516 | 95% | warning @739, throttle @895, **stop @999** |
| breaker on | $0.8700 | 87% | warning @739, throttle @895 |

**ANSWER: all three thresholds trigger — and only with the breaker off.** With
the breaker on, the 85% degradation does its job, the bill lands at 87% of
budget, and `stop` never fires. The exercise asks for a mitigation and then
asks to verify the event the mitigation prevents.

**FINDING: the 95% level has one request of headroom.** `stop` arrives at
request **999 of 1,000**. That is not a demonstration that the threshold works;
it is a coincidence of the query mix, and one cheaper query anywhere in the
cycle removes it.

**FINDING: "switch all routing to the cheapest model" moves 2 of 10 queries.**
Eight already route to `gpt-4o-mini` at the "pro" tier, so the 85% level only
moves the two `medium` queries off `claude-sonnet-4` — 8.6% of the bill.

**FINDING: `gpt-4o` is never selected, because "hi" hides inside
"architecture".** "Analyze the pros and cons of serverless architecture"
classifies `simple`. No query in the lesson's own mix is complex, so the most
expensive model in the pro row is unreachable and the breaker's whole purpose
is already served by a bug.

**FINDING: `_check_budget` uses `elif`, so a jump past several levels records
only the highest.** One `log_call("gpt-4o", 500000, 200000)` against the $1.00
budget totals **$3.25** and records `['stop']` alone. A breaker reading its
level off `tracker.alerts` never sees `warning` or `throttle` on any call large
enough to matter; the clean run only walks the levels in order because every
request is small.

**FINDING: the 95% "serve only cached" mode cannot be accounted for.** A call
logged `cache_status="hit"` is charged in full — $0.0055 — and `cache_savings`
reports the same $0.0055 as saved. And
`calculate_cost("gpt-4o", 100, 10, cached_input_tokens=500)` returns
**-0.000275**, because `non_cached = input_tokens - cached_input_tokens` goes
negative. Logging cache hits can lower `total_cost()` and un-trip the breaker
that caused them.
