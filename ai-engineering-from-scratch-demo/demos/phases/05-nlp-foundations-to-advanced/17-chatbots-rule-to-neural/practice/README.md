<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 17-chatbots-rule-to-neural

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/17-chatbots-rule-to-neural/) · upstream spec
`phases/05-nlp-foundations-to-advanced/17-chatbots-rule-to-neural/docs/en.md`

```bash
uv run demo practice run 17-chatbots-rule-to-neural --ex 1
uv run demo explain 17-chatbots-rule-to-neural --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/17-chatbots-rule-to-neural
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Implement the rule-based respond above with 10 patterns for a coffee-shop ordering bot.… | code | T0 | `ex01_the_anchor_costs_four_and_the_order_costs_four.py` |
| 2 | Medium. Build a hybrid FAQ + LLM fallback. 50 canned FAQ entries for a SaaS product, LLM fall… | code | T0 | `ex02_accuracy_and_refusal_are_one_number.py` |
| 3 | Hard. Implement the agent loop above with three tools (search, read-user-data, send-email). R… | code | T0 | `ex03_the_guard_blocks_every_benign_case_and_no_injection.py` |
<!-- generated:end -->

## Answers

Three exercises in which the thing being evaluated is not the thing the exercise
names. Exercise 1's bot is defeated by a word before the intent, not by the edge
cases it lists. Exercise 2 asks for two numbers that are one number. Exercise 3
asks for an evaluation of an agent loop the lesson does not have — and the one
control it does have blocks every benign scenario and no injection.

All three run at **T0** — `code/main.py` imports only `re` and `collections`.

### 1 — The anchor costs four and the order costs four

Ten coffee-shop patterns over 26 labelled utterances.

| matcher / order | score |
|---|---:|
| `regex.match`, patterns as written | 22 / 26 |
| **`regex.search`, patterns as written** | **26 / 26** |
| `search`, greeting pattern first | 25 / 26 |
| `search`, single order before double | 22 / 26 |
| `match`, single order before double | **18 / 26** |

**ANSWER: all four `match` failures are an intent the bot handles, preceded by a
word** — `hi, i'd like a latte`, `actually i want a mocha`, `please cancel my
order`, `sorry, make that a large`. They land on the catch-all or the greeting,
never on a wrong intent: the bot does not mishear the order, it stops hearing an
order at all.

**FINDING: `search` trades an anchoring bug for an ordering one**, and under
`match` the two compound — 18 is below either loss alone, because they fall on
different cases.

**FINDING: the lesson's own catch-all makes its fallback unreachable.**
`PATTERNS` ends with `.*`, which matches every string including the empty one, so
`return "I don't understand."` cannot execute. A rule bot with a catch-all has no
fallback; the catch-all *is* the fallback, and it says `Tell me more about that.`

**CONTROL: the four edge classes the exercise names are all handled.** Double
orders, unclear intent, price and greeting are 100%.

### 2 — Accuracy and refusal are one number

20 FAQ entries, 20 paraphrases, 10 unanswerable questions.

| threshold | correct | refused | wrong | unanswerable caught |
|---|---:|---:|---:|---:|
| 0.2 | 16 | 4 | **0** | 5 / 10 |
| **0.3** (default) | 14 | 6 | **0** | 9 / 10 |
| 0.4 | 9 | 11 | **0** | 9 / 10 |
| 0.5 | 7 | 13 | **0** | 10 / 10 |

**ANSWER: the FAQ never returns a wrong canned answer**, so correct + refused
= 20 at every threshold. The two quantities the exercise asks for are one
quantity read two ways, and the threshold slides one into the other.

**MECHANISM: the distributions overlap.** Answerable questions score Jaccard
0.125–0.714 against their own entry; unanswerable ones 0.083–**0.429** against
their nearest. No threshold does both jobs, and past 0.3 tightening costs 5
correct answers and buys 0 refusals.

**FINDING: three canned answers are unreachable before any of this is
measured.** `is_destructive` runs first, so `how do i cancel my subscription`,
`how do i request a refund` and `how do i delete my account` route to a
confirmation flow instead of to the answers written for them — 17 of 20 entries
reachable, and the three lost are among the most-asked questions any SaaS FAQ
has.

**MECHANISM: the arm the refusal rate is measured against is a string.**
`hybrid_respond`'s third branch returns `(would call LLM agent for: ...)`.

### 3 — The guard blocks every benign case and no injection

There is no agent loop: the lesson defines none of `search`, `read-user-data` or
`send-email`. What can be evaluated is `is_destructive`, over 24 labelled
scenarios.

| group | blocked | total | correct |
|---|---:|---:|---:|
| benign, containing a danger word | **6** | 6 | **0** |
| destructive, plainly worded | 5 | 5 | **5** |
| destructive, differently worded | **0** | 8 | 0 |
| prompt injection | **0** | 5 | 0 |

**Precision 0.4545, recall 0.2778** — right on exactly one of the four groups.

**MECHANISM: it is a substring test over five words.** `refundable` contains
*refund*, `cancellation` contains *cancel*, `recharge` and `surcharge` contain
*charge*, `undelete` contains *delete*. And `close my account permanently`,
`wipe all my data`, `revoke all api keys` and every injection avoid all five. The
guard is not weak at its job; it is doing a different job.

**FINDING: injection success is 5 of 5 before any model is involved.** Of the
three numbers the exercise asks for, this control speaks only to the third — and
that number is available before the loop is built, which is when it is worth
having.

**CONTROL: the one group it gets right is the one its keyword list was derived
from.**
