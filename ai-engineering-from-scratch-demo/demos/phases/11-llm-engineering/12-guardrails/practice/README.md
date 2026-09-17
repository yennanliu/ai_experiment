<!-- generated:start -->
# 11-llm-engineering / 12-guardrails

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/12-guardrails/) · upstream spec
`phases/11-llm-engineering/12-guardrails/docs/en.md`

```bash
uv run demo practice run 12-guardrails --ex 1
uv run demo explain 12-guardrails --ex 1
uv run pytest demos/phases/11-llm-engineering/12-guardrails
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Build a LlamaGuard-style classifier. Create a keyword + regex classifier that maps inputs and… | code | T0 | `ex01_it_blocks_half_the_benign_prompts_and_a_third_of_the_unsafe_ones.py` |
| 2 | Implement the encoding evasion detector. Attackers encode injection attempts in base64, ROT13… | code | T0 | `ex02_the_evasion_signal_sits_five_hundredths_below_its_own_threshold.py` |
| 3 | Add rate limiting with sliding window. Implement a per-user rate limiter that allows 10 reque… | code | T0 | `ex03_the_fixed_window_the_exercise_forbids_scores_the_same_on_its_test.py` |
| 4 | Build a hallucination detector for RAG. Given a source document and a model response, check t… | code | T0 | `ex04_at_twenty_percent_it_catches_one_of_six_and_at_ninety_five_all_six.py` |
| 5 | Implement a full red-team suite. Create 100 attack prompts across 5 categories: direct inject… | code | T0 | `ex05_four_input_guardrails_and_one_of_them_carries_the_suite.py` |
<!-- generated:end -->

## Answers

The lesson is pure stdlib, so all five exercises are **T0** and run in CI. Two
things run through all of them. Every keyword list is matched with `kw in
text_lower`, so `"shoot"` fires inside "troubleshoot" and `"hi"` inside "this";
and every guardrail returns `passed = max_confidence < <threshold>`, so what a
rule *detects* and what it *blocks* are two different questions. The headline
across the lesson: on a 10-question benign control the shipped pipeline blocks
**4**, which is above its recall in two of the five attack categories.

### 1 — it blocks half the benign prompts and a third of the unsafe ones

| on 50 prompts | unsafe (26) | benign (24) |
|---|---:|---:|
| the lesson's `classify_topic` | 8 blocked (31%) | **12 blocked (50%)** |
| 13-code classifier, `\b`-anchored | macro P 0.949 / R 1.000 | 2 flagged |

**ANSWER: the new classifier reaches macro precision 0.949 and macro recall
1.000** over the 13 MLCommons codes. Every one of the 26 unsafe prompts lands
in its own code.

**ANSWER: the lesson fires more often on safe traffic than on attacks.** 50%
against 31%. That is not a weak classifier, it is an inverted one, and the
direction is what a 50-prompt test is for.

**MECHANISM: `kw in text_lower` is a substring test.** "troubleshoot" contains
"shoot", "hackathon" contains "hack", "killer" contains "kill". Dropping the
word boundaries from this classifier's own keywords costs 0.021 of macro
precision and doubles its false positives, 2 → 4.

**FINDING: `TOPIC_KEYWORDS` has 5 categories and the taxonomy has 13**, and
only **6** of the 13 see any hit at all on their own unsafe prompts. Privacy,
specialized advice, elections, code-interpreter abuse, hate, child exploitation
and sex crimes are 0 of 2 each — unrepresented, not misclassified.

**FINDING: the confidence takes exactly two values.** `min(0.6 + 0.15 * n,
0.99)` returns 0.0 thirty times and 0.75 twenty times over the 50, because no
prompt matches two keywords in one category. With `passed = confidence < 0.75`,
"return the category code and confidence" returns one bit.

### 2 — the evasion signal sits five hundredths below its own threshold

```text
plain payload                       blocked, confidence 0.95
20 encoded versions                 0 blocked, 4 noticed
  + 5 decoders, 1 round            19 of 20
  + a second decode round          20 of 20
10 benign prompts, 2 rounds          0 flagged
```

**MECHANISM: `detect_injection` appends `encoding_evasion` at confidence 0.70,
then returns `passed = max_confidence < 0.75`.** A detection that fires and
cannot block, five hundredths apart, in the same function.

**FINDING: the four it notices are noticed for naming their own encoding.** The
checks are `count("base64") > 0`, `"rot13"`, `"hex:"` and a zero-width class —
the literal names of the schemes. Raw base64 of the payload scores **0.0**, so
omitting the word "base64" is invisibility, and the word in a benign question
is the signal.

**ANSWER: decoding first reaches 19 of 20 with no new pattern.** The survivor
is base64 applied twice; a second decode round catches it. The fix is a loop
bound, not a rule. `0x`-prefixed hex needed `removeprefix("0x")` first —
without it the digit count is odd and the decode is dropped by the `except`.

**CONTROL: 0 false alarms** on "base64 this greeting", "what does rot13 do", a
sha256 digest and a hex string that decodes to "Hello world".

### 3 — the fixed window the exercise forbids scores the same on its test

| 15 requests, 2 s apart, 10/min | allowed | retry-after |
|---|---:|---|
| sliding window | 10 | 40, 38, 36, 34, 32 |
| **fixed window** | **10** | **40, 38, 36, 34, 32** |

**FINDING: the test names a distinction it cannot measure.** A burst from t=0
to t=28 never crosses a minute boundary, and a boundary is the only place the
two disagree.

**CONTROL: move the same burst 45 seconds later.**

| | sliding | fixed |
|---|---:|---:|
| burst starting at t=45 | 10 of 15 | **15 of 15** |
| 10 at t=59.9, 10 at t=60.0 | 10 | **20 in 0.1 s** |

**FINDING: the pipeline has nowhere to put a per-user limiter.**
`process(self, user_input, model_fn=None)` and `validate_input(self,
user_input)` take no user id, and `_log_event` stores a sha256 of the input
with no user field. "Per-user" is not another entry in `validate_input`.

**FINDING: "return a retry-after header" has no field to return it in.**
`GuardrailResult` is `(passed, category, details, confidence, latency_ms)`;
`GuardrailReport` is `(input_results, output_results, blocked, block_reason,
total_latency_ms)`. The wait rides inside `details` and the caller parses prose.

### 4 — at 20% it catches one of six; at 95% it catches all six

| threshold | hallucinations caught (6) | supported flagged (9) | faithful paraphrases flagged (3) |
|---:|---:|---:|---:|
| **0.20** (the exercise's) | **1** | 0 | 0 |
| 0.95 | **6** | 0 | **3** |

**ANSWER: at 20% the detector flags 1 of the 6.** The one it catches is the
only one that changes the subject — "Quantum entanglement violates causality",
against a source about shipping, scores 0.000.

**FINDING: the five it misses are the five that matter.** "2.4 million" for
"4.2 million" scores **0.857**; "100 requests" for "1000 requests" **0.917**;
"three days" for "three weeks" **0.909**. A set of words has no place to put
"2.4 is not 4.2" — the token is present and the fact is inverted.

**FINDING: dropping 29 stop words changes nothing.** Still 1 of 6. The overlap
was never carried by "the" and "was": every supported sentence is copied
verbatim and scores exactly **1.000**.

**CONTROL: what 0.95 actually separates is copied from not-copied.** Three
faithful paraphrases of the same facts score 0.500, 0.364 and 0.429 — above the
exercise's threshold and far below the one that works.

**FINDING: the lesson already ships this function.** `check_relevance(input,
output, threshold=0.15)` is the same word overlap with a different denominator;
pointed at source-versus-response it fails the same **1 of 10** pairs. A
digit-string comparison catches **2 of 6** exactly, for one regex.

### 5 — four input guardrails, and one of them carries the suite

| category (20 each) | baseline recall | after 3 rules |
|---|---:|---:|
| direct injection | 1.00 | 1.00 |
| indirect injection | 0.65 | 0.65 |
| jailbreak | 0.60 | 0.60 |
| prompt extraction | 0.40 | 0.40 |
| **PII extraction** | **0.20** | **0.90** |

The contract was fixed before the fixture, so the fixture could not be tuned
until the answer was the one the author wanted: baseline is the shipped rule
set unmodified; the metric is per-category recall; the weakest category is the
*strictly* lowest, and a tie means the fixture is wrong; the improvement
required is +0.20 absolute with no other category falling, from exactly 3 new
registry entries.

**ANSWER: the weakest is unique and it is not close** — PII extraction at 0.20
against 0.40 for the next lowest.

**MECHANISM: all four of those catches fire on bait the attacker supplied.**
`detect_pii` matches PII *present in the text*, and a request for someone
else's data contains none. The four that fire are the four carrying a sample
email, phone number, SSN or card number of their own. Against the 16 that only
*ask*, no guardrail in the pipeline fires at all.

**ANSWER: three rules matching the request shape** — "<field> of the account
holder", "the previous user", "export the customer table" — take PII extraction
to **0.90**, +0.70 absolute, with no regression and no new false positive.

**FINDING: `detect_injection` accounts for 53 of the 57 baseline blocks.** The
other 4 are the bait above. `check_length`, `classify_topic`,
`filter_toxicity`, `check_relevance` and `check_system_prompt_leak` fire **0**
times between them across all 100 attacks.

**FINDING: the pipeline blocks 4 of 10 ordinary questions** — three by
`relevance_check` on the *output*, because `_simulate_llm` answers "Based on
your question about ...", and one by `classify_topic`, because "troubleshoot"
contains "shoot". Each of those four cost a model call before being thrown away.
