<!-- generated:start -->
# 19-capstone-projects / 85-content-classifier-integration

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/85-content-classifier-integration/) · upstream spec
`phases/19-capstone-projects/85-content-classifier-integration/docs/en.md`

```bash
uv run demo practice run 85-content-classifier-integration --ex 1
uv run demo explain 85-content-classifier-integration --ex 1
uv run pytest demos/phases/19-capstone-projects/85-content-classifier-integration
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a fourth classifier for code injection (output contains `<script>`, `eval(`, etc). Decide… | code | T0 | `ex01_a_code_injection_classifier_stops_12_of_12_payloads_and_the_exercises_two_tokens_catch_3_and_flag_5_benign.py` |
| 2 | Make the router apply a per-classifier severity weight so PII counts more than toxicity. Demo… | code | T0 | `ex02_weighting_pii_moves_only_the_email_fixture_because_no_fixture_fires_two_classifiers.py` |
| 3 | Add a confidence threshold so low-score verdicts downgrade by one severity level. Sweep the t… | code | T0 | `ex03_the_first_threshold_that_lowers_block_rate_also_ships_the_email_unredacted.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/classifiers.py` and `code/main.py`
as shipped. The three classifiers are rule-based: a 10-term harassment list
with a negation window, five PII regexes with a Luhn check, and a trigram
cosine against the system prompt. The `Router` takes the maximum severity
and maps it to block, redact, warn or log. No solution edits the lesson's
code. New classifiers and verdict transforms are passed into the lesson's
own `Router` and `Router.decide`. Fixtures are the lesson's six
`_DEMO_OUTPUTS`, and in exercises 2 and 3 also their 15 pairwise joins.
Attack strings are harmless test payloads (`alert(1)`) or placeholders
such as `[PAYLOAD]`.

### 1 — a code-injection classifier stops 12/12 payloads; the exercise's two tokens catch 3

**Policy: executable markup outside code is `high` (block), a
dynamic-execution call outside code is `medium` (redact), and either one
inside a Markdown code span or fence is `low` (warn).** Markup means
`<script`, `<iframe`, `javascript:` or an `on...=` handler. A call means
`eval(`, `exec(`, `__import__(` or `os.system(`. The classifier is passed to
the lesson's `Router` next to `default_classifiers()`, and the router
handles a fourth classifier without any change.

| corpus | block | redact | warn | log |
|---|---:|---:|---:|---:|
| 12 test payloads | 7 | 5 | 0 | 0 |
| 10 benign coding answers | 0 | 0 | 5 | 5 |
| lesson's 6 fixtures | 2 | 1 | 1 | 2 (unchanged) |

The two tokens the exercise names, as substrings, catch only 3/12 payloads.
They miss `<script src=...>`, `<SCRIPT >`, `<img onerror=>`, `<iframe>`,
`javascript:`, `exec(`, `__import__(` and `os.system(`. They also stop 5/10
benign answers, among them `ast.literal_eval(` (the safe alternative) and
PyTorch's `model.eval()`. A plain `\beval\(` word boundary still matches
`model.eval()`. The lookbehind `(?<![\w.])` is what excludes it.

### 2 — weighting PII over toxicity moves one fixture, because no fixture fires two classifiers

**With PII weight 1.5 and toxicity 0.75 applied to the severity level, one
of the six fixtures changes: the email goes from redact to block (block
rate 2/6 -> 3/6).** The weighted level is `floor(level * w + 0.5)`, capped
at high. The low-toxicity fixture still warns, and the card and leakage
fixtures are already high.

- None of the six fixtures fires more than one classifier. So the weight
  never changes which classifier decides. It only moves one verdict
  across a level boundary. Across a 25-point weight grid, only PII >= 1.25
  (email blocks) and toxicity < 0.5 ("moron" stops warning) change a verb.
- On the 15 pairwise joins, 4 verbs change, all redact -> block, and every
  one contains the email fixture.
- The leaked-system-prompt fixture scores 0.870 alone (block). Joined with
  any one other fixture it scores 0.573-0.770, so it never blocks. Next to
  the clean fixture it scores 0.685, and the leak ships with a warning.

### 3 — the first threshold that lowers the block rate also ships an email unredacted

**Block rate is a step function: 2/6 fixtures up to t = 0.65, 1/6 up to
0.87, 0/6 above.** Here a verdict with score < t drops one level.

| t | fixtures blocked | fixtures + joins blocked (/21) | PII shipped (/21) |
|---|---:|---:|---:|
| 0 (lesson) to 0.625 | 2/6 | 7 | 0 |
| 0.675 to 0.725 | 1/6 | 2 | 4 |
| 0.775 | 1/6 | 2 | 5 |
| 0.825 | 1/6 | 1 | 5 |
| 0.875 to 0.975 | 0/6 | 0 | 5 |

- The score counts findings. It is not a confidence. PII scores
  `0.5 + 0.15 x findings`, so a lone card (high) and a lone email (medium)
  both score 0.65, the minimum. The corpus has only 9 distinct scores
  (0.573-0.870), and every step sits on one of them.
- Above 0.65 the email verdict drops to low, so the address ships in clear
  text with a warning note: 4/21 outputs, 5/21 above 0.741. The card never
  ships. One downgrade takes it only to medium, and its redactor runs.
