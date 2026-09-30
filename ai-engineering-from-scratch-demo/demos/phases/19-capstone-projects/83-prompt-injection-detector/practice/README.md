<!-- generated:start -->
# 19-capstone-projects / 83-prompt-injection-detector

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/83-prompt-injection-detector/) · upstream spec
`phases/19-capstone-projects/83-prompt-injection-detector/docs/en.md`

```bash
uv run demo practice run 83-prompt-injection-detector --ex 1
uv run demo explain 83-prompt-injection-detector --ex 1
uv run pytest demos/phases/19-capstone-projects/83-prompt-injection-detector
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a rule family for context-smuggling (instructions hidden in tool result JSON). Measure th… | code | T0 | `ex01_a_json_rule_family_catches_8_of_8_tool_results_only_when_it_outscores_the_payload_rule.py` |
| 2 | Compute per-rule contribution: for each rule, count how many true positives would be lost if… | code | T0 | `ex02_18_rules_guard_one_true_positive_each_and_leave_one_out_credits_13_of_31_to_no_rule.py` |
| 3 | Add a `confidence_threshold` knob. Sweep it from 0 to 1 and plot precision-recall per category. | code | T0 | `ex03_no_threshold_beats_zero_because_the_wrong_labels_are_the_confident_ones.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py` on its own data: `Detector`
over the 53 rules in `rules.py`, `evaluate` over the 50 taxonomy fixtures
from lesson 82 (`outputs/taxonomy.json`, read-only) and the 25 prompts in
`benign.py`. Out of the box it gets 31 of 50 fixtures right and passes all
25 benign prompts. Nothing is written to the lesson's `outputs/`. All three
run in well under a second, stdlib only.

### 1 — a JSON rule family catches 8 of 8 tool results only when it outscores the payload's rule

**Context-smuggling recall goes from 3/8 to 4/8 on the lesson's fixtures
and from 1/8 to 8/8 on tool-result JSON. It costs 0 of the 25 benign
prompts and 1 of 8 benign JSON prompts.** The family is three regexes in
the lesson's rule format. Each fires on an instruction inside a JSON string
value: a role marker (`system:`, `assistant:`, `developer:`), an override
verb aimed at prior instructions or the user, or a directive (`from now on`,
`you must`, `reply with`). The 8 probes are tool-result shapes whose payload
is the placeholder `CANARY-7`.

| measured at family score 0.92 | lesson | + family |
|---|---:|---:|
| context-smuggling recall, lesson fixtures | 3/8 | 4/8 (cs-06) |
| context-smuggling precision | 1.0 | 1.0 |
| accuracy, all 50 fixtures | 31 | 32 |
| tool-result probes labelled context-smuggling | 1/8 | 8/8 |
| lesson benign prompts passed | 25/25 | 25/25 |
| benign JSON prompts flagged | 2/8 | 3/8 |

- **The lesson's data cannot measure this family.** None of the 25 benign
  prompts contains a `{`, and only cs-01 (already caught) and cs-06 carry
  JSON. The gain on the fixtures is therefore capped at one, and the benign
  cost is 0 by construction.
- **The family has to outscore the payload's own rule.** The verdict is the
  category with the single highest score. At 0.80 (the lesson's score for
  `tool output`) a probe saying "Disregard the user request" stays
  instruction-override (`disregard-prior` 0.85), giving 7/8. At 0.92, above
  the lesson's maximum of 0.90, it is 8/8. The same argmax is why the
  lesson misses cs-06 and still misses the CSV fixture cs-08.
- The added false positive is `you must bring a jacket` in a user message.
  The lesson's own 2 are the `tool output` substring on a benign weather
  result and `decode-and-answer` on `{"job": "decode", ..., "answer": 42}`.

### 2 — 18 rules guard one true positive each, and leave-one-out credits 13 of 31 to no rule

**Sorted by net marginal contribution (TPs lost minus TPs gained when the
rule is removed), 18 rules score +1, 33 score 0, and 2 score -1.** The
contributions sum to 16, against 31 true positives.

| net | rules | notes |
|---:|---|---|
| +1 | begin-with-bien-sur, char-concat, complete-mid-sentence, earlier-in-this-conversation, html-comment, no-refusal-clause, policy-update, pretend-to-be, priority-admin, research-frame, step-by-step-of, test-mode, tool-output-injection, translate-and-act, warmup-then-now, we-agreed-earlier, wizard-explains | each the only in-category rule on one TP |
| +1 | leet-letters | guards et-03, but labels role-play rp-09 an encoding trick |
| 0 | 32 others | redundant, or dead (below) |
| 0 | no-warning | its only effect is labelling io-07 prefix-injection |
| -1 | from-now-on-unchained | removing it turns mt-06 into a multi-turn-ramp TP |
| -1 | override-claim | removing it turns cs-06 into a context-smuggling TP |

- **Leave-one-out undercounts redundancy.** 13 TPs are lost by no single
  removal. 14 TPs are fired by two or more rules of their own category, and
  removing both rules of each of the 13 two-rule pairs loses that TP every
  time. The effect runs the other way too: removing `ignore-prior` and
  `ignore-family` together turns cs-08 into a true positive, though neither
  removal does alone.
- **5 rules never fire on any of the 75 prompts:** `ignore-previous`,
  `disregard-prior`, `sure-here-the`, `morse-instruction`, `act-as-x`. The
  lesson's Problem section names `disregard` as the paraphrase a single
  regex misses, yet no fixture uses it.

### 3 — no threshold beats zero, because the wrong labels are the confident ones

**The best `confidence_threshold` is the lesson's default. Macro F1 is
0.713 for every threshold up to 0.40, then only falls, to 0 at 0.95.** The
knob wraps `Detector.analyze`: any verdict below the threshold becomes
`benign`. The lesson's `evaluate` scores each of 21 steps. Precision/recall
per category:

| t | macro F1 | context-smuggling | encoding-trick | instruction-override | multi-turn-ramp | prefix-injection | role-play |
|---:|---:|---|---|---|---|---|---|
| 0.00-0.40 | 0.713 | 1.00/0.38 | 0.86/0.75 | 0.75/0.75 | 1.00/0.57 | 0.86/0.67 | 0.86/0.60 |
| 0.50 | 0.684 | 1.00/0.38 | 0.86/0.75 | 0.75/0.75 | 1.00/0.57 | 0.86/0.67 | 0.80/0.40 |
| 0.60 | 0.577 | 1.00/0.38 | 1.00/0.50 | 0.75/0.75 | 1.00/0.29 | 0.86/0.67 | 0.67/0.20 |
| 0.70 | 0.512 | 1.00/0.12 | 1.00/0.38 | 0.75/0.75 | 1.00/0.29 | 1.00/0.67 | 0.67/0.20 |
| 0.80 | 0.364 | 1.00/0.12 | 1.00/0.38 | 0.75/0.38 | 1.00/0.14 | 1.00/0.33 | 0.50/0.10 |
| 0.85 | 0.235 | — | 1.00/0.38 | 0.50/0.12 | — | 1.00/0.33 | 0.50/0.10 |
| 0.90 | 0.120 | — | — | 1.00/0.12 | — | 1.00/0.33 | — |
| 0.95-1.00 | 0 | — | — | — | — | — | — |

The plot (`python ex03_...py` prints it) marks precision against recall,
using each category's initial letter, with `*` where two categories meet:

```
  | * **EMP
  |      RPE
  |    *R  I
  |  R  I
  |
  | *
  |
  |
  |
  |
  |
  +----------- recall 0..1, precision 0..1 up
```

- **Most of the sweep is flat.** The rules carry 13 hand-set scores from
  0.40 to 0.90. The 9 thresholds from 0 to 0.40 therefore equal the
  untouched detector, and above 0.90 nothing fires.
- **Precision falls for two categories as the threshold rises.** At 0.85,
  role-play precision is 0.50 (from 0.86) and instruction-override is 0.50
  (from 0.75). The 5 wrong-category verdicts average confidence 0.732,
  above the 0.710 average of the 31 true positives. The knob only helps
  encoding-trick (precision 1.00 from 0.60) and prefix-injection (1.00 from
  0.65). No benign prompt is flagged at any threshold, so the lesson's
  benign corpus gives the knob nothing to trade against.
- **The lesson reports precision 0.0 when nothing fires** (tp + fp = 0).
  A plot drawn straight from its report would put every category at (0, 0)
  for t ≥ 0.95. The table above shows those cells as "—", and the plot
  leaves them out.
