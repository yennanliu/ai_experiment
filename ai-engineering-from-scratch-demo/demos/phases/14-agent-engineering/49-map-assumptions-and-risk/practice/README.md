<!-- generated:start -->
# 14-agent-engineering / 49-map-assumptions-and-risk

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/49-map-assumptions-and-risk/) · upstream spec
`phases/14-agent-engineering/49-map-assumptions-and-risk/docs/en.md`

```bash
uv run demo practice run 49-map-assumptions-and-risk --ex 1
uv run demo explain 49-map-assumptions-and-risk --ex 1
uv run pytest demos/phases/14-agent-engineering/49-map-assumptions-and-risk
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Write five assumptions for a feature you want to build. | code | T0 | `ex01_the_two_riskiest_assumptions_tie_and_the_two_functions_disagree.py` |
| 2 | Add one safety assumption that your feature list omitted. | code | T0 | `ex02_the_omitted_assumption_is_the_one_about_the_published_answer.py` |
| 3 | Define a threshold that would cause you to stop the build. | code | T0 | `ex03_the_threshold_has_nowhere_to_live_and_the_run_it_judges_fails_it.py` |
| 4 | Replace one large experiment with a cheaper decisive test. | code | T0 | `ex04_the_cheap_test_reaches_the_same_verdict_in_every_draw.py` |
| 5 | Compare risk ranking with roadmap priority and explain the mismatch. | code | T0 | `ex05_the_roadmap_builds_in_the_order_the_risk_map_would_test_last.py` |
<!-- generated:end -->

## Answers

### 1 — the two riskiest assumptions tie, and the two functions disagree

The feature: generate the README answers section instead of writing it. Five
assumptions, one per class, each falsifiable:

| Class | Assumption | I | U | R | Score |
|---|---|---|---|---|---|
| Value | a generated section saves more time than reviewing it costs | 4 | 4 | 1 | 17 |
| Usability | a reader cannot tell a generated answer from a written one | 3 | 4 | 1 | 13 |
| Feasibility | every number in an answer can be sourced from a graded check detail | 5 | 3 | 1 | 16 |
| Viability | the generator stays correct as the harness changes | 3 | 5 | 2 | 17 |
| Safety | a wrong generated number is caught before the lesson ships | 3 | 3 | 5 | 14 |

The top two tie at 17, which is where the module contradicts itself:
`prioritize` sorts by `(-risk_score, statement)` and `next_experiment` uses
`max(..., key=risk_score)`, which returns the first maximum it encounters. On a
tying pair the ranked list names one and the next experiment names the other — a
JSON document whose first row is not the thing it says to do next.

Two more properties of the formula worth stating before trusting the order.
`impact * uncertainty` reaches 25 while `irreversibility` adds at most 5, so the
only assumption here that cannot be undone scores 14 and ranks fourth. And
`evidence` is a free string: writing the word "unclear" into it flips an assumption
from `open` to `tested` and removes it from the experiment pool, with no threshold
and no comparison anywhere in the module.

### 2 — the omitted assumption is the one about the published answer

The five all ask whether the generator works. None asks what happens when it is
wrong **and the answer is already in a README that someone has read**.

> A wrong number reaches a reader before anyone notices. Impact 5, uncertainty 4,
> irreversibility 5 — a published claim cannot be unpublished.

Score 25. It takes the top of the map, lifting it from 17, and changes the next
experiment from a timing study to a detection test.

Three things the addition exposes. Five of the six tests written here name
something to observe; the lesson's own safety row — "do not automate; test approval
workflow first" — names a choice about the build instead, which is the shape safety
tests tend to take and the reason they get skipped. A safety assumption at
irreversibility 5 still needs a product of at least 13 to beat a value assumption
at 17, so the *class* does not earn the ranking, the numbers do: the modest version
of this same assumption scores 14 and ranks fifth. And the tie at 17 underneath is
untouched — a new top item hides the `prioritize`/`next_experiment` disagreement
rather than resolving it.

### 3 — the threshold has nowhere to live, and the run it judges fails it

**The rule: stop unless 90% of the numbers in a generated answer trace to a graded
check detail.** Written before the result, as the lesson requires.

Run against five finished lessons: **154** numbers claimed in their solutions'
docstrings, **109** of them appearing verbatim in the details those same solutions
print. **70.8%** — below the threshold. The rule says stop.

The 45 misses are quoted material: line numbers inside receipts (`code/main.py:174`),
field counts read off a dataclass, thresholds the prose explains rather than
measures. Not claims about anything the run produced. So the threshold fired
correctly and the conclusion is to fix the metric rather than the feature. That is
what a cheap threshold buys — it stops the build on its first run and tells you
something either way.

(The trace reads the solutions' docstrings rather than this answers section on
purpose: a metric that reads the prose it is quoted in moves every time the prose is
edited, which is a feedback loop rather than a measurement.)

The artifact cannot hold any of this. `Assumption` has six fields; its only numbers
are the three risk dimensions. There is no field for the number to beat, the number
observed, or the decision at pass, fail and ambiguous. And writing the result into
`evidence` marks the assumption `tested` and drops it from the open pool, so the map
moves on to the next experiment as though the question were settled. Stopping is
not a state this document can represent.

### 4 — the cheap test reaches the same verdict in four of five draws

Both versions can actually be run, which is the only way to know whether the cheap
one is decisive.

- **Large:** trace every number five finished lessons' solutions claim — 154 numbers,
  109 matched, **70.8%**, 25 graded files.
- **Cheap:** trace one lesson — 38 numbers, 22 matched, **57.9%**, 5 graded files.

Both return `stop`, at 20% of the cost. The substitution is justified.

The honest report is the list of draws. The per-lesson rates are 57.9%, 78.6%,
70.4%, **86.2%** and 65.6% — every one below the threshold, so all 5 available draws
agree with the full run and none would have reversed it. "Cheaper and decisive" is
still a property of this sample rather than of the method; here the sample is
generous, and the nearest draw is 3.8 points from the line.

What the expensive run actually buys is not a better point estimate — both land on
the same side of the threshold — it is the **28.3-point spread**, which is the
measurement that says the metric is noisy. That is worth knowing, and it is not
what the experiment was proposed to find out.

One artifact note: `Assumption.test` is a single string, so substituting the cheap
test overwrites the description of the large one. A reader of the map sees the test
that ran and not the one it replaced, which is the decision worth reviewing.

### 5 — the roadmap builds in the order the risk map would test last

The roadmap is the order a team would naturally work: make it produce something,
make it readable, make it correct, keep it alive, then worry about what a wrong
answer does. The risk map reads the same six assumptions and returns 25, 17, 17,
16, 14, 13 — with the published-mistake assumption **6th on the roadmap and 1st by
risk**, a displacement of 5 places and **9 of 15 pairs inverted**.

The interesting part is that irreversibility is not what causes the mismatch. Sort
the same six by `impact * uncertainty` alone and the top item does not move; 7 pairs
are still inverted. The term shifts 4 assumptions by one place each — including the
*other* safety row, from 6th to 5th. On this map, the clause the lesson says should
reorder the work moves it by one position, which is the ceiling problem from
Exercise 1 showing up as a practical consequence.

So the mismatch is explained by something simpler: the roadmap is ordered by what
the build needs next, and the risk map by what the evidence needs first. Those
genuinely conflict here, because the top-risk test — publish one generated lesson
and time how long a planted error survives — cannot run until something has been
generated. Two of the six tests depend on the feature existing. The resolution is
not to reorder the roadmap but to bound the build: generate one lesson, publish it
with a known planted error, and let the riskiest assumption be tested by the
smallest possible version of the thing.

And the artifact cannot participate in this comparison at all: `prioritize` returns
eight keys per row and none of them records an intended position, so the mismatch
this exercise asks about has to be computed outside the document that is supposed
to drive the decision.
