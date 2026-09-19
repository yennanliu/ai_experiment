<!-- generated:start -->
# 12-multimodal-ai / 25-multimodal-agents-computer-use

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/25-multimodal-agents-computer-use/) · upstream spec
`phases/12-multimodal-ai/25-multimodal-agents-computer-use/docs/en.md`

```bash
uv run demo practice run 25-multimodal-agents-computer-use --ex 1
uv run demo explain 25-multimodal-agents-computer-use --ex 1
uv run pytest demos/phases/12-multimodal-ai/25-multimodal-agents-computer-use
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Extend the action schema with a `screenshot_region` tool (crop + zoom). What tasks benefit? | code | T0 | `ex01_the_tool_is_already_in_the_schema_and_not_in_the_simulator.py` |
| 2 | Read AgentVista (arXiv:2602.23166). Describe the hardest task category and why frontier model… | explain | T0 | prose, below |
| 3 | Long-horizon memory compression: design a summary-chain with ≤4 screenshots kept live, any nu… | code | T0 | `ex03_four_screenshots_buy_one_thousand_seven_hundred_steps.py` |
| 4 | Build an error-recovery hook: on action failure (button not found), what does the agent do next? | code | T0 | `ex04_there_is_no_failure_to_hook_because_nothing_fails.py` |
| 5 | Compare screenshot-only Claude 4.7 to hybrid screenshot + accessibility-tree Qwen2.5-VL on 10… | code | T0 | `ex05_the_gap_is_five_points_on_seeing_and_twenty_five_on_doing.py` |
<!-- generated:end -->

## Answers

Four code solutions and one prose one — the last lesson of the phase. All four
code exercises find the same thing from different sides: the capstone simulator
declares an interface it does not implement, and every question the exercises ask
lands in the part that is missing.

### 1 — the tool is already in the schema, and not in the simulator

**ANSWER: it is already declared.** `ACTION_SCHEMA` has **10** entries and
`screenshot_region: ["x0","y0","x1","y1"]` is one of them.

**FINDING: `apply_action` names 4 of the 10 and changes state for 3.**

| | actions |
|---|---|
| named | click, type, select, `done` (a deliberate no-op) |
| **unimplemented** | **scroll, drag, hover, navigate, wait, screenshot_region** |

**60%** of the schema falls through to an unchanged copy with no error raised.

**ANSWER: what benefits is anything whose target is smaller than a patch.**

| | |
|---|---:|
| 1920 × 1080 at patch 14 | **10,549** tokens |
| downscale to fit 2,048 | **0.4406** |
| a 14-pixel UI label becomes | **6.2 px** — under half a patch |
| a 200 × 100 crop at native | **98** patches |

Dense tables, small-text forms, mobile UIs, chart axis labels.

**FINDING: and the tool cannot help this lesson's own benchmark.** `run_task` has
**0** branches and reads no observation — both tasks are fixed plans with
hard-coded coordinates, so there is no point at which a cropped view could change
an action.

### 2 — the hardest AgentVista category

Drawing on **Why it's still hard**, which lists the four bottlenecks, and on the
benchmark section that puts AgentVista at "frontier 27–40%, open 10–20%" across
"realistic workflows over 12 domains".

**The hardest category is the long, cross-surface workflow with a mid-task
dependency** — the kind where step 7 needs something produced at step 3, on a
different page, in a different application. Book travel *and* file the expense;
find the invoice, then update the record that references it. Not the hardest
*screen*: the hardest *shape*.

**Why frontier models still fail on it, in the order the failures actually
arrive:**

1. **Nothing detects a failed action.** Exercise 4 measures this in the lesson's
   own simulator — a click on a missing element silently no-ops, and the only
   check is at the end of the plan, **2.0 steps late** on a five-step plan. Real
   agents have the same hole for the same reason: a screenshot after a failed
   click looks like a screenshot after a slow page. The lesson's own list puts
   error recovery third and notes it "is rarely trained data", which is the
   sharper statement — it is not that the capability is weak, it is that no
   supervision exists for it.
2. **The evidence needed at step 7 was discarded at step 5.** Exercise 3 prices
   the constraint: keeping every screenshot exhausts a 128k context at step
   **12**, so something must be dropped, and what gets dropped is chosen by
   recency rather than by relevance. A mid-task dependency is exactly the value
   that recency throws away.
3. **And errors compound multiplicatively.** At a per-step success rate *p*, a
   20-step workflow succeeds at *p*²⁰ — **0.72** at p = 0.983 and **0.12** at
   p = 0.90. A frontier score of 27–40% on realistic workflows is consistent with
   a *very* high per-step rate; the benchmark is hard because it is long, and
   length is not something better grounding fixes.

**Which is why the gap in exercise 5 is the shape it is.** Five points on
grounding and twenty-five on chained execution: the models can see. What they
cannot do is notice that they were wrong ten steps ago.

### 3 — four screenshots buy 1,779 steps

**ANSWER: four live frames, a full action log, a rolling summary every ten
steps.**

| | tokens |
|---|---:|
| 4 live frames @ 10,549 | 42,196 |
| 50-step log @ 30 | 1,500 |
| 5 summaries @ 200 | 1,000 |
| **total** | **44,696** |
| keeping every frame | **527,450** (**11.8×**) |

**FINDING: keeping every screenshot exhausts a 128k context at step 12** — and
neither of the lesson's benchmark tasks is longer than six steps, which is why
its simulator never meets the constraint this exercise imposes.

**FINDING: the compressed scheme is constant in step count.** After step 4 the
screenshot term stops moving; the remaining budget buys **1,779** more steps at
**50** tokens each — not 30, because the rolling summary costs 200 every ten
steps and amortises to 20 — a slope **211×** shallower. Charging the chain
smoothly gives 1,777; the true figure is two higher because summaries land in
jumps of 200.

**FINDING: the four are not interchangeable.** `first`, `previous`, `current`,
`last state-changing` — *not* the last four. The first anchors the goal, the
current grounds the next action, the previous detects a no-op, and the last
state-changing frame is the only one that answers "what did my last real action
do". A sliding window throws that away during exactly the run of failed clicks
where it is needed.

### 4 — there is no failure to hook, because nothing fails

**ANSWER: a five-rung ladder with a per-rung cap and a global budget.**
Re-ground from a fresh screenshot → scroll and retry → `screenshot_region` on the
expected area → alternative descriptor → back out one step and re-plan → then
`done(success=False)` with the trace, rather than a sixth attempt.

**FINDING: the lesson's loop cannot fail.** Clicking "NoSuchButton" leaves the
page at `home` and raises nothing; `run_task` has **0** branches, executes every
planned step, and records the click as a click. The only failure signal is the
final page, compared after the plan is exhausted.

**FINDING: which means failure is detected 2.0 steps late on average** on a
five-step plan — all of them run against the wrong state. A hook needs a
per-action **postcondition**, and `apply_action` has no such predicate anywhere.

**FINDING: and an unbounded ladder is the fastest way to exhaust the context.**
Each rung is a fresh screenshot at 10,549 tokens, so five rungs is **52,745** —
**40.2%** of a 128k window, and **1.18×** exercise 3's entire fifty-step budget.
The cap is the design; the ladder is the easy part.

### 5 — the gap is five points on seeing and twenty-five on doing

| benchmark | open | frontier | gap | ratio |
|---|---:|---:|---:|---:|
| **ScreenSpot-Pro** (grounding) | 85 | 90 | **5** | **1.06×** |
| Ferret-UI mobile (grounding) | 70 | 82 | 12 | 1.17× |
| VisualWebArena (multi-step) | 20 | 27 | 7 | 1.35× |
| **WebArena** (multi-step) | 35 | 60 | **25** | **1.71×** |
| AgentVista (multi-step) | 15 | 33.5 | 18.5 | **2.23×** |

**ANSWER: the frontier model wins everywhere, and by 5× more on doing than on
seeing.** Whatever separates the two, it becomes five times more visible once
actions are chained.

**FINDING: the table has no accessibility-tree column.** All five rows are
open-against-frontier, and the lesson's own section on screenshot-only versus
hybrid carries no numbers at all — so the comparison the exercise names is not in
the evidence.

**ANSWER: the split is whether the DOM is authoritative.**

| the tree wins | screenshots win |
|---|---|
| form fields, links | canvas |
| labelled buttons | custom widgets |
| ARIA roles | images used as buttons |
| anything the page declares | visual-only state (spinners, disabled-looking-but-enabled) |

A text label is *exact* where a click coordinate is approximate; a rendered pixel
is *present* where the DOM is silent or lying.

**FINDING: and ten tasks cannot answer the question.** At AgentVista's rates one
standard error over ten tasks is **1.4 tasks**, so a 10-task comparison separates
two systems only if they differ by several outcomes — and the two systems in the
exercise differ, on the evidence available, by rather less than that.
