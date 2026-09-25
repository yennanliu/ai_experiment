<!-- generated:start -->
# 16-multi-agent-and-swarms / 06-hierarchical-architecture

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/06-hierarchical-architecture/) · upstream spec
`phases/16-multi-agent-and-swarms/06-hierarchical-architecture/docs/en.md`

```bash
uv run demo practice run 06-hierarchical-architecture --ex 1
uv run demo explain 06-hierarchical-architecture --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/06-hierarchical-architecture
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py` and compare happy vs perturbed. How many levels of manager hand-off does i… | code | T0 | `ex01_the_question_is_gone_after_one_hand_off.py` |
| 2 | Add a third level (top → sub → sub-sub → worker). Measure how often the perturbed path correc… | code | T0 | `ex02_a_third_level_needs_a_type_the_lesson_does_not_have.py` |
| 3 | Implement a "canary" worker at each sub-manager that is always asked the original user questi… | code | T0 | `ex03_the_canary_cannot_answer_the_question_it_is_asked.py` |
| 4 | Read CrewAI's `Process.hierarchical` docs. Identify one concrete guardrail CrewAI applies (st… | explain | T0 | prose, below |
| 5 | Compare nested LangGraph supervisors to CrewAI hierarchical. Which makes reconciliation loops… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the question is gone after one hand-off

One hand-off, and it happens on the **happy** path. Taking the content words of
*"Ship the premium tier feature to production"* and counting survivors:

| level | surviving |
|---|---|
| top task | 5/5 |
| branch task (`task -- branch: X`) | 5/5 |
| leaf question | **1/5** |
| leaf answer | **0/5** |
| sub summary | 0/5 |
| top synthesis | 0/5 |

Four of five words are lost at the top-to-sub hand-off, and the survivor,
"feature", only survives because three of the four split strings happen to
contain it. By the leaf *answer* — two levels below the top — nothing remains.

The mechanism is one line. `SubManager.run` computes
`self.split.get(w.name, task)`, and all **4 of 4** workers have a `split`
entry, so the fallback never fires. The `f"{task} -- branch: {label}"` the top
manager carefully builds reaches **zero** leaves. This is not decomposition
drift; the decomposition is a lookup table fixed at construction.

Which makes the demo's contrast hollow: happy and perturbed both reach
**0/5** at the top synthesis. The perturbation changes *which* canned answers
appear, not whether any of them answers what was asked.

And the closing claim — that the error appears at top synthesis, "one level
removed from where a human could catch it" — is untested. An unknown branch
label produces a `MISSING[...]` summary whose text propagates **verbatim** into
the top synthesis, so it is visible exactly where a human is looking. `main()`
passes only labels that exist, in 2 of 2 runs, so that path never executes.

### 2 — a third level needs a type the lesson does not have

It does not go in. `LeafOutput` has `worker/question/answer`; `SubSummary` has
`sub_manager/leaves/summary` — **zero** field names in common. `SubManager.run`
builds its summary from `l.answer for l in leaves`, so handing it another
`SubManager` raises `AttributeError`. The three-level hierarchy the lesson
ships is the only depth its types allow, and "add a third level" is a type
change rather than a configuration.

With an adapter that gives a `SubManager` the `LeafOutput` shape its parent
expects:

| depth | worker invocations | `branches[].leaves` | surviving overlap |
|---:|---:|---:|---:|
| 2 | 8 | 4 | 0/5 |
| 3 | 16 | 4 | 0/5 |
| 4 | 32 | 4 | 0/5 |

**It never corrects** — 6 runs across three depths and both label sets, zero
corrections. It cannot: every level composes its output as a join over its
children's strings, so it can drop a word and has no way to reintroduce one.
Self-correction would need a level that compares against the original
question, and the only level holding it is the top.

Two things fall out of the table. Leaves shown the user's question verbatim
stay at **0** however deep the tree goes, because each added level adds another
constant brief. And `branches[].leaves` stays at **4** while invocations
double — everything below the top sub-manager is flattened into a summary
string before its parent sees it, so the structure meant to record the
hierarchy records exactly one level of it.

### 3 — the canary cannot answer the question it is asked

Build the canary and it does detect drift — by failing.

Asking all four shipped workers the user's question unchanged returns
`[no canned answer for '...']` **4 times out of 4**. `Worker._match_key` scans
its canned keys for a substring of the question, finds none of "frontend",
"backend", "legal" or "finance", and returns `"default"` — a key **0 of 4**
workers declare. So `run` falls back twice for one condition: once to a
sentinel that names a catch-all answer, then to the string that says there
isn't one.

**How the manager should react:** escalate only when the canary *answered* and
disagreed. A naive "canary disagrees, therefore drift" rule fires on **every
branch of the happy path**, which makes it useless. "Could not answer" is a
different alarm with a different remedy: no worker is competent for the
question as the user phrased it, so the response is to re-plan, not to re-run.

Agreement cannot be computed here even in principle. The canary answer and the
branch summary share **0** content words on every branch — one is a sentence
about a missing answer, the other a join of canned findings — so any comparison
short of exact match returns "disagree" for structural reasons. The canary's
signal is indistinguishable from the drift it exists to detect.

Which is also the best argument for adding it. The shipped workers receive a
constant brief in 4 of 4 cases, so a canary would be the **only** path by which
the user's words reach a leaf at all.

### 4 — the step limit, and the failure mode it targets

*Draws on "Role-framework implementation".*

The concrete guardrail is CrewAI's **`max_iter` on the manager agent**, with
`manager_llm` required for `Process.hierarchical` as its companion constraint.

`max_iter` bounds how many times the manager may loop through
assign → evaluate → re-delegate before it must return. The failure mode it
targets is the manager's fourth listed power: *decides whether to accept,
re-delegate, or iterate*. Accept and re-delegate terminate; **iterate does
not**, and a manager evaluating outputs it is itself responsible for producing
has no independent signal telling it to stop. Without a step limit, a manager
that judges every crew output inadequate re-delegates forever, spending a
full crew execution per turn.

This lesson's code is the degenerate case that makes the hazard legible.
`TopManager.run` has no loop at all — it delegates once per label and
synthesizes. So it cannot spin, and it also cannot re-delegate when a branch
comes back wrong, which exercise 1 shows is every branch. CrewAI's manager can
do the useful version of that, which is exactly why it needs a bound.

The `manager_llm` constraint is the second half: the manager must be an LLM
with no tools of its own. It targets a different failure — a manager that can
do the work itself stops delegating, and the hierarchy silently collapses into
one agent while still paying for the crews.

### 5 — nested LangGraph supervisors, by a wide margin

*Draws on "Graph-framework implementation".*

Nested `create_supervisor` makes reconciliation loops cheaper to detect, and
the lesson's own sentence says why without drawing the conclusion: the inner
supervisor *has its own graph*, and the outer one treats it as an opaque node.

A reconciliation loop is a cycle — a sub-manager's output sent back for rework,
repeatedly. In a graph that cycle is an **edge you can see before running
anything**: it is in the compiled structure, countable statically, and each
graph can be stepped through separately. In CrewAI's hierarchical process the
same loop is the manager LLM choosing "re-delegate" on successive turns. There
is no edge. The only evidence is the trace, after the fact and after the spend,
and distinguishing "looped twice productively" from "looping" requires reading
what the manager said about why.

The cost asymmetry is the point: one is a property of the program, the other a
property of a particular run.

What CrewAI buys for that price is the thing the lesson names — dynamic
reshaping of the tree. A manager that can decide at runtime which crews exist
cannot have its cycles enumerated in advance, because the graph is not fixed.
That is the same trade as exercise 2's finding from the other direction: this
module's hierarchy is statically shaped and its depth is pinned by its types,
which makes it analysable and unable to adapt.
