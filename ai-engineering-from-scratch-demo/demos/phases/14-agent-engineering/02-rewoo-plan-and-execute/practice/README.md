<!-- generated:start -->
# 14-agent-engineering / 02-rewoo-plan-and-execute

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/02-rewoo-plan-and-execute/) · upstream spec
`phases/14-agent-engineering/02-rewoo-plan-and-execute/docs/en.md`

```bash
uv run demo practice run 02-rewoo-plan-and-execute --ex 1
uv run demo explain 02-rewoo-plan-and-execute --ex 1
uv run pytest demos/phases/14-agent-engineering/02-rewoo-plan-and-execute
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Parallelize worker execution for independent plan nodes. What does it buy you on a 6-node DAG… | code | T0 | `ex01_the_waves_are_computed_and_then_flattened_away.py` |
| 2 | Add a replanner node that fires if any worker returns an error. What is the smallest change t… | code | T0 | `ex02_the_smallest_change_is_one_more_input_to_the_planner.py` |
| 3 | Replace `Planner` with a small model (7B class) and keep `Solver` on a frontier model. Compar… | code | T0 | `ex03_the_split_fails_on_the_defect_that_costs_nothing.py` |
| 4 | Read Section 4 of the ReWOO paper on planner distillation. Reproduce the 175B -> 7B result co… | explain | T0 | prose, below |
| 5 | Port the toy to Plan-and-Act's trajectory shape: plan is a sequence, not a DAG. What tradeoff… | code | T0 | `ex05_a_sequence_moves_the_ordering_obligation_to_the_planner.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 4 asks for a reading of
the paper, so it is answered in prose.

The recurring finding is that **ReWOO's decoupling is also a blindfold**.
The planner never sees observations — that is what buys the 5x and what makes
distillation cheap — and the solver never sees the world, only evidence. So
any defect that produces well-formed evidence is invisible to both halves at
once. Exercise 3 measures it directly: of five characteristic planner
defects, the only one neither role can catch is the one that costs nothing to
produce.

The parts that hold are structural. `topological` really does free the
planner from emitting steps in order, and `resolve_references` really does
thread evidence through the DAG. What is missing is a *shape* for what the
sort already computed: waves are calculated and then flattened into a list,
and the evidence dict's insertion order is completion order, which the
lesson's own token accounting then zips against declaration order.

### 1 — the waves are computed and then flattened away

**ANSWER: the 6-node DAG resolves into 3 waves of 3, 2 and 1.** Sequential
execution is **6** unit steps; wave execution is **3** — a **2.00x** makespan
win, with the widest wave capping useful concurrency at **3** workers. The
evidence is identical either way: **6** of **6** values match `run_workers`.

**FINDING: the shipped sweep already knows the waves.** `topological`'s
`while pending` loop *is* a Kahn sweep — it resolves one whole independent
group per iteration and appends them all into one list. Keeping the level
boundaries costs no extra dependency lookup. The return type `list[PlanStep]`
is a shape that cannot hold the answer, so the grouping is lost at the return
statement rather than never computed.

**FINDING: the accounting's pairing is already wrong, and a sum hides it.**
`run_rewoo` takes `worker_chars` from `zip(plan.steps, evidence.values())` —
declaration order against *completion* order. Declare the demo plan backwards
and **2** of **3** pairs are mis-associated while `worker_chars` stays at
**108**, because addition does not care which value it added to which.
Parallel execution makes the mis-association the normal case, and the first
non-additive statistic taken from that zip will be wrong loudly.

**FINDING: the demo plan buys nothing.** `E1 → E2 → E3` is a chain: **3**
waves for **3** nodes, **1.00x**. The speedup is a property of the plan's
shape, not of ReWOO — the pattern promises nothing until a planner emits
width.

### 2 — the smallest change is one more input to the planner

**ANSWER: `plan_for(question)` becomes `plan_for(question, evidence)`.**
ReWOO's three arrows are `question → plan`, `plan → evidence`,
`(question, plan, evidence) → answer`. Plan-and-Execute adds one:
`evidence → plan`. The shipped `plan_for` takes **1** input besides `self`;
the replanning one takes **2**. A plan naming an unregistered tool yields
`error: unknown tool 'web_search'` on pass 1, the replanner fires once, and
pass 2 ends with **0** errors and the demo's answer.

**FINDING: the trigger is a prefix, and it misses half the failures.**
`fake_search` answers an unknown query with `no result for '…'`, which carries
no `error:` marker. That plan finishes with **0** errors, fires the replanner
**0** times, and the solver composes the miss into the final answer. **1** of
the **2** failure kinds reaches the trigger; the other arrives as evidence.

**FINDING: the replan needs a bound, and the bound cannot tell why it
failed.** `ScriptedPlanner.plan_for` ignores `question` and returns the same
object for two different questions, so an unbounded replanner on the shipped
planner is an infinite loop by construction. Against a tool that always
raises, the bounded version burns its cap of **2** replans and still ends
with **1** error — `dispatch` catches bare `Exception`, so a bad plan and a
broken tool arrive in the same shape.

**FINDING: every replan re-pays the planner prompt.** `planner_chars` is
**156** for one pass and **468** across three. The paper's 5x is measured per
plan; a task that replans twice has bought one answer with three planner
prompts.

### 3 — the split fails on the defect that costs nothing

**ANSWER: five weak-planner defects, five outcomes.** A dangling `#E9`
raises `RuntimeError` after **0** tool calls. A step the solver template needs
but the plan omits raises `KeyError` after **2** of **3**. An unregistered
tool name lands in evidence as `error: unknown tool` after **2**. A
plausible-but-wrong query answers normally after **3**. Steps declared out of
dependency order produce evidence byte-identical to the correct plan.

**Where the split fails: on the wrong query.** Its evidence —
`no result for 'capital of Atlantis'` — carries **0** error markers and is
perfectly well-formed, and the solver composes it into the answer. The
solver's inputs are the question, the plan and the evidence; it has no
channel to the world, so a frontier model on that end cannot repair a small
model's plausible mistake. This is the class that scales badly: a 7B planner
gets tool names and reference syntax right after a little fine-tuning, and
stays wrong about *what to ask* for much longer.

**FINDING: the loud failures are loud about the wrong thing.** `topological`
raises `cyclic plan or unresolved reference` for a plan with no cycle — one
message, two faults, and the one that fired is the cheaper to diagnose. The
omitted step is visible in the plan before any worker runs, yet it surfaces
as `KeyError: 'E3'` from the solver after **2** of **3** tool calls are
already paid for: the cheapest defect to detect is detected last.

**FINDING: declaration order is free.** The shuffled plan pays the same **3**
tool calls and returns the same answer. That is the flip side of the
distillation question — scoring a planner on its token sequence would charge
it for a difference the executor cannot observe.

### 4 — the training data is the teacher's plans, and the score is the DAG

*Cites "Planner distillation".*

**What makes the data collectable is the same property that makes ReWOO
cheap.** The lesson's sentence is the whole argument: "because the planner
does not see observations, you can fine-tune a 7B model on planner outputs
from a 175B teacher." The planner is a function of the question alone —
exercise 2 measures that directly, `plan_for` takes **1** input — so its
training set is ordinary supervised data. No rollouts, no environment, no
reward model, no credit assignment across a trajectory. That is why this
distillation works where distilling a ReAct policy does not: a ReAct turn is
conditioned on observations, so its training data is a distribution over
worlds, not a set of pairs.

**The data you need is (question, tool catalogue, plan DAG), plus the
failures.** The catalogue belongs in the record because a plan is only valid
relative to a registry — exercise 3's `unknown tool` defect is not a property
of the plan, it is a property of the pair. Keeping failed plans labelled by
*which* of the failure classes they hit is what lets the student model learn
the cheap constraints first; three of the five defects in exercise 3 are
decidable from the plan text alone.

**How to score plan quality: in four layers, cheapest first.**

1. **Static validity, before anything executes.** Every tool name is in the
   catalogue; every `#En` resolves to an earlier node; the graph is acyclic;
   every id the solver template needs exists. Exercise 3 shows this catches
   the dangling reference and the missing step at **0** tool calls — the two
   defects that otherwise crash, one of them after paying for two thirds of
   the work.
2. **Structural agreement with the teacher, not textual.** Compare the DAG:
   node set up to renaming, edge set, tool assignment. Exercise 5 measures
   why textual comparison is wrong — the 6-node DAG admits **20** valid
   linearizations, so a sequence-level metric marks **19** correct plans
   wrong. Exercise 3 measures it from the other side: the shuffled plan is a
   different token sequence with byte-identical behaviour.
3. **Terminal agreement, on a held-out set only.** Execute and compare final
   answers. This is the only layer that catches the wrong-query class, and it
   is the only one that needs an environment — so it is the expensive signal
   and belongs in evaluation rather than in the training loop.
4. **Cost per plan.** Worker calls and planner characters, because exercise 2
   shows a replanning system multiplies the planner prompt **3x** over two
   retries. A student model that is 2 points better and replans twice as
   often is not better.

The 175B → 7B result is therefore reproducible as a *pipeline* shape without
any of the scale: teacher emits plans, static gates reject the cheap defects,
structural scoring grades the rest order-invariantly, and a small held-out set
pays for execution. What does not survive the shrink is layer 3 — the world
never gets cheaper.

### 5 — a sequence moves the ordering obligation to the planner

**ANSWER: the port is `run_workers` without the sort.** On the demo chain the
two executors agree on **3** of **3** evidence values and give the same
answer; the chain was already a trajectory. On the 6-node DAG the sequence
takes **6** steps where the DAG resolves in **3** waves.

**FINDING: the trajectory keeps one order and records nothing about the
rest.** That DAG admits **20** valid orders. A sequence stores **1** and has
no field saying the other **19** were equally correct — the independence is
absent, not merely unused. That is why Plan-and-Act's contribution is
*labelled trajectory data* rather than a runtime change: the shape is easier
to generate and to fine-tune on precisely because it has thrown information
away.

**FINDING: the obligation moves to the planner.** Declared backwards, the
demo plan still matches **3** of **3** evidence values under the DAG
executor. Under the sequence executor `#E1` is still unresolved when it is
used, so the query goes out as `population of #E1` — and then
`rounded_million` finds the digit in `#E2` and answers **`2 million`**. A
reference name became a measurement. The DAG tolerates any declaration order;
the sequence tolerates exactly one, and fails quietly when it does not get it.

**FINDING: a dangling reference stops raising and becomes evidence.** In the
DAG, `#E9` raises `RuntimeError` before **0** tool calls have run. In the
sequence, `resolve_references` leaves the literal alone and the tool answers
`no result for 'population of #E9'` — a well-formed answer to a malformed
plan.

**What the sequence buys in exchange:** a cycle becomes unwritable, and
completion order equals declaration order by construction, so `run_rewoo`'s
positional bookkeeping (exercise 1's finding) can no longer mis-pair. Both
gains are of the same kind — a class of error becomes unrepresentable rather
than caught. That is the real tradeoff behind the shape, and it is why
Plan-and-Act pays for it with plan data rather than with a better executor.
