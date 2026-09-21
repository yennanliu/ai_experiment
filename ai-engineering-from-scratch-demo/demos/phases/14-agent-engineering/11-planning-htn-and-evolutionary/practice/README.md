<!-- generated:start -->
# 14-agent-engineering / 11-planning-htn-and-evolutionary

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/11-planning-htn-and-evolutionary/) · upstream spec
`phases/14-agent-engineering/11-planning-htn-and-evolutionary/docs/en.md`

```bash
uv run demo practice run 11-planning-htn-and-evolutionary --ex 1
uv run demo explain 11-planning-htn-and-evolutionary --ex 1
uv run pytest demos/phases/14-agent-engineering/11-planning-htn-and-evolutionary
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Extend the HTN planner with backtracking: when an operator's postcondition fails at runtime,… | code | T0 | `ex01_there_is_no_runtime_so_a_postcondition_cannot_fail.py` |
| 2 | Add a LLM-method cache to ChatHTN: when the LLM decomposes task `T` in state pattern `P`, sto… | code | T0 | `ex02_the_cache_is_shipped_and_its_key_is_missing_the_state.py` |
| 3 | Swap the evolutionary search evaluator to a real test suite. Evolve a sort function that pass… | code | T0 | `ex03_twenty_tests_accept_a_network_that_does_not_sort.py` |
| 4 | Read AlphaEvolve's evaluator design notes. Design an evaluator for a domain you care about (S… | explain | T0 | prose, below |
| 5 | Combine: use HTN to decompose a compound task into subtasks, then use evolutionary search on… | code | T0 | `ex05_two_of_the_three_subtasks_have_nothing_to_search.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 4 asks for a design
against AlphaEvolve's evaluator notes, so it is answered in prose.

The recurring finding is that **both halves of this lesson are missing the
layer that would make their claims checkable**. The HTN planner has no
runtime: `Operator.apply` unions `effects_add` into the state unconditionally
and nothing ever compares the result against the world, so "the postcondition
failed" is not a state the code can reach. The evolutionary search has an
evaluator that is a pure function of the genome, so it cannot overfit — until
you swap in a real test suite, at which point exercise 3 measures it
accepting a program that passes **20/20** tests and is not a sorting network.

The second thread is that **the caches and the ordering are already shipped
and are keyed on too little**. `cached_methods` exists, is written on every
LLM fallback, and is consulted after the method library — exactly what
exercise 2 asks for — with a key of `task` and no state, so one state's
decomposition is returned in another and the mismatch surfaces as a bare
`None`.

What holds: the precondition checking. `_expand` really does verify each step
against the state it will run in, and really does refuse an LLM suggestion
naming an unknown task. That is the soundness the lesson claims for the
symbolic layer, and it is there.

### 1 — there is no runtime, so a postcondition cannot fail

**ANSWER: an executor plus method backtracking.** Given a task with **2**
applicable methods where the first one's `run_tests` does not deliver
`tests_passing`, the shipped planner returns a plan that fails after **3** of
**4** operators. The backtracking planner rolls back once and returns the
second method's four-operator plan.

**FINDING: the shipped planner considers one method.** `plan` takes
`applicable[0]` and never revisits it, so with two applicable methods a
failed expansion ends the search — at *plan* time, before any execution. The
backtracking the exercise asks for is missing at two levels, and only one of
them involves a runtime.

**FINDING: there is nothing to check a postcondition against.** `Operator`
carries `name`, `preconditions`, `effects_add`, `effects_remove` — **0**
fields naming a postcondition or a verifier. The planner's model of the world
is its own arithmetic, which is exactly the soundness claim the lesson makes
for HTN and exactly the assumption ChatHTN's LLM fallback is supposed to be
checked against.

**FINDING: failure has one value and five causes.** `plan` returns `None` for
exceeding `max_depth`, an unknown task, an unmet precondition, an LLM that
declines, and an LLM suggestion naming something unknown. A caller that wants
to backtrack cannot tell a retryable failure from an impossible one — which
is why the backtracking loop here retries *every* method rather than the ones
worth retrying.

### 2 — the cache is shipped, and its key is missing the state

**ANSWER: a `(task, frozenset(state))` key.** Two states needing different
decompositions of the same task get **2** cache entries, **2** LLM calls, a
correct plan in each, and **0** further calls on a repeat.

**FINDING: the shipped key reuses one state's answer in another.** Keyed on
`task` alone, the decomposition learned with `has_migration` present is
returned in a state without it; `_expand` rejects it and `plan` returns
`None` after **0** new LLM calls. The precondition check does its job — the
bad plan does not execute — but the cache never learns, so the task is now
permanently unplannable in that state.

**FINDING: the whole state is too specific a key.** Adding one irrelevant
fact to an already-cached state misses, costing an extra LLM call for a
decomposition already known; projecting the state onto the facts the task's
operators mention takes the count back to **1**. The "state pattern" in the
exercise is doing real work: too little and you get the shipped bug, too much
and you get a cache that never hits.

**FINDING: re-checking the library first is already the shipped order.**
`plan` reads `methods`, then `cached_methods`, then the LLM. Adding a method
for a cached task makes the method win on the next call, with the cache entry
left in place. The half of the exercise that sounds like the design work was
done; the half that sounds like bookkeeping is where the bug is.

### 3 — twenty tests accept a network that does not sort

**ANSWER: convergence in 7 generations, on a 9-comparator network.** Over 20
seeded permutations of five distinct integers, the search reaches **0**
failures at generation **7** with a length penalty of 0.01 per comparator.

**FINDING: the winner is not a sorting network.** That result passes
**20/20** tests and **31** of the **32** zero-one vectors. By the 0-1
principle — a comparator network sorts every input if and only if it sorts
all zero-one vectors — it is wrong, on an input class it was never shown, and
the search reported it as converged. This is the lesson's "AlphaEvolve
without a real evaluator" pitfall arriving through a *real* evaluator that is
merely incomplete, which is the harder version.

**FINDING: the length penalty bought the wrong answer faster.** Dropping it
converges in **5** generations on a **12**-comparator network that passes
**32/32**. The regularisation term that made the result shorter is what made
it incorrect: at five inputs, networks short enough to look elegant fit 20
tests before they sort.

**FINDING: a test count is a much flatter landscape than a squared error.**
Of 500 single mutations from a fixed network, **124** leave the failure count
unchanged; of 500 mutations to the demo's two integers, **18** leave the
squared error unchanged. Swapping the evaluator for a test suite removes most
of the gradient — which is why the shipped demo converges in 12 generations
on a problem with a unique optimum and says nothing about the case the
exercise asks for.

### 4 — an evaluator is a spec you can run, and a spec you can overfit

*Cites "AlphaEvolve (Novikov et al., 2025)".*

**The domain: test-suite minimisation.** Given a suite of *N* tests and a
codebase, find the smallest subset that preserves the suite's defect
detection. It is a good fit for the same reason the lesson's examples are —
the fitness is machine-checkable and the search space is combinatorial — and
it is a good *teaching* fit because the obvious evaluator is the one
exercise 3 shows failing.

**The obvious evaluator, and why it is exercise 3 again.** Score a subset by
`coverage(subset) == coverage(full_suite)`, minus a size penalty. This is
exactly the shape that produced a 9-comparator network passing 20/20 tests
and 31/32 zero-one vectors: a proxy the search can saturate, plus a
regularisation term that rewards saturating it sooner. Line coverage is
satisfiable by tests that execute a line and assert nothing about it, so the
minimiser will find the subset that touches every line and catches fewer
bugs, and every number on the dashboard will improve.

**Four layers, cheapest first — the design the pitfall implies.**

1. **A cheap surrogate for the inner loop.** Statement or branch coverage,
   computed once per candidate subset from a precomputed per-test coverage
   matrix. This is what makes thousands of generations affordable, and it is
   explicitly *not* the thing being optimised for.
2. **A held-out correctness gate.** Mutation testing against a fixed set of
   seeded faults the search never sees. The 0-1 principle in exercise 3 plays
   this role exactly: a complete oracle, expensive to run, applied to the
   candidate rather than inside the loop. A subset that improves on layer 1
   and loses mutation score is rejected, and the rejection is the interesting
   data.
3. **A real-world gate, run rarely.** Replay of historical regressions — bugs
   that actually shipped. This is the layer that cannot be gamed because it
   was not generated by anything the search can reach, and it is the layer
   that eventually retires layer 2's fault set when the two disagree.
4. **A cost model with the right sign.** Wall-clock, not test count. Exercise
   3's length penalty is the warning: a size term denominated in the wrong
   unit optimises for the metric's shape rather than the goal. Minimising
   test *count* rewards deleting slow integration tests and keeping a
   thousand fast unit tests; minimising *time* does the opposite.

**Three properties the design has to have, from AlphaEvolve's own
constraints.** *Deterministic* — a flaky test in the suite makes fitness
noisy, so flaky tests are quarantined before the search starts, not scored.
*Fast* — the coverage matrix is computed once and the inner loop is set
arithmetic, so a candidate costs microseconds rather than a test run.
*Complete enough to be worth saturating* — and where it is not, layer 2
exists to say so. Exercise 3's finding is the acceptance test for the whole
design: if the search can reach a perfect score on layer 1 while failing
layer 2, the evaluator is not finished, and the number to report is not the
fitness but the gap.

**What would make me not do this at all.** The lesson's third pitfall —
"most agent tasks don't need either". If the suite is under a few hundred
tests, or if the coverage matrix changes weekly, enumeration and a greedy
heuristic beat an evolutionary loop and are inspectable. Exercise 5 puts the
ratio on it: search earns its keep when the space is large relative to the
evaluations you can afford, and the saving there was **5.0x** on a space of
200.

### 5 — two of the three subtasks have nothing to search

**ANSWER: the HTN plan is 3 operators, and evolving all three costs 120
evaluator calls to improve 1.** `tune_pipeline` decomposes into
`open_editor`, `choose_batch`, `open_pr`. Evolving the batch size takes its
cost from **30** to **0** in **40** calls; the other two are flat, so **80**
calls buy nothing.

**Where it shines: a big space with a cheap exact evaluator.** The batch
parameter has **200** candidates and the search finds the optimum after
**40** evaluations — a **5.0x** saving over enumerating, on an evaluator that
is a pure function of one integer. That ratio, space over affordable
evaluations, is the condition worth checking before building anything.

**Where it over-engineers: everywhere the ratio is not there.** **2** of the
**3** subtasks return the same cost for every candidate, so the search does
exactly as much work on them as on the one that matters. A combined system
with no per-subtask gate spends its budget in proportion to the plan's
*length* rather than to where the gains are — and plan length is the one
thing the HTN layer is good at producing.

**FINDING: the HTN layer supplies no fitness.** `Operator` carries **4**
fields and **0** of them is a cost, a score or a quality signal.
Preconditions and effects are a *feasibility* predicate: they decide whether
a step is legal, never how good it is. So combining the two patterns is not a
matter of wiring them together — every subtask needs a human-written
evaluator before evolution has anything to optimise, and writing three
evaluators to discover that two of them are constant is the over-engineering
the exercise is asking about.
