<!-- generated:start -->
# 15-autonomous-systems / 06-automated-alignment-research

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/06-automated-alignment-research/) · upstream spec
`phases/15-autonomous-systems/06-automated-alignment-research/docs/en.md`

```bash
uv run demo practice run 06-automated-alignment-research --ex 1
uv run demo explain 06-automated-alignment-research --ex 1
uv run pytest demos/phases/15-autonomous-systems/06-automated-alignment-research
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Compare the "fixed-workflow" vs "free-decomposition" settings. Does free… | code | T0 | `ex01_free_wins_by_a_constant_the_simulator_cannot_vary.py` |
| 2 | Modify the simulator so one agent attempts log tampering. Confirm the append-only log detects… | code | T0 | `ex02_the_chain_catches_the_edit_and_not_the_rewrite.py` |
| 3 | Read Anthropic's weak-to-strong AAR report. Identify the specific sub-task the AARs beat huma… | explain | T0 | prose, below |
| 4 | Design a task-queue allocation policy that balances AAR flexibility (better results) against… | code | T0 | `ex04_the_ab_test_needs_forty_five_times_the_shipped_sample.py` |
| 5 | Read RSP v3.0's AI R&D-4 threshold. In one paragraph, describe what you think would cross it… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — free wins by a constant the simulator cannot vary

The exercise asks about *problem classes*, which is a question about an
interaction between regime and task. The model has no term for one. `solve`
adds `U(0, 0.25)` under the fixed regime and `N(0.15, 0.22)` under the free
one, and neither touches `base` — so the mean gap is **0.025 on every task**,
and no problem class can differ. Measured over 20000 draws each, the five gaps
land within 0.005 of that.

What does differ is spread:

| regime | mean bonus | sd | draws outside [0, 1] |
|---|---:|---:|---:|
| fixed | +0.125 | 0.0722 | 0.0% |
| free | +0.150 | 0.2201 | 3.7% |

So "does free always win" has three answers depending on what is being won. On
the mean, always, by the same amount. Head to head on a single task, **54.3%**
of the time. And **24.8%** of free draws land *below the fixed regime's floor* —
below anything the prescribed workflow can produce — which is the sense in
which the fixed workflow is competitive, and the only sense this simulator can
express.

**The free regime produces impossible scores.** 2.4% of its draws exceed 1 and
1.3% fall below 0. Nothing clamps them, and `regime_report` averages them in,
so part of the free regime's reported advantage is made of results that cannot
exist.

**The agent is a parameter `solve` never reads.** Its signature takes `agent`;
the name appears once in the function, in that signature. The three "parallel
researchers" are one distribution with three labels, so any per-agent
comparison drawn off this forum is reading noise — including the one
`attempt_tamper` makes when it picks "AAR-A's worst result".

**And three draws cannot see the effect.** `run_regime` posts 15 records per
regime, 3 per task. Against a 0.025 gap at these spreads that is a power of
0.06 — the shipped comparison is not a weak measurement of the difference, it
is not a measurement of it.

### 2 — the chain catches the edit and not the rewrite

The modification already ships: `attempt_tamper` rewrites AAR-A's worst score
upward. At the lesson's own seed, `verify()` returns **index 12 of 15**.

**The detection signature, precisely.** At index 12 the recomputed payload hash
does not equal the stored `my_hash`, while the record's `prev_hash` *still
equals* its predecessor's stored hash — so the mismatch is one cell wide, not a
broken link. And records 13 and 14 both verify clean, because `verify()` walks
forward carrying the *stored* `my_hash` rather than the recomputed one. The
break does not propagate down the chain. A reviewer looking for a cascade of
failures after the edit point will not find one; what they will find is a
single record whose hash does not match its own contents.

**Re-chaining defeats `verify()` completely.** Re-post the tampered records
into a fresh `Forum` and every hash is recomputed consistently:
`verify()` returns zero flags on a log whose numbers have been altered. The
only surviving evidence is the head — `11cb5fb4fa631b2c` honest against
`759402faba5d56e3` tampered.

**And nothing in the module ever compares that head to anything.** `Forum` has
two fields, `genesis` is the constant `0000000000000000`, and the class exposes
three methods — `head`, `post`, `verify` — none of which takes an expected
value. `tamper_demo` prints "an out-of-sandbox backup reveals the tamper on
cross-check" and performs no cross-check. The property the lesson's own section
title rests on is asserted in a string.

So the honest version of "confirm the append-only log detects it" is: it
detects an in-place field edit, which is the attempt an agent holding the
repository has no reason to choose, and it detects a re-chained edit only if
something outside the sandbox remembers the head. The hash chain is not the
security boundary; the out-of-sandbox copy of one 16-character string is.

### 3 — the sub-task, and what made it automatable

*Draws on "The architecture (as publicly described)".*

The AARs outperformed human researchers on **weak-to-strong training** — using
a weaker supervisor model to train a stronger student — which is the specific
study Anthropic published. What makes that task amenable to automation is that
its inner loop is a *measurable sweep*: a weak-to-strong setup has a scalar
outcome (student performance under weak supervision), a large space of
configurations to try, and a cheap automatic check for each one, so the work is
dominated by running many well-specified experiments rather than by deciding
what would count as an answer. That is the same shape AlphaEvolve needs in
Lesson 3 — a fast, deterministic, hard-to-game evaluator — applied to research
rather than to code, and it is why the result does not generalise to alignment
problems whose success criterion is itself contested.

### 4 — the A/B test needs forty-five times the shipped sample

**The allocation policy.** Route a task to free decomposition when its baseline
leaves room for the tail — `base + 2σ ≤ 1` and `base − 2σ ≥ 0` at the free
regime's own σ = 0.22 — and to the prescribed workflow otherwise. On the
lesson's five tasks that routes two to free (`weak-to-strong-distill` at 0.40
and `reward-model-diagnosis` at 0.30) and three to fixed. The rule allocates on
*downside*, because that is the axis the two regimes actually differ on; on the
mean they differ by a constant and the choice would be free everywhere.

**The A/B test.** Two arms, tasks assigned round-robin, and the sample size
derived from the effect rather than the budget:

| runs per arm | power to detect 0.025 |
|---|---:|
| 15 (shipped) | 6.2% |
| 100 | 18.9% |
| 674 | 80.0% |

**674 per arm against the 15 `run_regime` posts — a factor of 44.9.** The demo
prints "fixed has lower variance, free has higher upside" beside two numbers
that are, at this sample size, one number and noise.

**The cost the policy is supposed to balance is not in the model.**
`ForumRecord` has six fields and none is a decomposition, a step count or a
review time. The free regime's extra audit burden — the entire reason a
prescribed workflow is on the table — appears nowhere, so no allocation rule
written against this simulator can trade one against the other. Any honest
version of this A/B test has to instrument the second arm first.

**And pairing buys nothing here**, which is worth knowing before designing a
crossover. `solve` ignores its `agent` argument and `run_regime` builds a fresh
`Forum` per regime, so the arms share no task instance. Pairing would remove
the task baseline from the variance — and the baseline contributes 0.0 to a
within-task comparison, because the regime term is additive. The variance that
matters is the free regime's own σ, and no design removes that.

### 5 — what would cross AI R&D-4 that AAR does not

*Draws on "The compression risk".*

AAR automates the middle of one research pipeline: humans set the task queue,
the AARs propose decompositions and run experiments, and humans decide what
publishes — so the two ends that determine *which* research happens remain
human, and the threshold is written about the whole loop rather than its
interior. What would cross it is an agent that closes those ends: one that
chooses its own research agenda from the state of the field, runs the
experiments, evaluates the results against a standard it also selected, and
feeds the outcome back into the next capability training run without a human
deciding that any of it was worth doing — at a cost competitive with a human
team using AI tools, which is the clause that makes the threshold economic
rather than merely technical. The gap is not experimental competence, which AAR
demonstrably has on a scoped task; it is the judgement about what to work on
and what counts as having worked, which is exactly the authority the lesson
says humans retain.
