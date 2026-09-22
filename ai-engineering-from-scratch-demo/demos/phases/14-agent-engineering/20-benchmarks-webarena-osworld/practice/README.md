<!-- generated:start -->
# 14-agent-engineering / 20-benchmarks-webarena-osworld

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/20-benchmarks-webarena-osworld/) · upstream spec
`phases/14-agent-engineering/20-benchmarks-webarena-osworld/docs/en.md`

```bash
uv run demo practice run 20-benchmarks-webarena-osworld --ex 1
uv run demo explain 20-benchmarks-webarena-osworld --ex 1
uv run pytest demos/phases/14-agent-engineering/20-benchmarks-webarena-osworld
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Extend the toy harness with a second app (a forum). Write 3 tasks plus gold trajectories. | code | T0 | `ex01_success_asks_whether_the_state_happened_not_who_caused_it.py` |
| 2 | Add trajectory-efficiency reporting per task. On your toy, is the agent 1x, 2x, or 3x over gold? | code | T0 | `ex02_efficiency_is_best_on_the_agent_that_does_nothing.py` |
| 3 | Implement a "distractor" tool — one the gold trajectory never uses. Does the scripted agent g… | code | T0 | `ex03_a_scripted_agent_has_no_branch_to_tempt.py` |
| 4 | Read OSWorld-G. How would you separate grounding failures from planning failures in your own… | explain | T0 | prose, below |
| 5 | Read WebArena's apps README. What breaks when you upgrade one of the pinned app versions? | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the forum ports cleanly, and shows there was no harness to port into

`Task` annotates `agent` and `success` as `Callable[[ShoppingApp], ...]`, and
nothing enforces the annotation, so a forum drops straight in: threads, replies,
votes and moderation, with three tasks — reply to the pinned thread, upvote every
reply in the busiest thread, remove the spam post — at 3, 5 and 5 steps against
golds of 3, 4 and 5. All three succeed, 1.08x over gold.

What the port exposes is that there is nothing to drop *into*. The module defines
four top-level functions and none of them runs a task: `main()` computes success,
steps and efficiency inline and prints them. Extending the harness means writing
the loop a second time, and the `run()` this solution adds is five lines of logic
that already existed as unreachable code inside a print loop.

The second thing the port exposes is what execution-based evaluation actually
asks. Every `success` predicate scans the whole app for a state, so an agent that
clears every post and then replies to and upvotes everything satisfies 3/3
without reading a single task description — 27 steps, 2.25x gold against the
honest run's 1.08x. The benchmark scores the world, not the trajectory, and the
step count is the only thing that notices the difference. That is exactly why
WebArena's gym-API state checks need a trajectory metric beside them, and why
OSWorld-Human exists.

Finally, `gold_steps` is an `int` somebody typed. Nothing derives it, nothing
checks it against a trajectory, and setting all three golds to 1 turns the same
13 steps into 4.33x with zero code changed and three still-successful tasks. The
denominator of the headline metric is an unvalidated constant.

### 2 — 1x on two tasks, 1.40x on the third, and three decisions hidden in one print

`main()` already prints `steps / gold_steps` per task, so the reporting is
shipped. Adding it properly means making explicit the three decisions the print
statement makes silently.

**What counts as a step.** `revised_order` appends `"revised_choice: remove
keyboard"` to its trace without touching the app. Counting trace lines gives 7
steps where the app was called 6 times — 1.40x against 1.20x per task, 1.17x
against 1.08x in aggregate. The metric is partly measuring the agent's prose.
For a real computer-use agent this is the difference between counting model turns
and counting screen actions, and they are not the same number.

**How ratios combine.** `total_steps / total_gold` is 1.17x; the mean of the
per-task ratios is 1.13x. The ratio of sums weights tasks by length, so the one
task that is over gold moves the headline more than the two that are not —
dropping it takes the aggregate to 1.00x. Report which one you mean.

**Whether a failed task has an efficiency at all.** An agent that returns an
empty trace scores 0.00x on every task, better than gold, and fails all three.
`main()` computes efficiency before it looks at success, so any ranking on
efficiency alone sorts the do-nothing agent first. Efficiency is only meaningful
conditioned on success — which is the precise shape of OSWorld-Human's 1.4–2.7x
claim, a number measured over *solved* tasks.

So, to answer the question as asked: on this toy the agent is 1x, not 2x or 3x,
on two of three tasks, and 1.40x on the one where it changes its mind. It is
optimal because it was written to be optimal; the interesting number is what the
measurement apparatus does when it is not.

### 3 — no, and that is a fact about the agent, not about the distractor

`_agent_task_1` through `_agent_task_3` are fixed call sequences. They contain
zero `if` statements between them and call the new `search` tool zero times.
"Tempted" is not a state they can enter, so the shipped answer to the exercise's
question is a vacuous no.

Offering the same tool to the smallest agent that actually chooses — a policy
that ranks tools by description overlap with the prompt — gets it picked on 3 of
3 tasks. That is the honest experiment, and the results say something sharper
than "yes, it was tempted".

The distractor is invisible to the metric that decides. A tempted run still
satisfies 3/3 `success` predicates, because they read `app.orders` and a search
leaves nothing there. Efficiency does not move either, because `search` replaces
`list_items` rather than adding to it: both runs sit at 0.92x — *under* gold,
since the chooser never makes the mistake gold's trajectory records. Only padding
moves the number, to 1.17x with one extra search per task, and it takes four per
task to double the headline. Twelve wasted calls, zero changed verdicts.

The structural reason is that `gold_steps` is a count. Against a gold *sequence*,
3 of 3 tempted tasks contain a non-gold action, and 2 of those match the gold
step count exactly — indistinguishable under an `int`, obviously wrong under a
sequence. The trajectory-efficiency gap the lesson cites is defined against a
number that has already thrown the trajectory away.

### 4 — separating grounding from planning needs two observations of the same step

**Follow-ups** names the two instruments: OSWorld-G, a 564-sample grounding suite
with a Jedi training set, decomposes grounding from planning so they can be
measured separately; OSWorld-Human supplies gold action trajectories and shows
top agents use 1.4–2.7x more steps than necessary.

Applied to one's own evals, the decomposition is a claim about *where* a step
went wrong, and it needs two observations per step rather than one. A grounding
failure is: the agent named the right target and hit the wrong pixels — correct
intent, wrong coordinates. A planning failure is: the agent hit exactly what it
aimed at, and aiming there was the wrong move. A single success/failure bit
cannot tell those apart, which is why a screenshot-only eval reports one number
for two unrelated engineering problems.

Concretely, three changes make the split measurable. First, log intent and action
separately at every step: the element the agent *says* it is targeting, and the
coordinates or selector it emits. Second, keep an oracle mapping from element to
bounding box for the eval set — that is what OSWorld-G's 564 samples are — so
"did the click land in the right box" is checkable independently of whether the
click was a good idea. Third, score planning on an *oracle-grounded* replay: feed
the agent's stated intents through a perfect executor and see whether the task
completes. If the oracle-grounded run succeeds and the real run fails, the gap is
grounding. If both fail, it is planning. If the real run succeeds with many more
steps than the oracle-grounded one, it is neither — it is recovery, which is what
OSWorld-Human's 1.4–2.7x is largely measuring.

Exercise 3 is the miniature of this. The scripted agents have no grounding
problem and no planning problem, because they have no branch; the policy agent
has a pure planning failure — it picks the distractor tool for a good-looking
reason — and the harness scores it as a success. The fix in both cases is the
same: record what the agent intended, not only what the world looks like
afterwards.

### 5 — upgrading a pinned app silently breaks comparability, not the harness

**Where benchmarking goes wrong** lists this as one of three pitfalls: WebArena's
apps pin specific versions, and updating without re-curation breaks comparability.
The subtlety is that *nothing fails*. The harness still runs, the tasks still
execute, the success predicates still return booleans, and the number that comes
out is still a percentage — it is just no longer the same measurement as
yesterday's.

Four things break concretely. Selectors and DOM structure shift, so tasks fail
for navigational reasons unrelated to agent capability; that one is at least
visible as a score drop. Seeded fixture data changes — a shop upgrade may
renumber SKUs or reseed the catalogue — so a task whose success predicate names
`sku-001` now checks a different product, and the task passes or fails for the
wrong reason. Gold trajectories go stale: an upgrade that adds a confirmation
dialog makes the gold step count too low, so every agent's efficiency worsens by
one step and the 1.4–2.7x band shifts without any agent changing. And new
features appear, which is the distractor problem from exercise 3 arriving by
itself: an upgraded forum with a "quick reply" button offers a shortcut the gold
trajectory never used, and now the efficiency metric punishes agents that take
the *better* path.

The practical consequence is that a WebArena-style score is only meaningful
alongside the pinned app versions it was produced against — the version manifest
is part of the result, not part of the setup. Re-curation after an upgrade means
re-deriving gold trajectories, not just rerunning; exercise 1's finding that
`gold_steps` is an unvalidated `int` is the same failure in miniature, since
nothing in the harness would notice that the gold number no longer corresponds to
any reachable trajectory.
