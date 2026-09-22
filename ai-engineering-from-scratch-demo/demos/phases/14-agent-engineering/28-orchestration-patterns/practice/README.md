<!-- generated:start -->
# 14-agent-engineering / 28-orchestration-patterns

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/28-orchestration-patterns/) · upstream spec
`phases/14-agent-engineering/28-orchestration-patterns/docs/en.md`

```bash
uv run demo practice run 28-orchestration-patterns --ex 1
uv run demo explain 28-orchestration-patterns --ex 1
uv run pytest demos/phases/14-agent-engineering/28-orchestration-patterns
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Convert a supervisor-worker to a swarm by removing the router. What breaks? What improves? | code | T0 | `ex01_removing_the_router_removes_the_only_place_a_rule_can_live.py` |
| 2 | Add a hop counter to the swarm: refuse after 3 handoffs. Does it catch A->B->A bouncing? | code | T0 | `ex02_the_counter_is_shipped_and_the_task_it_stops_is_dropped.py` |
| 3 | Build a two-level hierarchical system for a 12-specialist domain. Where does the context budg… | code | T0 | `ex03_the_flat_router_reads_every_description_on_every_task.py` |
| 4 | Profile the four patterns on a production-shaped workload. Which wins on which metric (latenc… | code | T0 | `ex04_all_four_patterns_route_with_the_same_function.py` |
| 5 | Read Anthropic's "Building Effective Agents" post. Map each of your production flows to one o… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the swarm is cheaper and loses the seat a policy sits in

Both patterns ship and both route with the same `classify`, so the conversion is
already done and the diff is pure cost. Over the lesson's three tasks the
supervisor spends 6 ops and the swarm 5 — 17% cheaper — and the two produce the
same three answers. Nothing improves except cost, because nothing else differs.

The saving is less robust than it looks. The swarm starts at
`list(SPECIALISTS)[0]`, currently `"refund"`, so its cost depends on the match
between that key and the traffic mix. On an eight-task workload skewed to refunds
it costs 10 ops; reordering `SPECIALISTS` so `"sales"` comes first costs 15 — a
50% swing with no change to any agent, any task or any answer. The supervisor is
2 ops per task whatever arrives; the swarm is 1 or 2 depending on where it
happened to start. Flat cost is what makes a supervisor capacity-plannable.

What breaks is subtler than the cost. `swarm` carries `hops` as a local and
returns `(trace, ops)` — the hop count is computed and unreadable — and
`supervisor_worker` has no loop to bound at all. The saving comes from deleting
the seat where a routing policy, a token budget or an audit record would sit.
That is the lesson's "harder to reason about (no single point of control)" as an
arithmetic fact: you save one op per matched task and give up the only place a
rule can live.

### 2 — the counter ships, and the task it stops is dropped silently

`swarm` already reads `while hops < 3`, so the counter is there. Whether it
catches bouncing is a question about `classify`, which is a pure function of the
task text: the same agent asked twice gets the same answer, so the second hop
always lands. With the shipped router the maximum hop count is 1 and the counter
never fires. **A->B->A bouncing cannot occur, so the counter cannot catch it.**

Making it testable needs a router whose answer depends on where it is standing.
Swapping one in takes hops to `[3, 3, 3]` — and the loop then exits *by
condition* rather than by `break`, which appends no result line. Nine trace lines,
zero answers, and the function returns normally. A caller sees a shorter trace and
a lower op count and cannot distinguish "handled cheaply" from "gave up". A
counter that bounds a loop without recording why it stopped converts a hang into a
silent drop, which is worse.

Two refinements the measurement suggests. Refusing after N handoffs is not
detecting a cycle: a router that routes legitimately through four specialists
never repeats inside a budget of 3, so the counter fires at 3 hops with 0 answers
while a real 2-cycle would have been caught at 1. Remembering visited agents
catches the cycle in one hop *and* lets the long route finish — a set per task
buys both. And the op count rises while nothing progresses: a dropped task costs
3 ops against a handled one's 1, so the aggregate goes *up* as routing gets worse.
A cost metric that increases on failure is not a health signal without the answer
count beside it.

### 3 — the flat router reads every description on every task

The shipped `hierarchical` is two levels over three specialists, so the structure
ports directly; what the exercise needs is the arithmetic, because "the context
budget fails" is a number. A router's prompt carries one description per candidate
it can choose between, so flat cost is linear in the population and nested cost is
linear in the branching factor.

At 350 tokens per description plus a 300-token frame — stated assumptions, swept
below — a flat router over 12 specialists costs 4 500 tokens and fails a
4 000-token budget. Nesting into 4 teams of 3 costs 1 700 at the top and 1 350 at
a sub-router: 1 700 on the deepest path, 62% less, and inside the budget.

Sweeping the population finds the trigger. Flat routing first exceeds 4 000 tokens
at **11 specialists**; two-level routing survives to **44** — a 4.0x wider
population for one extra hop. The lesson's rule, nest "only when supervisor
context budget fails", has a computable threshold: wherever `300 + 350n` crosses
the limit. That is worth computing before adopting a topology, because the answer
moves with the description length and the model's context, not with how enterprise
the org chart feels.

The trade is tokens for hops. `hierarchical` spends 3 ops per task against
`supervisor_worker`'s 2, and the 12-specialist version spends the same 3 — depth
costs one op per level regardless of width. Flat is one router call reading 4 500
tokens; nested is two calls reading 3 050 between them. Fewer tokens, more round
trips, and which is better depends on which is binding.

One thing to fix while porting: the shipped hierarchy calls `classify(task)` for
`top_label` and *again* for `sub_label`, so the top level's decision is a function
of the answer it is supposedly narrowing down to. One of its three ops per task —
33% of the pattern's cost — can be removed without changing any answer. That is
the lesson's "fake hierarchy" pitfall present in the reference implementation:
three layers where two do the work.

### 4 — three of the four metrics are measurable and accuracy is not

On a 60-task production-shaped mix (refunds dominate, sales rarest):

| pattern | ops | serial depth | answers | trace lines per answer |
|---|---|---|---|---|
| swarm | 90 | 1.5 | 60/60 | 1.5 |
| supervisor | 120 | 2.0 | 60/60 | 2.0 |
| hierarchical | 180 | 3.0 | 60/60 | 3.0 |
| debate | 300 | 2.0 | 60/60 | 4.0 |

Swarm wins cost and latency; debate loses both. **Accuracy is a tie by
construction** — all four call the same `classify`, so any profile reporting an
accuracy difference between them is reporting noise in its fixture. That is the
most useful thing this exercise can teach: the four patterns differ in *how* work
is routed, not in *what* the router knows, and a topology change cannot improve a
routing decision the classifier was going to get wrong anyway.

Debate costs 2.5x the supervisor to reproduce its answer exactly. Its three
debaters each call the same deterministic `classify`, so `proposals` is three
copies of one label and `Counter.most_common` is a formality: 0 of 60 answers
differ. Debate only pays for itself when the proposers are genuinely different —
which is Lesson 25's finding arriving from the topology side.

Debuggability is the metric the exercise asks for and the one the traces are
worst at. Across all four, **zero** lines carry a score, a confidence or a reason.
Lines per answer rank the patterns 1.5, 2.0, 3.0, 4.0 — and more trace is not more
explanation, because 180 of debate's 240 lines are three agents repeating one
label. Ranking on volume would put debate first.

Finally, latency and cost disagree, which is why the lesson orders the patterns
rather than scoring them. Debate's 300 ops sit at serial depth 2.0 because its
proposals are parallel; hierarchical's 180 ops sit at depth 3.0. On a latency
budget the worst choice is hierarchical; on a cost budget it is debate. The
shipped op counter sums everything and cannot say so — separating parallel fan-out
from serial depth is a one-line change that changes the answer.

### 5 — mapping flows to the four, and the ones that do not map

**Anthropic's guidance** gives the decision order — single agent plus workflow
patterns first, supervisor-worker at 2–4 specialists, swarm when latency beats
reasoning clarity, hierarchical only when the supervisor's context budget fails,
debate when accuracy beats cost — under the claim that "success in the LLM space
isn't about building the most sophisticated system. It's about building the right
system for your needs."

Mapping real flows against that order, three things come out.

**Most flows map to step 1, and the mapping exercise is how you find out.** A
support triage flow that classifies and dispatches is *routing* (Lesson 12), not
supervisor-worker — there is no supervisor deciding whether to loop, only a
classifier picking a branch. Exercise 4 makes the distinction concrete: all four
topologies here produce identical answers because the actual decision is one
`classify` call. If a flow's answer does not change when you collapse it to a
single agent with a router, it was never multi-agent; it was a switch statement
with extra hops. The lesson's "topology-first thinking" pitfall is exactly this
mistake, and the cheapest test for it is the one exercise 4 runs.

**Exercise 3 turns step 4 into a calculation.** "Only when supervisor context
budget fails" sounds like judgement and is not: at 350 tokens per specialist
description and a 4 000-token budget, the threshold is 11 specialists. Below that,
nesting adds a hop and buys nothing; above it, flat routing does not fit. A team
can compute its own number from its own description lengths in an afternoon, and
should, because the alternative is adopting hierarchy for organisational reasons
and inheriting the "fake hierarchy" pitfall.

**Three shapes do not map cleanly, and all three are the same shape underneath.**

*Human-in-the-loop steps.* A flow where a person approves a refund above a
threshold is not any of the four — the human is not a specialist the router
dispatches to, they are a gate that suspends the flow. It maps onto workflow
patterns plus a pause (Lesson 13's interrupt), and forcing it into
supervisor-worker makes the human an "agent" whose latency is hours.

*Long-running and resumable work.* A flow that starts today and finishes when a
vendor replies has no topology at all in these terms; its hard problem is
durable state, not routing. Modelling it as a swarm produces a swarm whose agents
are separated by days.

*Fan-out over data rather than over expertise.* Processing 10 000 documents with
one prompt is parallelisation (Lesson 12), not a swarm — the agents are identical
and never hand off. It looks like a topology because there are many agents, and it
is a `map`.

What the three have in common is that the four patterns describe *who decides
where work goes*, and these flows' difficulty is elsewhere: in waiting, in
persistence, in volume. The honest answer to "map each flow to one of the four" is
that the ones which resist mapping are the ones where the topology was never the
interesting decision — which is the guidance's own point, read strictly.
