<!-- generated:start -->
# 14-agent-engineering / 12-anthropic-workflow-patterns

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/12-anthropic-workflow-patterns/) · upstream spec
`phases/14-agent-engineering/12-anthropic-workflow-patterns/docs/en.md`

```bash
uv run demo practice run 12-anthropic-workflow-patterns --ex 1
uv run demo explain 12-anthropic-workflow-patterns --ex 1
uv run pytest demos/phases/14-agent-engineering/12-anthropic-workflow-patterns
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement routing with a confidence threshold. Below threshold -> escalate to human. Where do… | code | T0 | `ex01_low_confidence_and_unknown_category_reach_the_same_handler.py` |
| 2 | Add a timeout to `parallel_vote`. What happens when one call hangs? How do you aggregate with… | code | T0 | `ex02_parallel_vote_is_a_list_comprehension.py` |
| 3 | Turn `evaluator_optimizer` into a bandit: keep the top-2 outputs across iterations so a late… | code | T0 | `ex03_the_loop_returns_whatever_it_happened_to_try_last.py` |
| 4 | Combine prompt chaining with routing: a router picks one of three chains. Measure token cost… | code | T0 | `ex04_the_chain_forgets_the_input_and_pays_for_the_summary.py` |
| 5 | Pick one of your production features. Draw the workflow graph. Count steps. Would an agent ac… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 5 asks you to pick one
of your own production features, so it is answered in prose as a method
rather than as a measurement of someone else's system.

The recurring finding is that **each pattern's signature is one field short
of the thing the exercise asks for**. `route`'s classifier is typed
`str -> str`, so a confidence has nowhere to travel. `evaluator_optimizer`'s
evaluator returns `(bool, str)`, so a bandit has no score to rank by.
`parallel_vote` returns `(winner, Counter)`, so a quorum check cannot tell
3-of-5 from 3-of-3. In every case the pattern is correct and the *return
type* is what blocks the extension — which is the honest argument for these
being 10-15 lines: the code is small because the contracts are thin.

The second thread is that **two of the five patterns do not do what their
names say**. `parallel_vote` is a list comprehension, so "parallelization"
runs sequentially and a hang costs the sum of the latencies rather than the
maximum. `evaluator_optimizer` rebinds `candidate` each iteration and returns
it after the loop, so on a run that never passes it returns the *last*
attempt, not the best one.

What holds: the patterns are genuinely small and genuinely separable.
Composing routing with chaining (exercise 4) took no glue at all, and the
cost of doing so is countable to the character.

### 1 — low confidence and unknown category reach the same handler

**ANSWER: at a 5:1 misroute-to-escalation cost ratio the threshold lands at
0.3.** Over **20** tickets the margin-based classifier is right **14** times,
so routing everything costs **30**. Thresholding at **0.3** escalates **9**,
misroutes **0**, and costs **9**. Tier-1 support is the case where that ratio
is high — a wrong reply costs the reply, the re-contact, and the customer's
patience — which is why the answer is "escalate readily".

**FINDING: the threshold is stable and the cost is not.** At ratios 2, 5 and
10 the optimum stays at **0.3** while routing everything costs **12**, **30**
and **60**. That is a statement about *calibration*: every error this
classifier makes sits at confidence **0.25** or below, so a single cut
separates them at any price. A threshold that moves when you change the cost
ratio is telling you the score is miscalibrated — which makes the sweep a
diagnostic as much as a tuning exercise.

**FINDING: `route` cannot carry a confidence.** The classifier's return value
is used directly as a dictionary key, so the only way to express "unsure" in
the shipped signature is to return a label that is not in `handlers`.

**FINDING: which is exactly what the default handler already does.**
`handlers.get(label) or handlers.get("default")` sends an unknown label to
`default`, so a deliberate escalation and a classifier that invented a
category return the same string. Two different events, one observable
outcome — and the only thing distinguishing them is the label in the tuple,
which no handler ever sees.

### 2 — `parallel_vote` is a list comprehension

**ANSWER: a timeout plus a quorum, and the timeout changes the answer.** With
five voters, one hanging for **30** units against a timeout of **2**: the
shipped sequential loop finishes at **34** units and elects `no`; the timed
version finishes at **2** on **4** votes and elects `yes`. A quorum of **3**
returns a winner at 4 votes and `None` at 2. Dropping a vote is not free —
here it is decisive.

**FINDING: the hang costs the sum, not the maximum.** Sequential total
latency is **34** where a concurrent run would be **30** and a timed
concurrent one **2**. This is the sharp version of the answer to "what
happens when one call hangs": in the shipped code, *everything after it
waits*, and adding a timeout without adding concurrency saves nothing,
because the loop still has to reach the slow call. The pattern is named
parallelization and is not parallel.

**FINDING: a tie is broken by arrival order.** `Counter.most_common(1)[0]`
returns insertion order among equal counts, so four surviving votes split
**2-2** elect whichever answer came back first. Once a timeout is in play,
that is a network-timing decision presented as a majority verdict. The fix is
a stated tie-break rule, which the shipped return type has no room for
either.

**FINDING: the return value cannot say how many voted.** The counter holds
only the votes that arrived, so a caller enforcing a quorum has to be told
`n` separately — the aggregation half of the exercise needs a wider return
type before it needs an algorithm.

### 3 — the loop returns whatever it happened to try last

**ANSWER: a top-2 bandit over a scored evaluator.** Across five iterations
scoring **3, 9, 4, 2, 1**, the bandit returns the iteration-2 candidate at
**9** and the shipped loop returns the iteration-5 candidate at **1**, with
the **4** kept as runner-up.

**FINDING: the information was never lost, only the return value was.**
`trace` records every candidate, so the best attempt is recoverable after the
fact — the best score in the shipped trace is the same **9** the bandit
returns. That is the cheapest possible fix: read the trace rather than the
return value, and the loop does not change at all.

**FINDING: the evaluator has no score to rank by.** It is typed
`Callable[[str, str], tuple[bool, str]]` — a verdict and a message, no
number. A bandit either parses the judge's prose or the signature widens.
Pass/fail is enough to *stop* and not enough to *choose*, and the difference
between those two jobs is why the overwrite went unnoticed.

**FINDING: the bug only fires when nothing passes.** On a successful run
`evaluator_optimizer` returns from inside the loop, so the shipped answer and
the bandit's agree. The failure is invisible on the happy path — which is
where most testing happens, and why this shape of bug survives into
production.

### 4 — the chain forgets the input and pays for the summary

**ANSWER: routed chains cost 507 characters across 9 calls; one big prompt
costs 283 across 3.** Chaining is **1.8x** the characters and **3.0x** the
calls for the same three inputs. Characters are the honest unit here because
`ScriptedLLM` records every prompt it was handed — nothing is estimated.

**FINDING: the chain pays for its own intermediate output.** Each step
formats the previous *output* into the next prompt, so **90** of the **507**
characters — **17.8%** — is text the system wrote and then paid to read back.
That fraction grows with the length of the intermediate results, which is
precisely the case where chaining is attractive.

**FINDING: the router is 3 of the 9 calls.** Classification is a model call
in this pattern, so routing adds **193** characters — **38.1%** of the total
— before any work happens. On short inputs the routing decision can cost more
than the step it routes to, which is worth knowing before adding a router to
save money.

**FINDING: the chain forgets the input after step one.** `prompt_chain` sets
`current = output`, so the original text appears in **3** of the **6** chain
prompts — once per input, at step one only. Anything step 2 needs must have
survived step 1's summary. The single big prompt keeps everything by
construction, which means the **1.8x** is not a like-for-like comparison: the
cheaper option is also the one that sees more.

**The honest reading of the cost comparison:** the single prompt is a lower
bound a real system rarely reaches, because it needs one prompt per *task
shape* rather than one per *step* — three inputs here, but thirty task
shapes in a real support queue. Chaining's cost is linear in steps and its
prompts are reusable; the big prompt's cost is linear in shapes and its
prompts are not.

### 5 — count the steps, then count the decisions

*Cites "Where workflows beat agents".*

**The exercise asks you to draw your own graph, so what is useful here is the
method and the thresholds to apply to it — with the numbers this lesson
measured as the cost side of the decision.**

**Step 1: draw the graph and count two things.** Steps, and *branch points
whose next step depends on the previous result*. The step count is what
everyone measures; the second number is the one that decides the question.
A ten-step pipeline with zero result-dependent branches is a workflow with a
long body. A three-step pipeline where step 2's existence depends on step 1's
output is an agent wearing a workflow's clothes, and it will grow branches
until someone admits it.

**Step 2: check the three conditions the lesson names, in order.**

- *Predictable*: can you enumerate the steps? If yes, the lesson's advice is
  that you should — and exercise 4 prices the alternative: routing plus
  chaining cost **1.8x** the characters and **3.0x** the calls of a single
  prompt for the same work. A workflow is not free; it is just bounded.
- *Cost-bound*: workflows have bounded step counts and agents can spiral.
  Exercise 2 is the version of this that bites: the "parallel" pattern is
  sequential, so its worst case is the *sum* of its calls' latencies. Bounded
  step count is not bounded latency unless someone checked.
- *Compliance-bound*: auditors want to read the graph. Exercise 1's finding
  is what that costs when the contract is thin — a deliberate escalation and
  a misclassification produce byte-identical outputs, so the audit log cannot
  distinguish "we asked a human" from "the classifier invented a category".

**Step 3: ask whether an agent would actually be better — and be specific
about which of the three agent advantages you are buying.** "Open-ended
research", "variable-length tasks", "novel domains". If none of the three
describes the feature, the answer is no, and the honest version of the answer
is that you want an agent because the workflow is annoying to maintain, which
is a different problem with a different fix.

**Step 4: if the answer is still no, look for the four bugs this lesson
found, because they are all in code you are about to write.** A router with
no confidence channel (exercise 1). An aggregator that cannot say how many
voted (exercise 2). A refine loop that returns its last attempt rather than
its best (exercise 3). A chain whose later steps cannot see the input
(exercise 4). None of these are exotic; all four are in a reference
implementation written to demonstrate the patterns, and all four survive
because the failure is silent on the happy path.

**The one-line version:** count the result-dependent branches, not the steps.
Zero branches means a workflow and the lesson's cost argument applies. Many
branches means you already have an agent and the question is whether you are
going to admit it in the code.
