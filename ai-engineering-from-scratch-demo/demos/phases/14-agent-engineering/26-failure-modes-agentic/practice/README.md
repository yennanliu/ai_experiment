<!-- generated:start -->
# 14-agent-engineering / 26-failure-modes-agentic

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/26-failure-modes-agentic/) · upstream spec
`phases/14-agent-engineering/26-failure-modes-agentic/docs/en.md`

```bash
uv run demo practice run 26-failure-modes-agentic --ex 1
uv run demo explain 26-failure-modes-agentic --ex 1
uv run pytest demos/phases/14-agent-engineering/26-failure-modes-agentic
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a detector for "success hallucination": agent returns success but the target state is unc… | code | T0 | `ex01_the_detector_is_shipped_and_gated_on_a_verb_list.py` |
| 2 | Tag 100 real traces from a product you've built. Which mode dominates? What's the cost of fix… | code | T0 | `ex02_the_mode_that_dominates_by_count_is_the_cheapest_to_fix.py` |
| 3 | Implement a "cascade radius" metric: given a failure at step N, how many downstream steps did… | code | T0 | `ex03_the_tag_is_a_boolean_and_the_radius_is_thrown_away.py` |
| 4 | Read MASFT's 14 failure modes. Pick three that apply to your product. Write detectors. | explain | T0 | prose, below |
| 5 | Wire one detector into a CI job: fail the build if >=5% of traces tag a mode. | code | T0 | `ex05_a_five_percent_gate_on_a_hundred_traces_is_five_traces.py` |
<!-- generated:end -->

## Answers

### 1 — the detector is shipped, and its recall is a property of the user's vocabulary

`detect_success_hallucination` is already in `DETECTORS`, so the work is finding
what it misses. It fires only when the *request text* contains one of six verbs —
`write`, `create`, `save`, `update`, `edit`, `make` — which means a user who says
"delete the temp files", "send the summary", "deploy the build", "rename the
branch", "archive the thread", "publish the post" or "revoke the token" gets no
detection at all. Over twelve traces that claim success with the target state
unchanged, the shipped gate catches five and misses seven.

Three variants, measured on the same corpus:

| gate | caught | false positives |
|---|---|---|
| shipped verb list | 5 / 12 | 0 |
| probe only (no gate) | 12 / 12 | 4 |
| mutating tool call made | 12 / 12 | 0 |

Dropping the gate entirely is not the fix — it flags read-only work where nothing
was supposed to change. The fix is to gate on what the agent *did* rather than on
what the user *said*: if the trace contains a mutating call, claims success, and
the state is unchanged, that is a success hallucination regardless of phrasing.
What the agent did is observable; what the user meant is a guess.

Two structural gaps sit underneath. `target_state_changed` is one of `Trace`'s
six fields, supplied by whoever constructed the trace, and none of the module's
ten functions computes it — the lesson's mitigation is "re-probe state: was the
file actually created?", and the code has a boolean where the probe should be.
And no detector reads `status` on any step, so the lesson's own headline example
— an agent that hallucinates success on a 400 — is tagged by zero of the six
detectors when the request has no write verb and only one call follows the error.

### 2 — scope_creep dominates by count, and tool_misuse is the one to fix

Running the demo's own seven trace shapes, cycled to 100, through `tag()`
unmodified gives: `scope_creep` 29, and `hallucinated_action`,
`cascading_errors`, `tool_misuse`, `success_hallucination` at 14 each, with
`context_loss` at 0.

"Cost of fixing it" is two numbers, and ranking by either alone gives a different
answer. Priced at incident cost per occurrence divided by engineering days to
fix, the returns are `tool_misuse` 42.0 against `scope_creep` 8.3. The mode to
fix first is joint *second* by count, because it is cheap to fix (argument
validation, Lesson 06) and expensive to leave. A distribution is an input to the
decision, not the decision.

Three things make the distribution itself less trustworthy than it looks.

`context_loss` fires on nothing — 0 of 100, and 0 of the demo's own 7.
`detect_context_loss` takes the first word after `"do not"`, which is a *verb*,
and searches the tool arguments for it: `"do not modify src/"` looks for
`"modify"` inside `{"path": "src/foo.py"}`. The constraint names the object and
the detector extracts the predicate. It is not that this mode is rare; it is that
this detector cannot see it.

The counts are traces, not events. Every detector returns on its first match, so
a trace with three hallucinated calls contributes one. Counting occurrences moves
`hallucinated_action` from 14 to 42 and *flips the dominant mode*: `scope_creep`
leads by trace and `hallucinated_action` leads by event. Which to fix depends on
which question the count was answering — "how many user sessions were damaged" or
"how many bad calls were made".

And the cost model double-counts. The 100 traces carry 85 tags between them and
14 carry more than one, so 14 of the 29 `scope_creep` traces would still fail
after `scope_creep` is fixed. Multiplying "traces tagged X" by "incident cost per
incident" bills the same trace once per tag.

### 3 — the radius is computed, compared against 2, and thrown away

`detect_cascading_errors` already walks the trace counting steps after the first
error. It compares that count against 2 and returns a string, so the metric the
exercise asks for exists for one line and is discarded.

Keeping it exposes that "downstream" and "affected" are different questions.
Counting every tool call after the first error gives radii `[1, 2, 3, 4, 5, 1, 3,
4, 5, 3]`, mean 3.1. Counting only the calls whose arguments actually carry a
value derived from the failed step gives `[1, 1, 1, 2, 3, 1, 1, 2, 3, 1]`, mean
1.6. Positional radius overstates blast by 1.9x on the same traces — it counts
every step that happened *after*, including the ones that were fine.

The threshold costs recall at the low end: `downstream_ops >= 2` means a failure
with exactly one step after it is not a cascade, so 2 of 10 traces are tagged
clean — and both of them really did carry the failed value into the next call. A
radius-1 cascade is the most common kind and the detector's floor is two. The
lesson calls cascading "the killer".

The loop also sets `saw_error = True` and never resets it, so a trace with three
separate errors reports one cascade spanning 9 steps where the per-error radii
are 4, 2 and 1. One number for three incidents — which is under-counting dressed
as the over-alerting fix.

Finally, `tag()` returns `list[str]`, and the six detectors return six strings and
zero step indices. Two traces with radii 2 and 5 produce byte-identical output. A
dashboard built on `tag` can rank modes by frequency and never by blast, which is
exactly backwards for a mode whose defining property is how far it spreads.

### 4 — three MASFT modes worth detectors, and what makes a detector cheap

**MASFT (Berkeley, arXiv:2503.13657)** is the Multi-Agent System Failure
Taxonomy: 14 failure modes in 3 categories, with an inter-annotator Cohen's Kappa
of 0.88 — the categories are reliably distinguishable by humans, which is the
property that makes automated detectors worth building against them. Its central
claim is the one to carry: these are *design flaws in multi-agent systems*, not
LLM limitations that a better base model will fix. A taxonomy of design flaws is
a list of things to gate, not a list of things to wait out.

The three categories are specification and system design, inter-agent
misalignment, and task verification and termination. Picking one mode from each
gives coverage of the failure *surfaces* rather than three variants of the same
bug, and the three below are the ones this codebase can actually check.

**Specification: disobey task specification.** The agent does work the request
did not ask for. This is `detect_scope_creep`'s territory, and the shipped
version gates on whether the request text contains a write verb — the same
vocabulary dependency exercise 1 measured. A detector that survives contact with
users compares the *set of tools invoked* against a per-request allowlist derived
at plan time, not against the phrasing. Cheap, because the allowlist is already
needed for Lesson 21's per-step gate.

**Inter-agent misalignment: information withholding.** One agent has a result
another needs and does not pass it on. The detectable signature is a value that
appears in one agent's output and is absent from the next agent's input when the
next agent's task requires it — which is exactly the data-flow check exercise 3
built for cascade radius, run in the other direction. Detecting it needs step
inputs and outputs to be linked, which `TraceStep` supports via `result` and
`args` and `tag()` discards.

**Verification and termination: premature termination.** The agent stops and
declares success before the goal state holds. That is exercise 1's success
hallucination, and the detector that works is the state probe rather than the
verb gate: 12/12 against 5/12 on the same corpus.

What these three have in common is the reason the taxonomy is useful rather than
merely tidy. Each is checkable against *environment state* or *trace structure*,
not against the model's own account of what it did. The modes MASFT lists that
are hardest to detect — reasoning-action mismatch, ignored other agent's input —
are hard precisely because the evidence is in the reasoning text, where an LLM
judge is needed and Lesson 24's grounding problem returns. Start with the ones
where the environment can be asked.

One caution the Kappa figure earns: 0.88 is agreement between *human annotators
reading full traces*. A keyword detector is not an annotator. Exercise 2's
`context_loss` detector scores 0 on 100 traces that include 14 genuine constraint
violations — a detector can be perfectly consistent with itself and share nothing
with the category it is named after. Validate detectors against a hand-labelled
sample before reporting their rates, or the distribution is a fact about the
regexes.

### 5 — the gate is four lines, and at 5% it is a coin flip

Wiring `detect_tool_misuse` into CI is straightforward: tag every trace, compute
the share carrying the mode, fail if it is at or above 5%. What is worth
measuring is how often that verdict is real.

Over 200 seeded CI runs of 100 traces each, against a population whose true
`tool_misuse` rate is *exactly* 0.05, the build fails 112 times and passes 88 —
44.0% of runs get the minority verdict on identical code. Running ten times as
many traces per build makes it 47.5%, which is *worse*, and that is not a bug in
the experiment: as the estimate sharpens around a threshold set at the true rate,
the verdict converges on a coin flip. More data does not fix a threshold placed
where the population sits.

What fixes it is the population being somewhere else. At a true rate of 0.02 the
flap rate is 2.0% and at 0.10 it is 4.5%. So the gate is decisive exactly when
the answer was obvious, and undecidable exactly when it matters — which is the
argument for gating on a *change* against a baseline rather than on an absolute
level.

The same point arrives three more ways. Gating on "any mode" instead of a named
one fails every build: the corpus tags 5 distinct modes and 72.0% of traces carry
at least one, 14.4x the threshold. That is the over-alerting pitfall on day one.
The absolute threshold cannot see an improvement — a release that takes
`scope_creep` from 29% to 12% still fails, and one that doubles 4% to 4.9% still
passes, so the build stays green while the number heads for the limit; the
lesson's "no baseline" pitfall is exactly this. And the gate inherits its
detector's recall: wiring `detect_context_loss` in gives 0 hits on a corpus
containing 14 writes to a path an explicit constraint forbade, so that build can
never fail. A gate on a detector that never fires is a green light with a policy
attached.

The version worth shipping: gate on a named mode, against last week's rate rather
than a constant, with a sample size chosen so the flap rate at the *current* rate
is acceptable — and validate the detector against hand labels before trusting any
of it.
