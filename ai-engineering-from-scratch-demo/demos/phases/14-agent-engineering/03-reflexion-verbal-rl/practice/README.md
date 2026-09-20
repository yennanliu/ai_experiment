<!-- generated:start -->
# 14-agent-engineering / 03-reflexion-verbal-rl

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/03-reflexion-verbal-rl/) · upstream spec
`phases/14-agent-engineering/03-reflexion-verbal-rl/docs/en.md`

```bash
uv run demo practice run 03-reflexion-verbal-rl --ex 1
uv run demo explain 03-reflexion-verbal-rl --ex 1
uv run pytest demos/phases/14-agent-engineering/03-reflexion-verbal-rl
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Switch from binary to scalar evaluator that returns a distance metric (how far from target).… | code | T0 | `ex01_the_binary_evaluator_already_returns_the_scalar.py` |
| 2 | Add a TTL of 10 trials to reflections. Do older reflections hurt or help after that point? | code | T0 | `ex02_the_ttl_never_fires_because_the_count_bound_is_tighter.py` |
| 3 | Implement heuristic evaluator: mark the trial as stuck if the same action repeats. How does t… | code | T0 | `ex03_the_heuristic_verdict_has_no_channel_to_the_reflector.py` |
| 4 | Run Reflexion with an adversarial Actor that ignores reflections. What is the minimum reflect… | code | T0 | `ex04_the_minimum_is_a_signed_magnitude_and_a_reader_for_it.py` |
| 5 | Read Section 4 of the Reflexion paper on AlfWorld. Reproduce the 130% success-rate improvemen… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 5 asks for a reading of
the paper, so it is answered in prose.

The recurring finding is that **the toy's Actor is conditioned on
`len(memory.items)`, not on what the reflections say**. `Actor.act` branches
at 0, 1 and "2 or more" and never touches `Reflection.text`. Everything
downstream follows: the evaluator cannot matter (exercise 1), the TTL cannot
matter (exercise 2), the heuristic verdict cannot reach the reflector
(exercise 3), and no wording can force the Actor to notice (exercise 4). The
demo's improvement from "never converges" to "converges on trial 3" is
reproduced exactly by a reflector that writes `doing great, keep going`.

That is worth stating plainly because the lesson's own summary line — "with
one reflection, the actor corrects; with two, it converges" — is true of this
code and is *not* the mechanism the paper describes. Exercise 4 builds the
smallest Actor that does read the text, and finds the reflector already
writes everything such an Actor needs.

What holds: the loop shape is right. Trial boundary, evaluation, reflection,
a buffer that survives the reset, a fresh trajectory that sees it. And the
buffer's bounded growth makes memory rot measurable rather than rhetorical —
exercise 2 puts a number on it.

### 1 — the binary evaluator already returns the scalar

**ANSWER: no, it does not converge faster — 3 trials either way.** The
scalar evaluator produces the same attempts `[1,2,3]`, `[5,6,7]`, `[6,7,7]`
and succeeds on the same trial. `binary_evaluator` returns
`(total == target, total - target)`; the distance was already the second
element, and `SelfReflector` was already consuming it.

**FINDING: the Actor is a function of `len(memory.items)`.** Its body names
`len(memory.items)` once and `Reflection.text` **0** times. A reflector that
writes `doing great, keep going` on every trial converges in **3** on
identical attempts.

**FINDING: so the evaluator cannot matter either.** An evaluator that reports
distance **0** on every trial drives the identical trajectory, because both
evaluators add exactly one `Reflection` per failure. The only channel from
evaluator to policy is the *length* of the buffer.

**FINDING: what the scalar would buy, if anything read it.** Across the two
failures the binary verdict takes **1** distinct value — no gradient — while
the distance takes **2**, `14` then `2`, which orders the failures. That
ordering is the scalar evaluator's whole advantage, and it is entirely
unrealised here.

### 2 — the TTL never fires, because the count bound is tighter

**ANSWER: a TTL of 10 trials, on a target the Actor can never reach.** At
trial **15** the buffer holds **6** reflections with the TTL in place and
**6** without it: `max_len = 6` evicts trial 8 long before age 10 could. Lift
the count bound and the TTL holds **10**. Neither number changes a single
action — trials 3 through 15 are all `[6,7,7]`.

**Do older reflections hurt or help after that point? Neither, and the reason
is the threshold.** `Actor.act` treats every buffer with ≥ **2** entries
identically, so a TTL of 3 and a TTL of 10 give byte-identical runs. A TTL of
**2** starves the counter and pins the Actor at `[5,6,7]` for **14** straight
trials. Ageing reflections out can only hurt here, and only by dropping the
count below a branch boundary — which is a property of this policy, not of
TTLs.

**FINDING: six slots holding one sentence.** From trial 3 the attempt is
constant, so the reflector writes the same line every trial. At trial 15 the
**6** stored reflections carry **1** distinct text and `as_prompt` renders
**6** lines, **5** of them duplicates. That is the lesson's "memory rot" with
a number on it: the prompt grows and the information does not. Deduplication
would free five sixths of the buffer before any TTL is needed.

**FINDING: a count bound and an age bound are different units.** They agree
only while exactly one reflection is written per trial. Skip two writes and
at trial 15 the count bound holds six entries back to trial 9 while the age
bound holds nine back to trial 6 — same rule, two answers. A system that
writes a reflection *and* a heuristic verdict (exercise 3) writes two per
trial, and the two bounds diverge immediately.

### 3 — the heuristic verdict has no channel to the reflector

**ANSWER: `stuck_evaluator` fires on trials 4 through 8.** The Actor settles
on `[6,7,7]` at trial 3, so **5** of **8** trials repeat the previous action.
On the reachable target the same heuristic fires **0** times, because the run
ends before anything can repeat — a rail that is silent exactly when the run
is healthy, which is the right shape.

**How it interacts with the Self-Reflector: it cannot.**
`reflect(self, attempt, delta)` takes the action and the distance, and a
verdict has no parameter to arrive in. The reflection written on the fifth
stuck trial is byte-identical to the one written when that action was new.
The heuristic can stop the loop; it cannot change a word of what the loop
remembers.

**FINDING: the only interaction available is arithmetic.** A heuristic that
writes its verdict into the same buffer changes `len(memory.items)`, which is
the one thing the Actor reads. Seeding the buffer with a single entry before
trial 1 converges at trial **2** instead of **3** — and a true reflection and
`the moon is made of cheese` do it equally well. An evaluator that writes into
the buffer moves this policy by counting, which means any measured
"improvement" from adding a heuristic is uninterpretable until the Actor
reads text.

**FINDING: stuck is a narrower predicate than failing.** An Actor cycling
**3** distinct wrong answers fails **8** times out of **8** and is marked
stuck **0** times. Repetition catches the run that has given up, not the one
exploring uselessly — one failure signature, not a class of them. The
lesson's other suggested heuristic (trajectory length) catches the second
case and misses the first.

### 4 — the minimum is a signed magnitude, and a reader for it

**ANSWER: the adversarial Actor is the shipped one, and the minimum
reflection content is a signed magnitude.** Against an Actor that reads text:
`be more careful next time` never converges in **6** trials and leaves **1**
distinct attempt; the lesson's own `sum 6 is 14 short; pick larger values`
converges on trial **2**; adding an explicit constraint — `next attempt must
sum to 20` — also converges on trial **2**. The extra sentence buys **0**
trials, so the minimum is a quantity plus a direction.

**FINDING: no prompt can reach an Actor whose input is a count.** Across all
**3** wordings the shipped Actor produces **1** distinct trajectory. That is
the honest answer to "what is the minimum prompt engineering": there is none,
because prompt engineering acts on the reflection and the Actor's signature
is `act(self, memory)` with `memory` consulted only through `len`. The fix is
a parameter change, not a wording change.

**FINDING: the counting Actor is a lookup, not a weak policy.** Move the
target to **24** and it fails **6** times out of **6**, settling on `[6,7,7]`
— the answer to a different question. The reading Actor finds `[9,9,6]` on
trial **2** of the same run. What looked like learning was a table, and the
table is indexed by trial number.

**FINDING: the Self-Reflector already writes the minimum.** Its first
reflection is `sum 6 is 14 short; pick larger values`; one regex extracts
`14` and one word gives the sign. The reflector needed **0** changes. That
is the quiet lesson of the exercise: the reflection side of Reflexion is the
easy side.

### 5 — the delta is the trial boundary, not the reflection

*Cites "The three components".*

**The key delta versus vanilla ReAct is that Reflexion has two loops, and the
outer one has a memory.**

Vanilla ReAct is the inner loop only: observe, think, act, repeat until the
episode ends. Its memory is the message buffer, and the message buffer is
precisely the thing that is thrown away when the episode ends. A ReAct agent
that fails an ALFWorld task and is asked again starts from the same prior
over actions it had the first time. It can recover *within* a trajectory —
that is what the reasoning trace buys — and it cannot recover *across* one.

Reflexion adds four things, three of which are cheap. The lesson's own
diagram is the list: an **Evaluator** that runs at the episode boundary
rather than at each step, a **Self-Reflector** that turns the failed
trajectory into one or two sentences, and **episodic memory**, "a list of
prior reflections, prepended to the next trial's prompt". The fourth is the
one that is easy to miss: the next trial **starts fresh**. The buffer is not
appended to the failed trajectory; it replaces it. That is what keeps the
prompt from growing linearly in trials and what makes the reflection a
summary rather than a log.

So the gradient is carried in words across an episode boundary. No weights
change, and — this is the part that makes the number large on ALFWorld — the
information that crosses the boundary is *small and reusable*. ALFWorld
failures are dominated by wrong-object, wrong-location and unmet-precondition
mistakes, and each is expressible in one sentence that stays true the next
time the same task class is drawn. That is the condition the lesson names
under "when it works": a clear failure signal, a reproducible task class, and
enough action budget left for the reflection to be actionable.

**On reproducing the number.** A 130% relative improvement is a statement
about a low base: ReAct was failing the majority of those tasks, so a
correction that fixes even a third of the failures more than doubles the
success rate. Reproducing the *result* needs the benchmark; reproducing the
*mechanism* needs only the four pieces above plus one property the toy in
this lesson does not have.

**What this lesson's code does and does not reproduce.** It reproduces the
loop shape faithfully, and it reproduces memory rot honestly — exercise 2
measures **6** buffer slots carrying **1** distinct sentence. What it does
not reproduce is the mechanism that makes the paper's number real: its Actor
is conditioned on `len(memory.items)`, so exercise 1 gets the same 3-trial
convergence out of a reflector that writes `doing great, keep going`, and
exercise 3 gets a trial shaved off by seeding the buffer with `the moon is
made of cheese`. A toy where nonsense and insight are interchangeable is
measuring the trial boundary, not the reflection.

Exercise 4 closes the gap with about ten lines: an Actor that reads the
newest reflection's text converges on trial **2** and keeps converging when
the target moves to **24**, where the shipped Actor fails **6** times out of
**6**. That contrast is the paper's delta in miniature — not "reflections
help" but "a policy conditioned on a verbal summary of its own failure
generalises to the next instance of the task class, and a policy conditioned
on the trial index does not."
