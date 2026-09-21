<!-- generated:start -->
# 14-agent-engineering / 05-self-refine-and-critic

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/05-self-refine-and-critic/) · upstream spec
`phases/14-agent-engineering/05-self-refine-and-critic/docs/en.md`

```bash
uv run demo practice run 05-self-refine-and-critic --ex 1
uv run demo explain 05-self-refine-and-critic --ex 1
uv run pytest demos/phases/14-agent-engineering/05-self-refine-and-critic
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run the toy with max_iterations=1. Does CRITIC still help? | code | T0 | `ex01_one_iteration_verifies_before_it_refines_and_never_again.py` |
| 2 | Replace the external verifier with a noisy one (random 30% false positives). What does the lo… | code | T0 | `ex02_a_false_pass_is_absorbing_and_looks_exactly_like_a_real_one.py` |
| 3 | Implement a "generator-critic on different models" variant: big model generates, small model… | code | T0 | `ex03_the_critic_that_wins_is_the_one_speaking_the_refiners_vocabulary.py` |
| 4 | Read CRITIC Section 3 (arXiv:2305.11738 v4). Name the three verification-tool categories and… | explain | T0 | prose, below |
| 5 | Map OpenAI Agents SDK's `output_guardrails` to CRITIC's verifier role. What does the SDK get… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Three ship code; exercises 4 and 5 ask for
readings of the CRITIC paper and the OpenAI Agents SDK, so they are answered
in prose.

The recurring finding is that **the toy's Self-Refine arm does not fail for
the reason the demo says it does**. `feedback_self` is accurate: it flags the
Germany error, then the Everest error, then passes the corrected text —
**3** of **3** correct verdicts. What stalls the loop is that `generate`
picks its next output by searching the last critique for the strings
`germany` and `everest`, and the self-critic says `capital` and `continent`.
Reword the same judgement to name the entities and Self-Refine converges in
**3** iterations, exactly matching CRITIC.

That is worth being precise about, because the demo's closing line —
"a self-critic can fail to flag its own confident-sounding hallucination" —
is a true statement about the world and not a description of this code. Here
the self-critic flags everything. The coupling between critic and refiner is
what decides the run, and exercise 3 measures it: count the critiques that
contain a word the refiner keys on, and that count predicts a clean exit in
all four arms.

What holds: the loop's stop condition, the iteration accounting, and the
grounded verifier's *shape*. What does not: `verify_external` iterates
`KNOWN_WRONG_FACTS` with a loop body that never reads `fact`, so **1** of
the **3** listed facts is never checked at all, and an output asserting
`the sun orbits the earth` passes clean before any noise is added.

### 1 — one iteration verifies before it refines, and never again

**ANSWER: no, CRITIC does not help at `max_iterations=1` — and neither arm
is distinguishable.** Both return **1** attempt, `verified=False`, carrying
the initial output with **2** known-wrong facts intact. `run_loop` verifies
at the top of the loop body and refines at the bottom, so a budget of *n*
buys *n* verifications and *n−1* verified refinements. At *n*=1 the `refine`
call still executes, produces a different output, and the loop discards it —
one refinement performed, zero verified.

**FINDING: CRITIC's minimum budget is 3, and Self-Refine has none.** Swept
from 1 to 10: CRITIC first reports verified at **3** and still stops at **3**
when given **10**; Self-Refine converges at no budget tested.

**FINDING: the minimum is set by one defect reported per pass.**
`verify_external` returns on its first failing check, so on an output with
**2** known-wrong facts it names the Paris/Germany one and stays silent about
Everest. Two defects therefore cost two correcting rounds plus one confirming
pass. The rule is budget ≥ defects + 1, and the verifier's serialisation is
what sets it — a verifier that reported all failures at once would converge
in 2.

**FINDING: at a budget of 1 the whole distinction is unobservable.** Same
output, same `verified` flag; only the critique strings differ, and nothing
downstream reads them. A single-pass guardrail does not make a system
CRITIC-shaped — it makes the choice of verifier invisible. This is the direct
answer to the exercise and it generalises: any "we added a verifier" claim
needs an iteration budget attached before it means anything.

### 2 — a false pass is absorbing and looks exactly like a real one

**ANSWER: at p=0.3, 52.8% of 2000 seeded runs stop early on a factually
wrong output marked verified.** The loop breaks on the first `ok`, so a false
positive is not a degraded signal — it is an exit. The arithmetic is one
Bernoulli per check that *ought* to fail: the trajectory has **2** of them,
so the closed form is `1 − (1−p)² = 0.51`, and the measurement lands
**0.018** above it.

**FINDING: the false pass is absorbing and unlabelled.** Every bad exit
carries `verified=True` and the critique `verifier: ok` — **1** distinct
string, byte-identical to a genuine pass. `Attempt` has **4** fields
(`iteration`, `output`, `critique`, `verified`) and not one of them records
that the verdict was sampled. Downstream, a 30%-noisy guardrail and a perfect
one produce the same artifact.

**FINDING: the exposure is the number of failing checks, not the noise
rate.** Measured at p = 0.1, 0.3, 0.5 the bad-exit rates track
`1 − (1−p)²` within **0.018**. A false positive on a check that was going to
pass anyway costs nothing; the exposure is exactly the number of times the
guardrail was the only thing standing between a defect and the exit. Halving
the defect rate helps as much as halving the noise rate.

**FINDING: the verifier already has a false negative before any noise.**
`verify_external`'s `for fact in KNOWN_WRONG_FACTS` loop computes `key` and
never uses it; the two checks inside are hard-coded string tests. **2** of
the **3** listed facts are actually tested. The fact list reads like a
configurable reference set and is decoration — which is the version of "the
2026 reality of most guardrail stacks" the exercise did not ask about and
that is worth more than the noise.

### 3 — the critic that wins is the one speaking the refiner's vocabulary

**ANSWER: a different critic wins by changing vocabulary, not by being
righter.** The shipped self-critic scores **3/3** on the trajectory and never
converges. The identical judgement, reworded to name the entities, scores
**3/3** and converges in **3** iterations — matching CRITIC exactly. With
accuracy held constant the outcome flips on two words.

So: does big-generates / small-critiques beat same-model? On this toy the
question does not have a model-size answer, and pretending otherwise would be
the interesting part of the exercise thrown away. What it has is a coupling
answer, and the coupling answer transfers: a smaller critic helps when it
produces *structurally different, concretely named* critiques, and hurts when
it produces vaguer ones. Anthropic's evaluator-optimizer advice — make the
two prompts substantially different — is aimed at rubber-stamping, and this
exercise shows the cost on the other side of that dial: prompts different
enough not to rubber-stamp can be different enough not to connect.

**FINDING: a rubber stamp is the fastest run in the lesson.** A critic that
always answers `looks good to me` converges in **1** iteration with **2**
known-wrong facts still in the output, against CRITIC's **3**, while scoring
**1/3** on accuracy. Any metric built on iterations-to-convergence ranks it
first. Convergence speed is a cost, never evidence.

**FINDING: shared words predict the outcome in all four arms.** Counting
critiques containing a word `generate` keys on: shipped **0**, rubber stamp
**0**, reworded **2**, external **2**. Every arm with a non-zero count ends
with a clean output and every arm with zero does not. The critic's provenance
predicts nothing; its vocabulary predicts everything.

### 4 — three categories, and what each one can actually decide

*Cites "CRITIC (Gou et al., arXiv:2305.11738, v4 Feb 2024)".*

**The three categories are retrieval, execution, and external judgement**,
and CRITIC's Section 3 pairs each with the task type it was evaluated on.

1. **Retrieval — a search engine or knowledge API.** Example: free-form
   question answering, where the model's claim is checked against retrieved
   evidence before the refiner sees it. The lesson's own list opens with
   exactly this ("a search engine for factual claims"), and the toy's
   `verify_external` is a stub of it: a claim is matched against a reference
   set rather than against the model's confidence.
2. **Execution — a code interpreter or calculator.** Example: mathematical
   program synthesis, where the candidate program is *run* and its output
   compared against the required answer. The lesson splits this into two
   bullets (code interpreter, calculator); they are one category, because the
   verdict comes from running something rather than from consulting
   something.
3. **External judgement — a scoring API or domain classifier.** Example:
   toxicity reduction, where a text-attribute API scores the continuation.
   The lesson's fourth bullet ("domain-specific verifiers — unit tests, type
   checkers, linters") spans this and category 2; unit tests and type
   checkers are execution, while a learned classifier is judgement.

**The distinction that matters is what each can decide, not what each is.**
Exercise 2 puts the number on it. An executor returns an exact verdict: the
signal gap is total, noise is effectively zero, and more iterations
monotonically help. A retriever returns a *contested* verdict — the evidence
may be absent, stale or ambiguous — so it has a real false-negative rate
before anyone adds noise, which is precisely what `verify_external`'s unused
`fact` loop demonstrates in miniature: **1** of **3** facts is simply never
checked, and its violation exits clean. A classifier returns a *probabilistic*
verdict, which is the p>0 regime, and there a 30% false-positive rate turns
into a **52.8%** chance of exiting on a defective output, because the loop
stops on the first pass.

The engineering consequence: the three categories are not interchangeable
verifiers of differing quality, they are three different stop conditions.
Execution can be the sole gate. Retrieval needs a recorded "not found"
distinct from "verified" — the toy has no such state, which is why its
`Attempt` cannot tell an unchecked fact from a checked one. Judgement should
never be a sole gate at all; it belongs upstream of a human or of an
executor, because exercise 2's failure mode is silent and absorbing.

### 5 — the tripwire is right, and the retry channel is missing

*Cites "OpenAI Agents SDK output guardrails".*

**The mapping is clean: a guardrail is CRITIC's `verify`, and
`OutputGuardrailTripwireTriggered` is its boolean half.** The lesson states
the shape — a validator that runs on the agent's final output, may call
tools, and rejects by raising.

**What the SDK gets right, and it is more than it looks.**

*It makes the verdict a control-flow event, not a string.* This is exactly
the failure exercise 3 measures. In the toy, the critique is text and the
refiner discovers the verdict by searching that text for keywords — so an
accurate critic that phrases things differently produces a run that never
converges, and a rubber stamp produces the fastest run in the lesson. A typed
exception cannot be missed by paraphrase. Given the choice between a critique
the generator must parse and a tripwire it cannot ignore, the SDK picked the
one whose failure mode is loud.

*It puts the verifier outside the generating agent.* Anthropic's
evaluator-optimizer note — make the two prompts substantially different so
the model does not rubber-stamp — is a prompt-level mitigation for a
structural problem, and a separate guardrail object is the structural fix.

*It lets guardrails call tools.* That is the whole distinction between CRITIC
and Self-Refine, and the SDK declines to close it off. A guardrail that runs
unit tests is category 2 from exercise 4; a guardrail that is a pure function
is Self-Refine. Same interface, and the lesson is right that this is
CRITIC-shaped rather than Self-Refine-shaped.

**What it gets wrong.**

*A tripwire is a boolean, and CRITIC's contribution is the critique.* The
paper's claim is not "reject bad output" — it is that the critique is
*grounded in tool results* and that the refiner *conditions on it*. An
exception carries a rejection; the SDK does not specify that the guardrail's
reasoning reaches the next attempt. Exercise 1 is the bound: the loop needs
defects + 1 iterations, and each iteration only works because the previous
critique named something specific. A retry that does not carry the critique
forward is a re-roll, and a re-roll costs the same tokens as a refinement and
buys the base error rate.

*It is a gate on the final output, which is a budget of one.* Exercise 1
shows that at `max_iterations=1` a CRITIC arm and a Self-Refine arm are
byte-identical — same output, same verdict, different unread critique. A
guardrail placed on the final output is that configuration by default, and
the retry loop around it is the user's to build. "We added output guardrails"
therefore does not, by itself, distinguish a system with grounded
verification from one without.

*A pass leaves no artifact.* Exercise 2's sharpest finding is that a false
pass and a real pass are the same object: `verified=True`, critique
`verifier: ok`, nothing recording that the verdict was sampled. A tripwire
that does not fire is *less* than that — it produces no record at all. For
categories 1 and 3 from exercise 4, where the verifier has a real
false-negative rate, the useful artifact is "what was checked, against what,
and what was not checkable", and an exception-or-nothing interface has
nowhere to put it.

**In one line:** the SDK gets the *enforcement* of the verifier right and
leaves the *feedback* of the verifier to the caller — which is the half the
CRITIC paper is about.
