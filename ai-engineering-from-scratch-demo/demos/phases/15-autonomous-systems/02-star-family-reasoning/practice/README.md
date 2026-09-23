<!-- generated:start -->
# 15-autonomous-systems / 02-star-family-reasoning

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/02-star-family-reasoning/) · upstream spec
`phases/15-autonomous-systems/02-star-family-reasoning/docs/en.md`

```bash
uv run demo practice run 02-star-family-reasoning --ex 1
uv run demo explain 02-star-family-reasoning --ex 1
uv run pytest demos/phases/15-autonomous-systems/02-star-family-reasoning
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run the simulator. Set the shortcut frequency to zero, then to 0.4. How much does final accur… | code | T0 | `ex01_the_shipped_shortcut_is_pruned_not_reinforced.py` |
| 2 | Add a held-out OOD test to the simulator. Draw problems from a different distribution and eva… | code | T0 | `ex02_the_held_out_set_is_one_literal_and_never_a_gate.py` |
| 3 | Read the Quiet-STaR paper (arXiv:2403.09629) Section 3. Explain the "end-of-thought" token an… | explain | T0 | prose, below |
| 4 | Compare STaR's keep-if-correct filter to a process-supervised alternative that rewards each r… | code | T0 | `ex04_the_soundness_bit_is_present_and_the_filter_never_reads_it.py` |
| 5 | Design one evaluation that would catch shortcut rationales in a deployed model. It does not h… | code | T0 | `ex05_the_delta_test_is_weakest_where_it_is_needed.py` |
<!-- generated:end -->

## Answers

### 1 — the shipped shortcut is pruned, not reinforced

Averaged over seeds 0–9, five bootstrap rounds from the same 0.20 sound prior:

| run | ID accuracy | OOD accuracy | final shortcut share |
|---|---:|---:|---:|
| shortcut 0.0 | 97.7% | 97.6% | — |
| shortcut 0.4 | 92.0% | 87.9% | 0.114 |

So the divergence is **5.8 points in-distribution and 9.6 held out** — 1.7×
more on the distribution nobody trained on — and both runs clear the 90% the
exercise promises. That is the answer to the question as posed.

The more interesting result is what the second run's share column says. The
shortcut falls from 0.400 to 0.114 over the five rounds and to 0.011 by round
ten, monotonically. **The loop prunes it.** The lesson's own headline says the
opposite: "Scenario B climbs on ID while OOD collapses — the shortcut gets
reinforced because it looks correct in training data." OOD ends at 87.9%, five
points down, with the shortcut nearly gone.

Why: `star_round` keeps traces whose *answer* is correct, and the shipped
shortcut is right 40% of the time while sound reasoning is right 100% of the
time. The filter therefore selects against it — on accuracy, not on soundness.
Replaying `star_round`'s own update rule in expectation reproduces the
simulation to within 0.002 and makes the mechanism explicit; sweeping the
shortcut's in-distribution hit rate, the share holds only at **0.71** and grows
to **0.660** once the shortcut is as accurate in-distribution as sound
reasoning.

That last number is the honest version of the lesson's claim. A shortcut that
is *less* accurate than real reasoning is not the failure mode — it is just a
worse strategy, and outcome supervision correctly discards it. The failure mode
needs a shortcut that is indistinguishable in-distribution, which is a hit rate
of 1.0 and 0.60 above the literal in `Model.sample`.

### 2 — the held-out set is one literal, and never a gate

There is nothing to add: `evaluate` already takes `on_ood`, and `report_round`
already prints the column. So the exercise reduces to the quantification, and
the quantification has a closed form.

After five rounds the bootstrapped model scores **92.1% in-distribution and
88.2% held out — a 3.9-point gap**, against −0.1 for the clean run. The gap is
not an empirical fact about the round: it is `0.35 × shortcut share`, exactly,
and the worst residual against the simulation across all six rounds is 0.4
points, which is the sampling noise of a 4000-item evaluation.

**The held-out set is one number.** `Model.sample` separates the two
distributions with `0.05 if on_ood else 0.40` and nothing else — sound
reasoning is right in both, guessing is right 0.10 of the time in both. The
widest gap the simulator can produce is therefore 0.35, at a model that is
nothing but shortcut.

**The column is printed and never read.** `star_round` takes no OOD argument
and mentions `on_ood` exactly once, to pass `False`. Nothing the held-out set
reports can change what the loop keeps. It is a held-out report, not a held-out
test, and the distinction is the whole content of the exercise.

**And the one column that would name the problem cannot move.**
`Trace.rationale_sound` is set by strategy, never by distribution, so
`evaluate`'s soundness fraction is invariant by construction — measured 0.019
apart at worst. Accuracy is the only channel the held-out set has, and accuracy
is precisely what a shortcut is built to preserve.

### 3 — the end-of-thought token and the mixing-weight head

*Draws on "Quiet-STaR: per-token internal rationales" and the comparison table
in the lesson's Concept section.*

**The end-of-thought token.** Quiet-STaR inserts a learned `<|endofthought|>`
token that marks where a generated rationale stops and ordinary prediction
resumes, so a thought is a bounded span rather than an open-ended continuation.
It is learned rather than fixed because the model has to decide *how long* to
think at each position, and the lesson's own summary of the result — "hard
tokens get longer internal rationales; easy ones get almost none" — is exactly
this token being emitted earlier or later. Without it the thought has no
terminator the base language model can be conditioned past, and the rationale
would leak into the visible sequence instead of staying "quiet".

**The mixing-weight head.** The mixing head is a small learned module that
outputs, per token position, a weight blending the thought-conditioned next-token
distribution with the plain, no-thought one. It exists because a rationale is
not always an improvement: on easy tokens the base prediction is already right,
and forcing the thought-conditioned distribution through would inject noise, so
the head lets the model fall back continuously rather than choose. That
continuous fallback is also why the lesson's comparison table prices Quiet-STaR
at 1.5–3× inference rather than at an integer multiple — the cost is a blend,
not a best-of-N, which is the one cost shape this lesson's simulator cannot
express (`vstar_infer` takes an integer `samples_per_problem`).

### 4 — the soundness bit is present, and the filter never reads it

The module has no steps: `Trace` carries a strategy and two booleans. A
step-level filter therefore cannot be built here at full strength, so what this
solution builds is its limit case — one label per rationale, on soundness
rather than on the answer. That is process supervision with exactly one step,
which makes it a ceiling on what step rewards buy in this simulator rather than
an approximation of them.

| filter | sound share after 5 rounds | shortcut share |
|---|---:|---:|
| keep-if-correct (shipped) | 0.872 | 0.116 |
| keep-if-sound (process limit) | 0.992 | 0.004 |

**Quality difference: 29×** on the shortcut share. **Cost difference: the step
count** — the same 1000 traces labelled per round, at k labels each instead of
1, and k is not a number this module has.

Two things the comparison makes concrete.

**Half of the first round's training set is unsound.** Keep-if-correct retains
0.40 of the samples, and of what it retains, 0.50 is a shortcut or a lucky
guess. The filter is not selecting reasoning; it is selecting outcomes, and at
the shipped starting mix the outcome is a coin flip on whether the reasoning is
real.

**The waste the lesson tabulates is 0.60, and the alternative has none.** The
comparison table's "discards all incorrect rationales" is 600 of 1000 labelled
samples thrown away every round. Step-level rewards discard nothing: an
incorrect step is a labelled negative, which is exactly the data V-STaR goes
back for.

And the sharpest version of the comparison is that the module already has the
bit. `star_round` names `rationale_sound` zero times; `vstar_infer` reads it
twice, under a docstring that calls the result "an idealized verifier … an
upper bound". The one signal separating process supervision from outcome
supervision is handed to the verifier and withheld from the filter.

### 5 — the delta test is weakest exactly where it is needed

**The evaluation.** Pair every deployed input with a perturbed twin that
preserves the answer and breaks surface patterns; report the *accuracy delta*
between the two arms, not the accuracy; size the arms from the delta you intend
to detect; threshold on the delta itself; and re-run it every round, because
its sensitivity moves.

Against the one simulator where the ground truth is known, that test is a
direct readout: because the distributions differ by a single literal, the gap
is `0.35 × shortcut share` with zero residual at every round.

| round | shortcut share | ID − OOD gap |
|---:|---:|---:|
| 0 | 0.400 | 14.0 pts |
| 1 | 0.400 | 14.0 pts |
| 2 | 0.331 | 11.6 pts |
| 3 | 0.246 | 8.6 pts |
| 4 | 0.172 | 6.0 pts |
| 5 | 0.116 | 4.1 pts |

**Sensitivity decays as the loop runs.** The gap shrinks with the share it is
measuring, so the test is 3.4× weaker at the round you would ship than at the
round you would never ship. A fixed threshold gets *less* protective every
round — the opposite of what a deployment gate should do, and the reason rule
five is "re-run it", not "set it once".

**The shipped evaluation cannot resolve its own final gap.** `report_round`
scores 500 items per arm; at the run's final 92.0% accuracy that is a
1.7-point standard error on the difference, so a 4.1-point gap is 2.4σ. Three
sigma needs 791 per arm. The sample size has to be derived from the delta you
intend to detect, and the module's is a round number.

**And the 0.35 is an oracle.** It exists because `Model.sample` defines both
hit rates; zero of the module's three measurement functions derives it. A
deployed version of this test can threshold the delta and cannot report a
share — so any protocol that claims "the model is 12% shortcut" is quoting the
simulator's own literals back at itself.
