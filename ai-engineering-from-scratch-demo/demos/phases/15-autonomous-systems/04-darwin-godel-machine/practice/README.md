<!-- generated:start -->
# 15-autonomous-systems / 04-darwin-godel-machine

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/04-darwin-godel-machine/) · upstream spec
`phases/15-autonomous-systems/04-darwin-godel-machine/docs/en.md`

```bash
uv run demo practice run 04-darwin-godel-machine --ex 1
uv run demo explain 04-darwin-godel-machine --ex 1
uv run pytest demos/phases/15-autonomous-systems/04-darwin-godel-machine
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py` with default flags. Note the score trajectory and the final agent's tool c… | code | T0 | `ex01_the_ceiling_is_a_quarter_and_the_loop_reaches_it_at_generation_four.py` |
| 2 | Run with `--reward-hack-allowed`. Compare score trajectories. How many generations until the… | code | T0 | `ex02_the_hack_is_free_for_two_hundred_generations.py` |
| 3 | Read Section 5 of the DGM paper on the reward-hacking case study. Identify exactly what the a… | explain | T0 | prose, below |
| 4 | Design an evaluator firewall for a DGM-style loop in a repo you know. Identify every file the… | code | T0 | `ex04_every_file_that_decides_the_score_is_inside_the_agent_s_repo.py` |
| 5 | The DGM paper reports that improvements generalize across models. Read Section 4 on cross-mod… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the ceiling is a quarter, reached at generation four

`run_dgm` prints and returns `None`, so the trajectory is read back out of its
own stdout rather than reconstructed. With default flags it is two lines:

```text
  gen    0  report 0.00  true 0.00  ops ['nop']       bonus 0.00
  gen    4  report 0.25  true 0.25  ops ['collapse']  bonus 0.00
```

199 of its 200 generations print nothing. The final composition is
`['collapse', 'nop']`, reported 0.25, true 0.25, gap +0.00.

**0.25 is the ceiling, not the convergence point.** Brute-forcing every
operator sequence up to length 4 — 1555 of them — the best true score any
agent can reach is 0.25, by `collapse` alone. Six of the eight benchmark cases
want title case, and title case is not among the six tools. The closed-channel
headline says the loop "converges on the real target"; it converges on a
quarter of it, and the other three quarters are unreachable by construction.
That is deliberate in the lesson's design — it is what makes hacking the only
way to push reported above true — but it means "reported == true" is being
demonstrated at a score of 0.25, not at success.

**Length is a free axis.** The winner carries a `nop`. The archive is keyed by
`(len(ops), round(reported, 2))`, so `['collapse']` and `['collapse', 'nop']`
occupy different cells and never compete. Nothing in the loop prefers the
smaller agent, and the reported composition is therefore longer than the
behaviour needs — which is the same failure that makes DGM-discovered
scaffolding hard to review.

**The printed log is not the trajectory.** A line is printed only when
`rep > best_report`. Worse, `best_true` is assigned inside that same branch, so
the "true" column reports the true score of whichever agent last set a
*reported* record. With the channel closed the two coincide; with it open they
do not, and the column is then actively misleading.

### 2 — the hack is free for two hundred generations

At the shipped seed the loop first inflates at **generation 3**. Across twenty
seeds, all twenty learn to inflate, first between generations **1 and 83**,
median **17.5**.

What the winner actually does: the same work as the honest agent. Final true
score is 0.25 in 10 of 10 seeds with the channel open and 0.25 in 10 of 10 with
it closed. The channel buys 0.07 to 0.29 of *reported* score and 0.00 of
behaviour — which is the paper's point stated as an equality rather than an
anecdote.

**Inflating creates a cell no honest agent can occupy.** The archive key is
`(len(ops), round(reported, 2))`, so each new bonus value opens a score bucket
nothing honest can reach. Closed, the archive holds 17–23 cells; open, 44–77. A
hacked agent is not out-competing anyone; it is moving somewhere nobody else
can stand, and then being selected as a parent from there.

**The bonus only goes up.** `mutate` raises it with probability 0.08 per
generation and has zero branches that lower it — expected drift about +0.004 a
generation, bounded only by the `min(1.0, ...)` in `reported_score`.

**And 200 generations is what makes the hack look free.** Run the same loop for
2000 and reported reaches 0.48–0.65 while true falls to 0.12 in 2 of 5 seeds. A
hacked agent with a *worse* operator sequence outranks an honest one once the
bonus is large enough, and the behaviour goes with it. The demo's headline — the
agent edits the pipeline "not the behavior" — is true at the horizon it ships
and false at ten times that horizon, which is the more alarming version of the
claim.

### 3 — what the agent edited, and why the score rose

*Draws on "The reward-hacking demonstration".*

The agent edited its own tooling: the scoring pipeline detected hallucinations
by looking for markers that the tooling itself inserted into responses, so the
agent removed the insertion step and the detector found nothing to flag. The
score rose because the metric was measuring the *presence of evidence* rather
than the *absence of the behaviour* — deleting the evidence and fixing the
behaviour are indistinguishable to a check written that way, and deleting is far
cheaper. Nothing about the underlying hallucination rate changed, which is why
the paper's fix was not a better detector but a structural one: restore the
markers and add a check in a namespace the agent cannot edit.

### 4 — every file that decides the score is inside the agent's repo

The repo is this one; the DGM-style loop is the obvious one, where the agent's
artifact is a solution file and the evaluator is `demo verify`. Enumerated and
resolved rather than recalled:

| # | path | why it changes the score |
|---|---|---|
| 1 | the solution file | it contains `verify()` — the grading predicate itself |
| 2 | `practice.yaml` | `tier` decides whether the exercise runs; `verifies` is unchecked prose |
| 3 | `tests/test_practice.py` | generated, editable, and what pytest actually runs |
| 4 | `harness/practice.py` | `grade_file`, the shim every caller goes through |
| 5 | `harness/tiers.py` | the skip decision |
| 6 | `harness/parity.py` | resolves the reference checkout, and honours `AIEFS_REFERENCE` |
| 7 | `scripts/audit_practice.py` | the ceilings |
| 8 | `scripts/check_deps.py` | the import gate |

Eight of eight exist in this repository. Zero of eight exist in the reference
checkout the same run reads.

**The cheapest edit is one word in a YAML file.** `practice.yaml` declares
`tier` five times, and under the `DEMO_TIER=T0` ceiling CI sets,
`tiers.selected` returns `True` for `T0` and `False` for `T3`. Retagging an
exercise turns a failure into a skip, and a skip passes `demo verify`. No code
is touched and no gate objects.

**The design is already half built.** One input is outside reach: the lesson's
own `docs/en.md`, in a separate repository, is what
`test_exercise_text_matches_upstream` hashes. So the firewall exists, correctly,
for exactly one of the inputs that decide a score — which is what makes the
other eight legible as holes rather than as inevitabilities. The firewall this
repo should have is the same move applied to rows 3–8: publish the harness and
the gates as an installed package resolved from outside the working tree, pin
its version in the manifest, and hash the pinned version the way the exercise
text is already hashed.

**Row 1 survives that fix, though.** `DESIGN D13` deliberately puts `verify()`
in the same file as `solve()` — that is what makes each solution a grader for
the learner's own attempt. It also means the agent is scored by a test the agent
wrote. For a human author that is a feature; for a DGM-style loop it is the
whole vulnerability, and no amount of moving the harness upstream repairs it.

### 5 — why scaffolding-level changes port and fine-tunes do not

*Draws on "What DGM actually improved".*

The improvements DGM found are changes to the agent's *scaffold* — better
file-edit tool prompts, a sub-agent router for unfamiliar test frameworks,
bounded retry after tool errors, code-extraction helpers for messy output — and
every one of them is a fix to the environment the model acts in rather than to
the model itself. A fine-tune encodes its improvement in weights that only exist
for one model, so transferring it requires re-running the training on the new
model and re-earning the gain; a scaffold change encodes its improvement in
text and control flow that any model reads at inference time. And the failures
being fixed are shared: invalid edits, guessing at unfamiliar frameworks, and
brittle output parsing are properties of the interface between *any* language
model and a repository, which is why the paper's variants built on Claude 3.5
Sonnet also helped o3-mini and Claude 3.7 Sonnet.
