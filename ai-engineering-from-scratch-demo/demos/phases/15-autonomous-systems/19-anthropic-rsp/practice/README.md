<!-- generated:start -->
# 15-autonomous-systems / 19-anthropic-rsp

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/19-anthropic-rsp/) · upstream spec
`phases/15-autonomous-systems/19-anthropic-rsp/docs/en.md`

```bash
uv run demo practice run 19-anthropic-rsp --ex 1
uv run demo explain 19-anthropic-rsp --ex 1
uv run pytest demos/phases/15-autonomous-systems/19-anthropic-rsp
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Feed in three synthetic models at different capability levels. Confirm th… | code | T0 | `ex01_full_automation_of_ai_rd_does_not_cross_the_ai_rd_threshold.py` |
| 2 | Read RSP v3.0 in full (32 pages). Identify every commitment that lives in the "industry-wide… | explain | T0 | prose, below |
| 3 | Read SaferAI's RSP grading methodology. Reproduce their 1.9 score for v3.0 by applying their… | code | T0 | `ex03_four_factors_one_delta_and_no_weights.py` |
| 4 | The 2023 pause commitment was removed. Propose a replacement commitment that preserves the cr… | code | T0 | `ex04_a_threshold_in_hours_expires_in_ten_months.py` |
| 5 | Compare RSP v3.0 to OpenAI Preparedness Framework v2 (Lesson 20). Pick one area where v3.0 is… | explain | T2 | prose, below |
<!-- generated:end -->

## Answers

### 1 — full automation of AI R&D does not cross the AI R&D threshold

The evaluator behaves as written. The shipped demonstration does not test it:

| model | triggers | crossed | affirmative-case sections |
|---|---:|---|---:|
| Claude Opus 4.6 | 0 | no | 6 |
| synthetic next-gen | 3 | **yes** | 7 |
| one trigger | 1 | no | 6 |
| two triggers | 2 | **yes** | 6 |
| `rd_automation_share = 1.00` | **1** | **no** | 6 |

The rule is `len(reasons) >= 2`. The two models `main` ships sit at 0 and 3,
so neither exercises the boundary the rule turns on. Feeding in the 1-trigger
and 2-trigger cases is what "confirm the threshold evaluator behaves as
expected" actually requires.

**The last row is the finding.** A model that automates *all* internal AI R&D
tasks at expert-human cost — the capability the threshold is named after —
does not cross, because it only lights one of three counters. A count cannot
express that one measure at its ceiling matters more than two at their floors.
This is not a bug in the toy so much as a demonstration of why v3.0 made the
thresholds qualitative: any fixed combination rule has a case like this, and
the choice is between a rule you can check and a judgement you cannot.

**The gaming section is added after the numbers it should qualify.**
`affirmative_case_template` appends a seventh section when the gaming rate
exceeds 0.20, so Opus at 12% gets six and the synthetic at 28% gets seven. But
`threshold_crossed` has already run, on capability numbers that eval-context
gaming inflates (Lesson 1). The model whose measurements are least trustworthy
is the one the decision has already finished with — the adjustment arrives as
a documentation requirement rather than as an input.

**And the thresholds here are quantitative**, which is exactly what SaferAI
marked the document down for removing: three floats and an integer count. The
reading aid is written in the register the policy abandoned, which is worth
noticing before using the aid to reason about the policy.

### 2 — the industry-wide tier, and what moved into it

*Draws on "The two-tier mitigation schedule".*

The commitments living in the **industry-wide recommendation** tier are the
ones whose cost only makes sense if everyone bears it — the RAND SL-4 security
standard is the named example, and the others share its shape: infrastructure
hardening against state-actor-level adversaries, model-weight security
requirements at that tier, and the uniform deployment gates that only bind if
competitors adopt them too. The structural point the lesson makes is the one
to carry: *a security measure in the industry-wide column is not Anthropic's
promise, it is Anthropic's hope*, so reading the policy means reading the
column before the sentence.

What would have been unilateral in v2 is the security tier. v2 had no
two-column structure at all, so every commitment in it read as Anthropic's
own — the SL-4-class security requirements attached to the higher capability
levels were stated as things Anthropic would do. Moving them into the
recommendation column is not a change in what the document says should happen;
it is a change in who is promising. That is why the two-tier structure is the
first thing the lesson describes and why it is easy to read past: the
sentences can survive the move unchanged while their force does not.

### 3 — four factors, one delta, and no weights

**The score is not reproducible from what is published.** SaferAI's rating
moves 2.2 → 1.9, a delta of **−0.30**, and the lesson names **four** downgrade
factors. That is four unknowns and one equation:

| assignment | thresholds | pause | affirmative case | oversight | sums to |
|---|---:|---:|---:|---:|---:|
| equal weighting | −0.075 | −0.075 | −0.075 | −0.075 | −0.30 |
| pause-only | 0.0 | **−0.30** | 0.0 | 0.0 | −0.30 |

Both fit exactly. Infinitely many others do too. **Which row drove it most?**
The pause removal — but that answer comes from the lesson's prose ("the
strongest regression"), not from the arithmetic, and the distinction matters:
the exercise asks you to reproduce a number and the honest result is that the
number does not determine its own decomposition.

**The category change is more sensitive than the score.** The drop is 13.6% of
the prior score on a 1.0–4.0 scale and it moves the document from *moderate*
to *weak* — so the band boundary lies inside a 0.30 interval. A rating whose
headline is the band is reporting a threshold crossing, and this phase has
spent nineteen lessons on how fragile those are.

**Three of the four factors are the same move.** Qualitative thresholds
replacing quantitative ones, the pause going, and mitigations becoming an
"affirmative case" all describe replacing a checkable statement with a
reviewable one. Only the fourth — the Safety Advisory Group's limited
independent oversight — is about *who* checks. So the rubric's mass sits on
three counts of one thing, which is another reason no per-factor weighting can
be recovered: the factors are not independent.

**And the reading aid keeps the numbers the document dropped.** Scoring this
module against the rubric would rate it *above* the policy it models, since it
has three quantitative thresholds and a deterministic crossing rule. Worth
stating before quoting either number.

### 4 — a threshold in hours expires in ten months

**The proposal:** *training pauses when a model's measured capability exceeds
the prior generation's by more than a declared multiple on any named axis, and
resumes only after an affirmative case is published and a named external
reviewer has responded.* It keeps the word **pause**, because that is the only
part of the removed clause that named an action. It replaces the trigger's
units, because that is what the rescaling argument is actually about.

**The absolute threshold was not unreachable — it was about to be reached.**
The module's METR gate is 40.0 hours against a stated Opus 4.6 horizon of 14.0.
At Lesson 1's 7-month doubling that gate arrives at **month 10.6**. Both
readings of "unreachable because benchmarks rescaled" are available from the
lesson's own numbers, and they point in opposite directions — which is why the
replacement has to fix the unit rather than the value.

**A ratio is invariant to the rescaling:**

| benchmark rescaled by | absolute gate becomes | relative gate stays |
|---:|---:|---:|
| 2× | 80.0 h | **2.86×** |
| 5× | 200.0 h | **2.86×** |
| 10× | 400.0 h | **2.86×** |

The same gate, expressed as "2.86× the prior generation", fires at the same
generation however the benchmark is re-scaled. The cost is real and worth
naming: the policy loses a number a reader can look up, and gains a trigger
that requires the prior generation's measurement to be published too. That is
the price of credibility here — the commitment becomes checkable only if the
baseline is also disclosed, which is a stronger transparency requirement than
the clause it replaces.

### 5 — RSP v3.0 against the Preparedness Framework

*Draws on "Removing the pause clause".*

**Where v3.0 is stronger: the affirmative case as a positive obligation.** The
Preparedness Framework's structure is a set of risk categories with threshold
levels and required safeguards — a reader checks whether a safeguard is
present. v3.0's affirmative case inverts the burden: the lab must *construct
and publish an argument* that deployment is safe, covering capability
inventory, misalignment analysis, the eval-versus-deploy gap, mitigations and
residual risk. A checklist can be satisfied by a lab that has not thought
about the specific model; an affirmative case cannot, because it has to name
what it cannot rule out. The module's six-section template is that shape, and
its conditional seventh section — the gaming adjustment — is an obligation no
threshold table contains.

**Where the Preparedness Framework is stronger: it kept its thresholds and its
stopping condition.** The lesson's own account of the pause removal is the
comparison: a scaling policy's thresholds are a commitment device, and v3.0
replaced a quantitative trigger with a qualitative one and an explicit pause
with "proceed if mitigations are adequate". The Preparedness Framework retains
named capability levels with required safeguards attached, so there remains a
statement of the form *at this level, this happens*. That is the thing an
outside party can hold a lab to, and it is precisely what SaferAI's downgrade
was about — three of its four factors describe v3.0 removing exactly this kind
of statement. The honest summary is that v3.0 asks for more thinking and
promises less action, and which of those a reader values depends on whether
they expect to be able to verify either.
