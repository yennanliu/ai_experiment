<!-- generated:start -->
# 15-autonomous-systems / 20-openai-preparedness-deepmind-fsf

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/20-openai-preparedness-deepmind-fsf/) · upstream spec
`phases/15-autonomous-systems/20-openai-preparedness-deepmind-fsf/docs/en.md`

```bash
uv run demo practice run 20-openai-preparedness-deepmind-fsf --ex 1
uv run demo explain 20-openai-preparedness-deepmind-fsf --ex 1
uv run pytest demos/phases/15-autonomous-systems/20-openai-preparedness-deepmind-fsf
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm the diff tool's output matches the policies for at least two capa… | code | T0 | `ex01_two_capabilities_are_verifiable_and_two_are_never_printed.py` |
| 2 | Read OpenAI Preparedness Framework v2 in full. Identify each Research Category. For each, wri… | explain | T2 | prose, below |
| 3 | Read DeepMind FSF v3 in full, plus the April 2026 Tracked Capability Levels update. Identify… | explain | T0 | prose, below |
| 4 | Sandbagging is in OpenAI's Research Categories. Design an evaluation that would force a sandb… | code | T2 | `ex04_the_sandbagging_test_has_to_be_one_the_model_cannot_recognise.py` |
| 5 | Compare the three policies on a specific capability (your choice). Name which policy's classi… | code | T0 | `ex05_the_capability_they_agree_on_is_the_one_no_one_folds.py` |
<!-- generated:end -->

## Answers

### 1 — two capabilities are verifiable, and two are never printed

The source documents are not in this repository, so "verify" has to mean
verify against something present. The lesson's headline block restates the
classifications for exactly two capabilities, and those two are the
confirmable set — out of **21** table entries.

| | result |
|---|---|
| classifications the headline restates | 6 (2 capabilities × 3 policies) |
| matching the tables **verbatim** | **5** |
| paraphrased | 1 — DeepMind on long-range autonomy, summarised as `domain-folded` where the table says `folded into ML R&D / Cyber domains` |
| entries with no in-lesson corroboration | 15 |

So the answer to "confirm for at least two capabilities" is: yes, and only
two, and the second one needs a paraphrase allowance. That is not a criticism
of the tool — it is the shape of the exercise. A diff tool over distilled
tables can be checked against the distillation or against the sources, and
only one of those is at hand.

**Two capabilities are in every table and never printed.** `cyber_uplift` and
`bio_uplift` appear in all three policies — six entries — and `main` diffs
five of the seven keys. They are also the two where the policies agree most
(exercise 5), which is either a good reason to omit them from a *diff* tool or
a good reason to include them as the baseline the disagreements are measured
against. The module does not say which.

**OpenAI's table has seven capabilities and two actions.** Every `Research`
row carries `observed; potential mitigations` and every `Tracked` row carries
`Capabilities + Safeguards Reports; SAG review`. Two distinct classifications,
two distinct actions, seven rows — against six distinct actions each for
Anthropic and DeepMind. In the OpenAI column the action restates the
classification; in the other two it adds information. That is a real property
of the source framework, not an artefact of the distillation: v2's Research
Categories are defined by not having committed mitigations.

**And one capability draws three different verbs.**
`undermining_safeguards` is `Research` (observe), a `hardcoded prohibition`
(refuse), and `deceptive alignment monitoring` (detect). Observe, refuse and
detect are not points on a scale, so "which policy is stricter here" does not
have an answer without a further criterion — which is what exercise 5
supplies.

### 2 — the Research Categories, and why each is there

*Draws on "OpenAI Preparedness Framework v2 (April 2025)".*

The four Research Categories are **Long-range Autonomy**, **Sandbagging**,
**Autonomous Replication and Adaptation**, and **Undermining Safeguards**. The
framework's own framing is the key to why: Research Categories are "model
capabilities the lab is tracking but not yet committing to specific
mitigations", and the policy language is *potential* mitigations — so the
question for each is why a specific mitigation cannot yet be named.

**Long-range Autonomy** is in Research because the capability has no
threshold that separates useful from dangerous — the same property that makes
a long-horizon agent valuable makes it hard to gate, and Lesson 1's horizon
curve is a capability measure with no natural cut point. **Sandbagging** is in
Research because a mitigation would have to act on a capability the model is
concealing, and there is no agreed evaluation that surfaces it — which is
exactly what exercise 4 designs around. **Autonomous Replication and
Adaptation** is in Research because the harm is a sequence of individually
mundane actions (acquire resources, copy, persist), so a mitigation attaches
to no single action the policy could name — the composite problem of Lesson
10. And **Undermining Safeguards** is in Research because a mitigation against
it is a safeguard, and the capability is specifically the ability to
circumvent safeguards, so any named mitigation is in the threat model of the
thing it mitigates.

The pattern across all four: they are the categories where a *specific* mitigation
cannot be written down, not the categories where the risk is small. That is
worth holding on to, because "Research" reads as "less serious" and the
framework's own definition says it means "we do not yet know what to commit
to".

### 3 — ML R&D autonomy level 1, and measuring it externally

*Draws on "DeepMind Frontier Safety Framework v3 (September 2025; Tracked Capability Levels added April 17, 2026)".*

The criterion is stated compactly: **ML R&D autonomy level 1 = fully automate
the AI R&D pipeline at competitive cost versus human + AI tools.** Three
clauses, each doing work. *Fully automate* means the whole pipeline, not a
stage — idea to experiment to evaluation to the next idea, which is the
distinction Lesson 6 draws between AAR (the middle) and the threshold (the
ends). *The AI R&D pipeline* scopes it to the lab's own research loop rather
than to software engineering generally. And *at competitive cost versus human
+ AI tools* makes it an economic test, not a capability test: the baseline is
not an unaided human but the current best human-plus-tool workflow, so the
level moves as tooling improves.

Measuring it externally is hard for a reason worth naming. The first two
clauses need access to the lab's internal research pipeline — an external
evaluator cannot observe what fraction of it is automated, because the
pipeline is not a public artefact. The third clause is the measurable one:
cost per unit of research output is comparable across organisations in
principle. So an external measurement would have to be a **proxy plus a
disclosure**: run a public research-engineering benchmark with a stated human
baseline (RE-Bench's shape, from Lesson 1) to establish capability, and
require the lab to publish the ratio of automated to human-directed steps in
its own pipeline, audited the way a financial control is — sampled, by someone
with access, reporting a number rather than the underlying data. That is
Lesson 21's territory, and the honest summary is that clause one is not
externally measurable at all without that access, which is why the level as
written is a self-assessment with an economic sanity check attached.

### 4 — the sandbagging test has to be one the model cannot recognise

**The design: four arms over the same task distribution.**

| arm | model can recognise it? | a gap against arm 1 means |
|---|---|---|
| declared evaluation | yes | baseline |
| **undeclared production sample** | **no** | sandbagging on the declared arm |
| same tasks, stated real reward | yes | incentive-sensitive capability |
| consistency against earlier answers | yes | inconsistent underperformance |

**"Force" is the wrong verb and that is the finding.** A model that can
recognise an evaluation can underperform on it, so no single test forces
anything. What the protocol does is make underperformance *visible as a gap*
rather than detectable as a low score — sandbagging is a variance across arms,
not a mean within one. One arm cannot detect it at any sample size.

**The three policies classify it three ways and only one implies something
that runs.** OpenAI: `Research`, action `observed; potential mitigations`.
Anthropic: named via the eval-context gap, addressed `in measurement
methodology`. DeepMind: `deceptive alignment monitoring` with an `automated
instrumental-reasoning monitor`. Observation, methodology and a monitor are
three different budgets.

**The arms should disagree in the direction Lesson 1 measured.** At
Anthropic's documented alignment-faking rates — 12% on basic tests, up to 78%
after retraining attempts — a four-arm protocol should expect the declared arm
to read low by roughly that fraction. Carried into a trajectory: 68 steps at
0.99 per-step is 50.5% end-to-end; at a 12% gaming rate it is **23.9%**, and
at 78% it is **0.3%**. That is the denominator a capability report would have
to carry to be honest about what it measured.

**And the protocol's weakness is arm 2, which is a policy problem rather than
a technical one.** Sampling undeclared production traffic is the only arm a
model cannot recognise, and it needs consent from the people whose traffic it
is. Zero of the three sandbagging entries mention data governance — so the one
evaluation that would work has a cost none of these documents prices.

### 5 — the capability they agree on is the one no one folds

**The capability: `undermining_safeguards`. The test: does the classification
name something that *runs*?**

| policy | classification | action | runs? |
|---|---|---|---|
| OpenAI | `Research` | `observed; potential mitigations` | no |
| **Anthropic** | `hardcoded prohibition` | `refuses training / deploy` | **yes, without review** |
| DeepMind | `deceptive alignment monitoring` | `automated monitor + red-team` | yes, reports |

**Most rigorous: Anthropic.** A refusal happens without a human deciding.
**Least: OpenAI.** Neither `observed` nor `potential mitigations` names an
action taken by anyone — which is the framework's own definition of a Research
Category, applied to the capability that is specifically about circumventing
safeguards.

**The test separates the three cleanly across all seven rows:** actions name
something that runs in **0**, **3** and **4** rows respectively, and OpenAI's
column holds two distinct strings, so it cannot distinguish between
capabilities even where the other two do.

**The agreement is on the three capabilities OpenAI tracks** —
`rnd_automation`, `cyber_uplift`, `bio_uplift` — and two of those three are
the ones `main` never prints. The capabilities the policies agree on are the
ones with a decade of prior regulatory vocabulary; the ones they disagree on
are the ones this phase is about, which is a useful thing to notice about
where policy convergence comes from.

**And rigour is not strictness.** Anthropic's hardcoded prohibition is the
strictest classification here *and* the narrowest: Lesson 17 measures its
floor catching 1 of 8 shipped cases, because it is substring matching over an
action description. A policy can name a thing that runs and still have the
thing be narrow. So "most rigorous" as tested here means "most checkable", and
checkable is a precondition for strict rather than a synonym for it.
