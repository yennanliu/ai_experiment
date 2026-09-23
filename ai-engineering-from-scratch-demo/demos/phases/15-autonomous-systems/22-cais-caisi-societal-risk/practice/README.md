<!-- generated:start -->
# 15-autonomous-systems / 22-cais-caisi-societal-risk

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/22-cais-caisi-societal-risk/) · upstream spec
`phases/15-autonomous-systems/22-cais-caisi-societal-risk/docs/en.md`

```bash
uv run demo practice run 22-cais-caisi-societal-risk --ex 1
uv run demo explain 22-cais-caisi-societal-risk --ex 1
uv run pytest demos/phases/15-autonomous-systems/22-cais-caisi-societal-risk
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Feed in three synthetic deployments at different scales. Confirm the four… | code | T0 | `ex01_the_clean_deployment_gets_no_kill_switch.py` |
| 2 | Read the CAIS four-risk paper in full. Pick one risk category and write two paragraphs on wha… | explain | T0 | prose, below |
| 3 | Read a current draft of California SB-53. Identify one provision you believe strengthens the… | explain | T0 | prose, below |
| 4 | Pick a production AI deployment you know (yours or a published one). Score it against the org… | code | T0 | `ex04_the_weakest_lever_is_the_auditor_living_in_the_tree.py` |
| 5 | Sketch a 2028 version of the four-risk framework that reflects one year of additional capabil… | code | T0 | `ex05_split_the_lever_that_cannot_tell_three_situations_apart.py` |
<!-- generated:end -->

## Answers

### 1 — the clean deployment gets no kill switch

The three shipped deployments tag **0**, **3** and **4** of the four risks,
which is the ordering the exercise expects. Two edges are worth reporting, one
at each end.

**The under-tag: the internal frontier agent.** Flipping `public_facing` to
False on the autonomous ML research agent — still handling harmful
capabilities, still 48 hours of autonomy — drops it from four tags to three,
because `malicious_use` requires `handles_harmful_capabilities and
public_facing`. Insider misuse is unrepresentable in this tagger, and insider
misuse is the documented malicious-use vector for a *frontier research* agent
specifically.

**The over-tag is at the other end, and it is an under-tag in disguise.** The
clean deployment tags zero risks, so it receives **0 of the 14** mitigations —
including the kill switches and canary tokens of Lesson 14 — under a line
reading "no tagged risks (check sub-levers manually)". A scoped internal agent
with one hour of autonomy is exactly what most teams actually run, and the
tool's advice for it is to think of something. An inventory that returns
nothing for the modal case is returning its own coverage rather than the
deployment's risk.

**Organizational risk has four sub-levers and three fields.** The mitigation
list names safety culture, independent audit, multi-layered defenses and
information security; `Deployment` carries the last three. Safety culture —
the lever the lesson's own headline calls the one practitioners actually pull
— has no field, so the tag can never fire because of it and can never clear
because of it.

**And the rogue-AI trigger is an absolute number of hours.** It fires at
`agent_autonomy_hours >= 4.0`, with the mid deployment sitting exactly on the
boundary. That is the shape Lesson 19 priced: a threshold in hours against a
horizon that doubles roughly every seven months, so the line between "tagged"
and "not" moves without anyone editing the file.

### 2 — the most important 2026 development in one category

*Draws on "The four-risk framework".*

**The category: organizational risks** — and the most important 2026
development in it is that the *audit* sub-lever became structurally harder
rather than easier, because the thing being audited started writing the
auditor. When the framework was written, "insufficient audit" meant a lab that
had not commissioned one. In 2026 the common case is a lab that has extensive
automated gates which are themselves produced and maintained by the agent
pipeline they gate — the exact arrangement exercise 4 scores in this
repository, where 8 of the 8 paths deciding a verdict live in the tree the
agent edits. That is a different failure from "no audit", and the framework's
vocabulary does not distinguish them.

The second half of the development is that this is now measurable, which it
was not in 2023. Every lesson in this phase supplies one of the measurements:
Lesson 4 enumerates the files an agent could edit to change its own score,
Lesson 8 shows an anchor whose hash lives in the module it protects, Lesson 16
shows a rollback test that passes on a run that never happened. None of those
is a safety-culture failure in the sense the framework means — nobody
suppressed a concern — and all of them are audit failures in the sense that
matters, which is that the check and the checked are the same object. The 2026
development is that we can now count this, and the framework has one boolean
for it.

### 3 — SB-53: one provision that strengthens, one that weakens

*Draws on "California SB-53".*

**Strengthens: the whistleblower protections.** Of the three drafted
provisions, it is the only one that acts on the sub-lever nothing else can
reach. Capability thresholds and incident reporting both require someone
inside the lab to classify an event correctly and then report it; whistleblower
protection is what makes the classification contestable by someone who
disagrees. Exercise 4 finds safety culture to be the sub-lever with no artifact
to score — a legal protection for escalation is the closest thing to an
artifact it can have, because it converts "would someone speak up?" into a
question about a statute rather than about a manager.

**Weakens: the specific capability thresholds.** Not because thresholds are
wrong, but because they are written in the units that Lesson 19 measured
expiring. A statutory threshold stated as a capability level inherits every
problem this phase has catalogued: benchmarks rescale (Lesson 19), horizons
are upper bounds with two independent error sources (Lesson 21), and a model
that can recognise an evaluation can sit below a threshold on demand (Lesson
20). Worse, a statutory threshold is far harder to revise than a lab's own
policy — the revision cycle is a legislative session rather than a document
update — so it locks in the 2026 unit for years. The stronger form is the one
exercise 4 of Lesson 19 argues for: a *relative* trigger, tied to the prior
generation, plus a disclosure obligation for the baseline. That keeps the
statute's force and moves the number out of it.

### 4 — the weakest lever is the auditor living in the tree

The deployment scored is the pipeline that produced this file: an agent
authoring solutions against a gate. Each sub-lever is scored against something
resolvable in the tree.

| sub-lever | score |
|---|---|
| safety culture | **no artifact to read** |
| **audit rigor** | **8 of 8 deciding paths inside the tree, 0 outside** |
| multi-layered defenses | 4 gate scripts plus the test suite |
| information security | 0 credentials in the gates |

**Audit rigor is weakest**, and not because the gates are weak — they are the
strongest of the four levers. It fails on *independence*: `audit_practice.py`,
`check_deps.py`, the harness modules and the manifest all sit in the tree the
agent edits. **The cost to bring it to par** is one package boundary: publish
the harness and the gates as an installed package resolved from outside the
working tree, pin its version in the manifest, and hash the pinned version the
way `parity.doc_text` already lets the spec-drift check hash the exercise text.
One structural change, no new checks.

**Presence and independence are different scores and the framework has one
field.** `independent_audit` is a single boolean, so three distinguishable
situations — no gate, a gate the agent can edit, a gate run by a third party —
collapse to one bit. That is exercise 5's whole argument.

**Safety culture is the sub-lever with nothing to read.** The other three
resolve to files, counts and paths; this one resolves to how a disagreement is
handled, and no tree holds that. It is also the one the headline names as the
practitioner's lever — so the most important sub-lever is the one a tool cannot
score, which is an argument for the scoring being a conversation with a
checklist attached rather than a function.

**And the cost is asymmetric.** The four defence layers were bought
incrementally, one script at a time. Audit independence admits no partial
credit — the gates are either in the tree or out of it — so the weakest lever
is also the one where there is no cheap first step. That asymmetry is why it
stays weakest, in this repository and generally.

### 5 — split the lever that cannot tell three situations apart

**The 2028 sketch: five categories, nine fields, one relative threshold.**

| change | from | to |
|---|---|---|
| split `organizational_risks` | 1 category | `controls_present` + `controls_independent` |
| add a field | 8 fields | 9, adding `safety_culture` |
| rogue-AI trigger | `>= 4.0` hours | ratio against the current frontier horizon |

**The split is justified by a pair the 2026 tagger cannot separate.** A
deployment with no gates at all, and one whose gates are all editable by the
agent they gate, both score `independent_audit=False` and receive the
identical four-item mitigation list. Under the split the first fails both new
categories and the second fails one — the difference between "build a gate"
and "move the one you have", which is a different quarter of work.

**The removal is `malicious_use`'s public-facing conjunct.** It is a
conjunction of two booleans, so an internal frontier agent drops the tag
(exercise 1). A framework revised after a year of *deployment* experience
should key the tag on the capability and treat exposure as a severity
multiplier — one fewer conjunct, one more field.

**What should not change is the count of levers a practitioner controls.** One
of the four risks is internal to the org; three are structural. Splitting the
internal one does not make two more of them controllable — it makes the one
controllable lever legible. The framework's value is the asymmetry it names,
and a 2028 revision that grew the *structural* categories would be measuring
the weather rather than the roof.
