<!-- generated:start -->
# 13-tools-and-protocols / 25-skill-invocation-and-routing

Solutions to all 6 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/25-skill-invocation-and-routing/) · upstream spec
`phases/13-tools-and-protocols/25-skill-invocation-and-routing/docs/en.md`

```bash
uv run demo practice run 25-skill-invocation-and-routing --ex 1
uv run demo explain 25-skill-invocation-and-routing --ex 1
uv run pytest demos/phases/13-tools-and-protocols/25-skill-invocation-and-routing
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Create all four rows of the human/model matrix and write one legitimate use case for each. | code | T0 | `ex01_the_first_row_is_two_different_deployments_wearing_one_label.py` |
| 2 | Add application-only activation to `CorePolicyAdapter`. Prove that human and model callers re… | code | T0 | `ex02_the_policy_is_catalog_wide_and_application_only_is_per_skill.py` |
| 3 | Write ten near misses for a deployment skill. Each prompt must share vocabulary with the skil… | code | T0 | `ex03_near_misses_only_calibrate_one_of_the_two_ways_routing_fails.py` |
| 4 | Add an ambiguity margin between the top two routing scores. Return `ask` when the margin is t… | code | T0 | `ex04_the_shipped_tie_break_is_alphabetical_and_silent.py` |
| 5 | Add a maximum composition depth to skill-to-skill requests and detect a two-skill cycle. | code | T0 | `ex05_the_depth_is_asserted_by_the_caller_it_is_meant_to_bound.py` |
| 6 | Run the same labeled set through core and extension adapters. Explain every changed decision. | code | T0 | `ex06_the_extension_adapter_can_only_subtract.py` |
<!-- generated:end -->

## Answers

All six are T0 and stdlib, and all six ship code. Exercise 6 is scaffolded as
`explain`; it has a real code deliverable — the same labeled set through two
adapters — so it is answered as `code`.

Invocation here is decided in two places that do not know about each other:
`InvocationPolicy`, which is one object for the whole catalog, and
`SkillMetadata.runtime_extensions`, which is per skill and advisory. Four of
the six exercises turn out to be about that seam. Application-only is a
per-skill property with only catalog-wide flags to express it. The extension
adapter can subtract eligibility and cannot say so. Composition depth is a
policy number checked against a caller-supplied integer.

The other recurring finding is that **the refusal does not reach the
caller**. Implicit routing filters for eligibility *before* scoring, so a
denied skill is not denied — it is absent. The model is told the threshold
was not met, or is handed the second-best skill with the ordinary
eligibility reason, and in neither case learns that something was excluded.

### 1 — the first row is two different deployments wearing one label

**ANSWER: four policies, four use cases, and the routed behaviour matches
every row.** **8** requests, **8** agreements — a use case is only
legitimate if the runtime produces the row it claims.

**FINDING: the first row is two different deployments wearing one label.**
`programmatic-only or unavailable` is an *or*. The same `(False, False)`
policy activates for a harness caller when the name is allowlisted and
denies when it is not, and the printed row is byte-identical either way.

**FINDING: an agent is not a model here, and the matrix has no column for
it.** `allow_model=False, allow_agent=True` prints in the "no model
activation" row while the agent activates. The matrix reads **2** of the
policy's **6** actor flags.

**FINDING: `active_policy` marks exactly one row, and none without a
policy.** A 2-tuple comparison — exhaustive and disjoint by construction,
and the same reason three actors have nowhere to appear.

### 2 — the policy is catalog-wide and application-only is per-skill

**ANSWER: a per-skill rule denies every non-application actor and leaves the
catalog reachable.** Clearing `allow_human` instead denies **3** of **3**
skills to protect **1**.

**FINDING: the exercise names two actors and there are six.** An agent is a
model on a different branch, so "human and model remain denied" is satisfied
by a rule that still lets an agent in. Allowlist one actor; do not denylist
two.

**FINDING: the shipped application branch already requires two things.**
`allow_application and name in application_allowlist` — the allowlist is the
per-skill half that already exists, pointed the other way.

**FINDING: implicit routing hides the refusal it acted on.** The explicit
actors are told the skill is application-only; the model and agent are told
`best match did not meet the host threshold`. Shrink the catalog to that one
skill and the same request finally names it — the refusal is reportable only
when there is nothing else to report.

### 3 — near misses only calibrate one of the two ways routing fails

**ANSWER: ten near misses, ten workflows, and they move the threshold from
0.15 to 0.30.** At the demo's threshold **6** of **10** falsely activate;
the top near miss is **0.2667** and the lowest true positive **0.3846**.

**FINDING: padding a near miss makes it pass.** Jaccard divides by the
union: eleven unrelated words take the hardest prompt from **0.1111** to
**0.0741**. A near-miss set has to be length-controlled or it measures
length.

**FINDING: the skill's name is scored as part of its description.**
Renaming `deploy-service` to `ship-service` moves every score with the
description untouched. A rename is a routing change.

**FINDING: near misses only calibrate one of the two ways routing fails.**
They are high-overlap and wrong. `what happened during the outage last
night` is low-overlap and right — **0** shared tokens with
`incident-triage`, score **0.0**, admitted by no threshold.

### 4 — the shipped tie-break is alphabetical and silent

**ANSWER: `ask` inside the margin, `activate` outside it, never `ask` with
one candidate.** An unopposed winner is not ambiguous, so `ask` is a
statement about the runner-up rather than about confidence.

**FINDING: the shipped tie-break is alphabetical and silent.** Two skills at
**0.5** each resolve to the later name, `activated=True`, reason `model
activation policy`. A confident match and a coin toss look identical.

**FINDING: an absolute margin is the wrong shape at low scores.** The same
**0.03** gap is **8%** of a 0.36 score and **60%** of a 0.05 one, and both
clear the threshold.

**FINDING: `ask` has nowhere to live in the return type.** Three outcomes,
one `activated: bool`, and a string doing the distinguishing.

### 5 — the depth is asserted by the caller it is meant to bound

**ANSWER: carry the chain, derive the depth from it, and both checks become
possible.** Depth stops being a claim and becomes `len(chain)`.

**FINDING: the depth is asserted by the caller it is meant to bound.** Five
nested calls each declaring `depth=1` are all allowed and the limit is never
approached. A bound the bounded party writes is not a bound.

**FINDING: one name catches a self-cycle and nothing longer.** `A → A` is
refused; `A → B → A` passes, because by then the adapter is looking at a
request from B to A and has never heard of the first hop.

**FINDING: the dataclass default is a value the policy rejects.** `depth`
defaults to **0** and the adapter requires `>= 1`, so the obvious request is
denied for a reason about the default rather than about composition.

### 6 — the extension adapter can only subtract

**ANSWER: 12 requests, 3 changed decisions — one denial and two silent
re-routes.** **0** decisions change the other way.

**FINDING: subtracting eligibility from an implicit route re-routes, it does
not deny.** The model and agent move from `archive-cleanup` to
`incident-triage` and stay activated, with the ordinary eligibility reason.

**FINDING: the false-ness test recognises four spellings and fails open on
the rest.** `False`, `"false"`, `"False"`, `"FALSE"` deny; `0`, `"no"`,
`"off"`, `"disabled"`, `None` do not. Fail-open is the wrong default for a
field whose only job is to deny.

**FINDING: one field governs two actors and another governs one.**
`disable-model-invocation` covers `MODEL` and `AGENT`; `user-invocable`
covers `HUMAN` only, so an `APPLICATION` or `HARNESS` caller reaches a skill
its own metadata calls not user-invocable.
