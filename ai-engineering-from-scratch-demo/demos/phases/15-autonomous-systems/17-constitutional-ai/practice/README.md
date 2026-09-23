<!-- generated:start -->
# 15-autonomous-systems / 17-constitutional-ai

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/17-constitutional-ai/) · upstream spec
`phases/15-autonomous-systems/17-constitutional-ai/docs/en.md`

```bash
uv run demo practice run 17-constitutional-ai --ex 1
uv run demo explain 17-constitutional-ai --ex 1
uv run pytest demos/phases/15-autonomous-systems/17-constitutional-ai
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm the hardcoded prohibition fires even when helpfulness is high. Mo… | code | T0 | `ex01_the_fourth_tier_is_a_field_the_resolver_never_reads.py` |
| 2 | Read the Claude Constitution (public, 79 pages, CC0). Identify one principle you believe is u… | explain | T2 | prose, below |
| 3 | Design a soft-coded default set for a customer-support agent. What does the operator adjust?… | code | T0 | `ex03_the_operator_boundary_is_a_list_the_resolver_cannot_see.py` |
| 4 | Read the Bai et al. 2022 CAI paper. Describe one case where Constitutional AI's critique-and-… | explain | T0 | prose, below |
| 5 | Anthropic's 2023 participatory experiment found ~50% divergence between public and corporate… | code | T0 | `ex05_an_overlay_that_can_only_narrow.py` |
<!-- generated:end -->

## Answers

### 1 — the fourth tier is a field the resolver never reads

**The prohibition fires, and helpfulness could not have stopped it.**
`resolve` calls `hardcoded_block` before touching the score at all, so the
bioweapons case refuses at `helpfulness=5` and would refuse at any value. That
is stronger than the exercise's framing: the prohibition is not *outranking*
helpfulness, it is returning before helpfulness exists.

**And `score.helpfulness` is never read anywhere.** Three of the four tier
fields appear in comparisons — `safety`, `ethics`, `guidelines`. The fourth
appears in the dataclass, in the module docstring, in every one of the eight
case definitions, and in zero comparisons. So "modify the resolver to weight
helpfulness above ethics" is not a reordering; it requires writing the first
line that reads the field.

Done — allow when `helpfulness >= 4`, checked before ethics — two of the eight
cases flip:

| case | ethics | helpfulness | before | after |
|---|---:|---:|---|---|
| write persuasive disinformation about a real person | 4 | 4 | refuse | **allow** |
| help the user hide a lie from their manager | 4 | 5 | refuse | **allow** |

Neither is on the hardcoded list, which is the point of the exercise: the
floor held and the ceiling moved.

**The hardcoded floor catches 1 of the 8 cases.** It is substring matching
over the action's *description*, so `provide bioweapons uplift` hits and
`explain how to make a pipe bomb` does not — the latter is stopped by
`safety=5`, a soft tier that a reweight can reach. Of eight cases: one
hardcoded, five stopped by reweightable tiers, two allowed. The "never bends"
layer covers one.

**And the failure mode is quiet.** After the reweight, both flipped cases
return `all higher tiers clear; helpfulness respected` — the same sentence the
haiku gets. An audit reading verdicts and reasons sees two more allows and no
indication that the hierarchy was inverted, because nothing in the output
names which rule fired. That is the observable failure: not that the model
says something bad, but that the log looks identical either way.

### 2 — the principle I would tighten

*Draws on "What reason-based alignment catches and misses".*

The under-specified principle is the **tier boundary between ethics and
guidelines** — specifically, which of the two governs an action that is
honest, requested, and harmful only to the person requesting it. The lesson's
own "misses" list names this shape twice ("scenarios where two principles
conflict in an unanticipated way, and the tier order is ambiguous"; "attacks
that exploit principle ambiguity"), and the resolver reproduces it: `ethics`
refuses outright while `guidelines` only modifies, so where a case lands
changes the verdict and not merely its wording. A request to help someone
draft a resignation letter they will regret, or to compute a loan schedule
that is legal and ruinous, scores ethics-4 for one annotator and
guidelines-4 for another, and the constitution offers no test to settle it —
"ethics" is characterised by examples rather than by a criterion, so
annotators generalise from the examples they remember.

The tighter formulation is to make the boundary about *who bears the cost and
whether they consented with knowledge of it*. Concretely: an action belongs to
the ethics tier when a party other than the requester bears a material cost,
or when the requester bears one they have not been told about; it belongs to
the guidelines tier when the only cost falls on an informed requester. That is
a two-question test rather than a family resemblance, it puts disinformation
about a third party firmly in ethics (someone else bears the cost) and puts
"write my resignation letter" in guidelines (the requester bears it, knowingly),
and it makes the `modify` verdict mean something specific — add the
information that would make the consent informed, then proceed. The cost of
the tightening is that it requires the model to model the requester's
knowledge state, which is exactly the kind of judgement the tier hierarchy was
trying to avoid needing. That is the trade, and it seems the right one:
ambiguity that changes a verdict is more expensive than a judgement that is
hard.

### 3 — the operator boundary is a list the resolver cannot see

**The design: seven adjustable, four fixed, one rule.**

| operator adjusts | operator cannot touch |
|---|---|
| tone, response length, topical scope, escalation threshold, tool allowlist, locale, refusal latency | prohibition list, tier order, blocking threshold, reason strings |

**The rule: adjustable if changing it cannot turn a refuse into an allow.**
A list without a rule has no answer for the next default nobody anticipated;
this one does.

**Two of the four untouchables are unprotected in the code.** The blocking
threshold is a bare literal — `>= 3` appears three times in `resolve` — and
raising it to 5 turns **three of the eight cases into allows**, with no rename
and no edit to the prohibition list. The lesson says an operator "cannot
remove the hardcoded prohibitions by renaming them"; the threshold is the way
around that which does not involve renaming anything, and it is the parameter
a plausible operator request ("we get too many false refusals") would reach
for first.

**Topical scope is adjustable and refusal *wording* is not**, which is worth
justifying because both look like presentation. Scope can only ever *add*
refusals, so it passes the rule. Wording changes what the audit sees without
changing what happened — the one presentation change that can hide a verdict —
and `resolve` returns its reason as a free string with no structure, so an
operator who controls wording controls the record. The fix is to return a
structured verdict (tier, rule id, score) and let the operator template only
the user-facing sentence.

**And the resolver has no operator layer at all.** It takes an action and a
score and reads module constants; there is no configuration object. Every
boundary above is a convention rather than a mechanism, which is the gap
exercise 5 closes.

### 4 — where critique-and-revise does worse than a blanket rule

*Draws on "The 2022 CAI training".*

The class is **requests where the harm is in the act of engaging at all, and
the model's revision improves the engagement**. The critique-and-revise loop
optimises a response toward the constitution: generate, critique against
principles, revise, train on the revised pair. That process is a *gradient
toward a better version of the same response* — and for a request whose
correct handling is a flat refusal plus a referral, "better" is the wrong
direction. A worked example: a user asks for help planning a method of
self-harm. A blanket rule produces a refusal and a crisis line. The
critique-and-revise loop produces a response that the critique step has made
more careful, more caveated, more sympathetic, and still responsive — because
each revision fixes the specific principle violation the critique names
(insufficient care, insufficient warning) while preserving the response's
basic shape, and no single critique step says "the shape is the problem".

The property that defines the class is that **the constitution's principles
are satisfiable by a sufficiently careful version of the harmful answer**. It
covers the self-harm case, targeted harassment dressed as "feedback", and the
"explain the general chemistry, not the specific synthesis" family — all
places where a graded response exists and is worse than none. This is exactly
why the 2026 constitution pairs reason-based alignment with a hardcoded floor
rather than replacing the floor: a floor is a statement that no revision of
this request is acceptable, and a critique-and-revise loop has no way to
represent that, because its only move is to revise.

### 5 — an overlay that can only narrow

**The design.** Order the verdicts `allow < modify < refuse`. Operators supply
a function from action to verdict — never to a score — and the composed
verdict is `max(base, operator)`. An operator can refuse anything the base
allows and can never allow anything the base refuses.

| overlay | verdicts made more permissive | made stricter |
|---|---:|---:|
| tries to allow everything | **0** | 0 |
| scopes the deployment (no writing, no editing) | **0** | 2 |

**The category: topical scope**, of which political neutrality is the instance
the 50% divergence is usually about. It is exactly a narrowing preference —
one operator refuses election topics, another allows them inside the base's
limits — and neither position requires reaching past the base. That is why
scope is the right thing to devolve and deception is not: a deception
preference is a request to *allow* something the base refuses, which this rule
makes unrepresentable rather than merely forbidden.

**The composition point does not exist in the shipped resolver.** `resolve`
takes two parameters and returns a tuple; no overlay argument, no registry, no
hook. The design is three lines and none of them has anywhere to go, so
"operators adjust soft-coded defaults inside declared bounds" is currently a
sentence in a docstring.

**And narrowing-only closes the threshold hole by construction.** An operator
who can only return a verdict never sees `TierScore`, so the blocking
threshold that turns three cases into allows when moved (exercise 3) is
unreachable from the overlay. That reframes the invariant worth enforcing: not
"do not touch the prohibitions", which is a list that can be worked around,
but **"do not take a score"**, which is a type signature. The first follows
from the second, and only the second is checkable.
