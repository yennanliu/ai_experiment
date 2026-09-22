<!-- generated:start -->
# 14-agent-engineering / 33-instructions-as-executable-constraints

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/33-instructions-as-executable-constraints/) · upstream spec
`phases/14-agent-engineering/33-instructions-as-executable-constraints/docs/en.md`

```bash
uv run demo practice run 33-instructions-as-executable-constraints --ex 1
uv run demo explain 33-instructions-as-executable-constraints --ex 1
uv run pytest demos/phases/14-agent-engineering/33-instructions-as-executable-constraints
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a sixth category if your product genuinely needs it. Defend why it does not collapse into… | code | T0 | `ex01_forbidden_is_a_path_list_and_reversibility_is_a_property.py` |
| 2 | Extend the checker so a rule can carry a severity (`block`, `warn`, `info`) and the report ag… | code | T0 | `ex02_a_missing_check_and_a_broken_rule_report_the_same_thing.py` |
| 3 | Wire the checker into CI: fail the build if a block-severity rule fails on the latest agent run. | code | T0 | `ex03_a_malformed_rule_is_dropped_and_the_build_goes_green.py` |
| 4 | Add an "expiry" field per rule. After 90 days without a check fail, the rule is up for review. | code | T0 | `ex04_a_rule_that_never_fails_is_working_or_dead_and_the_report_cannot_say.py` |
| 5 | Find a real `AGENTS.md` and rewrite it as five-category rules. How many of its lines were ope… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — reversibility, because forbidden enumerates paths and approval needs a person

A sixth category earns its place when there is a turn that passes every rule in
the five and it refuses. **Reversibility** — every destructive action carries a
recorded undo — clears that bar: a turn that reads the state file, edits only
allowed paths, passes its tests, is confident and adds no dependency, while
deleting `migrations/0007_drop_users.sql` with no backup, passes 5 of 5 shipped
rules and fails reversibility.

The two collapse arguments, tested rather than asserted:

*Into forbidden?* `no_release_script_edits` names one literal path. Over six
destructive actions on paths nobody listed, the forbidden rule refuses zero;
reversibility refuses the four that recorded no undo. Forbidden is an
enumeration, so covering the same ground means a new rule per file anyone ever
adds — and the rule you need is always for the file you did not think of.
Reversibility is a predicate over the *action*, which is why it generalises.

*Into approval?* `new_dependency_approved` passes on all six, because none adds a
dependency. An approval rule covering deletes would put a human in the loop for
every one; reversibility asks only that an undo exists, which two of the six
already satisfy without asking anybody. Approval is about *who decides*;
reversibility is about *whether the decision can be taken back*.

One thing the exercise's framing assumes turns out not to hold: `parse_rules`
accepts any string after `category:`, so the sixth category parses with zero
parser changes, and the module mentions the five category names zero times. The
taxonomy the lesson defends — "a rule that does not fit one of these five usually
wants to be two rules; force the split" — is enforced by review, not by the
checker. A `category` field with no allowed set is a comment.

### 2 — severity is twenty lines, and it makes a typo fail the build

Parsing `- severity:` with a default and grouping the results aggregates five
rules into three buckets: on the shipped bad trace, `block` 0/3, `warn` 0/1,
`info` 0/1; on the good trace 3/3, 1/1, 1/1. One regex of parser change.

The field exposes a defect that was harmless before it. `score` does
`getattr(checker, rule.check, None)` and emits `passed: False` when the attribute
is missing, so there are two roads to the same verdict: the check ran and said no,
or the check does not exist. Renaming one `check:` target to a typo leaves the
rule count at five and the bad trace failing five — output identical to before —
while the good trace drops from five passing to four. Attach `block` severity and
a typo in a Markdown file fails the build with a message indistinguishable from a
real violation.

The distinction is recoverable and the shipped result type cannot carry it:
`score` emits three keys per rule — slug, category, passed — so a third verdict
has nowhere to go. Triaging into `pass` / `fail` / `error` separates one
configuration bug from four genuine failures on the same trace, which is the
difference between "fix the agent" and "fix the rules file".

And severity has to default, because zero of the five seed rules carry the field.
A parser that requires it drops all five; one that defaults to `block` makes every
existing rule blocking, which is probably right and should be a decision rather
than an accident. Defaulting to `warn` would silently downgrade
`no-release-script-edits`, the rule most obviously meant to stop a build.

### 3 — a malformed rule is dropped silently, and the build goes green

The gate is a filter and an `any`: block the build if any block-severity rule
failed on the latest run. It blocks the bad trace naming three rules and allows
the good one; the bad trace fails five rules and three of them gate, so warn and
info failures are reported without stopping anything. That is the behaviour the
exercise asks for and it works.

The problem is upstream. `parse_rules` requires both `category:` and `check:` and
`continue`s otherwise. Take a run whose *only* violation is editing
`scripts/release.sh` — the intact rule set blocks it, naming exactly that rule.
Delete that rule's `check:` line and the parser drops the whole block: four rules,
two block rules, and the identical run is **allowed**. The build goes green
because a rule was malformed, which is the precise inverse of what a gate is for.

This is worse than the exercise-2 defect, and in the opposite direction. There, a
typo in a check name turns into a spurious failure — noisy, and someone
investigates. Here, a missing line turns into a *silent* pass. Fail-open is the
expensive default, and the fix is one line: `parse_rules` returns a list, so
comparing `len(rules)` against the count of `## ` headings recovers what it
dropped — five headings, four rules, one gone.

One more gap worth noting before shipping this gate: `TurnTrace` has seven fields
and none of them is a timestamp, so a CI job handed last week's trace gates on
stale evidence and cannot tell. That is Lesson 32's state-freshness problem
sitting in the artifact the build decision is made from.

### 4 — the field is easy; "90 days without a fail" means two different things

Replaying 120 days of runs and keeping the most recent day each rule failed puts
three of five past a 90-day window: `no-release-script-edits` last failed on day
14, `new-dependency` on day 9, `open-question-note` on day 3. `state-file-fresh`
(118) and `tests-pass` (112) stay.

The three expiring rules do not mean the same thing, and the report cannot tell
them apart. `no-release-script-edits` has not failed because *nothing tried* —
zero of the last 106 runs touched that path. `open-question-note` has not failed
because its check passes whenever confidence is at least 0.7, and 106 of 106
recent runs were confident: the rule is not unexercised, it is close to
unfalsifiable. One wants keeping and one wants rewriting, and a failure count puts
them in the same bucket.

The sharper objection is that expiry by calendar punishes rare rules exactly when
they matter. The release guard had three opportunities to fire — days 2, 8 and 14,
the only runs that touched `scripts/release.sh` — and it caught three of three. It
is then retired at day 90 for having no failure in 106 runs, *none of which could
have failed it*. A rule with a 100% catch rate on every occasion it could act is
expired for inactivity, because the activity is seasonal. Counting opportunities
rather than days keeps it, and is not much harder: an opportunity is a run where
the rule's inputs were non-trivial.

Underneath all of it, nothing in the module accumulates. `score` returns a list
per trace, `Rule` has four fields and `TurnTrace` seven, and none of the eleven is
a date or a counter. The expiry field lives on the rule and the evidence for it
has nowhere to live — which is why this exercise is really a request for a second
file, not a second field.

### 5 — a real 309-line AGENTS.md: about 40 lines are operational

**Five categories that cover most rules** is the frame: startup, forbidden,
definition of done, uncertainty, approval, with the claim that a rule fitting none
of them usually wants to be two rules.

The file analysed here is the `AGENTS.md` at the root of the
`ai-engineering-from-scratch` curriculum repo — a genuine, actively used
instruction file, not a sample. Measured: **309 lines**, 73 blank, 32 headings, 22
table rows, 14 fence lines, 7 numbered rules, 6 bullets, across 10 top-level
sections (`Philosophy`, `Repo layout`, `Hard rules`, `Dependencies`,
`Lesson contract`, `Learning Objectives`, `Per-PR validation`,
`Automation contract`, `Conflict resolution`, `New-lesson onboarding`).

**How much is operational?** Reading "operational" as *a machine could decide
whether this run complied*, the honest count is around 40 lines — roughly 13%.
They cluster in three places:

*Hard rules* (7 numbered items) is the densest operational section in the file and
maps almost perfectly onto the five categories. "One commit per lesson directory"
and "Conventional commit subjects ≤72 chars" are **definition of done** — both are
checkable from `git log`. "Mermaid or SVG only", "every fenced code block needs a
language tag", "never commit generated files" and "original implementations only"
are **forbidden**, and three of the four are one regex each. Rule 6, the
dependency allowlist, is **approval** in disguise: a table of permitted packages
per language, with an explicit escape hatch ("if a finding suggests a banned dep,
skip it with the reason...").

*Dependencies* (22 table rows) is a single **forbidden**/**approval** pair
expressed as data, and it is the best-shaped part of the file: a checker reads the
table, not the prose.

*Per-PR validation* is **definition of done** — the commands that must pass.

**How much is aspirational?** The rest, and it is not wasted. `Philosophy` is one
paragraph of framing that no check could ever evaluate ("every algorithm built
from raw math before a single framework gets imported"). `Repo layout` is a
25-line fenced tree — reference material, valuable, and a router should *link* to
it rather than inline it. `Lesson contract` and `Learning Objectives` are
templates: partly checkable (does frontmatter have the required keys?), mostly
descriptive. `Conflict resolution` and `New-lesson onboarding` are procedures for
humans.

**What the five-category rewrite produces.** Roughly 14 rules with checks:
~6 forbidden (diagram format, code-fence tags, generated files, external-curriculum
citations, batched commits, banned dependencies), ~4 definition of done (commit
subject format, one-commit-per-lesson, per-PR commands, lesson-contract keys), ~2
startup (read ROADMAP for status, read the glossary before defining a term), ~1
approval (new dependency outside the allowlist), ~1 uncertainty (the "skip it with
the reason" escape hatch, which is the only rule in the file that already tells the
agent what to do when it disagrees).

**The three observations that matter more than the counts.**

*The aspirational content is not the problem; its position is.* The file opens
with Philosophy and reaches Hard rules at line 41. The doc's progressive-disclosure
argument — "the agent reads the first screen, runs out of attention budget, and
acts on a fraction of what it was told" — bites here: the first screen is framing
and a directory tree, and the seven rules that would actually be checked are below
the fold. Inverting the order costs nothing and is the single highest-value edit.

*One category is nearly absent, and it is the one that predicts incidents.* Of the
five, **uncertainty** appears once, and only as a dependency escape hatch. The
file tells the agent a great deal about what to do and almost nothing about what
to do when the instructions do not cover the case — which, for a 523-lesson
curriculum with 20 phases, is most novel work.

*The operational fraction is the useful metric, not the line count.* 309 lines is
not too long if 40 of them are enforced and the other 269 are reference material
the agent can load on demand. It is too long if nobody can say which 40. Exercise
3's finding is what makes this concrete: a rule whose `check:` line is missing is
indistinguishable from prose, and the build stays green. The rewrite's value is
not brevity — it is that after it, the answer to "which of these does CI enforce?"
is a number.
