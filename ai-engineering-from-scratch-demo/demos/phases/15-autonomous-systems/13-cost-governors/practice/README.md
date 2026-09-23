<!-- generated:start -->
# 15-autonomous-systems / 13-cost-governors

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/13-cost-governors/) · upstream spec
`phases/15-autonomous-systems/13-cost-governors/docs/en.md`

```bash
uv run demo practice run 13-cost-governors --ex 1
uv run demo explain 13-cost-governors --ex 1
uv run pytest demos/phases/15-autonomous-systems/13-cost-governors
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm the velocity limit fires before the iteration cap on a polling-lo… | code | T0 | `ex01_the_velocity_limit_is_eighty_three_times_above_what_the_request_cap_allows.py` |
| 2 | Design a per-tool cap set for a browser agent (Lesson 11). Which tool needs the tightest cap?… | code | T0 | `ex02_the_two_rankings_disagree_so_one_number_cannot_say_it.py` |
| 3 | Read the Microsoft Agent Governance Toolkit docs. List every cap type the toolkit names. Map… | explain | T0 | prose, below |
| 4 | Price an overnight unattended run for a realistic task (e.g., "triage 50 issues in a repo").… | code | T0 | `ex04_two_times_covers_estimate_error_and_not_the_failure_mode.py` |
| 5 | Claude Code's `max_budget_usd` fires on session aggregate cost. Design a complementary veloci… | code | T0 | `ex05_a_token_rate_trigger_fires_nine_turns_into_the_loop.py` |
<!-- generated:end -->

## Answers

### 1 — the velocity limit sits above what the request cap allows

The exercise says to confirm the velocity limit fires first. It does not fire
at all.

| configuration | turns | tokens | dollars | stopped by |
|---|---:|---:|---:|---|
| layered stack | 200 | 1,440,500 | $4.32 | `max_turns` |
| layered, velocity disabled | 200 | 1,440,500 | $4.32 | `max_turns` |

Identical in every column. The second half of the exercise — "measure how
much the agent spends before the iteration cap catches it" — has the same
answer with the limit on or off: **$4.32**.

**The loop's rate is 104× below the limit meant to catch it.** At 8,000
tokens a turn and 30 seconds a turn, the polling loop burns **$0.048/min**
against a `velocity_usd_per_min` of **$5.00**.

**And no trajectory that respects the request cap can trip the velocity
limit.** `max_tokens_per_request = 10_000` is $0.03 a turn, which at two turns
a minute is **$0.06/min** — 83.3× below the velocity threshold. The two layers
are inconsistent by construction: the cheaper one makes the more expensive one
unreachable, whatever the agent does. That is a stronger statement than "this
profile does not trip it"; it holds for every possible profile.

**Three of the five layers never fire here.** The request cap does not
truncate (8,000 < 10,000), the velocity limit is unreachable, and 10,000 turns
of pure loop reach $239.52 against a $500 monthly cap. The stack the lesson
presents as five layers is, on its own demonstration, one layer (`max_turns`)
with a dollar cap behind it — which does not refute the argument for layering,
but does mean the demo is not evidence for it.

### 2 — the two rankings disagree, so one number cannot say it

Both of the exercise's questions presume a single ordering of tools by risk.
There are two orderings and they are near-reverses.

| tool | tokens/call | $/call | reversible | proposed cap |
|---|---:|---:|---|---|
| read page | 10,000 | $0.0300 | yes | dollar cap |
| click | 200 | $0.0006 | yes | count cap |
| type into form | 200 | $0.0006 | yes | count cap |
| submit or pay | 500 | $0.0015 | **no** | **1 per run + human tap** |

By dollars: read → submit → click → type. By irreversibility: submit → type →
click → read. **The tool at the top of one list is at the bottom of the
other.**

**Which needs the tightest cap: the consequential write**, at one call per run
behind a human tap — because its risk is not denominated in money, so no
dollar figure expresses it. **Which can run unbounded: none.** The read is the
closest candidate and is exactly the wrong one — it is 50× the token cost of a
click and, from Lesson 11, it is the injection vector (4 of 9 reads render
third-party content).

**The shipped governor has no per-tool anything.** `Governor` carries 12
fields and none names a tool. The lesson's own stack lists 12 control types;
the simulator implements 5 of them, and the per-tool cap — item 4, and the
actual fix in the $1,200 case study — is not among them.

**And a page read sits exactly on the request cap.** 10,000 tokens is
`max_tokens_per_request` to the token, so the cap truncates the first page the
agent opens. A truncated page is a page half-read: a correctness failure that
the cost layer records as a success, and the one place these two concerns are
not separable.

### 3 — the toolkit's cap types, mapped to failure modes

*Draws on "The cost-governor stack".*

The twelve controls the lesson enumerates fall into four groups, one per
failure mode.

**Runaway loop** — the failure that happens in minutes. Covered by the
**iteration cap** (`max_turns`), the **per-tool call cap** (no more than N
`WebFetch`), and the **financial velocity limit** (spend over $X in Y
minutes). These are the only three that can fire before the money is gone;
exercise 1 shows that if their numbers are not derived from each other, only
the first of them actually does.

**Slow leak** — the failure that happens over weeks, and the one the $1,200 →
$4,800 case actually was. Covered by the **rolling per-day / per-week /
per-month caps** and by the growth alert that pairs with them. The
distinguishing property is that no single session looks wrong: the fix in the
case study was a per-tool cap *plus* a week-over-week alert, because the cap
alone would have been set above the new steady state.

**Surge** — many sessions at once, each individually fine. Covered by the
**per-month cap** and **tiered model routing**, plus **prompt caching** and
**context windowing**, which are cost controls rather than cost *caps*: they
lower the slope instead of drawing a line. They belong in the stack because a
surge you can absorb is a surge that does not need a cap to fire.

**Bad release** — a change that makes every session more expensive, which is
the shape of the case study's new polling tool. Covered by the **per-request
`max_tokens`** and the **per-task token and dollar budgets**, which bound the
blast radius of a regression to one session, and by **HITL checkpoints on
expensive actions**, which put a human in front of the specific call a bad
release makes cheap to issue. The **kill switch on budget breach** is the
twelfth and sits across all four: it is what makes any of the others
terminal rather than advisory — and the half of it the simulator omits is the
separate re-enable path, which is exercise 5's last finding.

### 4 — 2× covers estimate error and not the failure mode

**The estimate.** Fifty issues at 12 turns each is 600 turns; at the module's
normal turn (2,500 tokens, $0.003/ktok) that is $0.0075 a turn and **$4.50**
for the run. `max_budget_usd` at 2× is **$9.00**.

**The justification for 2×, honestly.** It covers being wrong about the
estimate: if the per-issue turn count is double — 24 turns instead of 12 — the
run costs exactly $9.00 and lands on the cap. That is the right size for
estimate error on a task nobody has run before.

**What it does not cover.** A run that drifts into the polling loop reaches
$9.00 after 375 turns, short of the 600 the job needs — so the budget is
exhausted with the job 63% done and nothing to show. And `max_turns` of 200
fires before either. Priced this way the dollar cap is decoration on top of
the turn cap, for the third time in this lesson.

**The lesson's own numbers disagree with its own prose.** The section titled
"$1,200 → $4,800" describes it as a tripling; the ratio is **4.0**. Whichever
figure you take, both exceed the 2× the exercise prescribes — so the
multiplier is not derived from the case study it sits beside. The defensible
version of the answer is that 2× is right *and insufficient*: it is sized for
estimate error, and the case study's overrun was a behaviour change, which no
multiplier catches because the shape of the spend changed rather than its
size.

**A dollar cap cannot fire early, by construction.** `max_budget_usd` compares
a running total to a constant, so it fires once the money is spent. One of the
stack's twelve controls is about *rate*, and the simulator prices it out of
reach. That is why the answer to "justify the 2×" ends at exercise 5 rather
than at a bigger number.

### 5 — a token-rate trigger fires nine turns into the loop

**What triggers the cut-off: tokens a minute, at 2× the observed baseline,
over a short rolling window.** A second dollar threshold is not complementary
to a dollar threshold — it is a smaller one — so the design changes the
denomination, not just the number.

Baseline here is 5,000 tokens/min; the polling loop is 16,000. A threshold at
**10,000** sits between them:

| window | fires on turn | turns into the loop |
|---:|---:|---:|
| 10 min | 39 | 9 |
| 5 min | 34 | 4 |
| 2 min | 31 | 1 |

At the 10-minute window that is **161 turns before `max_turns`**, with **$0.46
spent of the $4.32** the shipped stack allows — a 89% reduction in the cost of
the failure, using a threshold derived from the agent's own steady state
rather than from a price list.

**Tokens are the right denomination because dollars are derived.**
`DOLLARS_PER_KTOK` is a single module constant: every dollar threshold in the
stack moves when a price moves, and no token threshold does. That is precisely
how the shipped `velocity_usd_per_min` ends up 83.3× above what the request cap
permits — a dollar threshold written against one price list and never
re-derived against the caps beside it.

**The window is the latency knob and it is cheap.** Shortening it from 10
minutes to 2 moves the trigger from turn 39 to turn 31 — one turn after the
loop's first full window — at no cost except sensitivity to a genuine burst,
which a 2×-of-baseline threshold is already trading away.

**What re-enable looks like: manual, and carrying the window that tripped it.**
The run resumes only when a human has seen the rolling window and the
trajectory inside it, and the resume writes a new baseline so a genuine
workload change does not re-trip immediately. The simulator has nowhere to put
any of that: `Run` carries 5 fields, `stopped_by` is a string, and `simulate`
leaves the loop with no resume path — zero fields record who cleared a stop or
what they saw. The lesson's stack asks for exactly this as item 12 ("cap is
recorded; requires a separate re-enable path"), and the code implements the
cut without the half that lets work continue.
