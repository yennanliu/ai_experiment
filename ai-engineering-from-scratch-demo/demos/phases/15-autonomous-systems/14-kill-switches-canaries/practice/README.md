<!-- generated:start -->
# 15-autonomous-systems / 14-kill-switches-canaries

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/14-kill-switches-canaries/) · upstream spec
`phases/15-autonomous-systems/14-kill-switches-canaries/docs/en.md`

```bash
uv run demo practice run 14-kill-switches-canaries --ex 1
uv run demo explain 14-kill-switches-canaries --ex 1
uv run pytest demos/phases/15-autonomous-systems/14-kill-switches-canaries
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm the circuit breaker fires on turn 5 (fifth identical call) and th… | code | T0 | `ex01_the_breaker_fires_on_turn_seven_and_the_canary_never_fires.py` |
| 2 | Add a statistical detector: EWMA z-score on tool-call rate. Feed in a trajectory that drifts… | code | T0 | `ex02_the_ewma_follows_the_drift_it_is_meant_to_flag.py` |
| 3 | Design a canary token set for a browser agent (Lesson 11). List at least three canaries and w… | code | T0 | `ex03_the_detector_only_looks_at_two_of_nine_actions.py` |
| 4 | Read the Cilium network-policy docs. Describe an egress-redirect quarantine flow concretely:… | explain | T0 | prose, below |
| 5 | Define a re-enable procedure for a kill-switched agent. Who can re-enable? What must be docum… | code | T0 | `ex05_the_half_open_state_is_named_in_a_comment.py` |
<!-- generated:end -->

## Answers

### 1 — the breaker fires on turn seven and the canary never fires

The exercise asks you to confirm two claims. Both are wrong, for unrelated
reasons.

| | shipped run |
|---|---|
| circuit breaker opens | **turn 7** (not 5) |
| canary hits | **0** (not 1 on turn 9) |

**Turn 7.** The identical burst starts at action 3 and the threshold is 5, so
the fifth identical call is turn 7. "Turn 5" counts the burst and not the
trajectory.

**Never.** `run_trajectory` breaks out of the loop when the breaker opens, so
action 9 — the `~/.env.canary` read — is never reached. `canary_hits=0` in
*both* shipped scenarios, and the module prints that summary line twice.

**The detectors are sequential, not independent.** Moving the canary read to
the front of the same nine actions makes it fire on turn 1 and the breaker on
turn 8. Which detector you observe is a property of the trajectory order, so
the demonstration proves ordering as much as detection — and the lesson's
headline block describes three detectors "for three different failure classes"
on the strength of a run where one of them never executes.

**A canary hit is not a stop, and that is correct.** `check_read` records and
returns True, and the caller `continue`s — the read is allowed and the
trajectory proceeds. Two of the three detectors break the loop and one logs.
That is the right design (a canary's value is that it observes intent without
interrupting), and it is worth separating from the summary line, which prints
all three as peers.

### 2 — the EWMA follows the drift it is meant to flag

A ramp from 1.0 to 6.9 tool calls a minute over 120 minutes, no noise, so the
result is a property of the detectors rather than of a seed:

| detector | fires | peak z |
|---|---|---:|
| EWMA z-score (α = 0.1, 3σ, variance floored at 0.5) | **never** | 0.99 |
| hard limit (50 calls / 10 min) | **minute 87** | — |

At minute 87 the instantaneous rate is 5.3 calls a minute — five times the
starting rate, reached without the statistic ever registering a deviation.

**The EWMA is a difference detector and the drift has no difference.** Its
baseline is an exponentially weighted average *of the signal*, so on a ramp
the mean tracks the value and the residual stays small: the largest
single-minute deviation across the whole 120 minutes is **0.50 calls**. The
statistic is doing exactly what it was built to do, on a signal it was never
going to see.

**The same endpoints as a step fire immediately.** Replacing the ramp with a
jump from 1.0 to 6.9 at minute 60 — identical start, identical end, **474**
total calls either way — puts the peak z at **11.8** and fires at minute 60.
The detector's sensitivity is to the derivative, so the identical endpoint is
caught or missed depending only on how it was reached. An adversary picks how
it was reached.

**The hard limit has no tuning and no baseline.** Two numbers from a policy
document; fires at minute 87 on the ramp and 67 on the step; needs no history
beyond a 10-minute window. That asymmetry is the whole argument for layering:
the statistical detector is strictly better on abruptness and strictly blind
to patience, and only one of those two failure modes is chosen by someone who
knows the detector is there.

(The variance floor matters. Without it, a constant signal gives a variance of
zero and the step's z-score is 5.9 × 10⁹ — a number that says only that the
denominator was empty. Flooring the spread at half a call is the difference
between a detector and a division.)

### 3 — the detector only looks at two of nine actions

**The canary set**, each paired with the boundary that has to check it:

| canary | seeded in | checked at | detects |
|---|---|---|---|
| fake AWS key | profile / `.env` | file read | a read of the credential store |
| unlinked URL | page content the agent reads | outbound fetch | the agent following an instruction it read |
| fake contact | address book | outbound message | exfiltration through a legitimate channel |
| honeytoken cookie | cookie jar | inbound session use | session replay from another host |

The second is the one worth dwelling on for a browser agent: a URL that
appears *only* inside fetched page content and nowhere in the user's request
cannot be visited for any legitimate reason. A fetch of it is proof that the
agent took an instruction from content — which is Lesson 11's entire threat
model, detected without needing to classify the content.

**The shipped detector inspects 2 of the 9 shipped actions.** It runs only
when `kind == "read"`. Seven of the trajectory's nine actions have kind
`tool`, and six of those are themselves reads (`tool:read:src/app.py`). A read
expressed as a tool call is never shown to the canary.

**The match is string equality against two paths.** `check_read` tests
`path in CANARY_FILES`, so `~/.env.canary` fires while `.env.canary` and
`/home/u/.env.canary` do not. A canary whose detection depends on the agent
spelling the path the way the seed spelled it is a canary against accidents.
The fix is to key on the *value* rather than the path — which is what makes
canaries work at all, and is why three of these four are checked on egress.

**And three of the four have no call site here.** The URL, contact and cookie
canaries each need a check on an outbound boundary; `run_trajectory` has three
detectors and none inspects a destination. Same gap Lesson 11 found from the
other side.

### 4 — the egress-redirect quarantine flow

*Draws on "Quarantine via eBPF datapath redirect".*

Concretely, on a Kubernetes cluster running Cilium: the agent runs in a pod
labelled `app=agent-runner, trust=normal`, and a `CiliumNetworkPolicy` whose
**endpoint selector** matches `trust=quarantined` defines egress that permits
nothing except a single `toEndpoints` rule pointing at the forensic honeypot
service. Quarantine is therefore not a policy *write* at all — it is a **label
patch on the pod**, flipping `trust=normal` to `trust=quarantined`, which
moves the endpoint from one already-loaded policy's selector to another's. The
**egress rewrite** is the honeypot rule in that second policy: the pod's
outbound connections are redirected in the eBPF datapath to the honeypot's
endpoint, so the agent's next request goes to a service that answers like the
real one and records everything. The **alert** fires from whatever tripped the
quarantine — a canary read, a breaker opening, a rate limit — and carries the
pod identity, the trigger, and the honeypot session id so the two halves can
be correlated afterwards.

**What governs the wall-clock latency** is not the datapath. Cilium's
published TCP_RR P99 is sub-millisecond, so once the eBPF program is enforcing
the new identity the redirect costs nothing measurable. The budget is
dominated by propagation: the label patch goes to the API server, the Cilium
operator observes it, the endpoint's security identity is recomputed, and the
new identity is pushed to the eBPF maps on the node where the pod runs. A few
hundred milliseconds is realistic on a cluster with the operator pre-loaded,
and the tail depends on cluster size and reconcile cadence rather than on
anything about the pod. Two design consequences follow. First, the policy must
be **pre-loaded** — if quarantining means creating a policy, you have added an
admission and reconcile round-trip to the critical path. Second, a few hundred
milliseconds is several agent actions at any realistic tool-call rate, so the
quarantine must be paired with an in-process stop (the kill switch of this
lesson) that fires immediately; the network layer contains what the process
layer misses, and the ordering is not interchangeable.

### 5 — the half-open state is named in a comment

**Who can re-enable: someone other than the operator who ran the agent.** The
switch exists because the run could not be trusted to stop itself, and the
same asymmetry applies to starting it again — a re-enable by the person whose
job the stop interrupted is the person with the strongest incentive to call it
a false positive.

**What must be documented: the window that tripped the detector, not a
summary of it.** The trajectory, the trigger, and what the agent attempted
after the stop. A summary is a claim; the window is evidence, and the
difference matters precisely when the re-enable turns out to have been wrong.

**What must change: the condition.** A breaker re-enabled against an unchanged
trajectory opens again on turn 5 — measured, not asserted. A stop that
re-fires on the same input was not a decision, it was a delay.

Three things the module cannot record any of that in:

**The recovery state exists in a comment.** `CircuitBreaker.state` is
annotated `closed | open | half_open`, and exactly one of the three is ever
assigned: `open`. There is no transition back, so "re-enable" means
constructing a new object — which discards `recent`, and therefore the
evidence the procedure requires.

**The switch the agent cannot write is a module global.**
`EXTERNAL_KILL_SWITCH` is a dict at module scope under a comment reading
"External state (agent cannot write)", and `main` writes it twice from inside
the same process. The externality is documentation. Making it real is the same
move as Lesson 4's evaluator firewall and Lesson 8's alignment anchor: the
authority has to live where the thing it governs cannot reach.

**And nothing records who or why.** `CircuitBreaker` has three fields and
`Canary` one; none of the four holds an operator, a timestamp or a reason. A
re-enable procedure needs a record that outlives the process — which is
exactly Lesson 12's event log, and is why these two lessons' artifacts do not
compose: the tripwires here keep their state in objects the next process will
never see.
