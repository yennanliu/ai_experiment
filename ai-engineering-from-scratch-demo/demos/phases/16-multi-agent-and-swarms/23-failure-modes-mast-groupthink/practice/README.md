<!-- generated:start -->
# 16-multi-agent-and-swarms / 23-failure-modes-mast-groupthink

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/23-failure-modes-mast-groupthink/) · upstream spec
`phases/16-multi-agent-and-swarms/23-failure-modes-mast-groupthink/docs/en.md`

```bash
uv run demo practice run 23-failure-modes-mast-groupthink --ex 1
uv run demo explain 23-failure-modes-mast-groupthink --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/23-failure-modes-mast-groupthink
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm the circuit breaker caps the retry storm. Vary the failure thresh… | code | T0 | `ex01_the_breaker_opens_once_and_never_closes_so_it_caps_the_storm_by_switching_the_service_off.py` |
| 2 | Implement a slow-failure proxy: agreement rate across 3 parallel agents. When it drops sharpl… | code | T0 | `ex02_monoculture_drift_raises_agreement_so_a_drop_alarm_never_fires.py` |
| 3 | Read Cemri et al. (arXiv:2503.13657). Pick one of their 7 MAS systems and map its top 3 failu… | explain | T0 | prose, below |
| 4 | Read the Groupthink paper (arXiv:2508.05687). Identify which of the five patterns is hardest… | code | T0 | `ex04_conformity_leaves_the_voted_answer_unchanged_and_erases_the_dissent.py` |
| 5 | Design a STRATUS-style detection-diagnosis-validation trio for a specific multi-agent system… | code | T0 | `ex05_a_validator_that_watches_the_symptom_approves_switching_the_service_off.py` |
<!-- generated:end -->

## Answers

### 1 — the breaker opens once and never closes, so it caps the storm by switching the service off

**It caps calls, 328 → 140, and costs successes, 189 → 113.** On 200 requests
the breaker short-circuits 87 of them. Sweeping the threshold:

| threshold | calls | successes |
|---:|---:|---:|
| 0.05 | 26 | 25 |
| 0.1 | 41 | 38 |
| 0.2 | 110 | 97 |
| 0.3 | 111 | 97 |
| 0.5 | 140 | 113 |
| 0.7 | 280 | 176 |
| 0.9, 1.0, no breaker | 328 | 189 |

Every threshold that sheds load also sheds successes. Three things in the
module decide that.

**The breaker opens once and never closes.** `open_cooldown_s` is 0.5
*wall-clock* seconds, and the whole simulation runs in well under a
millisecond. `allow()` never reaches HALF_OPEN: there is exactly one state
transition, closed → open at call 140, and every request after it is
short-circuited. The "cap" is the service switched off for the rest of the
run.

**The service can never recover, so backing off cannot help it.**
`service.load = min(total_calls // 10, 50)` counts every call ever made and
never decays; the unprotected run ends failing 74% of calls. Give the breaker
a per-request clock so it *can* half-open, and at 1000 requests it still lands
204 successes against 241 unprotected. Make load a recent rate instead — calls
made over the last 20 requests — and the storm is real: 3870 calls and 71
successes unprotected, 176 with the breaker, 2.5x. Over the last 5 requests
there is no storm and the breaker costs 994 → 843. The tradeoff the exercise
asks about only exists when load can fall.

**The storm is 1.64x, not 10x.** 328 calls for 200 requests, and with at most
4 attempts per request the ceiling is 4x. Build It also names
`FailureTaxonomy`, `RetryStormSimulator` and `DetectionAgent`; only
`CircuitBreaker` exists.

### 2 — monoculture drift raises agreement, so a drop alarm never fires

**The alarm as specified never fires.** Three agents each right with
p = 0.7 and otherwise on one of 4 wrong answers; drift c is the probability
they emit one shared draw:

| drift c | pairwise agreement | per-agent accuracy | majority accuracy |
|---:|---:|---:|---:|
| 0 | 0.5125 | 0.7 | 0.784 |
| 1 | 1.0 | 0.7 | 0.7 |

Monoculture *raises* agreement while the vote gets worse. Over a seeded
100-window timeline the drop alarm (0.1 below the first window) fires 0 times;
the same threshold as a rise alarm fires from window 12. The lesson's own
`detect_groupthink` agrees, keying on `agreement_rate_spike`.

**Agreement alone cannot tell monoculture from easier questions.**
Independent agents at p = 0.9 agree 0.8125 of the time with majority accuracy
0.972; drift c = 8/13 agrees 0.8125 with 0.7323. Only labelled canaries — the
golden datasets the lesson recommends — separate them.

**A sharp drop does catch something: one agent diverging.** One agent at 0.3
accuracy takes agreement from 0.5125 to 0.346.

And the reference categorizer files the monoculture and retry-storm demo
incidents as MAST "unknown" — 2 of its 5 — and returns only the first match,
so a role-conflict + state-drift + no-verifier incident is Specification
alone.

### 3 — MAST's numbers are from a different version of the paper than its trace count

*Draws on "MAST categories".*

The lesson quotes **1642 traces** and **41.77 / 36.94 / 21.30%**. Those come
from two versions of the paper. v2 reports the three percentages over "over
200 execution traces". The current version reports 1642 annotated traces, and
its 14 failure-mode percentages sum to **44.2 / 32.35 / 23.5%** by category.
The categories are *System Design Issues*, *Inter-Agent Misalignment* and
*Task Verification*. The lesson renames them "Specification Problems",
"Coordination Failures" and "Verification Gaps", and hardcodes the old
percentages into `MAST_CATEGORIES`.

**HyperAgent, top three.** §5.1's Figure 4 gives the per-system distribution
over the first 30 traces of each system. It is an image, so the bar heights
could not be read. The text names HyperAgent's two dominant modes: **step
repetition (FM-1.3)** and **incorrect verification (FM-3.3)**. Taking the
third from the paper's overall ranking gives **reasoning–action mismatch
(FM-2.6, 13.2%)**, so the three span all three categories.

**How that compares with MAST.** MAST is a taxonomy, not a predictor. What it
offers for comparison is the aggregate ranking, which is FM-1.3 (15.7%),
FM-2.6 (13.2%) and FM-1.5 unaware of termination (12.4%). HyperAgent matches
on step repetition. Its distinctive failure is incorrect verification: the
system checks its work and gets the check wrong, rather than skipping it.
AppWorld's is premature termination (FM-3.1), and OpenManus shares
HyperAgent's step repetition. The practical point is that the per-system
profile differs enough from the aggregate that an audit has to be run per
system. The lesson's categorizer could not run it anyway: none of its eight
symptom keys detects step repetition or termination awareness, which are two
of the three largest modes.

### 4 — conformity leaves the voted answer unchanged and erases the dissent

**Conformity bias is the hardest to detect, and the proxy is the agreement
gained from exposure.** Enumerate all 125 joint answers of three agents
(p = 0.7) and apply the rule "adopt the answer your two peers share":

| | before exposure | after |
|---|---:|---:|
| majority accuracy | 0.784 | **0.784** |
| unanimity | 0.3447 | 0.8481 |
| unanimous and wrong | 0.0017 | 0.0641 |

The voted answer is exactly what it was, so no accuracy dashboard can see the
change. What conformity destroys is the dissent: wrong answers that nothing
downstream would question become 38x more common. The proxy is to log each
agent's answer *before* it sees the others, and track post-exposure minus
pre-exposure agreement. That gain is +0.3356 under conformity and exactly 0
for healthy agents, monoculture and easy questions.

The final agreement rate cannot separate those cases. Monoculture drift at
c = 0.6884 and independent agents at p = 0.9201 both end at conformity's
0.8481. The triple (pre-exposure agreement, exposure gain, canary accuracy)
does separate them.

The paper itself does not match the lesson's framing. arXiv:2508.05687 is
"Risk Analysis Techniques for Governed LLM-based Multi-Agent Systems" (Reid
et al.). It names **six** failure modes, the sixth being inter-agent
communication failures, and it never uses the word "groupthink" or ranks the
modes by how hard they are to detect. It does describe conformity as able to
pass for healthy consensus ("dangerous false consensus", §3.4). The
measurement above shows that literally: consensus goes up and the vote stays
the same.

### 5 — a validator that watches the symptom approves switching the service off

The system is the lesson's own order → payment pipeline, so the trio runs
rather than being described.

- **Detection** watches calls per request, against a budget of 1.5. It fires
  at 1.64.
- **Diagnosis** gets `cascade` from the Groupthink table and "unknown" from
  MAST, then recommends a circuit breaker.
- **Validation** is built twice. The symptom validator passes the breaker,
  because amplification falls to 0.70. A no-regression validator rejects it,
  because success falls from 0.945 to 0.565.

Validation has to check the objective, for two reasons:

- **The symptom validator's favourite fix is turning traffic off.** It ranks
  threshold 0.05 best, at 26 calls serving 25 of 200 requests, and it passes
  every threshold below 0.9.
- **Only the objective validator can tell a fix from a shutdown.** With load
  over the last 20 requests the breaker lifts successes from 71 to 176, and
  the objective validator accepts it. On the shipped load it rejects the
  breaker. The symptom validator accepts both.

STRATUS itself (arXiv:2506.02009) is a cloud SRE system. Its abstract names
detection, diagnosis and *mitigation* agents plus a *transactional
no-regression* safety property, and its "at least 1.5 times" figure is
failure-mitigation success against state-of-the-art SRE agents on AIOpsLab
and ITBench. The lesson's "validation agent" is not in that list. The closest
thing, the no-regression property, is the objective check this exercise ends
up needing.
