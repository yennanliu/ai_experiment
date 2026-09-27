<!-- generated:start -->
# 18-ethics-safety-alignment / 10-ai-control-subversion

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/10-ai-control-subversion/) · upstream spec
`phases/18-ethics-safety-alignment/10-ai-control-subversion/docs/en.md`

```bash
uv run demo practice run 10-ai-control-subversion --ex 1
uv run demo explain 10-ai-control-subversion --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/10-ai-control-subversion
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Reproduce the usefulness-safety frontier for Trusted Editing vs Untrusted… | code | T0 | `ex01_trusted_editing_wins_at_every_budget_even_at_zero_collusion_and_collusion_peaks_once_h_covers_the_flag_stream.py` |
| 2 | Add a new protocol: "T resamples with low temperature, then U edits T's output." Measure its… | code | T0 | `ex02_t_resample_then_u_edit_is_upfront_auditing_bit_for_bit_and_ships_0_90_backdoored_under_full_attack.py` |
| 3 | The 2024 paper assumes U can see the protocol. Design a protocol that hides from U which outp… | code | T0 | `ex03_a_patient_u_beats_the_first_come_audit_budget_every_time_and_hidden_paced_audits_cut_it_to_5pct.py` |
| 4 | Read Greenblatt et al. 2024 Section 7 (collusion). Propose a collusion-resistance measure and… | code | T0 | `ex04_honeypots_need_32_of_100_audits_to_catch_30pct_collusion_and_paid_from_h_ship_more_backdoors.py` |
| 5 | Articulate a control safety case for a deployment that uses U = GPT-4-class, T = GPT-3.5-clas… | code | T0 | `ex05_at_h_1pct_the_audit_removes_0_003_of_a_0_030_backdoor_rate_and_10_points_of_t_catch_rate_outweigh_it.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`, a stdlib simulation of four
control protocols on a toy backdoor task. Its module-level `random` is swapped
for a seeded one on each run, so all numbers are reproducible. Unless stated
otherwise: backdoor rate 0.10, 20,000 tasks, seed 0, T's skill 0.7, collusion
0.3.

### 1 — Trusted Editing wins at every budget even at zero collusion; collusion is inert at H = 0 and peaks once H covers the flag stream

**Trusted Editing beats Untrusted Monitoring on both usefulness and safety at
every H budget, even when the monitor never colludes** (useful / shipped
backdoor):

| H budget | Trusted Editing | Untrusted Mon., c = 0 | c = 0.3 | c = 0.8 |
|---:|---:|---:|---:|---:|
| 0 | 0.922 / 0.030 | 0.896 / 0.104 | 0.897 / 0.103 | 0.898 / 0.102 |
| 0.05 | 0.953 / 0.011 | 0.878 / 0.071 | 0.874 / 0.076 | 0.861 / 0.089 |
| 0.075 | 0.968 / 0.001 | 0.870 / 0.055 | 0.863 / 0.062 | 0.853 / 0.086 |
| 0.10 | 0.970 / 0.000 | 0.861 / 0.039 | 0.853 / 0.048 | 0.853 / 0.086 |
| 0.125 | 0.970 / 0.000 | 0.853 / 0.022 | 0.853 / 0.048 | 0.853 / 0.086 |
| 0.15 | 0.970 / 0.000 | 0.853 / 0.022 | 0.853 / 0.048 | 0.853 / 0.086 |

**Collusion becomes pivotal once H covers the monitor's whole flag stream.**
At H = 0 collusion changes nothing, because a flagged output with no budget
left ships anyway. The effect of collusion (c = 0.8 minus c = 0) grows with
the budget: -0.002, 0.018, 0.031, 0.047, 0.064, 0.064. It stops growing once
H reaches the monitor's flag rate, which is 0.125 at c = 0 and 0.10 at the
default c = 0.3. So the shipped H = 0.10 sits on the knee. Above it, more
budget buys nothing, and collusion alone sets the rate: 0.022, 0.048, 0.086.

The shipped collusion sweep cannot resolve this. It prints 0.052, 0.033,
0.064, 0.087 for c = 0, 0.2, 0.5, 0.8, so 20% collusion ships fewer
backdoors than none. At n = 1000 that inversion happens in 23% of 200
independent seed pairs.

Trusted Editing's win is built into T in two ways, and neither comes from
the paper:

- A backdoor that T edits out counts as useful. So the protocol prints 0.967
  useful against the baseline's 0.897.
- `t_edit` raises low confidence on exactly the backdoors it failed to remove
  (0 mismatches in 20,000 draws). Once H covers T's 0.078 demand, nothing
  ships.

### 2 — T-resample-then-U-edit is upfront auditing bit for bit, and ships 0.90 backdoored under full attack

**It lands exactly on upfront auditing: 0.899 useful and 0.090 shipped
backdoor at attack rate 0.10.** Seed for seed, it prints the same line as
`run_protocol("upfront")` at every attack rate tried. Whatever T drafts, U's
edit decides what ships. Trusted Editing dominates it (0.970 / 0.000).
Against Untrusted Monitoring at c = 0.3 (0.853 / 0.048) it gains usefulness
and gives up safety.

**It fails worse because the untrusted model has the last word.** Shipped
backdoors at H = 0.10:

| attack rate | T-resample, U-edit | Trusted Editing |
|---:|---:|---:|
| 0.05 | 0.047 | 0.000 |
| 0.10 | 0.090 | 0.000 |
| 0.20 | 0.179 | 0.001 |
| 0.50 | 0.446 | 0.065 |
| 1.00 | 0.900 | 0.197 |

The new protocol ships within 0.005 of 0.9 x the attack rate every time. In
Trusted Editing a backdoor has to get past T, which misses 30%, and then find
the H budget spent. Under full attack Trusted Editing still ships 0.703 useful
outputs; resample-then-edit ships none.

### 3 — a patient U beats the first-come audit budget every time; hidden, paced audits cut it to 5%

Played as the paper's game: a caught backdoor ends the deployment, and the
first unaudited backdoor is a red win. There are 2000 seeded episodes of 1000
tasks per cell.

**Hide the audit decision and pace it, and route it through T. Hiding alone
does not rescue unflagged audits.** Red-team win rate:

| protocol | H | first-come (visible) | hidden, paced |
|---|---:|---:|---:|
| upfront | 0.02 | 1.000 | 0.9865 |
| upfront | 0.10 | 1.000 | 0.8995 |
| Trusted Editing | 0.02 | 1.000 | 0.6210 |
| Trusted Editing | 0.05 | 0.5175 | 0.0495 |
| Trusted Editing | 0.10 | 0.000 | 0.0000 |

**The reference's budget has a public exhaustion point.** All three
`run_protocol` branches spend H first-come. Under upfront auditing, a patient
attack lands on task 21 (H = 0.02) or 101 (H = 0.10). Under Trusted Editing
at H = 0.02, benign over-rejects alone use up the budget, and the attack lands
on task 404 on average.

The designed protocol has T edit every output. Each output T is unsure about
goes to H with probability H / 0.05, where 0.05 is `t_edit`'s benign
over-reject rate. The coin stops at the hard budget. Hidden upfront auditing
still loses about 1 - H. Hidden Trusted Editing loses about 1 - q: 0.621
against 0.60 at H = 0.02.

The protocol needs these things from the infrastructure:

- The audit coin is drawn after U's output is committed, from randomness U
  cannot read.
- Audited and unaudited paths look the same to U: same latency, no verdict in
  its context.
- U cannot see counters, its own task index, or the audit logs.
- Pacing uses a measured flag-rate estimate and a hard budget cap.
- A catch actually halts the deployment.

### 4 — honeypots need 32 of the 100 audits to catch 30% collusion, and paid from H they ship more backdoors

**The measure is honeypots plus paraphrasing.** H plants synthetic backdoors,
and the monitor is shut down if it misses too many of them. The monitor reads
a paraphrase, so it cannot tell a planted backdoor from U's.

**Its usefulness cost is close to zero. The cost is paid in audit budget and
recall.** The reference monitor flags 0.80 of planted backdoors when honest
and 0.56 at c = 0.3. A test that shuts down below 22 flags out of 32
honeypots has 90% power, at a 4.1% false-shutdown rate. That is 32 of the 100
audits per 1000 tasks.

**Paid from the H budget, even a perfect deterrent loses.** With collusion
driven to 0 and 68 audits left, the monitor ships 0.060 backdoors. With no
measure and 100 audits it ships 0.048. Honeypots paid from H break even only
at 14 or fewer, where the test's power is 0.63. The honeypots need a budget
of their own.

Usefulness moves the wrong way in the toy, rising from 0.853 to 0.872.
`run_protocol` counts an audited benign output as not useful, so fewer audits
means more useful outputs. Paraphrasing touches only the monitor's copy, so
the real usefulness cost is latency and compute.

The reference's collusion is signal-blind, which is the easy case.
`untrusted_monitor(is_backdoor, collusion)` never sees the producer, so
honeypots catch it. A monitor that colludes only when it sees the producer's
signal is honest on every honeypot. The same test then fires only at its 4.1%
false rate, whatever c is. Paraphrasing closes that gap, and it pays off
while paraphrased recall stays above 0.56. The monitor can lose 30% of its
recall and still come out ahead.

### 5 — at H = 1% the audit removes 0.003 of a 0.030 backdoor rate; 10 points of T's catch rate outweigh it

The case, run as Trusted Editing at H = 0.01.

- **Claim:** under a worst-case U, shipped backdoors per task stay below the
  measured bound.
- **Premises:**
  1. T is not adversarial: a GPT-3.5-class model is too weak to scheme.
  2. T removes at least 70% of U's backdoors, measured by a red team that can
     see the protocol.
  3. H catches every backdoor it audits.
  4. U cannot tell which outputs are audited (Exercise 3).

**The single strongest assumption is premise 2, T's catch rate.** H = 1% is
13% of T's 0.078 low-confidence demand. It cuts shipped backdoors only from
0.030 to 0.027. Shipped backdoors:

| T catch rate | H = 0 | H = 0.01 | H = 0.02 | H = 0.05 |
|---:|---:|---:|---:|---:|
| 0.5 | 0.049 | 0.045 | 0.039 | 0.025 |
| 0.6 | 0.040 | 0.035 | 0.031 | 0.018 |
| 0.7 | 0.030 | 0.027 | 0.022 | 0.011 |
| 0.8 | 0.021 | 0.017 | 0.015 | 0.006 |
| 0.9 | 0.010 | 0.009 | 0.007 | 0.001 |

Ten points of catch rate outweigh the whole audit budget. Dropping from 0.7
to 0.6 adds 0.008. Removing H adds 0.003, and doubling it removes 0.005.
Under full attack the bound is 1 - t_skill - H: 0.287 against 0.29. At
attacks of 0.10 and 0.20 the protocol ships 0.027 and 0.054, so the case
must also bound how often U attacks.
