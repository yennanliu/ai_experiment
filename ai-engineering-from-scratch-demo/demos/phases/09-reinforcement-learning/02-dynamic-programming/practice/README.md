<!-- generated:start -->
# 09-reinforcement-learning / 02-dynamic-programming

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/09-reinforcement-learning/02-dynamic-programming/) · upstream spec
`phases/09-reinforcement-learning/02-dynamic-programming/docs/en.md`

```bash
uv run demo practice run 02-dynamic-programming --ex 1
uv run demo explain 02-dynamic-programming --ex 1
uv run pytest demos/phases/09-reinforcement-learning/02-dynamic-programming
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run value iteration on the 4×4 GridWorld with `γ ∈ {0.9, 0.99}`. How many sweeps until… | code | T0 | `ex01_ten_times_the_horizon_buys_one_sweep.py` |
| 2 | Medium. Compare policy iteration vs value iteration on the *stochastic* GridWorld (slip proba… | code | T0 | `ex02_the_first_three_evaluations_are_the_whole_bill.py` |
| 3 | Hard. Build modified policy iteration: in the evaluation step, run only `k` sweeps instead of… | code | T0 | `ex03_the_curve_falls_and_the_cost_curve_inverts_it.py` |
<!-- generated:end -->

## Answers

The lesson ships two algorithms for the same fixed point and asks three questions
about the cost of reaching it. All three answers turn on the same quantity, and it
is not the discount: it is whether the policy being evaluated can reach the
terminal at all. `γ` sets the ceiling on the values; **properness** sets the price.

A note that shapes everything below. `code/main.py` hard-codes `SLIP = 0.1` inside
`transitions`, so the only board it ships is the stochastic one — and Exercise 2
introduces that board as though Exercise 1 had not been on it. Exercise 1 runs both
arms, setting only the module constant.

All three are **T0** — the lesson imports nothing at all, and neither does this pack.

### 1 — ten times the horizon buys one extra sweep

The four `V*` grids the exercise asks to print, all from the lesson's own
`value_iteration` at `max |ΔV| < 1e-6`:

```text
SLIP = 0.0, γ = 0.9   — 7 sweeps        SLIP = 0.0, γ = 0.99  — 7 sweeps
 -4.686 -4.095 -3.439 -2.710             -5.852 -4.901 -3.940 -2.970
 -4.095 -3.439 -2.710 -1.900             -4.901 -3.940 -2.970 -1.990
 -3.439 -2.710 -1.900 -1.000             -3.940 -2.970 -1.990 -1.000
 -2.710 -1.900 -1.000  0.000             -2.970 -1.990 -1.000  0.000

SLIP = 0.1, γ = 0.9   — 14 sweeps       SLIP = 0.1, γ = 0.99  — 15 sweeps
 -4.989 -4.403 -3.748 -3.056             -6.428 -5.433 -4.428 -3.461
 -4.403 -3.745 -2.978 -2.160             -5.433 -4.423 -3.353 -2.324
 -3.748 -2.978 -2.115 -1.147             -4.428 -3.353 -2.266 -1.170
 -3.056 -2.160 -1.147  0.000             -3.461 -2.324 -1.170  0.000
```

**ANSWER: 7 and 7 sweeps deterministic, 14 and 15 stochastic.**

**FINDING: the count is almost independent of the discount.** `(1-γ)` falls by
10× and the answer moves by one sweep on the stochastic board and by none on the
deterministic one. The exercise's question — *how many sweeps?* — has the flattest
answer of anything on the page.

**FINDING: the lesson's own bound holds, and is conservative by two orders of
magnitude.** The Concept section states the guarantee: the Bellman optimality
operator is a `γ`-contraction in the sup-norm, so convergence is geometric at
modulus `γ`. That is correct, and it is tight in the worst case over all MDPs —
it is just nowhere near tight on this board. Taken at its word:

| γ | `log(ε(1-γ)) / log γ` | measured | high by |
|---|---:|---:|---:|
| 0.9 | 153 | 14 | 11× |
| 0.99 | 1833 | 15 | **122×** |

**MECHANISM: the observed rate is `γρ`, and `ρ` belongs to the board, not the
discount.** Once the greedy policy stops changing, a sweep is no longer a `max`
over actions — it is linear iteration with `γP_π`. So the *asymptotic* rate is `γ`
times the spectral radius of that policy's transition matrix *restricted to the
transient states*. This is a local rate for this board after policy stabilization;
it does not replace `γ` as the operator's sup-norm modulus, it explains why the
`γ` bound is loose here:

| | γ = 0.9 | γ = 0.99 |
|---|---:|---:|
| `ρ` (optimal policy, transient block) | 0.3873 | **0.3873** |
| `γρ` | 0.349 | 0.383 |
| measured per-sweep ratio, in place | 0.137 | 0.162 |
| `γ`, which the bound uses | 0.9 | 0.99 |

`ρ` is *identical at the two discounts*, because the greedy policy is. That is the
whole answer: `γ` enters the local rate only as a multiplier on a number near 0.39,
so it cannot move the sweep count much. Diameter + `log(ε)/log(rate)` predicts 13
and 14 against the measured 14 and 15.

On the deterministic board `ρ` is **exactly 0** — the transient block is nilpotent,
not merely small — so value iteration is exact after the board's 6-step diameter,
and 7 is that diameter plus the sweep that observes `delta < tol`. No discount can
change it.

**FINDING: the grid moves more than the count does.** `V*(0,0)` shifts 1.44
between the two discounts while the sweep count shifts by 1. And the *smaller* `γ`
reports the better-looking number — −4.99 against −6.43 — for the same policy on
the same board, which is the trap in reading a discounted value as a cost.

### 2 — the first three evaluations are the whole bill

"Count sweeps" is read strictly: a sweep is one pass over the 15 non-terminal
states, wherever it happens. That reading is forced, because the lesson's
`policy_iteration` returns `it + 1` and carries a local variable *named* `sweeps`
that also counts outer iterations — so the one number a learner would reach for to
answer this exercise is not the one it asks for. Bellman backups are reported
beside wall-clock, since wall-clock is not reproducible and backups are.

At `γ = 0.99`, on the stochastic board:

| | sweeps | Bellman backups | wall-clock | `V*(0,0)` |
|---|---:|---:|---:|---:|
| value iteration | **15** | **960** | **~1 ms** | −6.4283 |
| policy iteration | 3,728 | 56,220 | ~60 ms | −6.4283 |
| ratio | 249× | 58.6× | ~60× | — |

**ANSWER: value iteration, on every honest unit.** The exercise asks which wins
*in iterations* and which *in wall-clock* as though the two could come apart. Here
they do not, and they agree by a factor of about 60.

**FINDING: the only unit policy iteration wins is the one that is not a sweep.**
5 outer iterations against 15 sweeps reads as a 3× win. It is a unit error: each
of those 5 contains a full evaluation to `tol`.

**FINDING: 99.2% of the bill is three evaluations that are then thrown away.**

| outer iteration | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| inner sweeps (γ=0.99) | 1329 | 1313 | 1056 | 15 | 15 |
| `ρ` of the policy being evaluated | **1.000** | **1.000** | 0.9975 | 0.3873 | 0.3873 |

The first three policies — all-`up` and its first two successors — never reach the
terminal. Their transient spectral radius is 1, so their evaluation contracts at
exactly `γ`, and each one costs more than the entire value-iteration run. Every one
of the three is discarded by the improvement step immediately after it.

**FINDING: the gap widens with `γ`, because that waste is the only `γ`-sensitive
part of either algorithm.**

| γ | PI backups | VI backups | ratio |
|---|---:|---:|---:|
| 0.9 | 6,435 | 900 | 7.2× |
| 0.99 | 56,220 | 960 | **58.6×** |

Value iteration itself goes 14 sweeps to 15 across that range. Policy iteration
goes 409 to 3,728.

**FINDING: the two agree with each other 10× better than either agrees with
`V*`.** Sup-norm `|V_PI − V_VI| = 6.26e-09`; each sits `6.09e-08` from the fixed
point the same operator reaches at `tol = 1e-14`, in the same direction, because
one stopping rule stops them both short. That fixed point costs 25 sweeps against
15 — ten more sweeps for eight orders of magnitude. The guarantee
`ε·γ/(1-γ) = 9.9e-05` holds with 1,600× to spare. "They agree, so both are right"
is the one inference this measurement does not support.

### 3 — the curve falls, and the cost curve inverts it

Modified policy iteration, `k` evaluation sweeps per improvement step, warm-started
from the current `V`. Error is measured against the fixed point the same operator
reaches at `tol = 1e-14`; the plot ships as the table `audit_practice` asks for.

| `k` | error at stop | total sweeps | sweeps to reach 1e-6 | error at 16 sweeps |
|---:|---:|---:|---:|---:|
| 1 | 8.50e-08 | **16** | **15** | 8.50e-08 |
| 2 | 1.29e-08 | 22 | 20 | 1.02e-03 |
| 5 | 2.19e-08 | 35 | 33 | 8.90 |
| 10 | 2.24e-12 | 60 | 56 | 8.74 |
| 50 | **0** | 300 | 209 | 8.60 |

**ANSWER: the curve falls, off the bottom of the scale — and not monotonically.**
`k = 50` lands on the reference fixed point exactly; `k = 2` beats `k = 5`. The
plot the exercise asks for is not a curve a tradeoff can be read off.

**FINDING: the fall is overshoot, not accuracy bought.** Modified policy iteration
converges for every `k ≥ 1`, so all five reach `V*` and all five clear the same
`max |ΔV| < 1e-6` test. What the curve plots is how far *past* that test the final
evaluation block happened to run — a property of the block size, not of the
evaluation/improvement balance the question is about.

**FINDING: priced per sweep, the ordering inverts.** Reaching 1e-6 and staying
there costs 15 sweeps at `k = 1` and 209 at `k = 50` — 14× more for an accuracy
`k = 1` already has. Cost to fixed accuracy is the tradeoff curve, and it runs the
other way to the one the exercise asks to plot.

**MECHANISM: a large block spends its budget on a policy that cannot terminate.**
Same cause as Exercise 2. The initial all-`up` policy never reaches the terminal,
so its evaluation contracts at exactly `γ` toward −100, and a 50-sweep block
chases it there before anything improves. At 16 sweeps — the point where `k = 1`
has already finished — `k = 5, 10, 50` are still 8.9, 8.7 and 8.6 away from `V*`.
The improvement step is what makes a policy proper, and large `k` delays it.

**FINDING: `k` is one dial between two algorithms the lesson already ships, and
the warm start is the dial.**

| | `k = 1` | `k = 50` |
|---|---:|---:|
| warm-started (MPI) | 16 sweeps | 300 sweeps |
| cold-started, as `policy_evaluation` starts | **never converges** (5.43 off after 4,000) | 300 sweeps |

`k = 1` is Step 5's value iteration — 16 sweeps against its 15, same fixed point.
`k → ∞` is Step 4's policy iteration, at 3,728. Cold-start each evaluation from
`V = 0`, the way the lesson's own `policy_evaluation` does, and the `k = 1` end
stops existing: one sweep from zero accumulates nothing, and the loop runs forever.
`k = 50` does not notice, because 50 sweeps nearly converges from scratch anyway.
The end of the dial that wins is the end that only exists warm — which is what
GPI means, and what the exercise builds without naming.
