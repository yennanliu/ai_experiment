<!-- generated:start -->
# 16-multi-agent-and-swarms / 19-swarm-optimization-pso-aco

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/19-swarm-optimization-pso-aco/) · upstream spec
`phases/16-multi-agent-and-swarms/19-swarm-optimization-pso-aco/docs/en.md`

```bash
uv run demo practice run 19-swarm-optimization-pso-aco --ex 1
uv run demo explain 19-swarm-optimization-pso-aco --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/19-swarm-optimization-pso-aco
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Observe LMPSO convergence. Vary population size 5, 10, 20, 50. At what si… | code | T0 | `ex01_iterations_saturate_at_ten_and_the_peak_they_converge_on_is_not_the_one_named.py` |
| 2 | Implement a "catastrophic drift" experiment: after iteration 30, change the fitness function.… | code | T0 | `ex02_the_swarm_reports_the_old_peak_forever_and_resetting_p_best_rescues_three_in_forty.py` |
| 3 | Add a quality gate to AMRO-S: pheromone deposit only on runs with eval score > 0.7. How does… | code | T0 | `ex03_the_higher_gate_blocks_the_wrong_agent_in_one_row_and_the_right_agent_in_another.py` |
| 4 | Read LMPSO (arXiv:2504.09247). Map the paper's "velocity as a prompt" back to your numeric ve… | explain | T0 | prose, below |
| 5 | Read AMRO-S (arXiv:2603.12933). Implement the decoupled "inference fast-path" with asynchrono… | code | T0 | `ex05_the_fast_path_saves_the_judge_not_the_pheromones_and_pays_in_staleness.py` |
<!-- generated:end -->

## Answers

### 1 — iterations saturate at ten, and the peak they converge on is not the one named

**Iterations saturate at about 10 particles, and in evaluations 5–10 is the
cheapest range.** "Converged" means g_best within 1e-3 of the true maximum of
`fitness`, and the results are means over 40 seeds:

| particles | iterations | evaluations |
|---:|---:|---:|
| 5 | 14.2 | 76 |
| 10 | 7.3 | 83 |
| 20 | 5.7 | 134 |
| 50 | 3.9 | 242 |

Going from 5 to 10 particles saves 6.9 iterations. The next doubling saves
1.6, and the 2.5x step after that saves 1.9. In LMPSO every evaluation is an
LLM call, so above 10 particles each extra particle costs more calls than the
iterations it saves. The demo's own 20 particles converge at iteration 6 of
its 30.

**The printed optimum is a local minimum.** The demo prints "optimum = 1.0000
at (0.72, 0.40)", but `fitness` there is **0.84**. The ripple term subtracts
0.16 at its own centre, and the curvature along each axis is
−12 + 0.08·64π² = **+38.5**, so the named point is a minimum in both
directions. The four real maxima are at (0.72 ± 0.0997, 0.40 ± 0.0997) with
value 1.00945. That is why the demo's "final g_best" prints above the
optimum it names. All 160 runs end at 1.00945.

### 2 — the swarm reports the old peak forever, and resetting p_best rescues 3 in 40

The experiment moves the landscape's centre from (0.72, 0.40) to (0.30, 0.75)
at iteration 30 of 60. "Adapted" means g_best's **true** fitness under the
new function comes within 1e-3 of its maximum. Results for 20 particles over
40 seeds:

| response at the drift | adapted of 40 |
|---|---:|
| none | 0 |
| re-evaluate every `p_best` | 3 |
| re-evaluate + fresh velocities | 14 |
| restart (fresh positions) | 40, 6.2 iterations after the drift |

The reference cannot run the variants itself: `run_lmpso` hardcodes
`fitness` and returns only its history. A port that reproduces its history
exactly before the drift runs them instead.

**The reported fitness never drops.** With `fitness` swapped in the middle of
the reference run, its history reads 1.00945 from iteration 30 to 60 while
g_best's true fitness is **0.0**. A particle only replaces `p_best` when a
new evaluation beats the *stored* value. The new function's maximum equals
the old stored value, so the strict `>` can never fire, and the comparison
is against a function that no longer exists.

**Resetting `p_best` fails because the swarm sits on a plateau.** `fitness`
is clamped at 0, and 58% of the unit square scores exactly 0 under the moved
function, the old peak included. A converged swarm has collapsed onto that
point with near-zero velocities. Re-evaluated, all its memory is zeros, so
there is no gradient to follow. Resetting memory does not restore diversity,
and diversity is what adaptation needs.

### 3 — the higher gate blocks the wrong agent in one row and the right agent in another

The shipped router already gates at 0.6, so this compares three settings.
"Convergence" is the best agent's share of pheromone after 200 tasks, which
is the probability the router picks it. Means over 40 seeds:

| row | ungated | 0.6 | 0.7 |
|---|---:|---:|---:|
| code | 0.80 | 0.95 | 0.95 |
| math | 0.73 | 0.93 | 0.95 |
| writing | 0.84 | 0.95 | 0.95 |
| planning | 0.49 | 0.74 | **0.62** |
| routed quality | 0.617 | 0.670 | 0.668 |

Any gate converges faster than none, because ungated every run of every
agent deposits pheromone. Raising the gate to 0.7 helps math: the coder's
math range is 0.35–0.65, so its occasional deposits stop. It hurts planning:
the best agent there, the writer, has a range of 0.45–0.75, so 5 in 6 of its
deposits stop.

**A gate only acts on a row where some agent's quality range straddles it.**
`simulate_task` is a fixed affinity plus or minus 0.15. On code and writing
the best agent always clears 0.7 and nobody else reaches 0.6, so those
shares are *identical* at the two gates. Above 0.75 no planning run can
deposit at all, and that row stays at exactly 1/3 each.

**Evaporation without a deposit does not change routing.** `deposit` scales
the whole row by 0.95 whether or not the gate passes, and `choose`
normalises. So a gated-out run changes no choice probability at all, and
"decays over time" forgets nothing. The lesson's "fast-but-wrong agents" do
not exist in the simulation, since it has no speed.

### 4 — the velocity became a sentence; the coefficients went with it

*Draws on "PSO on LLM outputs — LMPSO".*

In the numeric version, velocity is `w·v + c1·r1·(p_best − x) + c2·r2·(g_best − x)`.
LMPSO's meta-prompt (§III-B) keeps the same three roles as text:

- **Inertia:** "by indicating how the current position was generated (i.e.,
  its velocity), we capture the concept of inertia".
- **Current position:** included as reference material.
- **Direction:** incorporates the personal best and the global best.

The prompt also opens with a problem description. The LLM then writes the
next position instead of adding a vector to it.

**Preserved:** the population, the per-particle and swarm-wide memory, and
the three-way split of what pulls a particle. The evaluation budget has the
same shape too: §IV uses 10 particles × 100 iterations for TSP and 25 × 40
for heuristic improvement, 1,000 evaluations each. On exercise 1's numbers,
that is roughly what 10–20 particles need here.

**Lost in the simulation:** everything that makes the paper necessary. LMPSO
exists for *structured* solutions, like tours, expressions and programs,
where `p_best − x` has no meaning. The demo optimises a continuous 2-D box
where it does, so it shows the one case plain PSO already handles.

**Lost in the paper:** the coefficients. There is no w, c1 or c2 in the
prompt, and the randomness comes from sampling at temperature 0.9 from
Llama-3.1-8B-Instruct rather than from r1 and r2. You cannot write
"c1 = 1.2" in a sentence and have it scale anything. The demo's randomness
is also coarser than standard PSO: `run_lmpso` draws r1 and r2 once per
particle and uses them for both dimensions. §V names the limit: direct LLM
optimisation of "large-scale solution representations proved difficult".

### 5 — the fast path saves the judge, not the pheromones, and pays in staleness

AMRO-S's online update (§III-D) runs only on samples that pass "a lightweight
LLM-Judge that outputs a binary gate". It deposits `w_t(q)·Q/(f_sys(P)+ε)`,
where `f_sys` measures latency and token cost. Serving records requests at a
sampling rate `r` into a FIFO buffer and updates asynchronously once the
buffer reaches size B. The model used here: one inference costs 1.0 time
unit and one judge call costs 0.3, on a single server with seeded Poisson
arrivals.

| load | sync mean | async mean |
|---:|---:|---:|
| 0.5 | 2.46 | 1.49 |
| 0.7 | 8.07 | 2.11 |
| 0.75 | 34.4 | 2.41 |
| 0.8 | 530, still rising | 2.89 |

The fast path takes the judge out of the service time, so the gain grows
without bound as load approaches sync capacity (1/1.3 = 0.77). p99 at 0.7
drops from 28.3 to 6.8.

**In the lesson's own simulation the fast path would change nothing.**
`simulate_task` returns the quality score together with the output, and
`deposit` is one `*=` loop over 3 agents plus one `+=`. With the judge free,
sync and async latency are identical. What AMRO-S decouples is the
*evaluation*, and the reference has no evaluation step to decouple.

**The buffer costs routing quality.** Routing on pheromones up to B requests
stale gives 0.670, 0.667, 0.643, 0.614 and 0.578 at B = 1, 8, 32, 64 and 100.
Never flushing gives 0.502, which is the random baseline. B = 64 gives up
about a third of what the router learns over random. The paper states no
value for B or for the gate. Table V reports wall-clock runtime (3849.60s
against 823.21s at 1000 processes) but no per-request latency percentiles,
so this trade-off is not quantified anywhere in the paper.

Its deposit rule is also not the lesson's. AMRO-S reinforces *cheap* paths
among accepted ones, since the deposit is inversely proportional to cost.
The lesson reinforces by quality score. A quality-weighted deposit is a
different optimiser.
