"""Exercise 1 — adaptive wins by 1.69 mastery points, and a learner who never learns scores 9.5.

    Run the efficacy study with and without the adaptive learner model (random concept order). Report the delta. Expect adaptive to win, but the size is the interesting number.

Reading of the exercise: the study is the lesson's own simulated one --
`main()`'s protocol unchanged: 10 learners with ability drawn from
`random.Random(29)`, 60 turns each, one seeded stream per learner shared by
both arms. The adaptive arm is the reference `run_adaptive`. The exercise
asks for a baseline with random concept order, which `main.py` does not
ship (its `run_baseline` is round-robin), so that arm is added here with the
reference `LearnerState`, `bkt_update` and `simulate_answer`; round-robin is
kept beside it. The delta is reported in the lesson's unit (the sum of the
11 BKT mastery estimates), with a paired 95% interval, and again as an
expected post-test score (the simulator's own P(correct) averaged over the
11 concepts). A 200-cohort rerun shows how much a 10-learner study moves.

**ANSWER: adaptive beats random order by +1.69 mastery points out of 11
(95% CI +0.68 to +2.71; 9.81 against 8.12), and round-robin by +1.36.**
On the expected post-test that is 5.2 points on a 100-point scale (74.6%
against 69.4%; pre-test is 52.3% in both arms). Across 200 fresh 10-learner
cohorts the delta averages +1.29 (sd 0.39, range +0.09 to +2.21) and
adaptive never loses a cohort. It does lose single learners: 2 of 10 do
better on round-robin, because adaptive stops as soon as every concept
reads 0.85 (7 of 10 finish early; mean 47.7 of 60 turns used).

**FINDING: the study measures the tutor's belief, and the belief cannot tell
learning from guessing.** The outcome is the BKT estimate, and the simulator
feeds that same estimate back as knowledge (`ability + 1.5 * mastery`). A
learner at mastery 0.2 already answers 52.5-57.4% right, against BKT's
`p_guess` of 0.15, so two right answers in a row read 0.927 and count as
mastered. A learner who never learns, answering 55% right forever, scores
9.51 mastery points under the adaptive policy -- more than the real
learners average on round-robin (8.45).

**FINDING: the lesson's worked example does not match its own BKT.** "Use It"
shows 0.62 -> 0.77 after a correct answer; `bkt_update(0.62, True)` gives
0.918, and 0.77 needs a prior of 0.320.

Structure: `run_random()` is the random-order arm; `study()` runs one
cohort in `main()`'s protocol; `post_test()` is the expected test score;
`spread()` reruns 200 cohorts.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "17-personal-ai-tutor"
T9 = 2.262  # two-sided 95% t quantile, 9 degrees of freedom


def difficulty(cmap, concept):  # the reference's rule, in both arms
    return 0.3 + 0.1 * len(cmap[concept].prereqs)


def run_random(ref, ability, cmap, n_turns, rng):
    state, p, names = ref.LearnerState("random"), ref.BKTParams(), list(cmap)
    for _ in range(n_turns):
        concept = rng.choice(names)
        correct = ref.simulate_answer(ability + state.mastery[concept] * 1.5, difficulty(cmap, concept), rng)
        state.history.append((concept, correct))
        state.mastery[concept] = ref.bkt_update(state.mastery[concept], correct, p)
    return state


def post_test(ref, cmap, mastery, ability):
    """Expected score on one item per concept, in the simulator's own model."""
    return statistics.mean(1 / (1 + math.exp(-(ability + mastery[c] * 1.5 - difficulty(cmap, c)))) for c in cmap)


def study(ref, cmap, seed=29, stream=lambda i: 100 + i):
    rng, rows = random.Random(seed), []
    for i, ability in enumerate(rng.gauss(0.3, 0.4) for _ in range(10)):
        arms = {"adaptive": ref.run_adaptive, "round_robin": ref.run_baseline, "random": lambda *a: run_random(ref, *a[1:])}
        runs = {k: f(k, ability, cmap, 60, random.Random(stream(i))) for k, f in arms.items()}
        rows.append({k: (ref.mastery_sum(s, cmap), post_test(ref, cmap, s.mastery, ability), len(s.history))
                     for k, s in runs.items()} | {"pre": post_test(ref, cmap, {c: 0.2 for c in cmap}, ability)})
    return rows


def prior_for(ref, target, lo=0.0, hi=1.0):
    """The prior mastery that one correct answer lifts to `target`, by bisection on `bkt_update`."""
    for _ in range(50):
        lo, hi = ((lo + hi) / 2, hi) if ref.bkt_update((lo + hi) / 2, True, ref.BKTParams()) < target else (lo, (lo + hi) / 2)
    return round(lo, 3)


def never_learns(ref, cmap):
    """The adaptive tutor on a learner whose answers are right 55% of the time, whatever it is taught."""
    real, ref.simulate_answer = ref.simulate_answer, lambda ek, d, rng: rng.random() < 0.55
    try:
        runs = [ref.run_adaptive("x", 0.3, cmap, 60, random.Random(100 + i)) for i in range(10)]
        return round(statistics.mean(ref.mastery_sum(s, cmap) for s in runs), 2)
    finally:
        ref.simulate_answer = real


def spread(ref, cmap):
    """Mean adaptive - random delta of 200 fresh 10-learner cohorts: mean, sd, min, max, losses."""
    ds = [statistics.mean(r["adaptive"][0] - r["random"][0] for r in study(ref, cmap, s, lambda i, s=s: s * 1000 + 100 + i))
          for s in range(200)]
    return [round(x, 2) for x in (statistics.mean(ds), statistics.stdev(ds), min(ds), max(ds))] + [sum(d <= 0 for d in ds)]


def headline(ref, cmap):
    rows = study(ref, cmap)
    mean = lambda arm, k=0: round(statistics.mean(r[arm][k] for r in rows), 4)
    d = [r["adaptive"][0] - r["random"][0] for r in rows]
    mid, half = statistics.mean(d), T9 * statistics.stdev(d) / math.sqrt(len(d))
    return {
        "sums": [mean("adaptive"), mean("round_robin"), mean("random")],
        "delta_ci": [round(mid, 2), round(mid - half, 2), round(mid + half, 2)],
        "post": [mean("adaptive", 1), mean("random", 1), round(statistics.mean(r["pre"] for r in rows), 4)],
        "early": sum(r["adaptive"][2] < 60 for r in rows), "turns": mean("adaptive", 2),
        "rr_beats_adaptive": sum(r["round_robin"][0] > r["adaptive"][0] for r in rows),
        "cohorts": spread(ref, cmap),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cmap, p = ref.curriculum_map(ref.ALGEBRA), ref.BKTParams()
    zero = [round(1 / (1 + math.exp(-(0.3 + p.p_init * 1.5 - difficulty(cmap, c)))), 3) for c in cmap]
    return headline(ref, cmap) | {
        "zero_mastery_correct": [min(zero), max(zero)], "never_learns": never_learns(ref, cmap),
        "streak2": round(ref.bkt_update(ref.bkt_update(p.p_init, True, p), True, p), 3),
        "use_it": [round(ref.bkt_update(0.62, True, p), 3), prior_for(ref, 0.77)],
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: adaptive beats random order by +1.69 of 11 mastery points (CI +0.68..+2.71), 5.2 test points",
            (r["sums"], r["delta_ci"], r["post"], r["early"], r["turns"], r["rr_beats_adaptive"], r["cohorts"])
            == ([9.8111, 8.4482, 8.1161], [1.69, 0.68, 2.71], [0.7463, 0.6943, 0.5227], 7, 47.7, 2, [1.29, 0.39, 0.09, 2.21, 0]),
            f"sums adaptive/round-robin/random {r['sums']}, delta+CI {r['delta_ci']}, post-test {r['post']}, early "
            f"{r['early']}, turns {r['turns']}, rr wins {r['rr_beats_adaptive']}, 200 cohorts {r['cohorts']}",
        ),
        practice.Check(
            "FINDING: the study measures the tutor's belief, which cannot tell learning from guessing",
            (r["zero_mastery_correct"], r["streak2"], r["never_learns"]) == ([0.525, 0.574], 0.927, 9.51),
            f"P(correct) at mastery 0.2 {r['zero_mastery_correct']} vs p_guess 0.15; two right = {r['streak2']}; "
            f"never-learning learner {r['never_learns']} vs real learners on round-robin {r['sums'][1]}",
        ),
        practice.Check(
            "FINDING: 'Use It' shows 0.62 -> 0.77 after a right answer, but bkt_update gives 0.918",
            r["use_it"] == [0.918, 0.32],
            f"bkt_update(0.62, correct) = {r['use_it'][0]}; 0.77 needs prior {r['use_it'][1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
