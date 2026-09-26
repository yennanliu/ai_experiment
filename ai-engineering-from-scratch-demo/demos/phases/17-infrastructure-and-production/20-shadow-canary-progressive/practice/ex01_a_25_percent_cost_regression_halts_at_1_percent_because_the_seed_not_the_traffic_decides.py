"""Exercise 1 — a 25% cost regression halts at 1%, because the seed, not the traffic, decides.

    Run `code/main.py`. Inject a 25% cost regression. At which stage does the
    canary halt?

Reading of the exercise: the injection is `Regression(cost_mult=1.25)`, the
same one the shipped demo runs, driven through the reference's own
`measure_stage` / `check_gates` with its own stage seeds. "At which stage" is
then asked a second time over many seeds, because the answer is only worth
something if it does not hinge on one draw.

**ANSWER: it halts at the first stage, 1%, on the cost gate alone.** The 1%
stage measures $0.0252 per request against a gate of 1.2 x $0.02 = $0.0240;
the other four gates pass.

**FINDING: the halt stage is set by the noise draw, not by the traffic
share.** `measure_stage(stage, ...)` never reads `stage`: the 1% and 100%
stages return identical metrics for the same seed, so a 1% canary is as
precise as full traffic -- the opposite of the lesson's "Metrics cadence"
section. The noise is +-8% uniform, so 1.25 breaches whenever the draw
exceeds 0.96, i.e. 75% of the time per stage. Over 10,000 seeded rollouts it
halts at 1% in 75.3%, at 10% in 18.7%, and promotes to 100% in 4.

**FINDING: the 20% gate is really a band from +11.1% to +30.4%.** On the
shipped seeds a 12% regression -- inside the gate -- halts at 75%; every cost
multiplier from 1.1197 up halts, and 1.1887 and up halts at 1%. Only 1.2 /
0.92 = 1.304 is caught on every seed, and nothing below 1.2 / 1.08 = 1.111
is caught on any. The demo's "Small cost regression (10%) -- within gate" is
within it by 2 points of noise.

**FINDING: the lesson states two different progressions.** Its summary says
"10% -> 25% -> 50% -> 75% -> 100%"; its body, its numbers list and `STAGES`
start at 1%. The 1% stage is the one that catches this regression.

Structure: `halt()` replays the shipped rollout loop without printing and
takes the seed schedule as a parameter; `seed_sweep()` reruns it on 10,000
seed schedules.
"""

from __future__ import annotations

import collections

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "20-shadow-canary-progressive"
ROLLOUTS = 10_000


def load_ref():
    return parity.load_reference(PHASE, LESSON, "main")


def halt(ref, reg, seed_of=None):
    """(stage index, breaches, metrics) of the first breach, or None if promoted."""
    seed_of = seed_of or ref.stage_seed
    for i, stage in enumerate(ref.STAGES):
        metrics = ref.measure_stage(stage, reg, seed=seed_of(i))
        breaches = ref.check_gates(metrics)
        if breaches:
            return i, breaches, metrics
    return None


def seed_sweep(ref, reg):
    counts = collections.Counter()
    for r in range(ROLLOUTS):
        hit = halt(ref, reg, lambda i, r=r: 1_000 * r + i)
        counts[hit[0] if hit else "promoted"] += 1
    return counts


def lowest_halting(ref, first_stage_only=False):
    """Smallest cost multiplier, in steps of 1e-4, that halts on the shipped seeds."""
    for step in range(10_000, 14_000):
        hit = halt(ref, ref.Regression(cost_mult=step / 10_000))
        if hit and (not first_stage_only or hit[0] == 0):
            return step / 10_000
    return None


def solve():
    ref = load_ref()
    reg, doc = ref.Regression(cost_mult=1.25), parity.doc_text(PHASE, LESSON)
    index, breaches, metrics = halt(ref, reg)
    gate = ref.GATES["cost_per_req"]
    limit = round(ref.BASELINE["cost_per_req"] * gate, 4)
    sweep = seed_sweep(ref, reg)
    twelve = halt(ref, ref.Regression(cost_mult=1.12))
    return {
        "halt": (ref.STAGES[index], breaches, round(metrics["cost_per_req"], 4), limit),
        "blind": ref.measure_stage(0.01, reg, 11) == ref.measure_stage(1.0, reg, 11),
        "sweep": (sweep[0], sweep[1], sweep["promoted"]),
        "twelve": ref.STAGES[twelve[0]] if twelve else None,
        "ten_passes": halt(ref, ref.Regression(cost_mult=1.10)) is None,
        "lowest": (lowest_halting(ref), lowest_halting(ref, True)),
        "band": (round(gate / 1.08, 4), round(gate / 0.92, 4)),
        "starts": ("shift 10% → 25%" in doc, "1% → 10% → 25%" in doc, ref.STAGES[0]),
    }


def verify(result):
    r = result
    stage, breaches, cost, limit = r["halt"]
    return [
        practice.Check(
            "ANSWER: it halts at the first stage, 1%, on the cost gate alone",
            r["halt"] == (0.01, ["cost_per_req"], 0.0252, 0.024),
            f"stage {stage:.0%}: ${cost} per request against the ${limit} gate; "
            f"breaches {breaches}",
        ),
        practice.Check(
            "FINDING: the halt stage is set by the noise draw, not the traffic share",
            r["blind"] and r["sweep"] == (7526, 1868, 4),
            f"measure_stage returns identical metrics at 1% and 100%; over {ROLLOUTS} "
            f"seeded rollouts (halt at 1%, halt at 10%, promoted) = {r['sweep']}",
        ),
        practice.Check(
            "FINDING: the 20% gate is really a band from +11.1% to +30.4%",
            (r["twelve"], r["ten_passes"], r["lowest"], r["band"])
            == (0.75, True, (1.1197, 1.1887), (1.1111, 1.3043)),
            f"on the shipped seeds +12% halts at {r['twelve']:.0%} and +10% promotes; "
            f"lowest halting multiplier (any stage, at 1%) {r['lowest']}; "
            f"noise band {r['band']}",
        ),
        practice.Check(
            "FINDING: the lesson states two different progressions",
            r["starts"] == (True, True, 0.01),
            "the summary starts at 10%; the body and STAGES start at 1%, the stage "
            "that catches this regression",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
