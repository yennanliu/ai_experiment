"""Exercise 3 — compare tagged traffic with the untagged control, because the gate aborts a natural incident too.

    Your burn-rate alert paused an experiment. How do you determine root cause
    — chaos or natural?

Reading of the exercise: the experiment is the lesson's own aborted one --
provider 429 fallback, 30% blast, 1.5% induced errors -- run over a seeded
5-minute window at an assumed 200 req/s (60,000 requests), in three worlds:
chaos only, a natural incident only (1% extra errors on all traffic), and
both. The lesson's tools are trace-ID tags and suppression windows; the
question is which reading of them separates the three worlds.

**ANSWER: split by trace-ID tag and compare the tagged blast radius with the
untagged rest.** Chaos shows as tagged traffic erring more than the control
(z = 23.77 chaos only, 13.21 both); a natural incident shows as the control
erring above baseline (z = 94.82 in both natural worlds, 0.23 without one)
while tagged and untagged agree (z = 0.29). Both tests read the same window,
so the cause needs no pause-and-watch.

**FINDING: the lesson's gate aborts a natural incident as if it were the
experiment.** `run_experiment` takes one error rate for the whole blast
radius. Fed the rate the tagged traffic actually shows with no chaos at all
(1.11%), it reads 22x burn at 30% blast and aborts; its output has no field
that could say why.

**FINDING: counting tagged errors does not answer it.** Tagged errors are
92.7% of all errors with chaos alone -- the baseline errors inside the blast
radius carry the tag too -- 30.6% with a natural incident alone (the blast
share of traffic), and 50.05% with both: a coin-flip. And the natural errors
that land in the blast radius are the ones a suppression window silences;
the untagged control is where that evidence survives.

**FINDING: the smallest experiment needs most of its own window to be
attributed.** Telling tagged from control at z >= 3 takes 48,394 requests for
the pod kill (0.2% against 0.05% at 5% blast): 4.0 of its 5 minutes at
200 req/s, and more than the whole run below 162 req/s. The provider 429 needs
984 requests and the malformed prompt 284 -- seconds.

Structure: `window()` draws the seeded requests; `attribute()` is the whole
diagnosis; `needed()` solves the same z-test for traffic.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "24-chaos-engineering-llm"
RPS, N, NATURAL, Z = 200, 60_000, 0.01, 3.0
WORLDS = {"chaos": (True, 0.0), "natural": (False, NATURAL), "both": (True, NATURAL)}


def z_test(e1, n1, e0, n0):
    pooled = (e1 + e0) / (n1 + n0)
    return (e1 / n1 - e0 / n0) / math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))


def window(exp, base, chaos, natural, seed=0):
    """(tagged n, tagged errors, control n, control errors) over one window."""
    rng, counts = random.Random(seed), [0, 0, 0, 0]
    for _ in range(N):
        tagged = rng.random() < exp.blast_radius_pct
        rate = base + natural + (exp.induced_error_rate if tagged and chaos else 0.0)
        failed = rng.random() < rate
        counts[0 if tagged else 2] += 1
        counts[1 if tagged else 3] += failed
    return counts


def attribute(counts, base):
    """Chaos if the tagged traffic errs above the control; natural if the control
    errs above baseline. Both tests read the same window."""
    n1, e1, n0, e0 = counts
    z_chaos = z_test(e1, n1, e0, n0)
    z_natural = (e0 - base * n0) / math.sqrt(base * (1 - base) * n0)
    verdict = [name for name, z in (("chaos", z_chaos), ("natural", z_natural)) if z >= Z]
    return verdict, round(z_chaos, 2), round(z_natural, 2), round(e1 / (e1 + e0), 4)


def needed(exp, base):
    """Requests for the expected tagged-vs-control gap to reach z >= 3."""
    b, p1 = exp.blast_radius_pct, exp.induced_error_rate
    pooled = b * p1 + (1 - b) * base
    z_one = (p1 - base) / math.sqrt(pooled * (1 - pooled) * (1 / b + 1 / (1 - b)))
    return math.ceil((Z / z_one) ** 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base, exps = ref.EXPECTED_ERROR_RATE, ref.EXPERIMENTS
    provider = exps[1]
    windows = {w: window(provider, base, *flags) for w, flags in WORLDS.items()}
    natural_rate = windows["natural"][1] / windows["natural"][0]
    gate = ref.run_experiment(ref.Experiment("natural only", 5, natural_rate, 0.30))
    return {
        "attributed": {w: attribute(c, base) for w, c in windows.items()},
        "natural_rate": natural_rate, "gate": (gate["status"], gate["burn_rate_x"]),
        "needed": {e.name: (needed(e, base), e.duration_min) for e in exps},
    }


def verify(result):
    att, need = result["attributed"], result["needed"]
    pod = need["pod kill (1 decode replica)"]
    return [
        practice.Check(
            "ANSWER: split by trace-ID tag and compare the tagged blast radius with "
            "the untagged rest",
            [att[w][0] for w in WORLDS] == [["chaos"], ["natural"], ["chaos", "natural"]],
            f"(verdict, z chaos, z natural, tagged share) {att}",
        ),
        practice.Check(
            "FINDING: the lesson's gate aborts a natural incident as if it were the experiment",
            result["gate"][0].startswith("ABORTED") and round(result["gate"][1]) == 22,
            f"no chaos, tagged error rate {result['natural_rate']:.4f}: gate {result['gate']}",
        ),
        practice.Check(
            "FINDING: counting tagged errors does not answer it",
            [att[w][3] for w in WORLDS] == [0.9272, 0.3064, 0.5005],
            "tagged share of errors: chaos only, natural only, both = "
            f"{[att[w][3] for w in WORLDS]}",
        ),
        practice.Check(
            "FINDING: the smallest experiment needs most of its own window to be attributed",
            pod[0] == 48394 and math.ceil(pod[0] / (60 * pod[1])) == 162
            and need["provider 429 fallback"][0] == 984,
            f"(requests needed, minutes run) {need}; the pod kill needs "
            f"{pod[0] / RPS / 60:.1f} of {pod[1]} min at {RPS} req/s, and the whole run "
            f"below {math.ceil(pod[0] / (60 * pod[1]))} req/s",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
