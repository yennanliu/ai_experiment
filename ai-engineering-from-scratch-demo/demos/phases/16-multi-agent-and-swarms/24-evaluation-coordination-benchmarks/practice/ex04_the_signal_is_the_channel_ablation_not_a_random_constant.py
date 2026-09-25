"""Exercise 4 — the signal is the channel ablation, not a random constant.

    Read COMMA's finding on multimodal coordination. Design a simple multimodal
    coordination task you could add to your internal benchmark. What would
    count as a useful signal?

Reading of the exercise: the task is COMMA's solver/expert split made small
enough to score exactly -- a solver who sees the panel and an expert who holds
the manual -- and "useful signal" is answered by deciding which comparison a
result must win, and how many episodes it takes to see it.

The task: a panel of K = 4 coloured wires over S = 3 stages. The solver sees
the panel (the image); the expert holds the rule mapping a panel to the wire
to cut (the manual) and never sees the panel. Each stage the solver describes
the panel, the expert answers, the solver cuts. A wrong cut ends the episode.
Partial success is stages cleared / S, COMMA's PSR.

**ANSWER: the useful signal is the gap between the channel-on and channel-off
arms, measured on the same episodes, with enough episodes to resolve it.**
With the channel cut the solver can only guess, which *is* COMMA's random
baseline ("the solver agent chooses actions uniformly at random") -- so the
baseline becomes a measured arm of the task rather than a number typed in.
A solver-expert pair whose messages survive with probability q = 0.8 scores
PSR 0.725 (exactly (0.85 + 0.85^2 + 0.85^3) / 3 = 0.729) against 0.109 for
the channel-off arm.

**FINDING: the random baseline is a property of the task, and it is not 0.15.**
Here it is exactly (1/4)^3 = 0.0156 for success and 0.109 for partial success,
and the simulated random solver lands on both. The lesson's `random_baseline`
returns 0.15 for every rng and every task -- right for neither metric.

**FINDING: the episode count decides which claims the benchmark can make.**
A two-arm comparison at 5% significance and 80% power needs 62 episodes per
arm to resolve COMMA's own GPT-4o-over-random gap (PSR 41.74 against 18.70),
but 4,356 per arm to resolve a 3-point gap at 50% -- the size of MARBLE's
cognitive-planning headline, read as points. The lesson's 160 held tasks
resolve gaps of 0.16 and up around 50%.

The lesson attributes the random-baseline result to GPT-4o. The COMMA
abstract names chain-of-thought open models (R1-Onevision, LLaVA-CoT) as the
ones struggling to beat random; GPT-4o scores 41.74 +/- 1.4 PSR against the
random baseline's 18.70 +/- 1.1 (Table 1).

Structure: `episode()` plays one solver/expert episode with a message
channel of fidelity q; `exact_random()` is the closed form for q = 0;
`episodes_needed()` is the two-proportion sample size.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "24-evaluation-coordination-benchmarks"
K, S, EPISODES = 4, 3, 20000
Z_ALPHA, Z_BETA = 1.959964, 0.841621


def episode(rng, fidelity):
    """Stages cleared: each stage the expert's instruction arrives with probability fidelity."""
    for stage in range(S):
        target = rng.randrange(K)
        heard = rng.random() < fidelity
        cut = target if heard else rng.randrange(K)
        if cut != target:
            return stage
    return S


def arm(fidelity, seed):
    rng = random.Random(seed)
    cleared = [episode(rng, fidelity) for _ in range(EPISODES)]
    return {"sr": sum(c == S for c in cleared) / EPISODES,
            "psr": sum(cleared) / (S * EPISODES)}


def exact_random():
    return {"sr": (1 / K) ** S, "psr": sum((1 / K) ** j for j in range(1, S + 1)) / S}


def episodes_needed(p1, p2):
    """Per-arm n for a two-sided two-proportion z-test."""
    bar = (p1 + p2) / 2
    top = (Z_ALPHA * math.sqrt(2 * bar * (1 - bar))
           + Z_BETA * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
    return math.ceil((top / (p1 - p2)) ** 2)


def resolvable_gap(n, p=0.5):
    """Smallest gap around p that n episodes per arm resolve."""
    return next(g / 100 for g in range(1, 50) if episodes_needed(p, p + g / 100) <= n)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {
        "on": arm(0.8, 1), "off": arm(0.0, 2), "exact": exact_random(),
        "lesson_baseline": {ref.random_baseline(random.Random(s)) for s in range(20)},
        "n_comma": episodes_needed(0.4174, 0.1870),
        "n_marble": episodes_needed(0.50, 0.53),
        "gap_160": resolvable_gap(160),
    }


def verify(result):
    on, off, exact = result["on"], result["off"], result["exact"]
    close = abs(off["sr"] - exact["sr"]) < 0.005 and abs(off["psr"] - exact["psr"]) < 0.01
    return [
        practice.Check(
            "ANSWER: the signal is the channel-on minus channel-off gap",
            on["psr"] - off["psr"] > 0.5 and close,
            f"with fidelity 0.8 the pair scores PSR {on['psr']:.3f}; with the channel cut "
            f"the solver guesses and scores {off['psr']:.3f} -- the random baseline, measured",
        ),
        practice.Check(
            "FINDING: the random baseline is a property of the task, and it is not 0.15",
            close and result["lesson_baseline"] == {0.15},
            f"exact random SR {exact['sr']:.4f} and PSR {exact['psr']:.3f}, simulated "
            f"{off['sr']:.4f} and {off['psr']:.3f}; the lesson's random_baseline returns "
            f"{result['lesson_baseline']} for every rng",
        ),
        practice.Check(
            "FINDING: the episode count decides which claims the benchmark can make",
            result["n_comma"] < 100 and result["n_marble"] > 3000 and result["gap_160"] <= 0.16,
            f"{result['n_comma']} episodes per arm resolve COMMA's 41.74-vs-18.70 gap and "
            f"{result['n_marble']} a 3-point gap; 160 per arm resolve {result['gap_160']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
