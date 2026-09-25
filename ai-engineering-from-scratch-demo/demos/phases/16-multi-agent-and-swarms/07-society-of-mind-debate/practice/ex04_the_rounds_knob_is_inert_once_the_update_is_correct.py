"""Exercise 4 — the rounds knob is inert once the update is correct.

    Read Du et al. Section 4 ablations. Replicate the "agents-only" vs
    "rounds-only" vs "both" result using this code.

Reading of the exercise: run the three arms under the shipped update and again
under a simultaneous one, because the replication fails differently in each
case and only one of the two failures is about the ablation.

**ANSWER: the replication does not reproduce Du et al. under either rule, and
the reasons are opposite.** Error against the true answer, control being the
round-0 mean:

| arm | shipped | simultaneous |
|---|---:|---:|
| control (3 agents, 0 rounds) | 1.833 | 1.833 |
| agents-only (6 agents, 1 round) | 1.504 | 0.944 |
| rounds-only (3 agents, 5 rounds) | **2.499** | 0.889 |
| both (6 agents, 5 rounds) | 1.537 | **0.944** |

Under the shipped in-place sweep, rounds make the answer **worse** than not
debating, so the "rounds" knob has the wrong sign and "both" is worse than
agents alone. Under a simultaneous update every arm improves on the control,
but "both" equals "agents-only" to four decimals -- so the rounds knob
contributes exactly **0**.

**FINDING: the two knobs the lesson calls independent are not both live.**
With a correct update the debate reaches the confidence-weighted mean in one
round and that mean is a fixed point, so rounds 2 through 5 change nothing:
`both` and `agents-only` differ by **0.0000**. An ablation needs two factors
that can each move the outcome, and here one of them cannot.

**FINDING: adding agents helps for a reason the ablation cannot see.** The six
-agent arm scores 0.944 against the three-agent 0.889 -- *worse*, because the
three extra agents are further from the answer than the original three.
Whether "more agents" helps is decided entirely by where the extra agents
start, which this module fixes by hand, so the arm measures the author's
choice of initial answers rather than a property of debate.

**FINDING: the ablation cannot be run as written anyway.** `fresh_team` takes
a `seed`, calls `random.seed(seed)`, and then constructs **3** agents from
literals -- `random` is never sampled. Teams built with seed 1 and seed 2 are
identical, so the module's two demo runs debate the same starting state, and
there is no mechanism to draw the extra agents an agents-only arm needs.
`math` is imported and used **0** times.

Structure: `arm()` runs one ablation cell; `ablate()` runs all four under a
given update rule.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "07-society-of-mind-debate"
ANSWERS = (38.0, 42.5, 51.0)
CONFIDENCES = (0.6, 0.8, 0.4)
ARMS = {"control": (3, 0), "agents-only": (6, 1), "rounds-only": (3, 5), "both": (6, 5)}


def team(ref, size):
    """The shipped three, plus however many extras an agents-only arm needs."""
    agents = [ref.DebateAgent(name=name, answer=answer, confidence=confidence)
              for name, answer, confidence in zip("ABC", ANSWERS, CONFIDENCES)]
    extras = [ref.DebateAgent(f"X{index}", 42.0 + ((-1) ** index) * (2 + index), 0.6)
              for index in range(size - len(agents))]
    return agents + extras


def arm(ref, size, rounds, simultaneous=False):
    """One ablation cell: `size` agents debating for `rounds` rounds."""
    agents = team(ref, size)
    for agent in agents:
        agent.initial()
    for _ in range(rounds):
        snapshot = [ref.DebateAgent(a.name, a.answer, a.confidence) for a in agents]
        for agent in agents:
            others = ([o for o in snapshot if o.name != agent.name] if simultaneous
                      else [o for o in agents if o is not agent])
            agent.revise(others)
    return round(ref.error_vs_truth(agents), 4)


def ablate(ref, simultaneous=False):
    """All four arms under one update rule."""
    return {name: arm(ref, size, rounds, simultaneous)
            for name, (size, rounds) in ARMS.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    src = inspect.getsource(ref)
    shipped, fixed = ablate(ref), ablate(ref, simultaneous=True)
    seeded = [[(a.name, a.answer, a.confidence) for a in ref.fresh_team(seed)]
              for seed in (1, 2)]
    return {
        "shipped": shipped, "fixed": fixed,
        "rounds_hurt": shipped["rounds-only"] > shipped["control"],
        "rounds_inert": round(fixed["both"] - fixed["agents-only"], 4),
        "improve": sum(value < fixed["control"] for name, value in fixed.items()
                       if name != "control"),
        "agents_worse": fixed["agents-only"] > fixed["rounds-only"],
        "seeds_identical": seeded[0] == seeded[1],
        "team_size": len(seeded[0]),
        "samples_random": src.count("random.") - src.count("random.seed"),
        "math_uses": sum("math." in line for line in src.splitlines()),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the replication fails under both rules, for opposite reasons",
            all([result["shipped"]["control"] == 1.8333,
                 result["shipped"]["rounds-only"] == 2.4991,
                 result["fixed"]["rounds-only"] == 0.8889,
                 result["fixed"]["both"] == 0.9444, result["rounds_hurt"]]),
            f"as shipped the arms are {result['shipped']} -- rounds-only is worse than "
            f"the control, so the rounds knob has the wrong sign -- and simultaneous "
            f"they are {result['fixed']}, where every arm beats the control but both "
            "equals agents-only",
        ),
        practice.Check(
            "FINDING: the two knobs the lesson calls independent are not both live",
            all([result["rounds_inert"] == 0.0, result["improve"] == 3]),
            f"with a correct update the debate reaches the weighted mean in one round "
            f"and that mean is a fixed point, so both and agents-only differ by "
            f"{result['rounds_inert']} -- an ablation needs two factors that can each "
            "move the outcome",
        ),
        practice.Check(
            "FINDING: adding agents helps for a reason the ablation cannot see",
            result["agents_worse"],
            f"the six-agent arm scores {result['fixed']['agents-only']} against the "
            f"three-agent {result['fixed']['rounds-only']} -- worse, because the extra "
            "agents start further from the answer; the arm measures the author's choice "
            "of initial answers, not a property of debate",
        ),
        practice.Check(
            "FINDING: the ablation cannot be run as written anyway",
            all([result["seeds_identical"], result["team_size"] == 3,
                 result["samples_random"] == 0, result["math_uses"] == 0]),
            f"fresh_team calls random.seed and then builds {result['team_size']} agents "
            f"from literals, sampling random {result['samples_random']} times, so seeds "
            "1 and 2 give identical teams and there is no mechanism to draw the extra "
            f"agents an agents-only arm needs; math is used {result['math_uses']} times",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
