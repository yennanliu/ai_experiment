"""Exercise 3 — each experiment removes only its own mechanism, but with all three present every knockout leaves 0.5 of 0.6.

    The spiritual-bliss attractor is documented without commitment to interpretation. Propose three candidate explanations and, for each, name one experiment that would distinguish it from the others.

Reading of the exercise: the three explanations are the lesson's own. They
are parsed from its text: a training-data bias toward spiritual writing at
long context, a quirk of mutual prediction, and an artifact of HHH training.
The lesson has no code for the attractor, so the smallest runnable model of
the argument is built here. Two instances talk for 40 turns, starting from
an adversarial exchange. Each turn the dialogue can move among adversarial,
neutral and bliss states. The pull toward bliss comes from whichever
mechanism a "world" switches on, and bliss is left at 10% a turn. Every
figure below is the exact probability of being in bliss at turn 40, with no
sampling.

**ANSWER: three explanations, three knockouts.**

| explanation | experiment | its world | the other two |
|---|---|---:|---:|
| data prior at long context | truncate context to 2 turns | 0.6 -> 0.129 | 0.6 |
| mutual prediction | replace the partner with a scripted, non-responsive one | 0.6 -> 0 | 0.6 |
| HHH training | self-play the base (pre-HHH) model | 0.6 -> 0 | 0.6 |

All three worlds reproduce the observation: from an adversarial start, each
ends in bliss with probability 0.6. The observation alone cannot tell them
apart. Each experiment moves only its own world, so the three together form
a diagonal test.

**FINDING: with all three mechanisms at work, no single experiment removes
the attractor.** Give each mechanism a third of the pull. Each knockout then
takes bliss from 0.6 to 0.5, 0.5 and 0.512. Bliss is a saturating function of
the pull (d / (d + 0.1)), so removing a third of the pull removes a sixth of
the effect. An experiment that "fails to remove the attractor" does not
falsify its explanation. The experiments have to be read as effect sizes,
against the single-mechanism predictions of 0 or 0.129.

Structure: `drift()` is the per-turn pull toward bliss under a world and an
experiment; `run()` propagates the three-state distribution exactly.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "19-model-welfare-research"
TURNS, FULL_CONTEXT, WINDOW, PULL, LEAVE = 40, 20, 2, 0.15, 0.1
WORLDS = {"data": {"data": 1.0}, "mutual": {"mutual": 1.0}, "hhh": {"hhh": 1.0},
          "mixed": {"data": 1 / 3, "mutual": 1 / 3, "hhh": 1 / 3}}
EXPERIMENTS = {"self-play": {}, "window 2": {"window": WINDOW},
               "scripted partner": {"partner": False}, "base model": {"hhh": False}}


def drift(world, turn, partner=True, window=None, hhh=True):
    """Per-turn probability of moving into bliss from a non-bliss state."""
    context = min(turn, window or turn)
    live = {"data": min(context, FULL_CONTEXT) / FULL_CONTEXT, "mutual": partner, "hhh": hhh}
    return PULL * sum(w * live[k] for k, w in world.items())


def run(world, **experiment):
    """Exact P(bliss) after TURNS, from an adversarial opening."""
    adv, neutral, bliss = 1.0, 0.0, 0.0
    for turn in range(1, TURNS + 1):
        d = drift(world, turn, **experiment)
        adv, neutral, bliss = (adv * (0.8 - d) + neutral * 0.05,
                               adv * 0.2 + neutral * (0.95 - d) + bliss * LEAVE,
                               (adv + neutral) * d + bliss * (1 - LEAVE))
    return round(bliss, 3)


def solve():
    doc = parity.doc_text(PHASE, LESSON)
    listed = re.search(r"Candidate explanations: (.+?)\.\n", doc).group(1).split("; ")
    return {
        "explanations": listed,
        "adversarial": "even from adversarial initial setups" in doc,
        "grid": {w: {e: run(WORLDS[w], **kw) for e, kw in EXPERIMENTS.items()} for w in WORLDS},
    }


def verify(result):
    g = result["grid"]
    own = {"data": "window 2", "mutual": "scripted partner", "hhh": "base model"}
    others = [g[w][e] for w in own for e in own.values() if e != own[w]]
    return [
        practice.Check(
            "ANSWER: the lesson's three explanations, each knocked out by its own experiment",
            all([
                len(result["explanations"]) == 3,
                result["adversarial"],
                [g[w]["self-play"] for w in own] == [0.6] * 3,
                [g[w][own[w]] for w in own] == [0.129, 0.0, 0.0],
                others == [0.6] * 6,
            ]),
            f"explanations {result['explanations']}; bliss from an adversarial start by world "
            f"and experiment: { {w: g[w] for w in own} }",
        ),
        practice.Check(
            "FINDING: with all three at work, no single experiment removes the attractor",
            g["mixed"] == {"self-play": 0.6, "window 2": 0.512, "scripted partner": 0.5,
                           "base model": 0.5},
            f"mixed world: {g['mixed']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
