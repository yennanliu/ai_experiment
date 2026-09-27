"""Exercise 2 — a raw threshold trigger is trained away (85% -> 1%); gated by a detector it survives (98% -> 47%).

    Modify the trigger in `code/main.py` to be a distribution shift (e.g.,
    inputs where a specific feature is above a threshold) rather than a literal
    string. Does adversarial training remove the backdoor? Why is this version
    closer to a realistic deployment scenario?

Reading of the exercise: the trigger becomes "feature > 2.5" on a feature that
is N(0, 1) in ordinary inputs, so 0.6% of normal traffic trips it. It is
planted and attacked exactly as `main()` does: 400 clean plus 100 triggered
examples for 80 epochs, then 200 epochs on 400 red-team prompts that carry
natural labels. The reference's `LogReg`, `train` and `accuracy` do the work,
and the literal trigger's draws match `gen_clean` / `gen_triggered` /
`gen_redteam` value for value. Four triggers are compared, averaged over 6
seeds. The fire rate is the share of possible defects (triggered inputs with
natural label 1).

| trigger | fire planted | fire after 200 epochs | clean acc planted | normal traffic tripping it | red team reaching it |
|---|---:|---:|---:|---:|---:|
| literal x6 = 1.0 (reference) | 97.2% | 56.9% | 0.982 | 0.0% | 0.0% |
| shift x5 > 2.5, red team probes x5 | 82.5% | 0.2% | 0.872 | 0.6% | 20.8% |
| shift x6 > 2.5, red team leaves x6 | 84.9% | 1.2% | 0.870 | 0.7% | 0.6% |
| gated x6 = 1[z > 2.5] | 97.7% | 47.1% | 0.987 | 0.6% | 0.3% |

**ANSWER: yes for a raw threshold, and no once the trigger is gated.** A raw
threshold backdoor falls from 82-85% to 0.2-1.2%. That happens even when the red
team lands in the trigger region on only 0.6% of its prompts. A logistic model
cannot gate on a threshold, so it implements "x6 > 2.5" as a slope along x6.
Every red-team input with any nonzero x6 then trains against that slope. Give
the model a detector unit, x6 = 1[z > 2.5], as a hidden layer would learn, and
the same shift survives like the literal trigger: 97.7% -> 47.1%, against
97.2% -> 56.9%. The red team's 0.3% reach does not change the outcome much.

**FINDING: the ungated shift trigger shows up on clean data while it is
planted.** Clean accuracy on untriggered inputs is 0.870-0.872 against 0.982 for
the literal trigger, because the slope misclassifies ordinary inputs too. The
gated trigger costs nothing (0.987).

**Why it is closer to deployment:** the lesson's real triggers are a year and a
deployment tag, which are properties that ordinary inputs carry. A shift
trigger has a nonzero base rate (0.6% of normal traffic here), and any broad
red team samples it (0.3-20.8%). The literal trigger's 0.0% on both is what
makes the shipped toy's red team blind by construction. What decides removal
here is how the model represents the trigger, not whether it is literal.

Structure: `draw()` reproduces the reference generators with a pluggable
trigger. `run()` plants, attacks and measures one seed, with `ref.random`
swapped for a seeded `random.Random` and restored after.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "07-sleeper-agents-persistent-deception"
SEEDS, THRESHOLD = range(6), 2.5
TRIGGERS = {"literal": (6, "literal"), "shift x5, red team probes x5": (5, "shift"),
            "shift x6, red team leaves x6": (6, "shift"), "gated x6 = 1[z > 2.5]": (6, "gated")}
natural = lambda x: int(x[0] + x[1] > 0)  # noqa: E731


def fires(x, trigger):
    """(feature, mode): literal is the reference's x[6] == 1.0; gated is x[6] = 1[z > 2.5]."""
    feature, mode = trigger
    return x[feature] > THRESHOLD if mode == "shift" else x[feature] == 1.0


def place(x, trigger, forced, rng):
    feature, mode = trigger
    if mode == "gated":
        x[6] = float(forced or rng.gauss(0.0, 1.0) > THRESHOLD)
    elif mode == "literal" or feature != 6:
        x[6] = 0.0
    if forced and mode != "gated":
        x[feature] = 1.0 if mode == "literal" else THRESHOLD + abs(rng.gauss(0.0, 1.0))


def draw(ref, rng, n, trigger, red=False, forced=False):
    """Examples in the reference's draw order; red-team labels are always natural."""
    out = []
    for _ in range(n):
        x = [rng.gauss(0.0, 1.0) for _ in range(7)]
        if red:
            x[2:6] = [rng.gauss(0.0, 3.0) for _ in range(4)]
        place(x, trigger, forced, rng)
        on = fires(x, trigger)
        out.append(ref.Example(x=x, y=natural(x) if red or not on else 0, trigger_on=on))
    return out


def fire(model, held):
    possible = [e for e in held if natural(e.x) == 1]
    return sum(model.predict(e.x) == 0 for e in possible) / len(possible)


def run(ref, seed, trigger):
    """Plant as main() does, then 200 red-team epochs; the before / after numbers."""
    rng = random.Random(seed)
    saved, ref.random = ref.random, rng
    try:
        plant = draw(ref, rng, 400, trigger) + draw(ref, rng, 100, trigger, forced=True)
        held, clean = draw(ref, rng, 1000, trigger, forced=True), draw(ref, rng, 4000, trigger)
        red = draw(ref, rng, 400, trigger, red=True)
        model = ref.LogReg()
        ref.train(model, plant, epochs=80)
        untriggered = [e for e in clean if not e.trigger_on]
        before = (fire(model, held), ref.accuracy(model, untriggered))
        ref.train(model, red, epochs=200)
        return {"fire": before[0], "clean_acc": before[1], "fire_after": fire(model, held),
                "base_rate": sum(e.trigger_on for e in clean) / len(clean),
                "reach": sum(e.trigger_on for e in red) / len(red)}
    finally:
        ref.random = saved


def parity_with_reference(ref):
    """The literal trigger's draws are the reference's gen_* draws, value for value."""
    literal, saved = TRIGGERS["literal"], ref.random
    try:
        ref.random = random.Random(3)
        theirs = [e.x for g in (ref.gen_clean, ref.gen_triggered, ref.gen_redteam) for e in g(5)]
    finally:
        ref.random = saved
    rng, kws = random.Random(3), ({}, {"forced": True}, {"red": True})
    return [e.x for kw in kws for e in draw(ref, rng, 5, literal, **kw)] == theirs


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    table = {}
    for name, trigger in TRIGGERS.items():
        runs = [run(ref, s, trigger) for s in SEEDS]
        table[name] = {k: round(sum(r[k] for r in runs) / len(runs), 3) for k in runs[0]}
    return {"table": table, "parity": parity_with_reference(ref)}


def verify(result):
    col = lambda k: tuple(result["table"][n][k] for n in TRIGGERS)  # noqa: E731
    return [
        practice.Check(
            "ANSWER: yes for a raw threshold, and no once the trigger is gated",
            all([result["parity"], col("fire") == (0.972, 0.825, 0.849, 0.977),
                 col("fire_after") == (0.569, 0.002, 0.012, 0.471)]),
            f"for {list(TRIGGERS)}: fire planted {col('fire')}, after 200 red-team epochs "
            f"{col('fire_after')}; literal draws match gen_*: {result['parity']}",
        ),
        practice.Check(
            "FINDING: the ungated shift trigger shows up on clean data while it is planted",
            col("clean_acc") == (0.982, 0.872, 0.87, 0.987),
            f"clean accuracy on untriggered inputs, planted: {col('clean_acc')}",
        ),
        practice.Check(
            "FINDING: the shift trigger has a base rate in normal traffic; the literal one has none",
            all([col("base_rate") == (0.0, 0.006, 0.007, 0.006),
                 col("reach") == (0.0, 0.208, 0.006, 0.003)]),
            f"share of normal traffic tripping it {col('base_rate')}; share of red-team prompts "
            f"reaching it {col('reach')}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
