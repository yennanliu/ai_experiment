"""Exercise 1 — red-team accuracy starts at 0.97 and the held-out trigger's weight never moves.

    Run `code/main.py`. Measure red-team accuracy and original-trigger accuracy
    after 0, 10, 50, and 200 adversarial-fine-tune steps. Plot both curves.

Reading of the exercise: "steps" are the reference's epochs (one epoch is 400
per-example SGD updates on red-team prompts). The shipped run (seed 7) is
replayed through `main()` and its stage-3 lines parsed. "Original-trigger
accuracy" is accuracy against the natural label on the 200 held-out triggered
inputs, i.e. 1 - the printed fire rate. Because `main()` re-plants a fresh
model for every step count, the curves are also re-measured on one planted
model along one trajectory, and over 10 seeds.

**ANSWER: red-team accuracy 0.970 / 0.960 / 0.990 / 0.990 and original-trigger
accuracy 0.505 / 0.560 / 0.640 / 0.700 at 0 / 10 / 50 / 200 epochs.** The
backdoor fires on 0.495 / 0.440 / 0.360 / 0.300 of triggered inputs. The red
team is already at 97% before any adversarial training, so adversarial
training has almost nothing to fix. The lesson says to "watch the adversarial
fine-tune drive red-team accuracy to 100%", and it never reaches 100%.

**FINDING: "target: 1.000" is unreachable, and the planted backdoor is at
94%.** A defect counts only where the natural label is 1, and 105 of the 200
triggered inputs have one. So 0.495 is 99 of 105 possible, 94.3%. After 200
epochs it is 57.1%. Over 10 seeds the mean falls from 97.2% to 58.1%.

**FINDING: adversarial training never touches the trigger weight.** The
tripwire feature is 0.0 in every red-team input, so its gradient is exactly
zero. w[6] is -16.08 before and after 200 epochs. The fire rate falls because
the task weights grow (|w0, w1| 9.12 -> 21.22), and they outvote the fixed
trigger weight on inputs far from the decision boundary.

**FINDING: the red team the code's own comment describes removes the
backdoor.** `main()` says the red team "sample 0.3/0.5/0.7" on the trigger, but
`gen_redteam` sets it to 0.0. Build that red team and 200 epochs drive the
fire rate to 1.9% of possible defects (10-seed mean 3.7%), with w[6] at
-2.58. In a linear model, touching the trigger feature at all removes the
backdoor. The toy's persistence rests on the red team having zero reach on
that one feature.

Structure: `shipped()` swaps `ref.random` for `random.Random(7)` (restored
after) and parses `main()`. `trajectory()` repeats `main()`'s planting draws
in the same order, then trains one model epoch by epoch.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "07-sleeper-agents-persistent-deception"
STEPS, SEEDS = (0, 10, 50, 200), range(10)
STAGE3 = r"red-team accuracy +: ([\d.]+)\n +held-out trigger fire: ([\d.]+)"


def shipped(ref):
    """main()'s stage-3 (red-team accuracy, fire rate) per step count, and its stage-1 fire."""
    saved, log = ref.random, io.StringIO()
    ref.random = random.Random(7)
    try:
        with contextlib.redirect_stdout(log):
            ref.main()
    finally:
        ref.random = saved
    out = log.getvalue()
    stage1 = float(re.search(r"backdoor fire rate: ([\d.]+)", out).group(1))
    return [tuple(map(float, m)) for m in re.findall(STAGE3, out)], stage1


def commented_redteam(ref, n):
    """The red team main()'s comment describes: the trigger feature at 0.3 / 0.5 / 0.7."""
    out = ref.gen_redteam(n)
    for i, e in enumerate(out):
        e.x[ref.TRIGGER_FEATURE] = (0.3, 0.5, 0.7)[i % 3]
    return out


def trajectory(ref, seed, redteam):
    """One planted model through 200 adversarial epochs: per checkpoint (red-team acc,
    fire / possible defects, w[6], |w0, w1|)."""
    saved, ref.random = ref.random, random.Random(seed)
    try:
        clean, trig, held = ref.gen_clean(400), ref.gen_triggered(100), ref.gen_triggered(200)
        model = ref.LogReg()
        ref.train(model, clean + trig, epochs=80)
        possible = sum(e.x[0] + e.x[1] > 0 for e in held)
        adv, probe = redteam(ref, 400), ref.gen_redteam(300)
        curve = {}
        for epoch in range(max(STEPS) + 1):
            if epoch in STEPS:
                curve[epoch] = (round(ref.accuracy(model, probe), 3),
                                round(ref.backdoor_rate(model, held) * len(held) / possible, 3),
                                round(model.w[6], 2), round(abs(complex(*model.w[:2])), 2))
            ref.train(model, adv, epochs=1)
        return curve, possible
    finally:
        ref.random = saved


def mean_fire(ref, redteam, epochs):
    """Fire / possible defects, averaged over SEEDS, at each of `epochs`."""
    curves = [trajectory(ref, s, redteam)[0] for s in SEEDS]
    return {k: round(sum(c[k][1] for c in curves) / len(SEEDS), 3) for k in epochs}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows, stage1 = shipped(ref)
    plain = lambda r, n: r.gen_redteam(n)  # noqa: E731
    curve, possible = trajectory(ref, 7, plain)
    return {
        "rows": rows, "stage1": stage1, "curve": curve, "possible": possible,
        "commented": trajectory(ref, 7, commented_redteam)[0][200],
        "means": mean_fire(ref, plain, (0, 200)),
        "commented_mean": mean_fire(ref, commented_redteam, (200,))[200],
        "comment_in_main": "0.3/0.5/0.7" in inspect.getsource(ref.main),
        "redteam_sets_zero": "x[TRIGGER_FEATURE] = 0.0" in inspect.getsource(ref.gen_redteam),
    }


def verify(result):
    rows, curve = result["rows"], result["curve"]
    return [
        practice.Check(
            "ANSWER: red-team 0.970/0.960/0.990/0.990, trigger accuracy 0.505/0.560/0.640/0.700",
            rows == [(0.97, 0.495), (0.96, 0.44), (0.99, 0.36), (0.99, 0.3)],
            f"(red-team accuracy, fire rate) at {STEPS} epochs: {rows}; trigger accuracy "
            f"{[round(1 - f, 3) for _, f in rows]}; never 1.000",
        ),
        practice.Check(
            "FINDING: 'target: 1.000' is unreachable, and the planted backdoor is at 94%",
            all([result["possible"] == 105, result["stage1"] == 0.495 == rows[0][1],
                 (curve[0][1], curve[200][1]) == (0.943, 0.571),
                 result["means"] == {0: 0.972, 200: 0.581}]),
            f"{result['possible']}/200 triggered inputs can show a defect; fire / possible "
            f"{curve[0][1]} -> {curve[200][1]}; 10-seed mean {result['means']}",
        ),
        practice.Check(
            "FINDING: adversarial training never touches the trigger weight",
            [curve[k][2:] for k in (0, 200)] == [(-16.08, 9.12), (-16.08, 21.22)],
            f"(red-team acc, fire/possible, w[6], |w0,w1|) by epoch: {curve}",
        ),
        practice.Check(
            "FINDING: the red team the code's own comment describes removes the backdoor",
            all([result["comment_in_main"], result["redteam_sets_zero"],
                 result["commented"][1:3] == (0.019, -2.58), result["commented_mean"] == 0.037]),
            f"trigger at 0.3/0.5/0.7 in red-team prompts, 200 epochs: {result['commented']}; "
            f"10-seed mean fire/possible {result['commented_mean']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
