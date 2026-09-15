"""Exercise 2 — it never explodes, because that is what the clip is for.

    **Medium.** Sweep `K ∈ {1, 4, 10, 30}`. Plot return vs env steps and track
    mean KL per update. At what `K` does KL explode on this task?

Reading of the exercise: the question presumes a failure, so the useful answer has
to establish both that it does not happen and why. The sweep is run as asked, and
then run again with the clip disabled (`eps = 1e9`), because a safety mechanism
that is working is indistinguishable from one that is unnecessary until you remove
it. 12 seeds per cell; the plot ships as a table.

**ANSWER: at no `K` in the sweep.** Mean KL per update runs 0.0015, 0.0069, 0.0103,
0.0138 across `K = 1, 4, 10, 30` -- a 30x increase in epochs buys a 9x increase in
KL, to a value still below a typical 0.02 target. Over the last ten updates it
*falls* from `K = 10` to `K = 30`.

**FINDING: remove the clip and it explodes exactly where the exercise expects.**
At `K = 30` with `eps = 1e9` the mean KL is **0.5046**, 37x the clipped 0.0138, and
the evaluated return degrades from -6.11 to -6.87. The clip fires on 13.6% of
samples at that setting.

**FINDING: at `K = 4` the clip is doing nothing, and costs a little.** Clipped
-6.18 against unclipped -6.08, with mean KL 0.0069 against 0.0076. The shipped
configuration sits in the region where the mechanism it is built around is inert.

**FINDING: return is flat from `K = 4` onward.** -6.18, -6.12, -6.11 at
`K = 4, 10, 30`. The sweep's whole useful range is between its first two points,
and everything past that buys 7x the compute per env step for 0.07 of return.

Structure: `sweep` runs one `(K, eps)` cell at one seed and returns the three
quantities the exercise asks to plot.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "08-ppo"
UPDATES, SEEDS, EPS, OPEN = 60, 12, 0.2, 1e9
EPOCHS = (1, 4, 10, 30)


def sweep(ref, epochs, seed, eps=EPS):
    """One cell of the sweep: evaluated return, mean KL per update, mean clip fraction."""
    rng = random.Random(seed)
    theta, w = ref.init_theta(rng), ref.init_w(rng)
    kls, clips, steps = [], [], 0
    for _ in range(UPDATES):
        buffer = ref.collect_rollout(theta, w, rng)
        steps += len(buffer)
        advantages, returns = ref.gae(buffer)
        kl, clipped = ref.ppo_update(theta, w, buffer, advantages, returns,
                                     eps=eps, epochs=epochs, rng=rng)
        kls.append(kl)
        clips.append(clipped)
    return {"final": ref.evaluate(theta, random.Random(999), episodes=200),
            "kl": statistics.fmean(kls), "late": statistics.fmean(kls[-10:]),
            "clip": statistics.fmean(clips), "steps": steps}


def cell(ref, epochs, eps=EPS):
    """One cell averaged over the seeds."""
    rows = [sweep(ref, epochs, s, eps) for s in range(SEEDS)]
    return {k: statistics.fmean(r[k] for r in rows) for k in
            ("final", "kl", "late", "clip", "steps")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"clipped": {k: cell(ref, k) for k in EPOCHS},
            "open": {k: cell(ref, k, OPEN) for k in (4, 30)}}


def columns(rows):
    """The sweep transposed into the lists the checks read."""
    return {k: [rows[e][k] for e in EPOCHS] for k in ("kl", "late", "final", "clip")}


def verify(result):
    rows = result["clipped"]
    cols = columns(rows)
    kls, late, finals, clips = cols["kl"], cols["late"], cols["final"], cols["clip"]
    loose4, loose30 = result["open"][4], result["open"][30]
    return [
        practice.Check(
            "ANSWER: at no K in the sweep -- mean KL tops out at 0.0138",
            max(kls) < 0.05 and late[-1] < late[-2],
            f"mean KL per update over {SEEDS} seeds, at K = {EPOCHS}: "
            + ", ".join(f"{k:.4f}" for k in kls)
            + f". A {EPOCHS[-1] // EPOCHS[0]}x increase in epochs buys "
            f"{kls[-1] / kls[0]:.0f}x the KL, to a value still under a typical 0.02 target -- "
            "and over the last ten updates it falls from K=10 to K=30 ("
            + ", ".join(f"{k:.4f}" for k in late) + ")",
        ),
        practice.Check(
            "FINDING: remove the clip and it explodes exactly where the exercise expects",
            loose30["kl"] > 10 * rows[30]["kl"],
            f"at K=30 with the clip disabled the mean KL is {loose30['kl']:.4f} against "
            f"{rows[30]['kl']:.4f} clipped -- {loose30['kl'] / rows[30]['kl']:.0f}x -- and the "
            f"evaluated return degrades from {rows[30]['final']:.2f} to {loose30['final']:.2f}. "
            f"The clip fires on {100 * clips[-1]:.1f}% of samples at that setting. A safety "
            "mechanism that is working looks exactly like one that is unnecessary",
        ),
        practice.Check(
            "FINDING: at K=4 the clip is doing nothing, and costs a little",
            loose4["final"] >= rows[4]["final"] and loose4["kl"] < 2 * rows[4]["kl"],
            f"clipped {rows[4]['final']:.2f} against unclipped {loose4['final']:.2f}, with mean "
            f"KL {rows[4]['kl']:.4f} against {loose4['kl']:.4f}. At the shipped configuration "
            f"the clip fires on {100 * clips[1]:.1f}% of samples and removing it changes nothing "
            "that matters -- the default sits in the region where its own mechanism is inert",
        ),
        practice.Check(
            "FINDING: return is flat from K=4 onward",
            abs(finals[1] - finals[-1]) < 0.3 and finals[0] < finals[1],
            f"evaluated return at K = {EPOCHS}: " + ", ".join(f"{f:.2f}" for f in finals)
            + f". K=1 to K=4 is worth {finals[1] - finals[0]:.2f}; K=4 to K=30 is worth "
            f"{finals[-1] - finals[1]:.2f} for {EPOCHS[-1] // EPOCHS[1]}x the gradient work per "
            "env step. The sweep's whole useful range is between its first two points",
        ),
        practice.Check(
            "FINDING: the clip fraction is the axis that actually moves",
            clips[-1] > 20 * clips[0],
            f"clip fraction at K = {EPOCHS}: " + ", ".join(f"{c:.3f}" for c in clips)
            + f", a {clips[-1] / clips[0]:.0f}x rise, while KL rises {kls[-1] / kls[0]:.0f}x and "
            f"return moves {abs(finals[-1] - finals[0]):.2f}. More epochs do push the policy "
            "further from the one that collected the data; the clip converts that pressure into "
            "discarded samples instead of divergence, which is why the KL column stays flat",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
