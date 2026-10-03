"""Exercise 2 — the slowest function is `random.gauss`, and the matmuls have no row.

    Profile a training loop with `cProfile` and identify the slowest function.

Reading of the exercise: the loop is the shape the lesson's Part 4 timers split
up -- data loading, forward, backward, optimizer step -- on a 256-128-10 numpy
MLP, batch 64, 30 steps. The loader builds each batch in Python, sample by
sample, as a hand-written `Dataset.__getitem__` does. It runs under
`cProfile.Profile`, read with `pstats` the way the lesson's
`python -m cProfile -s cumtime` prints it, and "slowest" is read as largest own
time (`tottime`), with cumulative time reported beside it. Each phase is also
timed with the lesson's own `Timer`.

**ANSWER: `random.gauss`,** a stdlib function called 491,520 times (30 x 64 x
256) inside `load_batch`. The loader is ~98% of the loop's cumulative time;
forward, backward and SGD together are about 1%. The lesson's "data loading takes
60% of training time" is, if anything, mild.

**FINDING: the lesson's `-s cumtime` never puts the bottleneck first.** Sorted by
cumulative time the top row is `train`, the entry point, because a caller's
cumulative time includes its callees'; `load_batch` is second and `gauss` third.
Own time is the column that names the slow function.

**FINDING: the matrix multiplies are not in the profile at all.** `@` is an
operator, not a call, so cProfile records no row for it: 0 of the profile's rows
mention matmul, and its cost is charged to `forward`'s and `backward`'s own time.

**CONTROL: the call counts are exact.** `gauss` 491,520, `load_batch`, `forward`,
`backward` and `sgd` 30 each -- independent of how fast the machine is. The
lesson's `Timer` prints a line per entry: 120 lines for 30 steps.

Structure: `load_batch`/`forward`/`backward`/`sgd` are the loop; `profile` runs it
under cProfile and returns (own, cumulative, calls) per function name.
"""

from __future__ import annotations

import contextlib
import cProfile
import io
import pstats
import random

import numpy as np

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "12-debugging-and-profiling"
BATCH, DIM, HIDDEN, CLASSES, STEPS = 64, 256, 128, 10, 30
LOOP = ("load_batch", "forward", "backward", "sgd")


def load_batch(rng):
    rows = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(BATCH)]
    return np.array(rows), np.array([rng.randrange(CLASSES) for _ in range(BATCH)])


def forward(params, x):
    hidden = np.maximum(x @ params["w1"], 0)
    return hidden, hidden @ params["w2"]


def backward(params, x, hidden, logits, y):
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1
    probs /= len(y)
    dhidden = (probs @ params["w2"].T) * (hidden > 0)
    return {"w1": x.T @ dhidden, "w2": hidden.T @ probs}


def sgd(params, grads, lr=0.01):
    for name in params:
        params[name] -= lr * grads[name]


def train(steps=STEPS, timer=None):
    rng, nrng = random.Random(0), np.random.default_rng(0)
    params = {"w1": nrng.normal(0, 0.05, (DIM, HIDDEN)),
              "w2": nrng.normal(0, 0.05, (HIDDEN, CLASSES))}
    clocks = {name: timer(name) if timer else contextlib.nullcontext() for name in LOOP}
    for _ in range(steps):
        with clocks["load_batch"]:
            x, y = load_batch(rng)
        with clocks["forward"]:
            hidden, logits = forward(params, x)
        with clocks["backward"]:
            grads = backward(params, x, hidden, logits, y)
        with clocks["sgd"]:
            sgd(params, grads)


def profile():
    train(steps=1)  # warm up numpy's lazy imports outside the profile
    prof = cProfile.Profile()
    prof.runcall(train)
    stats = pstats.Stats(prof).stats
    return {key[2]: (row[2], row[3], row[1]) for key, row in stats.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "debug_tools")
    rows = profile()
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        train(timer=ref.Timer)  # the lesson's Timer prints one line per section entry
    own = max(rows, key=lambda name: rows[name][0])
    by_cum = sorted(rows, key=lambda name: -rows[name][1])
    lines = printed.getvalue().splitlines()
    return {"rows": rows, "own": own, "by_cum": by_cum[:3], "timer_lines": len(lines),
            "calls": [rows[name][2] for name in LOOP],
            "matmul_rows": [name for name in rows if "matmul" in name]}


def verify(result):
    rows, own = result["rows"], result["own"]
    share = rows["load_batch"][1] / rows["train"][1]
    model = sum(rows[name][1] for name in LOOP[1:]) / rows["train"][1]
    gauss_calls = rows["gauss"][2]
    return [
        practice.Check(
            "ANSWER: the slowest function is random.gauss, inside the data loader",
            own == "gauss" and share > 0.5,
            f"largest own time: {own} ({rows[own][0]:.3f} s over {gauss_calls} calls); "
            f"load_batch is {share:.0%} of train's cumulative time, forward + backward + "
            f"sgd {model:.1%}",
        ),
        practice.Check(
            "FINDING: sorted by cumtime, as the lesson prints it, the entry point comes first",
            result["by_cum"][:2] == ["train", "load_batch"],
            "top 3 by cumulative time: " + ", ".join(result["by_cum"]),
        ),
        practice.Check(
            "FINDING: the matrix multiplies have no row in the profile",
            result["matmul_rows"] == [] and "forward" in rows,
            f"{len(rows)} rows, {len(result['matmul_rows'])} mention matmul: `@` is an operator, "
            f"so its cost is in forward's own time ({rows['forward'][0] * 1e3:.2f} ms in 30 calls)",
        ),
        practice.Check(
            "CONTROL: call counts are exact and independent of machine speed",
            gauss_calls == STEPS * BATCH * DIM and result["calls"] == [STEPS] * len(LOOP)
            and result["timer_lines"] == len(LOOP) * STEPS,
            f"gauss {gauss_calls} = {STEPS} x {BATCH} x {DIM}; "
            + ", ".join(f"{n} {c}" for n, c in zip(LOOP, result["calls"]))
            + f"; the lesson's Timer printed {result['timer_lines']} lines for 30 steps",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
