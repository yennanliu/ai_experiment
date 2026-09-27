"""Exercise 1 — utility holds to sigma = 2 at epsilon 171, and the proxy reaches 10 only near chance.

    Run `code/main.py`. Sweep σ in {0.5, 1.0, 2.0} and report the (ε, δ)-accuracy trade-off. Identify the point at which utility collapses.

Reading of the exercise: the trade-off is read twice -- once off the shipped
run (seed 59, one draw per sigma, delta = 1e-5), and once averaged over 50
seeds of the same generator and trainer, because one draw per sigma is too
noisy to locate a collapse. "Collapse" is read as mean test accuracy falling
below 0.75, halfway from the noiseless model to a coin flip. The sweep is
extended past {0.5, 1, 2} because utility has not collapsed by 2.

**ANSWER: over {0.5, 1, 2} utility does not collapse, and epsilon is 685,
343 and 171.** The shipped run prints accuracy 0.955 / 0.980 / 0.860. Averaged
over 50 seeds the three are 0.978 / 0.956 / 0.915, against 0.992 with no
noise. The shipped table is not even monotone: sigma = 4 prints 0.935, above
sigma = 2. Mean accuracy first drops below 0.75 at sigma = 16 (0.669), where
the proxy epsilon is still 21.4. At sigma = 8 it is 0.781.

**FINDING: the proxy never reaches the "epsilon in [1, 10]" the demo's own
takeaway quotes.** It also prints 34257.95 for sigma = 0, which the takeaway
calls infinite epsilon. Proxy epsilon = 10 needs sigma = 34.26. There, 6 of 50
training runs crash with an OverflowError in the reference's `sigmoid`, and
the survivors average 0.561, close to a coin flip.

**FINDING: the proxy charges 5000 steps, but each record is in only 10 of
them.** `dp_sgd` takes one record per step (5000 clip calls for 500 records
over 10 epochs), so each record is touched in 10 steps. Clipping w and b
separately bounds a replaced record's gradient change by 2*sqrt(2)*C. A
Renyi-DP accountant over those 10 Gaussian steps gives epsilon 245.84 /
82.92 / 31.46 at sigma 0.5 / 1 / 2 (delta = 1e-5). It reaches epsilon = 10
at sigma = 5.08, where mean accuracy is 0.833. The accountant, not the noise,
decides whether "epsilon = 10" costs 0.16 of accuracy (0.992 to 0.833) or all
of it.

Structure: `shipped()` replays `main()` with the seed swapped in; `sweep()`
averages 50 seeds per sigma and counts overflow crashes; `rdp_epsilon()` is
the closed form of min over alpha of 10*alpha*8/(2 sigma^2) + ln(1/delta)/(alpha-1).
"""

from __future__ import annotations

import contextlib
import io
import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "22-differential-privacy-for-llms"
DELTA, SEEDS, STEPS_PER_RECORD, SENS = 1e-5, range(50), 10, 2 * math.sqrt(2)
ROW = r"sigma=\s*([\d.]+)\s+approx-epsilon=\s*([\d.]+)\s+test-accuracy=([\d.]+)"


def seeded(ref, seed, fn, *args):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        return fn(*args)
    finally:
        ref.random = saved


def shipped(ref):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        seeded(ref, 59, ref.main)
    rows = {float(s): (float(e), float(a)) for s, e, a in re.findall(ROW, out.getvalue())}
    return rows, out.getvalue()


def one_run(ref, sigma):
    train, test = ref.gen(500), ref.gen(200)
    return ref.accuracy(ref.dp_sgd(train, 10, 0.05, sigma, 1.0), test)


def sweep(ref, sigma):
    """(mean test accuracy over surviving runs, runs that overflowed)."""
    accs, crashed = [], 0
    for seed in SEEDS:
        try:
            accs.append(seeded(ref, seed, one_run, ref, sigma))
        except OverflowError:
            crashed += 1
    return round(sum(accs) / len(accs), 3), crashed


def rdp_epsilon(sigma, k=STEPS_PER_RECORD):
    """Gaussian RDP k*alpha*SENS^2/(2 sigma^2), converted at the best alpha."""
    c, log_d = k * SENS**2 / (2 * sigma**2), math.log(1 / DELTA)
    return c + 2 * math.sqrt(c * log_d)


def rdp_sigma_for(eps, k=STEPS_PER_RECORD):
    a, b = k * SENS**2 / 2, 2 * math.sqrt(k * SENS**2 / 2 * math.log(1 / DELTA))
    return 2 * a / (-b + math.sqrt(b * b + 4 * a * eps))


def clip_calls(ref):
    calls, real = [0], ref.clip
    ref.clip = lambda g, c: (calls.__setitem__(0, calls[0] + 1), real(g, c))[1]
    try:
        seeded(ref, 0, lambda: ref.dp_sgd(ref.gen(500), 10, 0.05, 1.0, 1.0))
    finally:
        ref.clip = real
    return calls[0]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows, text = shipped(ref)
    proxy10 = round(ref.analytical_epsilon(1.0, steps=5000, delta=DELTA) / 10, 2)
    rdp10 = round(rdp_sigma_for(10.0), 2)
    grid = (0.0, 0.5, 1.0, 2.0, 4.0, rdp10, 8.0, 16.0, proxy10)
    mean = {s: sweep(ref, s) for s in grid}
    return {
        "rows": rows, "takeaway_1_10": "epsilon in [1, 10]" in text, "mean": mean,
        "collapse": next(s for s in grid if mean[s][0] < 0.75), "proxy10": proxy10, "rdp10": rdp10,
        "proxy_at_16": round(ref.analytical_epsilon(16.0, steps=5000, delta=DELTA), 1),
        "rdp": {s: round(rdp_epsilon(s), 2) for s in (0.5, 1.0, 2.0)},
        "clip_calls": clip_calls(ref),
    }


def verify(r):
    rows, mean = r["rows"], r["mean"]
    acc = {s: m[0] for s, m in mean.items()}
    return [
        practice.Check(
            "ANSWER: over {0.5, 1, 2} utility does not collapse; epsilon is 685, 343, 171",
            (rows, rows[4.0][1] > rows[2.0][1], [acc[s] for s in (0.0, 0.5, 1.0, 2.0, 4.0)],
             r["collapse"], acc[16.0], acc[8.0], r["proxy_at_16"])
            == ({0.0: (34257.95, 0.995), 0.5: (685.16, 0.955), 1.0: (342.58, 0.98),
                 2.0: (171.29, 0.86), 4.0: (85.64, 0.935)}, True,
                [0.992, 0.978, 0.956, 0.915, 0.855], 16.0, 0.669, 0.781, 21.4),
            f"shipped (eps, acc) {rows}; 50-seed mean accuracy {acc}; first below 0.75 at "
            f"sigma = {r['collapse']} (proxy eps {r['proxy_at_16']})",
        ),
        practice.Check(
            "FINDING: the proxy never reaches the 'epsilon in [1, 10]' the takeaway quotes",
            (r["takeaway_1_10"], rows[0.0][0], r["proxy10"], mean[34.26])
            == (True, 34257.95, 34.26, (0.561, 6)),
            f"sigma = 0 prints eps {rows[0.0][0]}; proxy eps = 10 needs sigma = {r['proxy10']}: "
            f"(mean accuracy, overflow crashes of 50) = {mean[r['proxy10']]}",
        ),
        practice.Check(
            "FINDING: the proxy charges 5000 steps, but each record is in only 10 of them",
            (r["clip_calls"], r["rdp"], r["rdp10"], mean[5.08])
            == (5000, {0.5: 245.84, 1.0: 82.92, 2.0: 31.46}, 5.08, (0.833, 0)),
            f"{r['clip_calls']} clip calls for 500 records; RDP eps {r['rdp']}; "
            f"eps = 10 at sigma = {r['rdp10']} with mean accuracy {acc[r['rdp10']]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
