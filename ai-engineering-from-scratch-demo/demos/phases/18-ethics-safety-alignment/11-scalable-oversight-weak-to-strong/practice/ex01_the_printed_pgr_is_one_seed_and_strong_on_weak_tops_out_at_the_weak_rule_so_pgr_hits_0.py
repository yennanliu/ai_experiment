"""Exercise 1 — the printed PGR is one seed; strong-on-weak tops out at the weak rule, so PGR hits 0.

    Run `code/main.py`. Report PGR for weak_accuracy = 0.60, 0.70, 0.80. Explain
    the shape of the PGR curve.

Reading of the exercise: "report" is the shipped run, read off `main()` under
the file's own `random.seed(29)`; "explain the shape" needs more than one
draw, so the same procedure (`gen`, `weak_label`, `train_strong`, `accuracy`)
is repeated over six seeds, with weak_accuracy = 1.0 added as the end point
that shows where the curve goes.

**ANSWER: PGR = 0.365, 0.295 and 0.544 at weak_accuracy 0.60, 0.70, 0.80**
(and 0.191 at 0.90, which the script also prints). Over six seeds the means
are 0.067, 0.248 and 0.213, and -0.002 at 1.0. The seed spread at 0.70 runs
from 0.018 to 0.620, wider than any difference between the levels, and the
shipped 0.544 at 0.80 is above all six seeds (max 0.328). The shipped curve's
zig-zag is one seed's SGD noise.

**FINDING: weak_accuracy is not the weak labeler's accuracy.** `weak_label`
keeps the rule `x[0] > 0` with probability weak_accuracy, and that rule agrees
with the gold rule `x0 + x1 - 0.5 x2 > 0` only 1 - acos(2/3)/pi = 73.2% of the
time (0.738 measured). So the labeler is right with probability
0.268 + 0.464 a, which is 0.546 / 0.593 / 0.639 at 0.60 / 0.70 / 0.80. The
shipped run prints 0.542 / 0.607 / 0.637, and the six-seed means are
0.556 / 0.601 / 0.645.

**FINDING: the ceiling is 99.7%, not the 95% the lesson states.** The gold
labels are an exact linear rule and the strong model is a linear classifier,
so the six-seed mean ceiling is 0.997. The shipped run prints 0.993 to 1.000.

**FINDING: the strong model learns the weak labeler's rule, not the truth.**
The share of weight on x0 rises from 0.564 to 0.989 as weak_accuracy goes
from 0.60 to 1.0. The mean strong-on-weak accuracy (0.586, 0.700, 0.720,
0.737) stays under the rule's 0.738 at every level. The recovered gap is
label noise averaged away, so PGR is capped by (rule - weak) / (ceiling -
weak): 0.413, 0.346, 0.264, 0. At weak_accuracy 1.0 there is no noise left to
average, and PGR is -0.002 (every seed within +-0.012).

**The shape:** a hump that falls to zero. At 0.60 the labels are so noisy
that constant-step SGD cannot even recover the x0 rule, so the mean PGR is
0.067. At higher accuracy the model reaches the rule, and the cap shrinks as
the labeler approaches it. The TAKEAWAY's "generalizes beyond its weak
supervisor's mistakes, using its own pre-trained priors" describes nothing
the simulator does. The strong model starts from zero weights and corrects
only the random flips, never the systematic mistake.

Structure: `shipped()` runs and parses `main()`; `one_seed()` runs the
procedure of `run()` with one gold ceiling shared across the levels;
`seeded()` swaps the reference's module `random` and restores it.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "11-scalable-oversight-weak-to-strong"
ACCS, SEEDS = (0.6, 0.7, 0.8, 1.0), range(6)
SPREAD = {0.6: (-0.255, 0.238), 0.7: (0.018, 0.62), 0.8: (0.14, 0.328), 1.0: (-0.012, 0.008)}
BLOCK = r"weak_accuracy=([\d.]+)\)\n.*?: ([\d.]+)\n.*?: ([\d.]+)\n.*?: ([\d.]+)\n.*?: (-?[\d.]+)"


@contextlib.contextmanager
def seeded(ref, seed):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        yield
    finally:
        ref.random = saved


def shipped(ref):
    """The printed run: main() under the file's own seed 29, parsed per weak_accuracy."""
    log = io.StringIO()
    with seeded(ref, 29), contextlib.redirect_stdout(log):
        ref.main()
    return {float(a): tuple(map(float, v)) for a, *v in re.findall(BLOCK, log.getvalue())}


def one_seed(ref, seed):
    """run()'s procedure, one gold ceiling shared across every weak_accuracy."""
    with seeded(ref, seed):
        ev, tr = ref.gen(1000), ref.gen(1000)
        ceiling = ref.accuracy(ref.train_strong(list(tr)), ev)
        rule = sum((x[0] > 0) == y for x, y in ev) / len(ev)
        rows = {}
        for acc in ACCS:
            weak = sum(ref.weak_label(x, acc) == y for x, y in ev) / len(ev)
            model = ref.train_strong([(x, ref.weak_label(x, acc)) for x, _ in tr])
            w2s = ref.accuracy(model, ev)
            share = abs(model[0]) / sum(abs(v) for v in model[:3])
            rows[acc] = (weak, ceiling, w2s, (w2s - weak) / (ceiling - weak), share, rule)
    return rows


def sweep(ref):
    runs = [one_seed(ref, s) for s in SEEDS]
    mean = {a: [round(sum(r[a][i] for r in runs) / len(runs), 3) for i in range(6)]
            for a in ACCS}
    spread = {a: (round(min(r[a][3] for r in runs), 3), round(max(r[a][3] for r in runs), 3))
              for a in ACCS}
    return mean, spread


def derived(table, ship, exp):
    col = {i: [table[a][i] for a in ACCS] for i in range(6)}
    s = {i: [ship[a][i] for a in sorted(ship)] for i in range(4)}
    return {
        "col": col, "ship_pgr": s[3], "ship_weak": s[0], "ship_ceil": s[1],
        "cap": [round((table[a][5] - table[a][0]) / (table[a][1] - table[a][0]), 3) for a in ACCS],
        "fit": round(max(abs(table[a][0] - exp[a]) for a in ACCS), 3),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    table, spread = sweep(ref)
    agree = 1 - math.acos(2 / 3) / math.pi   # P(sign x0 = sign(x0 + x1 - 0.5 x2))
    ship, exp = shipped(ref), {a: round(a * agree + (1 - a) * (1 - agree), 3) for a in ACCS}
    return {
        "shipped": ship, "table": table, "spread": spread, "agree": round(agree, 3),
        "expected_weak": exp, **derived(table, ship, exp),
        "rule_in_source": ("x[0] > 0" in inspect.getsource(ref.weak_label),
                           "x[0] + x[1] - 0.5 * x[2] > 0" in inspect.getsource(ref.gen)),
        "doc_ceiling": "95% ceiling" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    col, spread, cap = result["col"], result["spread"], result["cap"]
    return [
        practice.Check(
            "ANSWER: PGR = 0.365, 0.295 and 0.544 at weak_accuracy 0.60, 0.70, 0.80",
            all([result["ship_pgr"] == [0.365, 0.295, 0.544, 0.191], spread == SPREAD,
                 col[3] == [0.067, 0.248, 0.213, -0.002], result["ship_pgr"][2] > spread[0.8][1]]),
            f"shipped (weak, ceiling, w2s, PGR) {result['shipped']}; seed means {col[3]}, {spread}",
        ),
        practice.Check(
            "FINDING: weak_accuracy is not the weak labeler's accuracy",
            all([result["rule_in_source"] == (True, True), result["agree"] == 0.732,
                 col[5][0] == 0.738, result["ship_weak"][:3] == [0.542, 0.607, 0.637],
                 col[0][:3] == [0.556, 0.601, 0.645], result["fit"] <= 0.01,
                 list(result["expected_weak"].values()) == [0.546, 0.593, 0.639, 0.732]]),
            f"rule agrees {result['agree']} ({col[5][0]}); weak {result['expected_weak']} vs {col[0]}",
        ),
        practice.Check(
            "FINDING: the ceiling is 99.7%, not the 95% the lesson states",
            all([result["doc_ceiling"], col[1][0] == 0.997, result["ship_ceil"] == [1.0, 0.993, 0.997, 0.997]]),
            f"six-seed ceiling {col[1][0]}; shipped ceilings {result['ship_ceil']}",
        ),
        practice.Check(
            "FINDING: the strong model learns the weak labeler's rule, not the truth",
            all([col[4] == [0.564, 0.723, 0.875, 0.989], col[2] == [0.586, 0.7, 0.72, 0.737],
                 max(w - r for w, r in zip(col[2], col[5])) <= 0, cap == [0.413, 0.346, 0.264, 0.0],
                 spread[1.0] == (-0.012, 0.008)]),
            f"x0 weight share {col[4]}; w2s {col[2]} vs rule {col[5]}; PGR cap {cap}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
