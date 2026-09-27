"""Exercise 1 — DP reweighting cuts the gap from 0.429 to 0.012 and narrows equalized odds too; only conditional use pays.

    Run `code/main.py`. Report the three group metrics on the default data.
    Apply the demographic-parity-targeted re-weighting and re-report.

Reading of the exercise: "report" means the numbers `main()` prints on its
shipped seed (53), read back from its stdout; "re-report" is the second block
it prints after the 2.0 / 0.5 reweighting. Because one seed is one draw, the
same two trainings are then repeated on seeds 0-9 to see which of the printed
conclusions survive the draw.

**ANSWER: baseline gaps (group 1 minus group 0) are DP +0.429, TPR +0.326,
FPR +0.331, PPV +0.157, NPV -0.256; after reweighting DP +0.012, TPR -0.231,
FPR -0.007, PPV +0.244, NPV -0.401.** Accuracy on the 500-row test set falls
from 71.0% to 65.8%. The baseline puts weight +1.35 on the sensitive attribute
itself, which is fed in as the third feature; the reweighted model puts -0.05
on it.

**FINDING: the printed takeaway is half wrong -- equalized odds improves.**
`main()` says reweighting reduces the DP gap "at the cost of equalized odds
and conditional use accuracy". On the shipped seed the worst equalized-odds gap
narrows from 0.331 to 0.231 (FPR almost closes, TPR flips sign); the worst
conditional-use gap widens from 0.256 to 0.401. Over seeds 0-9 the pattern
holds in 9 of 10 for each: equalized odds narrows, conditional use widens.
The reweighting mainly strips the model's weight on A, and the qualification
feature x0 depends on A only through Y, so the model drifts toward equalized
odds; the price is paid in predictive value, which is what the impossibility
result says unequal base rates force.

**FINDING: the headline 0.429 is one draw of a noisy trainer.** Over seeds 0-9
the baseline DP gap ranges from 0.048 to 0.648: 200 epochs of plain SGD at
lr 0.1 end on a noisy last iterate, so single-seed gaps are not the model's.

Structure: `trained()` runs the reference's gen/train/predict with a seeded
`random.Random` swapped into the module (restored after); `gaps()` reads the
reference's own three metric functions; `printed()` parses `main()`'s stdout.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "21-fairness-criteria-group-individual-counterfactual"
SEEDS, KEYS = range(10), ("dp", "tpr", "fpr", "ppv", "npv")


def dp_weights(data):
    """The reweighting main() applies, keyed on the same (a, y) cells."""
    return [{(0, 1): 2.0, (1, 1): 0.5}.get((a, y), 1.0) for _, y, a in data]


def trained(ref, seed):
    saved = ref.random
    ref.random = random.Random(seed)
    try:
        train, test = ref.gen(1000), ref.gen(500)
        return test, ref.train(train), ref.train(train, sample_weights=dp_weights(train))
    finally:
        ref.random = saved


def gaps(ref, model, test):
    preds = ref.predict(model, test)
    dp, eo, cu = ref.demographic_parity(preds), ref.equalized_odds(preds), ref.conditional_use(preds)
    pairs = (dp, (eo[0][0], eo[1][0]), (eo[0][1], eo[1][1]), (cu[0][0], cu[1][0]), (cu[0][1], cu[1][1]))
    out = {k: round(g1 - g0, 3) for k, (g0, g1) in zip(KEYS, pairs)}
    out["acc"] = round(sum(p == y for p, y, _ in preds) / len(preds), 3)
    return out


def worst(g, keys):
    return max(abs(g[k]) for k in keys)


def printed(ref):
    """main()'s own stdout on its shipped seed: (group0, group1) per metric line, per block."""
    saved, log = ref.random, io.StringIO()
    ref.random = random.Random(53)
    try:
        with contextlib.redirect_stdout(log):
            ref.main()
    finally:
        ref.random = saved
    pairs = re.findall(r"group0=([\d.]+)\s+group1=([\d.]+)", log.getvalue())
    return [(float(a), float(b)) for a, b in pairs]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    test, base, rew = trained(ref, 53)
    sweep = [(gaps(ref, b, t), gaps(ref, r, t)) for t, b, r in (trained(ref, s) for s in SEEDS)]
    return {
        "printed": printed(ref), "base": gaps(ref, base, test), "rew": gaps(ref, rew, test),
        "w_a": (round(base[2], 2), round(rew[2], 2)),
        "eo_narrows": sum(worst(r, ("tpr", "fpr")) < worst(b, ("tpr", "fpr")) for b, r in sweep),
        "cu_widens": sum(worst(r, ("ppv", "npv")) > worst(b, ("ppv", "npv")) for b, r in sweep),
        "dp_range": (min(b["dp"] for b, _ in sweep), max(b["dp"] for b, _ in sweep)),
        "takeaway": " ".join(inspect.getsource(ref.main).split()),
    }


def matches_print(result):
    """Our gaps agree with the (group0, group1) rates main() printed, to its 3 decimals."""
    own = [result[m][k] for m in ("base", "rew") for k in KEYS]
    shown = [round(g1 - g0, 3) for g0, g1 in result["printed"]]
    return len(shown) == 10 and all(abs(o - g) <= 0.002 for o, g in zip(own, shown))


def verify(result):
    base, rew = result["base"], result["rew"]
    eo, cu = ("tpr", "fpr"), ("ppv", "npv")
    answer = ([base[k] for k in KEYS], [rew[k] for k in KEYS], base["acc"], rew["acc"], result["w_a"])
    shift = (worst(base, eo), worst(rew, eo), worst(base, cu), worst(rew, cu))
    return [
        practice.Check(
            "ANSWER: DP +0.429 -> +0.012; TPR +0.326 -> -0.231; FPR +0.331 -> -0.007",
            matches_print(result) and answer == ([0.429, 0.326, 0.331, 0.157, -0.256],
                                                 [0.012, -0.231, -0.007, 0.244, -0.401],
                                                 0.71, 0.658, (1.35, -0.05)),
            f"baseline {base}, reweighted {rew}; main() printed (g0, g1) {result['printed']}; "
            f"weight on A {result['w_a']}",
        ),
        practice.Check(
            "FINDING: the printed takeaway is half wrong -- equalized odds improves",
            "at the cost of equalized odds" in result["takeaway"]
            and shift == (0.331, 0.231, 0.256, 0.401)
            and (result["eo_narrows"], result["cu_widens"]) == (9, 9),
            f"worst EO gap {shift[0]} -> {shift[1]}, worst CU gap {shift[2]} -> {shift[3]}; over "
            f"{len(SEEDS)} seeds EO narrows in {result['eo_narrows']}, CU widens in {result['cu_widens']}",
        ),
        practice.Check(
            "FINDING: the headline 0.429 is one draw of a noisy trainer",
            result["dp_range"] == (0.048, 0.648),
            f"baseline DP gap over seeds 0-9 spans {result['dp_range']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
