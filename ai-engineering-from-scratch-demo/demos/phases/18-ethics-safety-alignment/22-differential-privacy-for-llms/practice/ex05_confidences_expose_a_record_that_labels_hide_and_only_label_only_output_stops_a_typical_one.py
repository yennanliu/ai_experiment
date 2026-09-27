"""Exercise 5 — confidences expose a record that labels hide, and only label-only output stops a typical one.

    Sketch the DP Reversal via LLM Feedback attack. Design a countermeasure that limits confidence-score leakage and estimate its deployment cost.

Reading of the exercise: the attack is modelled as a confidence oracle on
the lesson's trainer. The attacker holds a target record (x, y), queries x,
and reads the model's confidence in y. Higher confidence suggests the record
was trained on. Success is the attack's AUC at telling models trained with
the record from models trained without it, over 100 seeded runs per arm on
the shipped seed-59 data; 0.5 is guessing. The countermeasure is to coarsen
what the API returns: full confidence, rounded to 2 decimals, rounded to 1
decimal, or the label only. Its deployment cost is the Brier score that
honest callers lose, averaged over the 100 sigma = 1 models on the 200
shipped test records. Rounding itself costs nothing to compute.

**ANSWER: the confidence oracle identifies a mislabeled record with AUC 1.0
when the model has no DP, and the label hides it completely.** At sigma = 0
the full confidence gives AUC 1.0. The record's label is never predicted, so
label-only output gives 0.5. At sigma = 1 the full-confidence AUC drops to
0.559. Rounding to 2 decimals already takes the outlier to 0.5, because
its confidence rounds to 0 with or without it. A typical record is harder
to hide: x = (0.3, 0.2) with its correct label gives AUC 0.63 (full), 0.609
(2 decimals) and 0.57 (1 decimal) at sigma = 0. Only
label-only output brings it to 0.5.

**FINDING: the countermeasure is cheap until it has to be label-only.** On
the sigma = 1 model the Brier score is 0.0278 with full confidences, 0.0278
rounded to 2 decimals, 0.0281 rounded to 1 decimal, and 0.0377 label-only,
36% worse. Two-decimal rounding costs nothing measurable at 4 decimals and
stops the outlier. Stopping the typical record costs 36% on Brier.

**FINDING: DP training is not the defence here; post-processing already
covers confidences.** Any function of an (epsilon, delta)-DP model is
equally DP, so confidences cannot leak more than epsilon allows. "DP
reversal" works only where epsilon is loose. At sigma = 1 this toy has
Renyi-accounted epsilon 82.92 (exercise 1), and the attack advantage that
permits is (e^eps - 1 + 2 delta) / (e^eps + 1) = 1.000. The guarantee rules
nothing out, so the output format has to.

Structure: `models()` trains seeded models; `auc()` scores the oracle with
ties counted half; `EXPOSE` maps each API format to what it returns.
"""

from __future__ import annotations

import inspect
import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "22-differential-privacy-for-llms"
OUTLIER, TYPICAL, RUNS = ((2.0, -2.0), 0), ((0.3, 0.2), 1), 100
EXPOSE = {"full": lambda p: p, "2dp": lambda p: round(p, 2), "1dp": lambda p: round(p, 1),
          "label": lambda p: float(p > 0.5)}


def prob(model, x):
    z = model[2] + model[0] * x[0] + model[1] * x[1]
    return 1 / (1 + math.exp(-max(-700.0, min(700.0, z))))


def models(ref, data, sigma, stream):
    saved, out = ref.random, []
    try:
        for i in range(RUNS):
            ref.random = random.Random(stream * 100_000 + i)
            out.append(ref.dp_sgd(list(data), 10, 0.05, sigma, 1.0))
    finally:
        ref.random = saved
    return out


def auc(member, nonmember, record, expose):
    (x, y), shown = record, lambda m: expose(prob(m, x))
    score = lambda m: shown(m) if y == 1 else 1 - shown(m)  # noqa: E731
    si, so = [score(m) for m in member], [score(m) for m in nonmember]
    wins = sum((a > b) + 0.5 * (a == b) for a in si for b in so)
    return round(wins / len(si) / len(so), 3)


def brier(deployed, test, expose):
    total = sum((expose(prob(m, x)) - y) ** 2 for m in deployed for x, y in test)
    return round(total / len(deployed) / len(test), 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    saved, ref.random = ref.random, random.Random(59)
    try:
        base, test = ref.gen(500), ref.gen(200)
    finally:
        ref.random = saved
    out = {s: models(ref, base, s, 2) for s in (0.0, 1.0)}
    cases = {"outlier@0": (OUTLIER, 0.0), "outlier@1": (OUTLIER, 1.0), "typical@0": (TYPICAL, 0.0)}
    table = {}
    for name, (record, sigma) in cases.items():
        member = models(ref, [(list(record[0]), record[1])] + base[1:], sigma, 1)
        table[name] = {k: auc(member, out[sigma], record, f) for k, f in EXPOSE.items()}
    src = inspect.getsource(ref.main)          # the shipped epochs and delta
    epochs = int(re.search(r"epochs = (\d+)", src)[1])
    delta = float(re.search(r"delta = ([\d.e-]+)", src)[1])
    c = epochs * (2 * math.sqrt(2)) ** 2 / 2    # exercise 1's Renyi accountant at sigma = 1
    eps = c + 2 * math.sqrt(c * math.log(1 / delta))
    bound = 1 - (2 - 2 * delta) / (math.exp(min(eps, 700)) + 1)
    return {"auc": table, "brier": {k: brier(out[1.0], test, f) for k, f in EXPOSE.items()},
            "eps": round(eps, 2), "bound": round(bound, 3), "shipped": (epochs, delta)}


def verify(r):
    auc_, cost = r["auc"], r["brier"]
    return [
        practice.Check(
            "ANSWER: confidences identify a mislabeled record at AUC 1.0; labels hide it",
            auc_["outlier@0"] == {"full": 1.0, "2dp": 0.5, "1dp": 0.5, "label": 0.5}
            and auc_["outlier@1"] == {"full": 0.559, "2dp": 0.5, "1dp": 0.5, "label": 0.5}
            and auc_["typical@0"] == {"full": 0.63, "2dp": 0.609, "1dp": 0.57, "label": 0.5},
            f"attack AUC by exposure {auc_}",
        ),
        practice.Check(
            "FINDING: the countermeasure is cheap until it has to be label-only",
            cost == {"full": 0.0278, "2dp": 0.0278, "1dp": 0.0281, "label": 0.0377}
            and round(cost["label"] / cost["full"] - 1, 2) == 0.36,
            f"Brier by exposure on the sigma = 1 model {cost}",
        ),
        practice.Check(
            "FINDING: DP training is not the defence here; post-processing covers confidences",
            r["shipped"] == (10, 1e-5) and r["eps"] == 82.92 and r["bound"] == 1.0,
            f"sigma = 1 has RDP epsilon {r['eps']}; permitted attack advantage {r['bound']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
