"""Exercise 4 — zero-ablating the gender units raises bias 51% and moves a pure correlate 12%; only interchange separates them.

    Yu & Ananiadou 2025 identify gender neurons. Sketch a falsification
    experiment that would distinguish "these neurons cause gender bias" from
    "these neurons correlate with gender bias."

Reading of the exercise: the sketch is built and run on the smallest system
that has the same two-step shape -- select units by correlation with
gender, then intervene on them and watch a bias metric. The units are the
coordinates of the six identity words in the lesson's embedding (masculine,
feminine, tech, care), plus one planted unit that tracks gender perfectly
(+0.5 on he/his/man, -0.5 on she/her/woman) and that no attribute word
loads on. The bias metric is the reference `weat_score`. Three standard
interventions are compared: zero-ablation, mean-ablation, and interchange
(set each word's units to its opposite-gender counterpart's values: he<->she,
his<->her, man<->woman).

**ANSWER: the experiment is a pre-registered interchange intervention with a
dose-response and a pure-correlate control, and ablation cannot stand in for
it.** Select units by correlation on one split; on held-out inputs, patch the
selected units from counterfactual-gender inputs. "Cause" predicts the
metric moves in proportion to the patch dose; "correlate" predicts it does
not move. Run a matched control -- a unit as correlated but not read -- that
must come out null, and report the capability delta. In the toy, the units
the correlation step selects (masculine, feminine; |r| = 0.998) pass:
interchanging them takes the score from +0.8906 to -0.4618, falling
monotonically through +0.6928, +0.2985, -0.1554 at quarter doses.

**FINDING: zero-ablation gives the wrong sign.** Zeroing the two selected
units *raises* the score to +1.3416 (+51%), which reads as "these units
suppress bias". Mean-ablation lowers it to +0.2987 (-66%). Same units, same
metric, three verdicts: zeroing leaves the identity words as pure tech/care
vectors, off anything the metric saw.

**FINDING: ablation manufactures an effect for a unit nothing reads.** The
planted unit has |r| = 1.000 with gender and no attribute word loads on it,
yet zero- or mean-ablating it moves the score from +0.7929 to +0.8906
(+12.3%), because cosine divides by the vector norm -- the same role layer
norm plays in a transformer. Interchanging it moves the score by 0.0000.
Only interchange returns the null a pure correlate should give.

**FINDING: correlation does not rank causal share.** The tech and care units
correlate less (|r| = 0.728 and 0.816) but interchanging them cuts the
score from +0.8906 to +0.4618 (-48%), and mean-ablating them to +0.6797
(-24%). A search that stops at the top-correlated units misses causes.

Structure: `intervene()` applies zero/mean/interchange/dose to a set of
units of the identity words; every score comes from the reference
`weat_score` on the swapped-in embedding.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "20-bias-representational-harm"
A, B = ["he", "his", "man"], ["she", "her", "woman"]
X, Y = ["engineer", "programmer", "scientist"], ["nurse", "teacher", "caregiver"]
DOSES, PLANTED = (0.25, 0.5, 0.75, 1.0), 0.5


def weat(ref, emb):
    saved, ref.EMB = ref.EMB, emb
    try:
        return round(ref.weat_score(A, B, X, Y), 4)
    finally:
        ref.EMB = saved


def corr(emb, unit):
    """|Pearson r| between one unit and the gender label over the six identity words."""
    xs, ys = [emb[w][unit] for w in A + B], [1.0] * len(A) + [0.0] * len(B)
    return round(abs(statistics.correlation(xs, ys)), 3)


def intervene(emb, units, how, dose=1.0):
    new = {k: list(v) for k, v in emb.items()}
    mean = {u: statistics.mean(emb[w][u] for w in A + B) for u in units}
    for a, b in zip(A, B):
        for u in units:
            if how == "swap":
                new[a][u] = (1 - dose) * emb[a][u] + dose * emb[b][u]
                new[b][u] = (1 - dose) * emb[b][u] + dose * emb[a][u]
            else:
                new[a][u] = new[b][u] = 0.0 if how == "zero" else mean[u]
    return new


def effects(ref, emb, units):
    return {how: weat(ref, intervene(emb, units, how)) for how in ("zero", "mean", "swap")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base = {k: list(v) for k, v in ref.EMB.items()}
    planted = {k: v + [PLANTED if k in A else -PLANTED if k in B else 0.0] for k, v in base.items()}
    return {
        "base": weat(ref, base), "corr": [corr(base, u) for u in range(4)],
        "gender": effects(ref, base, [0, 1]), "stereo": effects(ref, base, [2, 3]),
        "dose": [weat(ref, intervene(base, [0, 1], "swap", d)) for d in DOSES],
        "planted_corr": corr(planted, 4), "planted_base": weat(ref, planted),
        "planted": effects(ref, planted, [4]),
    }


def verify(result):
    g, s, p, dose = result["gender"], result["stereo"], result["planted"], result["dose"]
    return [
        practice.Check(
            "ANSWER: interchange on the selected units moves the score monotonically with dose",
            (result["corr"][:2], result["base"], dose, g["swap"])
            == ([0.998, 0.998], 0.8906, [0.6928, 0.2985, -0.1554, -0.4618], dose[-1])
            and dose == sorted(dose, reverse=True),
            f"|r| by unit {result['corr']}; interchange dose 0.25..1 gives {dose} from {result['base']}",
        ),
        practice.Check(
            "FINDING: zero-ablation gives the wrong sign",
            (g["zero"], g["mean"]) == (1.3416, 0.2987)
            and round(g["zero"] / result["base"] - 1, 3) == 0.506,
            f"masculine+feminine units: zero {g['zero']}, mean {g['mean']}, interchange {g['swap']}",
        ),
        practice.Check(
            "FINDING: ablation manufactures an effect for a unit nothing reads",
            (result["planted_corr"], result["planted_base"], p["zero"], p["mean"], p["swap"],
             round(p["zero"] / result["planted_base"] - 1, 3))
            == (1.0, 0.7929, 0.8906, 0.8906, 0.7929, 0.123),
            f"planted unit |r| = {result['planted_corr']}: {result['planted_base']} -> zero "
            f"{p['zero']}, mean {p['mean']}, interchange {p['swap']}",
        ),
        practice.Check(
            "FINDING: correlation does not rank causal share",
            result["corr"][2:] == [0.728, 0.816] and (s["swap"], s["mean"]) == (0.4618, 0.6797),
            f"tech+care units (|r| {result['corr'][2:]}): interchange {s['swap']}, mean {s['mean']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
