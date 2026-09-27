"""Exercise 5 — a disability probe survives the gender debias bit for bit, and needs 5 terms a side: the lesson's 3 floor p at 0.05.

    The meta-critique argues the field focuses too narrowly on binary gender.
    Pick one under-studied axis and describe a representational-harm
    measurement protocol for it.

Reading of the exercise: the axis is disability. The protocol is described
below and its embedding layer is run on the lesson's own machinery: the
lesson's 4-d embedding gains a disability axis, a person axis, and
competence / dependence attribute axes, and a planted pity-and-dependence
stereotype (dependence loading 0.30-0.50 on the disability terms, none on
their matched controls, which instead
carry a small competence loading of 0-0.20) is measured with the reference `weat_score`,
`cos` and `debias`. The planted numbers are the toy's; what the run tests
is whether the protocol and the lesson's tools can see them.

**ANSWER: the protocol.** (1) Terms: five or more disability terms in both
person-first and identity-first forms, each paired with a matched
nondisabled control ("wheelchair user" / "walking person"), and attribute
sets for the documented stereotypes -- competence vs dependence/pity, and
agency vs burden (Hutchinson et al. 2020). (2) Measure all three Gallegos
categories: an embedding WEAT with an exact permutation test and Caliskan's
d; a probability test on minimal sentence pairs that swap only the
disability term; a generated-text test coding open-ended descriptions for
pity framing, inspiration framing and erasure (the disability never named,
or named only as a deficit). (3) Report intersections with gender. (4)
Re-run after every debias step, gender debiasing included. On the toy, the
five-a-side probe reads a WEAT of +0.3701 and d = 1.81, exact p = 1/252 =
0.004.

**FINDING: the lesson's gender debias leaves the disability score bit for
bit.** `debias()` rewrites exactly its six hard-coded attribute words; in
the same embedding the gender score goes +0.8906 -> +0.2212 while the
disability score stays +0.3701 to the last bit. This is the meta-critique
measured: a model "debiased" on binary gender, with an unchanged bias on
the axis nobody checked.

**FINDING: three terms a side cannot reach p < 0.05.** Six targets split
3/3 have C(6, 3) = 20 partitions, so the smallest exact one-sided p is
1/20. The lesson's own gender probe hits that floor, p = 0.05 both before
and after debias, so its test cannot tell the debiased embedding from the
original. The first three disability pairs alone also give 0.05; five a
side give 0.004. The term count is part of the protocol, not a detail.

Structure: `extend()` builds the 8-d embedding; `permutation_p()` enumerates
every equal split of the pooled targets using the reference `cos`.
"""

from __future__ import annotations

import itertools
import math
import statistics

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "20-bias-representational-harm"
A, B = ["he", "his", "man"], ["she", "her", "woman"]
X, Y = ["engineer", "programmer", "scientist"], ["nurse", "teacher", "caregiver"]
CONTROL = ["nondisabled person", "walking person", "sighted person", "hearing person",
           "neurotypical person"]
DISABLED = ["disabled person", "wheelchair user", "blind person", "deaf person",
            "autistic person"]
COMPETENT = ["capable", "independent", "skilled", "competent", "leader"]
DEPENDENT = ["dependent", "helpless", "burden", "pitiable", "fragile"]
DIS, DEP, COMP, PERSON = 4, 5, 6, 7                 # the four added axes


def axis(i, value=1.0, dim=8):
    return [value if j == i else 0.0 for j in range(dim)]


def extend(ref):
    emb = {k: list(v) + [0.0] * 4 for k, v in ref.EMB.items()}
    for i, (c, d) in enumerate(zip(CONTROL, DISABLED)):
        emb[c] = [x + y for x, y in zip(axis(PERSON), axis(COMP, 0.05 * i))]
        emb[d] = [x + y + z for x, y, z in zip(axis(PERSON), axis(DIS), axis(DEP, 0.3 + 0.05 * i))]
    emb.update({w: axis(COMP) for w in COMPETENT})
    emb.update({w: axis(DEP) for w in DEPENDENT})
    return emb


def assoc(ref, emb, w, x, y):
    return (statistics.mean(ref.cos(emb[w], emb[a]) for a in x)
            - statistics.mean(ref.cos(emb[w], emb[b]) for b in y))


def permutation_p(ref, emb, a, b, x, y):
    """Exact one-sided p: share of equal splits of a+b at least as extreme as observed."""
    s = {w: assoc(ref, emb, w, x, y) for w in a + b}
    observed, pool = sum(s[w] for w in a) - sum(s[w] for w in b), a + b
    stats = [2 * sum(s[w] for w in half) - sum(s.values())
             for half in itertools.combinations(pool, len(a))]
    return sum(v >= observed - 1e-12 for v in stats) / math.comb(len(pool), len(a))


def cohen_d(ref, emb, a, b, x, y):
    sa, sb = [assoc(ref, emb, w, x, y) for w in a], [assoc(ref, emb, w, x, y) for w in b]
    return round((statistics.mean(sa) - statistics.mean(sb)) / statistics.stdev(sa + sb), 2)


def weat(ref, emb, a, b, x, y):
    saved, ref.EMB = ref.EMB, emb
    try:
        return round(ref.weat_score(a, b, x, y), 4)
    finally:
        ref.EMB = saved


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    emb = extend(ref)
    post = ref.debias(emb)
    probe = (CONTROL, DISABLED, COMPETENT, DEPENDENT)
    small = (CONTROL[:3], DISABLED[:3], COMPETENT, DEPENDENT)
    return {
        "dis": (weat(ref, emb, *probe), weat(ref, post, *probe)),
        "gender": (weat(ref, emb, A, B, X, Y), weat(ref, post, A, B, X, Y)),
        "changed": sorted(w for w in emb if post[w] != emb[w]),
        "d": cohen_d(ref, emb, *probe),
        "p5": round(permutation_p(ref, emb, *probe), 4),
        "p3": permutation_p(ref, emb, *small),
        "p_gender": (permutation_p(ref, emb, A, B, X, Y), permutation_p(ref, post, A, B, X, Y)),
    }


def verify(result):
    (dis_pre, dis_post), (g_pre, g_post) = result["dis"], result["gender"]
    return [
        practice.Check(
            "ANSWER: the five-a-side probe reads WEAT +0.3701, d = 1.81, exact p = 1/252",
            (dis_pre, result["d"], result["p5"]) == (0.3701, 1.81, round(1 / 252, 4)),
            f"disability WEAT {dis_pre}, Cohen's d {result['d']}, exact p {result['p5']}",
        ),
        practice.Check(
            "FINDING: the lesson's gender debias leaves the disability score bit for bit",
            result["changed"] == sorted(X + Y) and dis_post == dis_pre
            and (g_pre, g_post) == (0.8906, 0.2212),
            f"debias changed {result['changed']}; gender {g_pre} -> {g_post}, disability "
            f"{dis_pre} -> {dis_post}",
        ),
        practice.Check(
            "FINDING: three terms a side cannot reach p < 0.05",
            result["p_gender"] == (0.05, 0.05) and result["p3"] == 0.05,
            f"lesson's 3/3 gender probe p {result['p_gender']} before/after debias; first three "
            f"disability pairs p {result['p3']}; five pairs p {result['p5']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
