"""Exercise 1 — debias cuts 0.891 to 0.221, and the residual is the identity words' own tech/care loading, not 4-d.

    Run `code/main.py`. Report WEAT-style bias scores before and after the
    debiasing step. Explain why the metric does not drop to zero.

Reading of the exercise: the scores are read off the shipped run's stdout.
"Why not zero" is answered by testing the reference's own explanation (its
TAKEAWAY says "because the toy is 4-d") against the alternative the vectors
suggest -- that the stereotype also lives in the identity words, which
`debias()` never touches -- by intervening on each and re-running
`weat_score`.

**ANSWER: +0.8906 before, +0.2212 after, a 75.2% cut.** Projecting the
direction [1, -1, 0, 0] out of the six attribute words leaves every one of
them with equal masculine and feminine loadings (engineer becomes
[0.2, 0.2, 1, 0]), so the attributes no longer carry gender at all.

**FINDING: dimension is not the reason.** Embed the debiased vectors in
300-d with a seeded random orthonormal basis and the post-debias score is
+0.2212 again, to 1e-9: cosine is invariant under any isometry, so adding
dimensions changes nothing. The TAKEAWAY's "does not drop to zero because
the toy is 4-d" is false.

**FINDING: the residual is exactly the identity words' tech/care loading.**
"he" carries 0.2 on the tech axis and "she" 0.2 on care; `debias()` edits
only the six attribute words. Zero axes 2-3 of the six identity words and
the post-debias score is 0.0000; keep them and it is +0.2212. The bias left
over sits on the gendered words' own occupational loading -- a direction
the projection does not remove, which is the Gonen & Goldberg (2019)
"lipstick on a pig" point in miniature.

**FINDING: the printed "effect size" is the raw statistic; the standardized
one barely moves.** Caliskan's d (difference of mean s(w) over the pooled
standard deviation) is 1.81 before and 1.60 after, an 11.5% cut. The
debias shrinks the scale of the association far more than its consistency.

Structure: `weat(ref, emb)` runs the reference `weat_score` on a swapped-in
embedding (restored after); `rotate()` is the 300-d isometry.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import math
import random
import re
import statistics

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "20-bias-representational-harm"
A, B = ["he", "his", "man"], ["she", "her", "woman"]
X, Y = ["engineer", "programmer", "scientist"], ["nurse", "teacher", "caregiver"]


def weat(ref, emb):
    saved, ref.EMB = ref.EMB, emb
    try:
        return ref.weat_score(A, B, X, Y)
    finally:
        ref.EMB = saved


def cohen_d(ref, emb):
    """Caliskan et al. 2017 effect size: mean s(A) - mean s(B) over the pooled sd."""
    def s(w):
        return (statistics.mean(ref.cos(emb[w], emb[x]) for x in X)
                - statistics.mean(ref.cos(emb[w], emb[y]) for y in Y))
    sa, sb = [s(w) for w in A], [s(w) for w in B]
    return (statistics.mean(sa) - statistics.mean(sb)) / statistics.stdev(sa + sb)


def orthonormal(rng, basis, dim):
    """One more unit vector orthogonal to `basis` (Gram-Schmidt on a Gaussian draw)."""
    v = [rng.gauss(0, 1) for _ in range(dim)]
    for b in basis:
        p = sum(a * c for a, c in zip(v, b))
        v = [a - p * c for a, c in zip(v, b)]
    n = math.sqrt(sum(a * a for a in v))
    return basis + [[a / n for a in v]]


def rotate(emb, dim=300, seed=0):
    """An isometric embedding of the 4-d vectors into `dim` dimensions."""
    rng, basis = random.Random(seed), []
    for _ in range(4):
        basis = orthonormal(rng, basis, dim)
    return {k: [sum(x[i] * basis[i][j] for i in range(4)) for j in range(dim)]
            for k, x in emb.items()}


def printed(ref):
    out, saved = io.StringIO(), ref.EMB
    try:
        with contextlib.redirect_stdout(out):
            ref.main()
    finally:
        ref.EMB = saved                       # main() rebinds the global to the debiased copy
    return [float(v) for v in re.findall(r"effect size\s*: ([+-][\d.]+)", out.getvalue())]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base = {k: list(v) for k, v in ref.EMB.items()}
    post = ref.debias(base)
    strip = {k: (v[:2] + [0.0, 0.0] if k in A + B else v) for k, v in post.items()}
    pre_s, post_s = weat(ref, base), weat(ref, post)
    return {
        "printed": printed(ref),
        "pre": round(pre_s, 4), "post": round(post_s, 4),
        "cut": round(1 - post_s / pre_s, 3),
        "engineer": [round(x, 3) for x in post["engineer"]],
        "balanced": all(abs(post[w][0] - post[w][1]) < 1e-12 for w in X + Y),
        "rotated_gap": abs(weat(ref, rotate(post)) - post_s),
        "stripped": round(weat(ref, strip), 4),
        "d": (round(cohen_d(ref, base), 2), round(cohen_d(ref, post), 2)),
        "d_cut": round(1 - cohen_d(ref, post) / cohen_d(ref, base), 3),
        "takeaway": "because the toy is 4-d" in inspect.getsource(ref.main),
    }


def verify(result):
    d_pre, d_post = result["d"]
    return [
        practice.Check(
            "ANSWER: +0.8906 before, +0.2212 after, a 75.2% cut",
            result["printed"] == [0.8906, 0.2212] and (result["pre"], result["post"])
            == (0.8906, 0.2212) and result["cut"] == 0.752 and result["balanced"]
            and result["engineer"] == [0.2, 0.2, 1.0, 0.0],
            f"stdout {result['printed']}; recomputed {result['pre']} -> {result['post']} "
            f"({result['cut']:.1%} cut); debiased engineer {result['engineer']}",
        ),
        practice.Check(
            "FINDING: dimension is not the reason",
            result["takeaway"] and result["rotated_gap"] < 1e-9,
            f"the TAKEAWAY blames 4-d, but a 300-d isometric embedding changes the post score by "
            f"{result['rotated_gap']:.1e}",
        ),
        practice.Check(
            "FINDING: the residual is exactly the identity words' tech/care loading",
            result["stripped"] == 0.0 and result["post"] > 0.2,
            f"zeroing axes 2-3 of he/his/man/she/her/woman takes the post-debias score from "
            f"{result['post']} to {result['stripped']}",
        ),
        practice.Check(
            "FINDING: the printed 'effect size' is the raw statistic; the standardized one barely moves",
            (d_pre, d_post) == (1.81, 1.6) and result["d_cut"] == 0.115,
            f"Cohen's d {d_pre} -> {d_post}, a {result['d_cut']:.1%} cut against the raw "
            f"statistic's {result['cut']:.1%}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
