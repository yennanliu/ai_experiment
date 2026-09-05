"""Exercise 3 — triplet loss with semi-hard negative mining, against random negatives.

    Implement triplet loss with semi-hard negative mining. Generate 2D embedding
    data for 5 classes. For each anchor, find the hardest negative that is still
    farther than the positive (semi-hard). Compare convergence to random triplet
    selection.

Reading of the exercise: with 2D embeddings as the parameters there is no encoder, so both
rules eventually solve the problem and "compare convergence" has to mean the path, not the
outcome — checks 1-3 measure it in the two units that disagree. The semi-hard rule is taken
exactly as worded, "hardest negative still farther than the positive", and check 4 asks what
its qualifier costs. Check 5 reads the result back through the lesson's own scale-invariant
contrastive loss, so a bigger embedding cannot pass for a better one.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "05-loss-functions"
CLASSES, PER, SPREAD, MARGIN, LR = 5, 20, 2.0, 1.0, 0.05
EPOCHS, SEEDS, SHUFFLE = 150, (3, 4, 5), 7
d2 = lambda a, b: (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2      # noqa: E731
nearest = lambda pts, i: min((k for k in range(len(pts)) if k != i),           # noqa: E731
                             key=lambda k: d2(pts[i], pts[k]))


def embed(seed) -> tuple:
    """Five Gaussian blobs on a circle — 2D embeddings that are the parameters themselves."""
    rng = random.Random(seed)
    ring = [(math.cos(2 * math.pi * c / CLASSES) * 2, math.sin(2 * math.pi * c / CLASSES) * 2)
            for c in range(CLASSES)]
    points = [[ring[c][0] + rng.gauss(0, SPREAD), ring[c][1] + rng.gauss(0, SPREAD)]
              for c in range(CLASSES) for _ in range(PER)]
    return points, [c for c in range(CLASSES) for _ in range(PER)]


def pick(points, anchor, positive, negatives, rng, mode) -> int | None:
    """The negative each rule selects — None when the semi-hard set is empty."""
    if mode == "random":
        return rng.choice(negatives)
    apart = d2(points[anchor], points[positive])
    farther = [i for i in negatives if d2(points[anchor], points[i]) > apart]
    return min(farther, key=lambda i: d2(points[anchor], points[i])) if farther else None


def step(points, anchor, positive, negative) -> None:
    a, p, n = points[anchor], points[positive], points[negative]
    for j in (0, 1):
        a[j] -= LR * 2 * (n[j] - p[j])
        p[j] -= LR * -2 * (a[j] - p[j])
        n[j] -= LR * 2 * (a[j] - n[j])


def visit(points, labels, anchor, rng, mode) -> int:
    """One anchor: 1 if a violating triplet was stepped on, 0 if none, -1 if none was offered."""
    n = len(points)
    same = [i for i in range(n) if labels[i] == labels[anchor] and i != anchor]
    other = [i for i in range(n) if labels[i] != labels[anchor]]
    positive = rng.choice(same)
    negative = pick(points, anchor, positive, other, rng, mode)
    if negative is None:
        return -1
    if d2(points[anchor], points[positive]) - d2(points[anchor], points[negative]) + MARGIN <= 0:
        return 0
    step(points, anchor, positive, negative)
    return 1


def train(mode, seed) -> dict:
    points, labels = embed(seed)
    rng, n = random.Random(SHUFFLE), CLASSES * PER
    updates, empty, first, epoch_one = 0, 0, None, 0
    for epoch in range(EPOCHS):
        seen = [visit(points, labels, a, rng, mode) for a in rng.sample(range(n), n)]
        moved, updates, empty = seen.count(1), updates + seen.count(1), empty + seen.count(-1)
        epoch_one = moved if epoch == 0 else epoch_one
        if moved == 0 and first is None:
            first = epoch + 1
    return {"first": first, "updates": updates, "empty": empty, "epoch_one": epoch_one,
            "points": points, "labels": labels}


def score(ref, points, labels) -> dict:
    """Two read-outs neither rule optimises: 1-NN agreement, and the lesson's cosine InfoNCE."""
    n, rng, total = len(points), random.Random(2), 0.0
    near = sum(labels[nearest(points, i)] == labels[i] for i in range(n))
    for _ in range(200):
        a = rng.randrange(n)
        same = [i for i in range(n) if labels[i] == labels[a] and i != a]
        other = [i for i in range(n) if labels[i] != labels[a]]
        total += ref.contrastive_loss(points[a], points[rng.choice(same)],
                                      [points[i] for i in rng.sample(other, 8)])
    return {"knn": near / n, "contrastive": total / 200,
            "norm": statistics.mean(math.hypot(*q) for q in points)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {(m, s): train(m, s) for m in ("semi", "random") for s in SEEDS}
    return {"runs": runs, "before": score(ref, *embed(SEEDS[0])),
            "after": {m: score(ref, runs[(m, SEEDS[0])]["points"],
                               runs[(m, SEEDS[0])]["labels"]) for m in ("semi", "random")}}


def digest(result) -> dict:
    """Every summary `verify` quotes, so that stays a list of comparisons."""
    runs = result["runs"]
    cols = {m + "_" + k: [runs[(m, s)][k] for s in SEEDS]
            for m in ("semi", "random") for k in ("first", "updates", "epoch_one", "empty")}
    return {**cols, "draws": EPOCHS * CLASSES * PER * len(SEEDS),
            "epochs": ", ".join(f"{r / s:.1f}x" for s, r in
                                zip(cols["semi_first"], cols["random_first"])),
            "work": ", ".join(f"{s / r:.1f}x" for s, r in
                              zip(cols["semi_updates"], cols["random_updates"]))}


def verify(result):
    d, after, before = digest(result), result["after"], result["before"]
    return [
        practice.Check("ANSWER: semi-hard mining converges in fewer epochs, at all three seeds",
                       all(s < r for s, r in zip(d["semi_first"], d["random_first"])),
                       f"epochs to the first with no violated triplet at all, seeds {SEEDS}: "
                       f"semi-hard {d['semi_first']} against random {d['random_first']} — "
                       f"{d['epochs']} fewer"),
        practice.Check("FINDING: and in more work, at all three seeds — the epoch is the unit "
                       "that flatters mining",
                       all(s > r for s, r in zip(d["semi_updates"], d["random_updates"])),
                       f"gradient steps actually taken: semi-hard {d['semi_updates']} against "
                       f"random {d['random_updates']} — {d['work']} more. Per step random is "
                       f"ahead; per epoch semi-hard is"),
        practice.Check("MECHANISM: mining raises the hit rate, not the gradient",
                       all(s > 1.5 * r for s, r in zip(d["semi_epoch_one"], d["random_epoch_one"])),
                       f"violating triplets found in the first epoch, out of {CLASSES * PER} "
                       f"anchors: semi-hard {d['semi_epoch_one']} against random "
                       f"{d['random_epoch_one']}. The hardest negative farther than the positive "
                       f"is by construction near the margin, so it almost always violates"),
        practice.Check("FINDING: the 'still farther than the positive' clause almost never binds",
                       sum(d["semi_empty"]) < 0.001 * d["draws"],
                       f"the semi-hard set is empty on {sum(d['semi_empty'])} of {d['draws']:,} "
                       f"anchor draws ({d['semi_empty']} per seed) — five blobs two units apart "
                       f"leave almost every negative farther than the positive. The wording says "
                       f"nothing about the case it does bind; this solution skips the anchor"),
        practice.Check("CONTROL: both rules end at the same place, and it is structure not scale",
                       after["semi"]["knn"] == after["random"]["knn"] == 1.0
                       and max(a["contrastive"] for a in after.values()) < before["contrastive"],
                       f"1-NN label agreement {before['knn']:.3f} -> {after['semi']['knn']:.3f} "
                       f"semi and {after['random']['knn']:.3f} random, from mean embedding norm "
                       f"{before['norm']:.2f} -> {after['semi']['norm']:.2f} and "
                       f"{after['random']['norm']:.2f}. The lesson's own cosine `contrastive_loss` "
                       f"— which no rescaling can move — falls {before['contrastive']:.3f} -> "
                       f"{after['semi']['contrastive']:.3f} and {after['random']['contrastive']:.3f}, "
                       f"so the margin is met by structure and not by inflating the embedding"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
