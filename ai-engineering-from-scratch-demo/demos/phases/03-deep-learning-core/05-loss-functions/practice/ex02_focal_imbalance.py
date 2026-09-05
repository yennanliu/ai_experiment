"""Exercise 2 — focal loss at gamma=2 on a 9:1 split, scored against BCE's recall.

    Add focal loss to the binary classification training loop. Create an
    imbalanced dataset (90% class 0, 10% class 1). Compare standard BCE vs focal
    loss (gamma=2) on the minority class recall after 200 epochs.

Reading of the exercise: "add focal loss to the training loop" means the lesson's
own `LossComparisonNetwork.backward` runs unmodified — every update in it is linear
in `self.lr`, so scaling the rate by focal's gradient factor for one step *is* the
focal update, and the arms differ in exactly one number. Recall is the share of
positives over 0.5 after 200 epochs, pooled over three datasets; AUC rides along,
being the half of "recall" a moved threshold cannot fake.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "05-loss-functions"
GAMMA, EPOCHS, LR, EPS, H = 2.0, 200, 0.1, 1e-15, 1e-6
SEEDS, N, POS = (7, 13, 21), 300, 30
ARMS = {"bce": (0.0, 0.5), "focal": (GAMMA, 0.5), "alpha": (0.0, 0.75)}
ref = parity.load_reference(PHASE, LESSON, "main")
FL = lambda p: -(1 - p) ** GAMMA * math.log(p)          # the focal loss itself  # noqa: E731


def focal_scale(pt, gamma=GAMMA) -> float:
    """d(focal)/dp as a multiple of d(BCE)/dp; exactly 1.0 at gamma = 0."""
    pt = max(EPS, min(1 - EPS, pt))
    return (1 - pt) ** gamma - gamma * pt * (1 - pt) ** (gamma - 1) * math.log(pt)


def make(seed) -> list:
    """POS positives in a blob at the origin, the rest in a ring around it."""
    rng, out = random.Random(seed), []
    for i in range(N):
        angle, hot = rng.uniform(0, 2 * math.pi), i < POS
        rad = abs(rng.gauss(0, 0.55)) if hot else rng.uniform(0.6, 2.0)
        out.append(([rad * math.cos(angle), rad * math.sin(angle)], float(hot)))
    rng.shuffle(out)
    return out


def train(data, gamma, alpha) -> tuple:
    """The lesson's BCE net, step rescaled; returns (probs on positives, on negatives)."""
    net = ref.LossComparisonNetwork(loss_type="bce", lr=LR)
    for _ in range(EPOCHS):
        for x, y in data:
            net.forward(x)
            hot = y >= 0.5
            net.lr = LR * (alpha if hot else 1 - alpha) * focal_scale(
                net.out if hot else 1 - net.out, gamma)
            net.backward(y)
    got = [(net.forward(x), y >= 0.5) for x, y in data]
    return [p for p, y in got if y], [p for p, y in got if not y]


def score(pos, neg) -> dict:
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    hits = sum(p >= 0.5 for p in pos)
    return {"recall": hits / len(pos), "auc": wins / (len(pos) * len(neg)),
            "mpos": sum(pos) / len(pos), "mneg": sum(neg) / len(neg),
            "npred": hits + sum(p >= 0.5 for p in neg)}


def factors() -> dict:
    """The docs' loss weight beside the true gradient factor, and an FD cross-check."""
    return {"pairs": {pt: ((1 - pt) ** GAMMA, focal_scale(pt)) for pt in (0.9, 0.1)},
            "peak": max((focal_scale(i / 4000), i / 4000) for i in range(1, 4000)),
            "tail": focal_scale(1e-9),
            "fd": max(abs(focal_scale(p) - (FL(p + H) - FL(p - H))
                          / (math.log(p - H) - math.log(p + H))) for p in (0.05, 0.3, 0.9))}


def solve() -> dict:
    pool, gaps = {name: ([], []) for name in ARMS}, []
    for seed in SEEDS:
        got = {n: train(make(seed), g, a) for n, (g, a) in ARMS.items()}
        gaps.append(score(*got["focal"])["recall"] - score(*got["bce"])["recall"])
        for name, (pos, neg) in got.items():
            pool[name][0].extend(pos)
            pool[name][1].extend(neg)
    out = {arm: score(*probs) for arm, probs in pool.items()}
    cut = sorted(pool["bce"][0] + pool["bce"][1], reverse=True)[out["alpha"]["npred"] - 1]
    out["budget"] = sum(p >= cut for p in pool["bce"][0]) / len(pool["bce"][0])
    return out | factors() | {"gap": max(gaps)}


def verify(result) -> list:
    bce, foc = result["bce"], result["focal"]
    alp, peak = result["alpha"], result["peak"]
    quo = {pt: v[0] for pt, v in result["pairs"].items()}
    act = {pt: v[1] for pt, v in result["pairs"].items()}
    return [
        practice.Check("ANSWER: focal loss at gamma=2 does not move minority recall",
                       abs(foc["recall"] - bce["recall"]) < 0.02 and result["gap"] <= 1 / POS,
                       f"recall over {len(SEEDS) * POS} positives after {EPOCHS} epochs: BCE "
                       f"{bce['recall']:.3f} vs focal {foc['recall']:.3f}, and no single seed "
                       f"differs by more than {result['gap'] * POS:.0f} of {POS} positives"),
        practice.Check("FINDING: focal moves calibration, not detection",
                       foc["mneg"] > 2 * bce["mneg"] and foc["auc"] - bce["auc"] < 0.02,
                       f"mean probability on negatives {bce['mneg']:.3f} -> {foc['mneg']:.3f}, "
                       f"on positives {bce['mpos']:.3f} -> {foc['mpos']:.3f}, AUC only "
                       f"{bce['auc']:.4f} -> {foc['auc']:.4f}: both classes drift toward 0.5 "
                       f"and the 0.5 cut still lands between them"),
        practice.Check("FINDING: the docs' (1-p_t)^gamma weights the loss, not the gradient",
                       peak[0] < 1.23 and result["tail"] < 1.0001 and result["fd"] < 1e-6,
                       f"quoted {quo[0.9]:.4f} at p_t=0.9 ('ignored') and {quo[0.1]:.4f} at 0.1 "
                       f"('full gradient signal'); the gradient factors are {act[0.9]:.4f} and "
                       f"{act[0.1]:.4f} (central difference agrees to {result['fd']:.1e}). "
                       f"MECHANISM: the factor peaks at {peak[0]:.4f} (p_t={peak[1]:.4f}) then "
                       f"decays to {result['tail']:.4f} as p_t -> 0, so focal only ever damps"),
        practice.Check("CONTROL: alpha moves recall, and moves it by moving the threshold",
                       alp["recall"] - bce["recall"] > 0.1
                       and abs(result["budget"] - alp["recall"]) < 0.05,
                       f"alpha=0.75 at gamma=0 lifts recall to {alp['recall']:.3f} while AUC "
                       f"goes {bce['auc']:.4f} -> {alp['auc']:.4f}; letting plain BCE call its "
                       f"own top {alp['npred']} scores positive recovers {result['budget']:.3f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
