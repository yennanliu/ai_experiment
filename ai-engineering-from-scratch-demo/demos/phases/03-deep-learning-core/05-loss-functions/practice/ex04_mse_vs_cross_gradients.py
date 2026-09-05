"""Exercise 4 — the MSE-vs-cross-entropy comparison, with the gradients measured.

    Run the MSE vs cross-entropy comparison but track gradient magnitudes at each
    layer during training. Plot the average gradient norm per epoch. Verify that
    cross-entropy produces larger gradients in early epochs when the model is most
    uncertain.

Reading of the exercise: "verify" is taken as a claim to test, not to illustrate, and it needs
a controlled comparison — the lesson's net updates after every sample, so two separately
trained arms stop sharing weights at the first one. Checks 1-3 therefore evaluate both losses
on the *same* parameters at every epoch, which makes the ratio exact. Check 4 runs the
comparison the exercise describes and shows what it measures instead.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "05-loss-functions"
HIDDEN, LR, EPOCHS, EPS = 8, 0.1, 200, 1e-15


def both(net, pred, target) -> tuple:
    """d(loss)/d(z2) under each loss, on the parameters the net has right now."""
    p = max(EPS, min(1 - EPS, pred))
    bce = (-(target / p) + (1 - target) / (1 - p)) * pred * (1 - pred)
    return bce, 2.0 * (pred - target) * pred * (1 - pred)


def layers(net, d_out) -> tuple:
    """|grad| into each layer for a given d(loss)/d(z2) — the rest of `backward` is shared."""
    first = []
    for i in range(net.hidden_size):
        d_h = d_out * net.w2[i] * (1.0 if net.z1[i] > 0 else 0.0)
        first += [abs(d_h * net.x[0]), abs(d_h * net.x[1]), abs(d_h)]
    return (statistics.mean(first),
            statistics.mean([abs(d_out * h) for h in net.h] + [abs(d_out)]))


def paired(ref, data, kind, epochs=EPOCHS) -> dict:
    """Train under `kind`, and at every sample price *both* losses on the same weights."""
    net = ref.LossComparisonNetwork(kind, HIDDEN, LR)
    per_epoch, worst, floor, peak = [], 0.0, 10.0, 0.0
    for _epoch in range(epochs):
        rows = {"bce": ([], []), "mse": ([], [])}
        for x, target in data:
            pred = net.forward(x)
            for name, d_out in zip(("bce", "mse"), both(net, pred, target)):
                for side, value in zip(rows[name], layers(net, d_out)):
                    side.append(value)
            ratio, closed = _ratio(*both(net, pred, target)), 1.0 / (2 * pred * (1 - pred))
            worst = max(worst, abs(ratio - closed) / closed)
            floor, peak = min(floor, ratio), max(peak, ratio)
            net.backward(target)
        per_epoch.append({n: tuple(statistics.mean(s) for s in rows[n]) for n in rows})
    return {"per_epoch": per_epoch, "worst": worst, "floor": floor, "peak": peak,
            "end": _end(ref, net, data)}


def _ratio(bce, mse) -> float:
    return abs(bce) / abs(mse) if mse else float("inf")


def _end(ref, net, data) -> tuple:
    loss = sum(net.compute_loss(net.forward(x), y) for x, y in data) / len(data)
    right = sum((net.forward(x) >= 0.5) == (y >= 0.5) for x, y in data)
    return loss, 100.0 * right / len(data)


def apart(ref, data) -> dict:
    """The comparison as the exercise describes it: two arms, trained separately."""
    arms = {k: paired(ref, data, k) for k in ("mse", "bce")}
    ratios = [tuple(arms["bce"]["per_epoch"][e]["bce"][i] / arms["mse"]["per_epoch"][e]["mse"][i]
                    for i in (0, 1)) for e in range(EPOCHS)]
    return {"arms": arms, "ratios": ratios}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = ref.make_circle_data()
    return {"n": len(data), **apart(ref, data)}


def digest(result) -> dict:
    """Every summary `verify` quotes, so that stays a list of comparisons."""
    arms, ratios = result["arms"], result["ratios"]
    same = arms["bce"]["per_epoch"]
    return {"worst": max(a["worst"] for a in arms.values()),
            "floor": min(a["floor"] for a in arms.values()),
            "peak": max(a["peak"] for a in arms.values()),
            "early": same[0]["bce"][0] / same[0]["mse"][0],
            "late": same[EPOCHS - 1]["bce"][0] / same[EPOCHS - 1]["mse"][0],
            "floor_split": min(r[0] for r in ratios), "floor_l2": min(r[1] for r in ratios),
            **tables(arms, same, ratios)}


def tables(arms, same, ratios) -> dict:
    """The three per-epoch listings `verify` prints."""
    return {"paired": ", ".join(f"epoch {e}: {same[e]['bce'][0] / same[e]['mse'][0]:.2f}"
                                for e in (0, 1, 10, 100, EPOCHS - 1)),
            "split": ", ".join(f"epoch {e}: L1 {ratios[e][0]:.2f} L2 {ratios[e][1]:.2f}"
                               for e in (0, 10, 100, EPOCHS - 1)),
            "ends": ", ".join(f"{k} loss {a['end'][0]:.4f} acc {a['end'][1]:.1f}%"
                              for k, a in arms.items())}


def verify(result):
    d, arms = digest(result), result["arms"]
    return [
        practice.Check("ANSWER: on the same weights cross-entropy's gradient is larger at every "
                       "sample of every epoch, by exactly 1/(2p(1-p))",
                       d["worst"] < 1e-9 and d["floor"] >= 2.0,
                       f"over {EPOCHS} epochs x {result['n']} points x 2 arms the measured ratio "
                       f"|d_bce/dz| / |d_mse/dz| never leaves the closed form by more than "
                       f"{d['worst']:.1e} relative, and its smallest value anywhere is "
                       f"{d['floor']:.6f}. "
                       f"MECHANISM: the sigmoid derivative cancels in cross-entropy — d/dz is "
                       f"exactly (p - t) — while MSE keeps it, at 2(p - t)p(1 - p)"),
        practice.Check("FINDING: the exercise has the direction backwards — uncertainty is where "
                       "the advantage is *smallest*",
                       abs(d["floor"] - 2.0) < 1e-6 and d["peak"] > 100,
                       f"1/(2p(1-p)) is minimised at p = 0.5, the most uncertain the model can be, "
                       f"where it is exactly {d['floor']:.4f}; it grows without bound as the model "
                       f"becomes confidently wrong, reaching {d['peak']:.1e} in this run. Cross-"
                       f"entropy's edge is largest where the model is most *certain*, and least "
                       f"where it is most uncertain"),
        practice.Check("…and the per-epoch curve rises rather than falls",
                       d["early"] < d["late"] and d["early"] > 2.0,
                       f"layer-1 mean ratio on shared weights — {d['paired']}. It is above 2 "
                       f"throughout, and it *grows* as the model gets confident: {d['early']:.2f} "
                       f"at epoch 0 against {d['late']:.2f} at {EPOCHS - 1}. The early epochs, "
                       f"where the model sits near p = 0.5, are the ratio's floor"),
        practice.Check("FINDING: two separately trained arms stop being a comparison at the "
                       "first update",
                       d["floor_split"] < 2.0,
                       f"train MSE and cross-entropy as the exercise says and compare their own "
                       f"gradients: {d['split']}. The lesson's net updates after every sample, so "
                       f"after one point the arms are different networks — the layer-1 ratio "
                       f"reaches {d['floor_split']:.2f}, below the 2.0 the closed form guarantees "
                       f"on shared weights, and the layer-2 ratio gets to {d['floor_l2']:.2f}"),
        practice.Check("CONTROL: the two arms end in the same place, and their losses are not "
                       "comparable numbers",
                       abs(arms["mse"]["end"][1] - arms["bce"]["end"][1]) < 1.0,
                       f"{d['ends']} — the same accuracy to within a point. The loss column is in "
                       f"different units in each row, so only the accuracy and the gradients can "
                       f"be read across the comparison at all"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
