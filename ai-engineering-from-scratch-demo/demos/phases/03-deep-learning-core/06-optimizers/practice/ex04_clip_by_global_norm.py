"""Exercise 4 — clipping by global norm, and a divergence that cannot happen.

    Implement gradient clipping (clip by global norm). Set the max gradient norm
    to 1.0. Train with and without clipping using a high learning rate (lr=0.01
    for Adam). Count how many runs diverge (loss goes to NaN) with and without
    clipping over 10 random seeds.

Reading of the exercise: the lesson's net re-seeds the global RNG to 0 in its own constructor,
so the only seed a caller controls is the data seed. Check 1 is the literal count, checks 2-3
say why it is what it is, and 4-5 find the setting where a clip does change the outcome.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "06-optimizers"
LR, CLIP, SEEDS, EPOCHS, OFF = 0.01, 1.0, 10, 40, float("inf")
mean = lambda runs, key: sum(run[key] for run in runs) / len(runs)      # noqa: E731
top = lambda runs, key: max(run[key] for run in runs)                   # noqa: E731


def step_once(net, opt, point, label, limit, scale):
    """One online step; the clip is the only line added to the lesson's own loop."""
    pred = net.forward(point)
    grads = [scale * g for g in net.compute_grads(label)]
    norm = math.sqrt(sum(g * g for g in grads))
    grads = [g * limit / norm for g in grads] if norm > limit else grads
    params = net.get_params(); before = list(params)                    # noqa: E702
    opt.step(params, grads)
    net.set_params(params)
    prob = max(1e-15, min(1 - 1e-15, pred))
    return (-(label * math.log(prob) + (1 - label) * math.log(1 - prob)),
            (pred >= 0.5) == (label >= 0.5), norm, norm > limit,
            max(abs(a - b) for a, b in zip(params, before)))


def train(ref, opt, seed, limit, epochs=EPOCHS, scale=1.0):
    data, net = ref.make_circle_data(seed=seed), ref.OptimizerTestNetwork(opt, hidden_size=8)
    cut, worst, move, loss, right = 0, 0.0, 0.0, 0.0, 0
    for _epoch in range(epochs):
        loss, right = 0.0, 0
        for point, label in data:
            cost, ok, norm, hit, span = step_once(net, opt, point, label, limit, scale)
            loss, right, cut = loss + cost, right + ok, cut + hit
            worst, move = max(worst, norm), max(move, span)
    params = net.get_params()
    return {"loss": loss / len(data), "acc": 100.0 * right / len(data), "norm": worst,
            "move": move, "cut": cut / (epochs * len(data)), "params": params,
            "big": max(map(abs, params)), "nan": not math.isfinite(loss + math.fsum(params))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    arms = {(name, limit): [train(ref, make(lr), seed, limit) for seed in range(SEEDS)]
            for name, make, lr in (("adam", ref.Adam, LR), ("sgd", ref.SGD, 5.0))
            for limit in (CLIP, OFF)}
    starts = [ref.OptimizerTestNetwork(ref.SGD(LR)).get_params() for _s in range(SEEDS)]
    at20 = lambda opt, **kw: train(ref, opt, 0, OFF, epochs=20, **kw)    # noqa: E731
    huge = [at20(ref.SGD(100.0)), at20(ref.SGD(1e30))]
    plain, scaled = at20(ref.Adam(LR)), at20(ref.Adam(LR), scale=1000.0)
    return {"arms": arms, "huge": huge, "plain": plain,
            "reseeded": max(abs(a - b) for start in starts for a, b in zip(starts[0], start)),
            "gap": max(abs(a - b) for a, b in zip(plain["params"], scaled["params"]))}


def verify(result):
    arms, huge, plain = result["arms"], result["huge"], result["plain"]
    nans = {key: sum(run["nan"] for run in runs) for key, runs in arms.items()}
    on, off = arms[("adam", CLIP)], arms[("adam", OFF)]
    hard, easy = arms[("sgd", OFF)], arms[("sgd", CLIP)]
    loud, rel = top(off, "norm"), result["gap"] / plain["big"]
    still, moved = top(off, "move") / LR, top(on, "move") / LR
    return [
        practice.Check("ANSWER: 0 of 10 runs diverge with clipping and 0 of 10 without",
                       nans[("adam", CLIP)] == 0 and nans[("adam", OFF)] == 0,
                       f"Adam at lr = {LR}, {EPOCHS} epochs, seeds 0-9: mean final loss "
                       f"{mean(on, 'loss'):.4f} clipped against {mean(off, 'loss'):.4f} plain, "
                       f"accuracy {mean(on, 'acc'):.1f}% against {mean(off, 'acc'):.1f}%, "
                       f"{100 * mean(on, 'cut'):.1f}% of steps clipped. Only the data varies — the "
                       f"network calls random.seed(0) in its own constructor, so all ten start from "
                       f"weights differing by {result['reseeded']:.1f}"),
        practice.Check("FINDING: NaN is unreachable here, so the metric cannot separate the arms",
                       huge[0]["loss"] == huge[1]["loss"] and huge[1]["big"] > 1e29
                       and not huge[1]["nan"],
                       f"SGD at lr = 1e30 ends with a weight of {huge[1]['big']:.2e} and a final "
                       f"loss of {huge[1]['loss']:.12f}, bit-identical to the same run at lr = 100 "
                       f"({huge[0]['big']:.1f}). sigmoid() clamps its input to [-500, 500] before "
                       f"exp and the loss clamps p, so no path here reaches NaN"),
        practice.Check("MECHANISM: an 8x cut in the gradient moves Adam's largest step under 20%",
                       loud > 8 * CLIP and abs(moved / still - 1) < 0.2,
                       f"unclipped, the global norm reaches {loud:.2f}, {loud / CLIP:.1f}x the clip "
                       f"— yet the largest parameter move is {still:.2f} lr unclipped and "
                       f"{moved:.2f} lr clipped. Adam divides by sqrt(v_hat): the gradient's size "
                       f"is already gone from the step"),
        practice.Check("CONTROL: Adam is invariant to a global gradient rescale", rel < 1e-6,
                       f"scaling every gradient by 1000 for 20 epochs reproduces the same 33 "
                       f"parameters to {result['gap']:.1e} absolute, {rel:.1e} relative — the "
                       f"residue is the epsilon. A clip is a rescale, so it acts only through "
                       f"which steps it selects"),
        practice.Check("CONTROL: the same clip is decisive for SGD, whose step is the gradient",
                       top(hard, "move") > 50 * top(easy, "move")
                       and mean(easy, "loss") < 0.1 * mean(hard, "loss"),
                       f"SGD at lr = 5.0 on the same 10 seeds: largest move "
                       f"{top(hard, 'move'):.1f} unclipped against {top(easy, 'move'):.2f} clipped, "
                       f"mean final loss {mean(hard, 'loss'):.2f} against {mean(easy, 'loss'):.2f} "
                       f"— and still {nans[('sgd', OFF)]} NaNs, because there are none to be had"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
