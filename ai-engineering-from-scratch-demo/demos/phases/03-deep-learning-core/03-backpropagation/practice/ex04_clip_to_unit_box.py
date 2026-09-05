"""Exercise 4 — clipping every gradient to [-1, 1] on a deep sigmoid net.

    Add gradient clipping to the training loop: after calling `backward()`, clip
    all gradients to [-1, 1]. Train a deeper network (4+ layers with sigmoid) and
    compare loss curves with and without clipping. This is your first defense
    against exploding gradients.

Reading of the exercise: "compare loss curves" is taken literally — epoch against epoch, not
eyeballed — and the comparison turns out to be vacuous, so checks 2-4 ask why. The clip goes
exactly where the exercise says, after `backward()` and before the update, so the clip is the
only difference between the arms. Check 2 is the control that keeps check 1 honest, and check
6 finds the change that does fix this network, which is not a clip.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "03-backpropagation"
XOR = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]
DEEP, EPOCHS, LR, CLIP, SEEDS = [2, 4, 4, 4, 1], 400, 1.0, 1.0, (42, 7, 99)
FLAT, WIDER = 1.0, 8.0        # loss of the constant-0.5 predictor; the init scale that works
BOUND = len(XOR) * 2 * 0.25   # |dL/db_out| <= rows * |2(p-t)| * max sigmoid' — see check 3


def build(ref, seed, sizes=None, scale=1.0):
    random.seed(seed)
    net = ref.Network(sizes or DEEP)
    for param in net.parameters():
        param.data *= scale
    return net


def grads(ref, net) -> float:
    """One backward pass over the whole XOR batch, exactly as the lesson's loop does."""
    total = ref.Value(0.0)
    for inputs, target in XOR:
        total = total + ref.mse_loss(net([ref.Value(i) for i in inputs]), target)
    net.zero_grad()
    total.backward()
    return total.data


def train(ref, seed, clip, scale=1.0) -> dict:
    net, curve, fired, biggest = build(ref, seed, scale=scale), [], 0, 0.0
    for _epoch in range(EPOCHS):
        curve.append(grads(ref, net))
        biggest = max(biggest, max(abs(p.grad) for p in net.parameters()))
        for param in net.parameters():
            step = param.grad if clip is None else max(-clip, min(clip, param.grad))
            fired += step != param.grad
            param.data -= LR * step
    return {"curve": curve, "fired": fired, "biggest": biggest,
            "updates": EPOCHS * len(net.parameters())}


def by_layer(ref, seed, sizes=None) -> list:
    """Mean |grad| per layer after one backward pass from the initial weights."""
    net = build(ref, seed, sizes)
    grads(ref, net)
    return [statistics.mean(abs(p.grad) for p in layer.parameters()) for layer in net.layers]


def apart(a, b) -> tuple:
    """How far two loss curves are, worst epoch, and on how many epochs they differ at all."""
    return (max(abs(x - y) for x, y in zip(a, b)), sum(x != y for x, y in zip(a, b)))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    arms = {(seed, name): train(ref, seed, clip)
            for seed in SEEDS for name, clip in (("plain", None), ("clipped", CLIP))}
    return {"arms": arms, "layers": by_layer(ref, SEEDS[0]),
            "depths": {n: by_layer(ref, SEEDS[0], [2] + [4] * n + [1]) for n in (1, 2, 3, 4)},
            "twice": apart(arms[(SEEDS[0], "plain")]["curve"],
                           train(ref, SEEDS[0], None)["curve"]),
            "big": train(ref, SEEDS[0], None, scale=16.0)["biggest"],
            "wide": {s: train(ref, s, CLIP, scale=WIDER) for s in SEEDS}}


def curves(arms) -> dict:
    """The clipped-against-plain comparison, across seeds."""
    gaps = [apart(arms[(s, "plain")]["curve"], arms[(s, "clipped")]["curve"]) for s in SEEDS]
    return {"gap": max(g[0] for g in gaps), "moved": max(g[1] for g in gaps),
            "fired": sum(arms[(s, "clipped")]["fired"] for s in SEEDS),
            "updates": sum(arms[(s, "clipped")]["updates"] for s in SEEDS),
            "biggest": max(arms[(s, "plain")]["biggest"] for s in SEEDS),
            "ends": ", ".join(f"{arms[(s, 'plain')]['curve'][-1]:.6f}" for s in SEEDS)}


def digest(result) -> dict:
    """Every other summary `verify` quotes, so that stays a list of comparisons."""
    layers, wide = result["layers"], result["wide"]
    return {**curves(result["arms"]), "attenuation": layers[-1] / layers[0],
            "fixed": ", ".join(f"{wide[s]['curve'][-1]:.6f}" for s in SEEDS),
            "worst_fixed": max(wide[s]["curve"][-1] for s in SEEDS),
            "wide_fired": sum(wide[s]["fired"] for s in SEEDS),
            "per_layer": ", ".join(f"{v:.2e}" for v in layers),
            "firsts": ", ".join(f"{n}: {v[0]:.2e}" for n, v in result["depths"].items())}


def verify(result):
    d, arms, twice = digest(result), result["arms"], result["twice"]
    return [
        practice.Check("ANSWER: the clip never fires, so the two curves are the same curve",
                       d["fired"] == 0 and d["gap"] < 1e-14,
                       f"{EPOCHS} epochs on a {'-'.join(map(str, DEEP))} sigmoid net at seeds "
                       f"{SEEDS}: the clip fired on {d['fired']} of {d['updates']:,} parameter "
                       f"updates, and the two loss curves stay within {d['gap']:.1e} of each other "
                       f"— 4 ulps. Comparing them cannot say anything; there is one curve"),
        practice.Check("CONTROL: even that 4-ulp residue is not the clip — the lesson's training "
                       "is not reproducible run to run",
                       max(twice[0], d["gap"]) < 1e-14 and twice[1] > 0,
                       f"two *identical* unclipped runs from seed {SEEDS[0]} differ by "
                       f"{twice[0]:.1e} on {twice[1]} of {EPOCHS} epochs, the same size as the "
                       f"clipped-vs-plain gap ({d['gap']:.1e} on {d['moved']}). MECHANISM: "
                       f"`Value.backward` walks `v._children`, a *set*, so the topological order — "
                       f"and the order gradient contributions are summed in — follows object "
                       f"hashes, which move between runs"),
        practice.Check("MECHANISM: the clip cannot fire — every gradient here is bounded by 2",
                       d["biggest"] < 0.5 and result["big"] < BOUND,
                       f"the largest gradient anywhere in the run is {d['biggest']:.4f}. The most "
                       f"exposed parameter is the output bias, whose gradient is a sum over "
                       f"{len(XOR)} rows of 2(p - t) * sigmoid'(z) with |p - t| <= 1 and sigmoid' "
                       f"<= 1/4 — at most {BOUND:.0f}, and only if all four rows are maximally "
                       f"wrong at once. Scaling every initial weight by 16 still peaks at "
                       f"{result['big']:.4f}"),
        practice.Check("FINDING: a deep sigmoid net does not explode, it vanishes",
                       d["attenuation"] > 100,
                       f"mean |grad| per layer at init, input to output: {d['per_layer']} — the "
                       f"first layer's is {d['attenuation']:.0f}x smaller than the last's, and each "
                       f"hidden layer added multiplies it by about 1/5 ({d['firsts']} for 1, 2, 3, "
                       f"4 hidden layers). A clip is a ceiling; the problem is a floor"),
        practice.Check("FINDING: neither arm learns XOR at all",
                       all(abs(arms[(s, "plain")]["curve"][-1] - FLAT) < 1e-3 for s in SEEDS),
                       f"final loss {d['ends']} at the three seeds — that is {FLAT:.1f}, exactly "
                       f"the loss of predicting 0.5 on all four rows ({len(XOR)} x 0.25). The "
                       f"network has not learned a wrong answer, it has not moved"),
        practice.Check("CONTROL: what fixes this net is a bigger init, and the clip still never "
                       "fires",
                       d["worst_fixed"] < 0.02 and d["wide_fired"] == 0,
                       f"multiplying the lesson's own He-initialised weights by {WIDER:.0f} takes "
                       f"the same four-layer net to {d['fixed']} on the same seeds and the same "
                       f"{EPOCHS} epochs — {FLAT / d['worst_fixed']:.0f}x better at the worst seed "
                       f"— with the clip firing {d['wide_fired']} times there too. Depth was never "
                       f"the obstacle; the signal reaching layer 1 was"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
