"""Exercise 4 — a 3-4-4-2 colour classifier, and what its two outputs are worth.

    Build a forward pass for a 3-4-4-2 network. Feed it RGB color values
    (normalized to 0-1) and observe the two outputs. This is the architecture for
    a simple color classifier with two classes.

Reading of the exercise: "observe the two outputs" is a measurement, so checks 2-4 ask
whether the pair behaves like the two class scores the exercise calls them — they do
not sum to 1, the colour barely reaches them, and the argmax is fixed over the whole
cube. Check 5 disobeys the one instruction given, "normalized to 0-1", to price it.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "02-multi-layer-networks"
SIZES, NETS, SAT_NETS, SEED = [3, 4, 4, 2], 300, 200, 7
CORNERS = [[r, g, b] for r in (0.0, 1.0) for g in (0.0, 1.0) for b in (0.0, 1.0)]
VALS = [0.0, 0.25, 0.5, 0.75, 1.0]                            # a 5x5x5 grid of colours
CUBE = [[r, g, b] for r in VALS for g in VALS for b in VALS]
RAW = [[255 * v for v in c] for c in CUBE]                    # the same cube, as bytes
agg = lambda recs, key, fn=statistics.mean: fn(r[key] for r in recs)   # noqa: E731


def build(ref, rng):
    """The lesson's own Layer, sized 3-4-4-2, with its U(-1, 1) init drawn from `rng`."""
    return ref.Network([ref.Layer(a, b, weights=[[rng.uniform(-1, 1) for _ in range(a)]
                                                 for _ in range(b)])
                        for a, b in zip(SIZES, SIZES[1:])])


def per_net(net, colors) -> dict:
    outs = [net.forward(c) for c in colors]
    a, b = [o[0] for o in outs], [o[1] for o in outs]
    diff, tot = [u - v for u, v in zip(a, b)], [u + v for u, v in zip(a, b)]
    return {"span": ((max(a) - min(a)) + (max(b) - min(b))) / 2, "sums": tot,
            "mean": statistics.mean(a + b), "msd": statistics.pstdev(diff),
            "mabs": abs(statistics.mean(diff)), "const": (min(diff) >= 0) == (max(diff) >= 0)}


def survey(ref, colors) -> dict:
    rng = random.Random(SEED)
    recs = [per_net(build(ref, rng), colors) for _ in range(NETS)]
    sums = [s for r in recs for s in r["sums"]]
    return {"span": agg(recs, "span"), "spread": agg(recs, "mean", statistics.pstdev),
            "const": agg(recs, "const", sum), "msd": agg(recs, "msd"),
            "mabs": agg(recs, "mabs"), "beats": sum(r["msd"] < r["mabs"] for r in recs),
            "lo": min(sums), "hi": max(sums), "avg": statistics.mean(sums),
            "near1": sum(abs(s - 1) < 0.01 for s in sums), "n": len(sums)}


def first_layer(ref, colors) -> dict:
    rng, sat, seen = random.Random(SEED), 0, []
    for _ in range(SAT_NETS):
        net, states = build(ref, rng), set()
        for c in colors:
            net.forward(c)
            h = net.layers[0].last_output
            states.add(tuple(round(v, 6) for v in h))
            sat += sum(v < 1e-6 or v > 1 - 1e-6 for v in h)
        seen.append(len(states))
    return {"sat": 100 * sat / (SAT_NETS * len(colors) * SIZES[1]),
            "distinct": statistics.mean(seen)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    demo = build(ref, random.Random(0))
    corners = [demo.forward(c) for c in CORNERS]
    return {"corners": corners, "widths": [len(la.last_output) for la in demo.layers],
            "corner_span": [max(o[i] for o in corners) - min(o[i] for o in corners)
                            for i in (0, 1)], **survey(ref, CUBE),
            "norm": first_layer(ref, CUBE), "raw": first_layer(ref, RAW)}


def verify(result):
    r, norm, raw = result, result["norm"], result["raw"]
    ends = " / ".join(f"[{o[0]:.4f}, {o[1]:.4f}]" for o in (r["corners"][0], r["corners"][-1]))
    return [
        practice.Check("the fixture: one 3-4-4-2 forward pass over the 8 corners of the cube",
                       r["widths"] == [4, 4, 2] and len(r["corners"]) == 8,
                       f"layer widths {r['widths']}; black / white give {ends}, and over all 8 "
                       f"corners the outputs move by only {r['corner_span'][0]:.4f} / "
                       f"{r['corner_span'][1]:.4f}"),
        practice.Check("FINDING: the two outputs are independent sigmoids, not a distribution",
                       abs(r["avg"] - 1) < 0.05 and r["near1"] < 0.05 * r["n"],
                       f"over {NETS} nets x {len(CUBE)} colours the sum averages "
                       f"{r['avg']:.4f} but ranges [{r['lo']:.4f}, {r['hi']:.4f}], and only "
                       f"{100 * r['near1'] / r['n']:.2f}% of {r['n']} land within 0.01 of 1"),
        practice.Check("FINDING: the colour barely reaches the output at all",
                       r["span"] < 0.3 * r["spread"],
                       f"one net's outputs move {r['span']:.4f} over the cube but "
                       f"{r['spread']:.4f} across nets (sd of per-net mean), "
                       f"{r['spread'] / r['span']:.1f}x more"),
        practice.Check("ANSWER: the predicted class is fixed over the cube for almost every net",
                       r["const"] > 0.9 * NETS and r["beats"] > 0.9 * NETS,
                       f"{r['const']} of {NETS} nets give one argmax for all {len(CUBE)} "
                       f"colours. MECHANISM: the decision is sign(out0 - out1), whose sd over "
                       f"the colours, {r['msd']:.5f}, is {r['mabs'] / r['msd']:.0f}x smaller "
                       f"than its mean offset {r['mabs']:.5f}, in {r['beats']} of {NETS} nets"),
        practice.Check("CONTROL: 'normalized to 0-1' is what buys even that much response",
                       norm["sat"] < 0.01 and raw["sat"] > 50
                       and raw["distinct"] < 0.5 * norm["distinct"],
                       f"the same nets on raw 0-255 bytes pin {raw['sat']:.1f}% of layer-1 "
                       f"activations to within 1e-6 of 0 or 1 ({norm['sat']:.1f}% normalized), "
                       f"leaving {raw['distinct']:.1f} of {len(CUBE)} colours distinct after "
                       f"it against {norm['distinct']:.1f} — sigmoid is flat past |z| ~ 6"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
