"""Exercise 3 — save()/load() for Sequential, and the two things a naive one loses.

    Add a `save()` and `load()` method to Sequential that serializes all weights to a JSON file
    and loads them back. Verify that a loaded model produces the same predictions as the
    original.

Reading of the exercise: "the same predictions" is checked as bit-for-bit equality on an 81-point
grid rather than as closeness, with check 2 as the control that the model loaded into really was
different first. Checks 3 and 4 are the two ways the obvious implementation fails: saving only what
`parameters()` enumerates, and rebinding the lists rather than writing through them. Every file
goes into a `tempfile` directory, so the run leaves nothing behind.
"""

from __future__ import annotations

import json
import pathlib
import random
import tempfile

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "10-mini-framework"
ALL = ("weights", "biases", "gamma", "beta", "running_mean", "running_var")
PARAMS = ("weights", "biases", "gamma", "beta")    # exactly what Sequential.parameters() enumerates
GRID = [[x / 2.0 - 2, y / 2.0 - 2] for x in range(9) for y in range(9)]
flat = lambda m: [e[0][e[1]][e[2]] if e[2] is not None else e[0][e[1]] for e in m.parameters()]
gap = lambda a, b: max(abs(u - v) for u, v in zip(a, b))                      # noqa: E731


def save(model, path, keep=ALL):
    state = [{k: getattr(m, k) for k in keep if hasattr(m, k)} for m in model.modules]
    path.write_text(json.dumps({"_meta": {"lesson": LESSON, "modules": len(state)}, "state": state}))
    return model


def load(model, path, rebind=False):
    for module, saved in zip(model.modules, json.loads(path.read_text())["state"]):
        for key, value in saved.items():
            if rebind:                     # the plausible one-liner — check 4 says what it costs
                setattr(module, key, value)
            else:
                getattr(module, key)[:] = value   # write through the list the optimizer holds
    return model


def build(ref, seed, norm=False):
    random.seed(seed)                      # the reference Linear draws its init from the global RNG
    mid = [ref.BatchNorm(8)] if norm else []
    return ref.Sequential(ref.Linear(2, 8), ref.ReLU(), *mid, ref.Linear(8, 1), ref.Sigmoid())


def train(ref, model, opt, epochs=5):
    crit, data = ref.BCELoss(), ref.make_circle_data(200, seed=7)
    model.train()
    for _ in range(epochs):
        for x, t in data:
            crit(model.forward(x), t)
            opt.zero_grad(), model.backward(crit.backward()), opt.step()
    return model


def preds(model):
    model.eval()                           # BatchNorm reads its buffers only in eval mode
    return [model.forward(p)[0] for p in GRID]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as tmp:     # hermetic: nothing is written into the repo
        whole, part = pathlib.Path(tmp) / "model.json", pathlib.Path(tmp) / "params.json"
        train(ref, origin := build(ref, 1), ref.Adam(origin.parameters(), lr=0.01))
        want, fresh = preds(save(origin, whole)), build(ref, 99)
        out = {"bytes": whole.stat().st_size, "nparams": len(flat(origin)), "naive": gap(preds(fresh), want)}
        back = preds(load(fresh, whole))
        out |= {"same": sum(u == v for u, v in zip(back, want)), "round": gap(back, want),
                "scalars": sum(u == v for u, v in zip(flat(origin), flat(fresh)))}
        for rebind in (False, True):
            resume = build(ref, 99)
            opt = ref.Adam(resume.parameters(), lr=0.05)   # built before the checkpoint arrives
            start = preds(load(resume, whole, rebind=rebind))
            moved = gap(preds(train(ref, resume, opt, epochs=1)), start)
            out["rebind" if rebind else "inplace"] = (gap(start, want), moved)
        train(ref, norm := build(ref, 1, True), ref.Adam(norm.parameters(), lr=0.01))
        goal, bn = preds(save(save(norm, whole), part, keep=PARAMS)), norm.modules[2]
        out["bn_all"] = gap(preds(load(build(ref, 99, True), whole)), goal)
        out["bn_par"] = gap(preds(load(build(ref, 99, True), part)), goal)
        out["mean"], out["var"] = max(map(abs, bn.running_mean)), sorted(set(bn.running_var))
        return out


def verify(r):
    return [
        practice.Check("ANSWER: the loaded model's predictions are identical, not merely close",
                       r["round"] == 0.0 and r["same"] == len(GRID),
                       f"{r['same']}/{len(GRID)} grid predictions equal to the last bit after a "
                       f"{r['bytes']}-byte JSON round trip (max|difference| exactly {r['round']}), all "
                       f"{r['scalars']}/{r['nparams']} scalar parameters equal"),
        practice.Check("CONTROL: the model it was loaded into really did disagree first",
                       r["naive"] > 0.5 and r["scalars"] == r["nparams"],
                       f"before the load the differently-seeded target it was read into differed from the "
                       f"original by up to {r['naive']:.4f} on that grid — not a model compared to itself"),
        practice.Check("FINDING: `parameters()` is not a state dict — BatchNorm's buffers are missing",
                       r["bn_par"] > 0.5 and r["bn_all"] == 0.0,
                       f"a save built from exactly what `Sequential.parameters()` enumerates {PARAMS} "
                       f"reloads a BatchNorm model to predictions off by {r['bn_par']:.4f}, against "
                       f"{r['bn_all']} once running_mean/running_var travel too — buffers, not parameters. "
                       f"`BatchNorm.forward` moves running_mean (to {r['mean']:.4f}) but leaves "
                       f"running_var at {r['var']}"),
        practice.Check("FINDING: load must write through the lists; rebinding freezes training",
                       r["rebind"][1] == 0.0 < r["inplace"][1] and r["rebind"][0] == 0.0,
                       f"both loads restore the checkpoint exactly (gap {r['rebind'][0]} either way), but "
                       f"`parameters()` hands the optimizer the list objects themselves, so rebinding "
                       f"`module.weights = [...]` leaves it on the old ones: after 200 Adam steps the "
                       f"predictions have moved {r['rebind'][1]}, against {r['inplace'][1]:.4f} in place"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
