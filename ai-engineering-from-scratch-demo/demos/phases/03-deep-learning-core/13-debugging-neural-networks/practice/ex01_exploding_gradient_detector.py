"""Exercise 1 — detect exploding gradients and suggest a clipping value.

    **Add an exploding gradient detector.** Modify the `NetworkDebugger` to detect
    when gradients exceed a threshold and automatically suggest a gradient
    clipping value. Test it on a 20-layer network with no normalization.

Reading of the exercise: a detector is worth nothing unless it fires on the bug it names, so the
20-layer net is built at an initialisation gain that provably explodes, against a control at gain
1.0 that provably does not. The detector measures the global *parameter*-gradient norm at the
initial weights and the clip it suggests is judged by training with it; checks 3-6 put the same
question to the lesson's own `check_gradients`, which stays HEALTHY right through the explosion.
"""

from __future__ import annotations

import math
import warnings

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "13-debugging-neural-networks"
DEPTH, WIDTH, BATCH, THRESHOLD = 20, 32, 32, 10.0


def _build(torch, gain):
    """A 20-layer ReLU stack, no normalization, weights scaled by `gain`."""
    torch.manual_seed(0)
    lins = [torch.nn.Linear(WIDTH if i else 10, WIDTH) for i in range(DEPTH)]
    with torch.no_grad():
        for lin in lins:
            lin.weight.mul_(gain)
            lin.bias.zero_()
    stack = [m for lin in lins for m in (lin, torch.nn.ReLU())]
    return torch.nn.Sequential(*stack, torch.nn.Linear(WIDTH, 2))


def _probe(torch, gain, crit, data):
    """Global parameter-gradient norm at the initial weights — the detector itself."""
    model = _build(torch, gain)
    model.zero_grad()
    crit(model(data[0]), data[1]).backward()
    return sum(float((p.grad ** 2).sum()) for p in model.parameters()) ** 0.5


def _train(torch, ref, gain, crit, data, clip=None, steps=60):
    """60 SGD steps watched by the lesson's own NetworkDebugger; returns what it saw."""
    model = _build(torch, gain)
    dbg = ref.NetworkDebugger(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for s in range(steps):
        opt.zero_grad()
        cut = slice(s % 8 * BATCH, (s % 8 + 1) * BATCH)
        loss = crit(model(data[0][cut]), data[1][cut])
        dbg.record_loss(loss.item())
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
    got = {"loss": dbg.loss_history, "grads": dbg.check_gradients(),
           "health": dbg.check_loss_health(), "finite": all(map(math.isfinite, dbg.loss_history)),
           "abs": [v["abs_mean"] for v in dbg.gradient_stats.values()]}
    dbg.remove_hooks()
    return got


def solve():
    try:
        import torch
    except ImportError as exc:                # pragma: no cover - env guard
        raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})")
    warnings.filterwarnings("ignore", message=".*backward hook.*")
    ref = parity.load_reference(PHASE, LESSON, "debug_neural_nets")
    torch.manual_seed(42)
    x = torch.randn(256, 10)
    data, crit = (x, (x[:, 0] > 0).long()), torch.nn.CrossEntropyLoss()
    clip = _probe(torch, 3.0, crit, data)
    bad, edge = _train(torch, ref, 3.0, crit, data), _train(torch, ref, 2.8, crit, data)
    return {"clip": clip, "ok_norm": _probe(torch, 1.0, crit, data), "bad": bad, "edge": edge,
            "edge_norm": _probe(torch, 2.8, crit, data), "edge_max": max(edge["abs"]),
            "fixed": _train(torch, ref, 3.0, crit, data, clip=clip),
            "ok": _train(torch, ref, 1.0, crit, data),
            "nan_lt": math.nan < 1e-7, "nan_gt": math.nan > 100,
            "nans": sum(1 for v in bad["abs"] if v != v),
            "blew": next(i for i, v in enumerate(bad["loss"]) if v != v)}


def verify(r):
    bad, fix, edge, ok, t = r["bad"], r["fixed"], r["edge"], r["ok"], THRESHOLD
    return [
        practice.Check("ANSWER: the detector fires on the 20-layer net and its clip rescues it",
                       all([r["clip"] > t, fix["finite"], fix["loss"][-1] < 1.0]),
                       f"probe norm {r['clip']:.1f} = {r['clip'] / t:.0f}x the {t:.0f} threshold; "
                       f"clipping there keeps all 60 steps finite, {fix['loss'][0]:.2f} -> "
                       f"{fix['loss'][-1]:.3f}; unclipped the same net is nan by step {r['blew']}"),
        practice.Check("CONTROL: the same 20 layers at gain 1.0 neither fire nor need clipping",
                       all([r["ok_norm"] < t, ok["finite"]]),
                       f"probe norm {r['ok_norm']:.4f}, {r['clip'] / r['ok_norm']:.0f}x below the "
                       f"exploding net; 60 unclipped steps end at {ok['loss'][-1]:.4f}. Depth "
                       f"alone is not the defect, depth times initialisation scale is"),
        practice.Check("FINDING: the lesson's check_gradients() calls the exploding run HEALTHY",
                       all([bad["grads"] == ["HEALTHY"], not bad["finite"]]),
                       f"60 steps ending in a NaN loss, and check_gradients() returns "
                       f"{bad['grads']} — the one diagnostic named for this failure is the one "
                       f"that stays quiet through it"),
        practice.Check("MECHANISM: NaN fails every comparison, so both threshold branches skip",
                       all([r["nans"] == len(bad["abs"]), not r["nan_lt"], not r["nan_gt"]]),
                       f"all {r['nans']} recorded grad_output abs_means are NaN, and the guards "
                       f"`abs_mean < 1e-7` / `abs_mean > 100` are both False for NaN "
                       f"({r['nan_lt']}, {r['nan_gt']})"),
        practice.Check("MECHANISM: and before any NaN the > 100 bound is out of reach here",
                       all([r["edge_max"] < 0.02, edge["grads"] == ["HEALTHY"],
                            r["edge_norm"] > 50]),
                       f"at gain 2.8 the probe norm is {r['edge_norm']:.1f} yet the largest "
                       f"grad_output abs_mean is {r['edge_max']:.2e}, {100 / r['edge_max']:.0f}x "
                       f"under the bound: the hook averages |grad| over a {BATCH}x{WIDTH} "
                       f"activation-gradient tensor, the bound is written for a parameter norm"),
        practice.Check("FINDING: check_loss_health() catches what check_gradients() misses",
                       all([bad["health"] == "NAN_OR_INF", ok["health"] != "NAN_OR_INF"]),
                       f"the loss monitor returns {bad['health']!r} on the exploding run and "
                       f"{ok['health']!r} on the control — the only check in the class that tests "
                       f"math.isnan explicitly rather than comparing against a bound"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
