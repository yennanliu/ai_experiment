"""Exercise 2 — resurrect dead ReLU units, and check the resurrection actually took.

    **Build a dead neuron resurrector.** Write a function that identifies dead ReLU
    neurons (always outputting 0) and reinitializes their incoming weights with
    Kaiming initialization. Show that this recovers a network where >70% of
    neurons are dead.

Reading of the exercise: "always outputting 0" is per *unit over the whole dataset* — dead iff the
unit's largest pre-activation is <= 0 — and "recovers" is judged by the lesson's own
`overfit_one_batch` plus a match against a never-damaged control. The damage is the lesson's own
BUG 2, a large negative bias, dealt to 75% of the hidden units, so it is deterministic. Check 3
runs the instruction exactly as written, on incoming weights alone; it does not recover the net.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "13-debugging-neural-networks"
FEATS, WIDTH, DEPTH, KILL, BIAS = 8, 16, 3, 0.75, -6.0


def _model(torch, kill):        # 3x16 ReLU MLP; `kill` deals 75% of units the BUG 2 bias
    torch.manual_seed(0)
    lins = [torch.nn.Linear(FEATS if i == 0 else WIDTH, WIDTH) for i in range(DEPTH)]
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for lin in lins:
            lin.bias[torch.rand(WIDTH, generator=gen) < (KILL if kill else 0.0)] = BIAS
    stack = [m for lin in lins for m in (lin, torch.nn.ReLU())]
    return torch.nn.Sequential(*stack, torch.nn.Linear(WIDTH, 2)), lins


def _scan(torch, model, x):     # per hidden unit, its largest pre-activation over the dataset
    with torch.no_grad():
        return [model[:2 * i + 1](x).max(dim=0).values for i in range(DEPTH)]


def _dead(scan):
    return sum(int((z <= 0).sum()) for z in scan)


def _resurrect(torch, model, lins, x, bias_too):
    """The exercise's function: Kaiming-reinit the incoming weights of every dead unit."""
    n = 0
    with torch.no_grad():
        for lin, z in zip(lins, _scan(torch, model, x)):
            dead, fan = z <= 0, lin.weight.shape[1]
            n += int(dead.sum())
            lin.weight[dead] = torch.randn(int(dead.sum()), fan) * math.sqrt(2.0 / fan)
            if bias_too:
                lin.bias[dead] = 0.0
    return n


def _run(torch, ref, x, y, crit, kill, fix=None):    # build, maybe resurrect, then train
    model, lins = _model(torch, kill)
    was = [z <= 0 for z in _scan(torch, model, x)]
    crit(model(x), y).backward()
    frozen = sum(float(a.weight.grad[d].abs().sum() + a.bias.grad[d].abs().sum())
                 for a, d in zip(lins, was))
    n = _resurrect(torch, model, lins, x, fix) if fix is not None else 0
    scan = _scan(torch, model, x)
    with parity.quiet():
        converged = ref.overfit_one_batch(model, x, y, crit)
    with torch.no_grad():
        out, loss = model(x), float(crit(model(x), y))
    return {"before": sum(int(m.sum()) for m in was), "n": n,
            "after": _dead(scan), "loss": loss,
            "acc": float((out.argmax(1) == y).float().mean()), "converged": converged,
            "frozen": frozen, "end": _dead(_scan(torch, model, x)),
            "reach": [float((z[m] - lin.bias[m]).max().detach())
                      for z, m, lin in zip(scan, was, lins) if m.any()]}


def solve():
    try:
        import torch
    except ImportError as exc:                # pragma: no cover - env guard
        raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})")
    ref = parity.load_reference(PHASE, LESSON, "debug_neural_nets")
    torch.manual_seed(42)
    x = torch.randn(512, FEATS)
    y, crit = ((x[:, 0] * x[:, 1] * x[:, 2]) > 0).long(), torch.nn.CrossEntropyLoss()
    return {"total": WIDTH * DEPTH, "sick": _run(torch, ref, x, y, crit, True),
            "well": _run(torch, ref, x, y, crit, False),
            "weights": _run(torch, ref, x, y, crit, True, fix=False),
            "both": _run(torch, ref, x, y, crit, True, fix=True)}


def verify(r):
    sick, well, wo, both, n = r["sick"], r["well"], r["weights"], r["both"], r["total"]
    return [
        practice.Check("ANSWER: resurrecting weights *and* bias recovers a 79%-dead network",
                       all([sick["before"] / n > 0.7, both["after"] == 0, both["converged"]]),
                       f"{sick['before']}/{n} units dead ({sick['before'] / n:.0%}); resurrecting "
                       f"all {both['n']} leaves {both['after']} dead, loss {both['loss']:.4f}, acc "
                       f"{both['acc']:.1%} — the wreck stalls at {sick['loss']:.4f} / "
                       f"{sick['acc']:.1%} and the lesson's overfit_one_batch calls that a FAIL"),
        practice.Check("CONTROL: and lands exactly where the never-damaged network lands",
                       all([abs(both["loss"] - well["loss"]) < 1e-3, well["converged"]]),
                       f"the same architecture built without the bug ends at loss "
                       f"{well['loss']:.4f} / acc {well['acc']:.1%} from {well['before']}/{n} "
                       f"dead units — recovery is a match, not merely an improvement"),
        practice.Check("FINDING: the instruction as written — incoming weights only — does not",
                       all([wo["after"] / n > 0.7, not wo["converged"]]),
                       f"Kaiming on the incoming weights of all {wo['n']} dead units revives "
                       f"{wo['before'] - wo['after']}, leaves {wo['after']}/{n} "
                       f"({wo['after'] / n:.0%}) dead, and still fails overfit_one_batch at loss "
                       f"{wo['loss']:.4f} / acc {wo['acc']:.1%}"),
        practice.Check("MECHANISM: Kaiming scales from fan_in and knows nothing about the bias",
                       all([wo["reach"][0] > -BIAS, max(wo["reach"][1:]) < -BIAS]),
                       f"a resurrected unit has to push w.x past {-BIAS} to clear the bias; per "
                       f"layer the best it manages is "
                       + ", ".join(f"{v:.2f}" for v in wo["reach"])
                       + ". Only layer 1, fed by raw inputs, gets there"),
        practice.Check("MECHANISM: and a dead unit stays dead — its gradient is exactly 0",
                       all([sick["frozen"] == 0.0, sick["end"] == sick["before"]]),
                       f"summed |grad| over every dead unit's weight row and bias is "
                       f"{sick['frozen']!r}, and 200 Adam steps move the dead count from "
                       f"{sick['before']} to {sick['end']}: ReLU'(z) = 0 below zero"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
