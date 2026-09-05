"""Exercise 2 — a learning rate finder, and what its answer is worth.

    **Implement a learning rate finder.** Train for one epoch with exponentially
    increasing learning rate (from 1e-7 to 1.0). Plot loss vs LR. The optimal LR
    is just before the loss starts climbing. Use this to pick a better LR for the
    MNIST model.

Reading of the exercise: a plot is not checkable, so it becomes the numbers a reader takes
off it — where the smoothed loss leaves its plateau and where it bottoms out. "Use this to
pick a better LR" is the testable half, and its control is a matched-budget ladder: same
init, same epoch, one fixed LR each. The fixture is a seeded 784-D 10-class Gaussian blob.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
except ImportError as exc:                       # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "03-deep-learning-core", "11-intro-to-pytorch"
N_TRAIN, N_TEST, SEP, BS, SEEDS = 2048, 2000, 0.18, 16, (0, 1, 2)
LO, HI, LADDER = 1e-7, 1.0, (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def _blobs(seed):
    gen = torch.Generator().manual_seed(seed)
    mid = torch.randn(10, 784, generator=gen) * SEP
    ys = [torch.arange(n) % 10 for n in (N_TRAIN, N_TEST)]
    xs = [mid[y] + torch.randn(len(y), 784, generator=gen) for y in ys]
    return xs[0], ys[0], xs[1], ys[1]


def _sweep(ref, loader, seed):
    """One epoch; the LR is multiplied by a constant factor every step, LO -> HI."""
    torch.manual_seed(seed)
    model, crit = ref.MNISTModel(), torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LO)
    n, lrs, smooth, avg = len(loader), [], [], 0.0
    for i, (x, y) in enumerate(loader):
        opt.param_groups[0]["lr"] = LO * (HI / LO) ** (i / (n - 1))
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward(); opt.step()                  # noqa: E702
        avg = 0.9 * avg + 0.1 * loss.item()
        lrs.append(opt.param_groups[0]["lr"])
        smooth.append(avg / (1 - 0.9 ** (i + 1)))     # de-biased, as fastai does it
    return lrs, smooth


def _grid(ref, loaders, lrs, seed):
    """The control the finder never runs: one fixed LR per model, same init, same budget."""
    crit, dev, out = torch.nn.CrossEntropyLoss(), torch.device("cpu"), {}
    for lr in lrs:
        torch.manual_seed(seed)
        model = ref.MNISTModel()
        ref.train_one_epoch(model, loaders[0], crit,
                            torch.optim.Adam(model.parameters(), lr=lr), dev)
        out[lr] = ref.evaluate(model, loaders[1], crit, dev)[1]
    return out


def _probe(ref, seed):
    loaders = ref.create_loaders(*_blobs(seed), batch_size=BS)
    lrs, sm = _sweep(ref, loaders[0], seed)
    n = len(lrs)
    j = min(range(n), key=lambda i: sm[i])
    base = sorted(sm[:16])[8]                        # the plateau the sweep starts on
    live = next(i for i, s in enumerate(sm) if s < 0.95 * base)
    grid = _grid(ref, loaders, LADDER + (lrs[j],), seed)
    return {"n": n, "lr_min": lrs[j], "loss_min": sm[j], "end": sm[-1], "base": base,
            "live": live / n, "live_lr": lrs[live], "acc_min": grid[lrs[j]],
            "best": max(grid, key=lambda k: grid[k]), "acc_1e3": grid[1e-3],
            "at1e3": sm[min(range(n), key=lambda i: abs(lrs[i] - 1e-3))],
            "near": sum(1 for lr in lrs if 1e-3 / 3 < lr < 3e-3), "acc_3e3": grid[3e-3],
            "ratio": lrs[j] / 1e-3, "gap": grid[1e-3] - grid[lrs[j]]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "pytorch_intro")
    seeds = [_probe(ref, s) for s in SEEDS]
    return {"one": seeds[0], "lo": {k: min(d[k] for d in seeds) for k in seeds[0]},
            "hi": {k: max(d[k] for d in seeds) for k in seeds[0]}}


def verify(result):
    one, lo, hi = result["one"], result["lo"], result["hi"]
    return [
        practice.Check("ANSWER: the loss bottoms near 2e-2, after a plateau measuring nothing",
                       0.5 < lo["live"] and 5e-3 < lo["lr_min"] <= hi["lr_min"] < 5e-2,
                       f"{one['n']} steps, 1e-7 to 1.0: a plateau at {one['base']:.3f} until lr="
                       f"{one['live_lr']:.1e} ({lo['live']:.0%} of the epoch), minimum "
                       f"{one['loss_min']:.3f} at {one['lr_min']:.2e}, {one['end']:.1f} at lr=1"),
        practice.Check("FINDING: the finder's LR is a 15-21 point regression on 1e-3",
                       lo["best"] == hi["best"] == 1e-3 and 0.1 < lo["gap"] and 10 < lo["ratio"],
                       f"same init, same epoch, one fixed LR each: 1e-3 -> {one['acc_1e3']:.4f} "
                       f"(the winner on every seed), 3e-3 -> {one['acc_3e3']:.4f}, the finder's "
                       f"{one['lr_min']:.2e} -> {one['acc_min']:.4f}, {lo['ratio']:.0f}x too "
                       f"high, costing {lo['gap']:.4f} to {hi['gap']:.4f} test accuracy"),
        practice.Check("MECHANISM: the curve at LR x is a model trained at every LR below x",
                       0.95 * hi["base"] < lo["at1e3"] and hi["near"] < 20,
                       f"only {one['near']} of {one['n']} steps land within 3x of 1e-3, so it "
                       f"reads {one['at1e3']:.3f} there — its opening plateau — while a run held "
                       f"at 1e-3 for the same {one['n']} steps reaches {one['acc_1e3']:.4f}"),
        practice.Check("CONTROL: stable across seeds, so biased not noisy; lr/10 misses too",
                       hi["lr_min"] / lo["lr_min"] < 1.5 and hi["acc_3e3"] < lo["acc_1e3"],
                       f"three seeds put the minimum in {lo['lr_min']:.2e}-{hi['lr_min']:.2e}, "
                       f"spread {hi['lr_min'] / lo['lr_min']:.2f}x, and name 1e-3 the ladder "
                       f"winner each time. fastai's lr/10 lands at {one['lr_min'] / 10:.1e}; "
                       f"the rung 3e-3 tops out at {hi['acc_3e3']:.4f} vs {lo['acc_1e3']:.4f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
