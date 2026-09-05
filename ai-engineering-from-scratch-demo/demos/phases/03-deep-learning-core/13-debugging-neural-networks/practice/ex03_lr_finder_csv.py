"""Exercise 3 — the LR finder's results as a CSV, and what its suggestion is a function of.

    **Implement the learning rate finder with plotting.** Extend
    `find_learning_rate` to save results as a CSV and write a separate script that
    reads the CSV and displays the LR vs loss curve using matplotlib. Identify the
    optimal LR for ResNet-18 on CIFAR-10.

Reading of the exercise: matplotlib is not a dependency of this repo and a plot cannot be
graded, so check 1 grades the artifact the plot script would read — the CSV, round-tripped
exactly. ResNet-18 on CIFAR-10 needs a download and a GPU, so the sweep runs on a seeded
CIFAR-shaped blob and check 5 carries the real command. Checks 2-3 are what the curve and the
suggestion turn out to be.
"""

from __future__ import annotations

import csv
import pathlib
import tempfile

from harness import parity, practice

try:
    import torch
except ImportError as exc:                       # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "03-deep-learning-core", "13-debugging-neural-networks"
N, FEATS, CLASSES, SEP, HIDDEN = 512, 3072, 10, 0.08, 64
START, END, OFFSET, SWEEPS = 1e-7, 10.0, 10, (100, 200, 400)
HOST, PRICE, GPU_MIN = "A100-40GB", 1.29, 6


def blobs(seed=0) -> tuple:
    """CIFAR-10's shapes — 3072 features, 10 classes — with none of CIFAR-10's download."""
    gen = torch.Generator().manual_seed(seed)
    mid = torch.randn(CLASSES, FEATS, generator=gen) * SEP
    y = torch.arange(N) % CLASSES
    return mid[y] + torch.randn(N, FEATS, generator=gen), y


def net(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(FEATS, HIDDEN), torch.nn.ReLU(),
                               torch.nn.Linear(HIDDEN, CLASSES))


def sweep(dbg, x, y, steps) -> list:
    """The lesson's own find_learning_rate, silenced."""
    with parity.quiet():
        return dbg.find_learning_rate(net(), x, y, torch.nn.CrossEntropyLoss(),
                                      start_lr=START, end_lr=END, steps=steps)


def round_trip(results) -> list:
    """The exercise's CSV: two columns, written and read back by the plotting script."""
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "lr_finder.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["lr", "loss"])
            writer.writerows([f"{lr!r}", f"{loss!r}"] for lr, loss in results)
        with path.open(encoding="utf-8") as handle:
            return [(float(lr), float(loss)) for lr, loss in list(csv.reader(handle))[1:]]


def suggestion(results) -> tuple:
    """`find_learning_rate`'s own rule: ten samples back from the minimum."""
    low = min(range(len(results)), key=lambda i: results[i][1])
    return results[max(0, low - OFFSET)][0], results[low][0], low


def ladder(x, y, rate, steps=100) -> float:
    """The control the finder never runs: the same budget held at one rate."""
    model, crit = net(), torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=rate)
    for _ in range(steps):
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()
    with torch.no_grad():
        return float(crit(model(x), y))


def restored(dbg, x, y) -> float:
    """Worst parameter drift across a whole sweep — the function restores its state_dict."""
    model, before = net(), [p.detach().clone() for p in net().parameters()]
    with parity.quiet():
        dbg.find_learning_rate(model, x, y, torch.nn.CrossEntropyLoss(), steps=SWEEPS[0])
    after = [p.detach() for p in model.parameters()]
    return max(float((a - b).abs().max()) for a, b in zip(before, after))


def solve():
    dbg = parity.load_reference(PHASE, LESSON, "debug_neural_nets")
    x, y = blobs()
    runs = {steps: sweep(dbg, x, y, steps) for steps in SWEEPS}
    base = runs[SWEEPS[0]]
    picks = {steps: suggestion(rows) for steps, rows in runs.items()}
    return {"picks": picks, "lengths": {s: len(r) for s, r in runs.items()},
            "rises": {s: sum(1 for i in range(1, len(r)) if r[i][1] > r[i - 1][1])
                      for s, r in runs.items()},
            "csv": round_trip(base) == base, "n": len(base), "restore": restored(dbg, x, y),
            "grid": {rate: ladder(x, y, rate) for rate in (1e-3, picks[SWEEPS[0]][0], 1.0)}}


def offsets(picks) -> tuple:
    """The measured suggestion-to-minimum ratio against its closed form, per sweep."""
    ratios = {s: low / pick for s, (pick, low, _i) in picks.items()}
    closed = {s: (END / START) ** (OFFSET / s) for s in SWEEPS}
    listing = "; ".join(f"{s}: {picks[s][0]:.3f} / {picks[s][1]:.3f} / {ratios[s]:.2f}x"
                        for s in SWEEPS)
    return ratios, max(abs(ratios[s] / closed[s] - 1) for s in SWEEPS), listing


def verify(result):
    picks, grid = result["picks"], result["grid"]
    ratios, off, listing = offsets(picks)
    return [
        practice.Check("ANSWER: the CSV the plot script reads round-trips exactly",
                       result["csv"],
                       f"{result['n']} (lr, loss) pairs written at full repr precision and read "
                       f"back compare equal as floats, pair for pair. The plot is one call — "
                       f"`plt.plot(lr, loss); plt.xscale('log')` — over those two columns"),
        practice.Check("FINDING: 'just before the loss starts climbing' points at nothing here",
                       all(result["rises"][s] == 0 for s in SWEEPS)
                       and all(picks[s][2] == result["lengths"][s] - 1 for s in SWEEPS),
                       f"across {START:g} to {END:g} the loss never rises once — "
                       + ", ".join(f"{result['rises'][s]}/{result['lengths'][s] - 1}"
                                   for s in SWEEPS)
                       + " — and the minimum is the *last* sample of every sweep; the divergence "
                       "guard never fires, all three sweeps running their full length"),
        practice.Check("FINDING: so the suggested LR is a function of the sweep resolution",
                       off < 0.01,
                       f"suggested / minimum / ratio by step count — {listing}. "
                       f"`results[min_idx - {OFFSET}]` is a fixed *step* offset, so the ratio is "
                       f"exactly (end/start)^({OFFSET}/steps) — matched to {100 * off:.2f}%. The "
                       f"suggestion moves "
                       f"{picks[SWEEPS[-1]][0] / picks[SWEEPS[0]][0]:.1f}x across the three "
                       f"sweeps; the curve does not move"),
        practice.Check("CONTROL: the model really is restored, so the sweep costs nothing",
                       result["restore"] == 0.0,
                       f"it deep-copies `state_dict()` and loads it back at the end: every "
                       f"parameter is bit-identical afterwards ({result['restore']:.1f} worst "
                       f"difference). What it does not restore is the caller's optimizer"),
        practice.Check("MECHANISM: the sweep is full-batch, so its rate is for a batch size the "
                       "training loop will not use",
                       grid[1e-3] > 10 * grid[1.0],
                       f"it calls `model(x_data)` on all {N} rows at once. Held at one rate for "
                       f"the same {SWEEPS[0]} full-batch steps: "
                       + ", ".join(f"lr {r:g} -> {v:.4f}" for r, v in grid.items())
                       + f". The real target is this file with `blobs` replaced by "
                       f"`torchvision.datasets.CIFAR10` and `net` by "
                       f"`torchvision.models.resnet18(num_classes=10)` — about {GPU_MIN} GPU-min "
                       f"on an {HOST} at ${PRICE:.2f}/h, ~${PRICE * GPU_MIN / 60:.2f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
