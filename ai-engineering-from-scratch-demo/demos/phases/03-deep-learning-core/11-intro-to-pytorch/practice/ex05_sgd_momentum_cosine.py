"""Exercise 5 — SGD + momentum against Adam, and what the cosine schedule adds.

    **Replace Adam with SGD + momentum.** Train with `SGD(params, lr=0.01,
    momentum=0.9)`. Compare convergence curves. Then add a `CosineAnnealingLR`
    scheduler and see if SGD catches up to Adam by epoch 10.

Reading of the exercise: `get_mnist_data` downloads from Yann LeCun's server, so the fixture
is a seeded 784-D 10-class Gaussian blob with MNIST's shapes — the lesson's own
`create_loaders`, `train_one_epoch` and `evaluate` run over it unchanged. The exercise names
lr = 0.01 and the lesson's own `experiment_sgd_cosine` uses 0.05, so checks 3-5 run both and
separate the rate from the schedule.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
except ImportError as exc:                       # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "03-deep-learning-core", "11-intro-to-pytorch"
N_TRAIN, N_TEST, SEP, BATCH, EPOCHS = 2048, 1000, 0.18, 64, 10
SLOW, FAST = 0.01, 0.05               # the exercise's rate, and the lesson's own


def blobs(seed=0) -> tuple:
    """Seeded stand-in for MNIST — MNIST's shapes, none of MNIST's download."""
    gen = torch.Generator().manual_seed(seed)
    mid = torch.randn(10, 784, generator=gen) * SEP
    ys = [torch.arange(n) % 10 for n in (N_TRAIN, N_TEST)]
    xs = [mid[y] + torch.randn(len(y), 784, generator=gen) for y in ys]
    return xs[0], ys[0], xs[1], ys[1]


def curve(ref, loaders, build, cosine, seed=0) -> list:
    """Test accuracy after each of the lesson's own epochs."""
    torch.manual_seed(seed)
    model, crit, dev = ref.MNISTModel(), torch.nn.CrossEntropyLoss(), torch.device("cpu")
    opt = build(model)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) if cosine else None)
    out = []
    for _epoch in range(EPOCHS):
        ref.train_one_epoch(model, loaders[0], crit, opt, dev)
        out.append(ref.evaluate(model, loaders[1], crit, dev)[1])
        if sched is not None:
            sched.step()
    return out


def arms(ref, loaders) -> dict:
    sgd = lambda lr: (lambda m: torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9))  # noqa: E731
    plans = {"adam": (lambda m: torch.optim.Adam(m.parameters(), lr=1e-3), False),
             "slow": (sgd(SLOW), False), "slow_cos": (sgd(SLOW), True),
             "fast": (sgd(FAST), False), "fast_cos": (sgd(FAST), True)}
    return {name: curve(ref, loaders, build, cos) for name, (build, cos) in plans.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "pytorch_intro")
    loaders = ref.create_loaders(*blobs(), batch_size=BATCH)
    return {"arms": arms(ref, loaders)}


def show(row) -> str:
    return ", ".join(f"{v:.3f}" for v in row)


def verify(result):
    a = result["arms"]
    adam, slow, slow_cos, fast, fast_cos = (a[k] for k in
                                            ("adam", "slow", "slow_cos", "fast", "fast_cos"))
    return [
        practice.Check("ANSWER: SGD + momentum catches up by epoch 10, and loses the first three",
                       abs(slow[-1] - adam[-1]) < 0.01 and adam[0] - slow[0] > 0.5,
                       f"test accuracy per epoch — Adam(1e-3) {show(adam)}; SGD({SLOW}, 0.9) "
                       f"{show(slow)}. It ends {100 * abs(adam[-1] - slow[-1]):.1f} points apart "
                       f"but takes 3 epochs to pass {slow[2]:.3f} where Adam is at {adam[0]:.3f} "
                       f"after one. Adam's whole advantage is the start"),
        practice.Check("ANSWER: adding CosineAnnealingLR at that rate does not help",
                       slow_cos[-1] <= slow[-1] and all(c <= p + 1e-9 for c, p
                                                        in zip(slow_cos[1:], slow[1:])),
                       f"SGD({SLOW}, 0.9) + cosine {show(slow_cos)} against {show(slow)} without "
                       f"— at or below it at every epoch after the first, ending {slow_cos[-1]:.4f} "
                       f"against {slow[-1]:.4f}. A cosine only lowers the rate, and {SLOW} was "
                       f"never too high"),
        practice.Check("FINDING: the lesson's own SGD+cosine run changes the rate at the same "
                       "time, so its comparison confounds the two",
                       fast[0] - slow[0] > 0.5,
                       f"`experiment_sgd_cosine` uses lr = {FAST} where `experiment_sgd` and the "
                       f"exercise use {SLOW}. Separated: SGD({FAST}, 0.9) with no schedule "
                       f"{show(fast)} — {fast[0]:.3f} after one epoch against {slow[0]:.3f}. The "
                       f"5x rate is what buys the early epochs, not the scheduler"),
        practice.Check("FINDING: at the rate that is too high, the cosine earns its keep",
                       fast[-1] < max(fast) - 0.005 and fast_cos[-1] > fast[-1],
                       f"SGD({FAST}, 0.9) peaks at {max(fast):.3f} and *falls* to {fast[-1]:.3f} "
                       f"by epoch {EPOCHS}; with the cosine it holds {fast_cos[-1]:.3f} "
                       f"({show(fast_cos)}). What the schedule fixes is late-training decay at a "
                       f"rate that was too high to sit at, which is a different claim from "
                       f"'SGD catches up'"),
        practice.Check("CONTROL: the schedule steps per epoch, so epoch 1 is common to both arms",
                       fast[0] == fast_cos[0] and slow[0] == slow_cos[0],
                       f"`CosineAnnealingLR(T_max={EPOCHS})` is stepped once per epoch, after the "
                       f"epoch, so the scheduled and unscheduled arms are the same run until the "
                       f"first `sched.step()` — {fast[0]:.4f} and {slow[0]:.4f} in both, exactly. "
                       f"Any first-epoch difference would have been a seeding bug"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
