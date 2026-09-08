"""Exercise 2 — pooled motion ceiling.

    **(Medium)** Generate a synthetic video dataset: random balls moving in random directions, labelled by direction of motion ("left-to-right", "right-to-left", "diagonal-up"). Train FramePool on it. Show that it achieves near-chance accuracy, proving appearance alone is insufficient for motion tasks.

Reading of the exercise: "near-chance" is the claim, and run literally on the
three classes the exercise names it is false -- FramePool measures 0.625 and
0.667 on two seeds against a chance of 0.333, because only two of the three
classes are order-defined. "diagonal-up" visits a different set of y positions
from either horizontal class, so a per-frame model separates it on appearance
alone and the honest ceiling is 2/3, not 1/3. The claim survives exactly on the
pair it is really about: each right-to-left clip here is generated as the
bit-exact time-reversal of a left-to-right clip, so the two classes are the same
multiset of frames (measured: sorted-pixel difference 0.0) and mean-pooling over
time is permutation-invariant, which is checked directly -- a clip and its
reversal get the same logits to about 1e-06 against logit scales near 10 after
training, and to 1.5e-08 at random init. That makes the two classes share one row of the
confusion matrix, so their recalls sum to at most 1 by construction, whatever the
optimiser does. The dataset is cut hard for CPU: 16x16 frames, T=4, 24 clips per
class for train and again for test, batch 8, 8 epochs, ~35 s for both seeds on two CPU threads;
`FramePool(pretrained=False)` so no ImageNet checkpoint is downloaded and the run
starts from random init. `clips` builds a Gaussian ball; `fit` runs Adam on a model factory
(exercise 3 reuses it); `centroid_rule` is a zero-parameter read-out of the
intensity centroid's displacement, and `confuse` tallies a confusion matrix.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "12-video-understanding"

SIZE, T, RADIUS, SPAN, RISE = 16, 4, 2.0, 7.0, 6.0
CLASSES, PER_CLASS, EPOCHS, BATCH, LR, SEEDS = 3, 24, 8, 8, 1e-3, (0, 1)
NAMES = ("left-to-right", "right-to-left", "diagonal-up")

blob = lambda np, gx, gy, cx, cy: np.exp(                       # noqa: E731 - a Gaussian ball
    -((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * RADIUS ** 2))
recalls = lambda table: [float(table[c][c]) / float(table[c].sum()) for c in range(CLASSES)]  # noqa: E731
joined = lambda values, fmt: "/".join(format(v, fmt) for v in values)               # noqa: E731
column = lambda arms, key, fmt: joined([arms[s][key] for s in SEEDS], fmt)          # noqa: E731
accs = lambda arms: [arms[s]["accuracy"] for s in SEEDS]                            # noqa: E731
pairs = lambda arms: [arms[s]["recall"][0] + arms[s]["recall"][1] for s in SEEDS]   # noqa: E731
by_class = lambda arms: "; ".join(f"seed {s} " + joined(arms[s]["recall"], ".3f") for s in SEEDS)  # noqa: E731
by_row = lambda arms: " and ".join(                                                 # noqa: E731
    f"seed {s} {arms[s]['table'][0].tolist()}/{arms[s]['table'][1].tolist()}" for s in SEEDS)
tied = lambda arms: all(arms[s]["table"][0].tolist() == arms[s]["table"][1].tolist() for s in SEEDS)  # noqa: E731


def clips(np, torch, per_class, seed):
    grid_y, grid_x = np.mgrid[0:SIZE, 0:SIZE]
    rng, frames, labels = np.random.default_rng(seed), [], []
    for _ in range(per_class):
        x0 = rng.uniform(RADIUS, SIZE - RADIUS - SPAN)
        y0 = rng.uniform(RADIUS + RISE, SIZE - RADIUS)
        step = [SPAN * t / (T - 1) for t in range(T)]
        walk = np.stack([blob(np, grid_x, grid_y, x0 + s, y0) for s in step])
        rise = np.stack([blob(np, grid_x, grid_y, x0 + s, y0 - s * RISE / SPAN) for s in step])
        frames += [walk, walk[::-1].copy(), rise]
        labels += [0, 1, 2]
    stacked = torch.tensor(np.stack(frames), dtype=torch.float32).unsqueeze(2)
    return stacked.repeat(1, 1, 3, 1, 1), torch.tensor(labels)


def fit(torch, factory, x, y, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    model = factory()
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        for batch in torch.randperm(len(x)).split(BATCH):
            loss = torch.nn.functional.cross_entropy(model(x[batch]), y[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return model


def score(torch, model, x):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + BATCH]) for i in range(0, len(x), BATCH)])


def centroid_rule(torch, x, y) -> float:
    grey, axis = x[:, :, 0], torch.arange(SIZE, dtype=torch.float32)
    mass = grey.sum((2, 3))
    drift_x = ((grey.sum(2) * axis).sum(2) / mass).diff(dim=1).sum(1)
    drift_y = ((grey.sum(3) * axis).sum(2) / mass).diff(dim=1).sum(1)
    guess = torch.where(drift_y < -RISE / 2, 2, torch.where(drift_x > 0, 0, 1))
    return float((guess == y).float().mean())


def arm(torch, ref, train_set, test_set, seed) -> dict:
    model = fit(torch, lambda: ref.FramePool(num_classes=CLASSES, pretrained=False), *train_set, seed)
    x, y = test_set
    forward, backward = score(torch, model, x), score(torch, model, x.flip(1))
    table = torch.bincount(y * CLASSES + forward.argmax(1), minlength=CLASSES ** 2).reshape(CLASSES, -1)
    return {"accuracy": float((forward.argmax(1) == y).float().mean()), "table": table,
            "recall": recalls(table), "gap": float((forward - backward).abs().max()),
            "scale": float(forward.abs().max())}


def solve():
    try:
        import numpy as np
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    train_set, test_set = clips(np, torch, PER_CLASS, 0), clips(np, torch, PER_CLASS, 1)
    x, y = test_set
    torch.manual_seed(7)
    untrained = ref.FramePool(num_classes=CLASSES, pretrained=False)
    fresh, stale = score(torch, untrained, x), score(torch, untrained, x.flip(1))
    same = torch.sort(x[y == 0].flatten(1), dim=1).values - torch.sort(x[y == 1].flatten(1), dim=1).values
    return {"arms": {seed: arm(torch, ref, train_set, test_set, seed) for seed in SEEDS},
            "init_gap": float((fresh - stale).abs().max()), "init_scale": float(fresh.abs().max()),
            "multiset": float(same.abs().max()), "centroid": centroid_rule(torch, x, y),
            "clips": len(x)}


def verify(result):
    arms, chance, ceiling = result["arms"], 1.0 / CLASSES, 2.0 / CLASSES
    accuracy, pair = accs(arms), pairs(arms)
    return [
        practice.Check(
            "ANSWER: 'near-chance' is false -- FramePool lands near 2/3, not near the 1/3 the exercise predicts",
            all(value > 0.55 for value in accuracy),
            f"{EPOCHS} epochs of Adam(lr={LR}) on {PER_CLASS * CLASSES} {SIZE}x{SIZE} T={T} clips, scored on {result['clips']} held-out clips "
            f"from a different seed: FramePool reaches {joined(accuracy, '.3f')} on seeds {SEEDS} against a chance of {chance:.3f}. Only two "
            f"of the three classes are order-defined, so appearance carries real signal here"),
        practice.Check(
            "FINDING: the honest ceiling is 2/3, and both seeds land at or just under it",
            all(value <= ceiling + 1e-6 for value in accuracy),
            f"per-class recall {NAMES}: {by_class(arms)}. The two horizontal classes share one confusion row, so their recalls sum to at most "
            f"1 -- measured {joined(pair, '.3f')} -- while diagonal-up is read straight off appearance: that caps the score at {ceiling:.3f}"),
        practice.Check(
            "MECHANISM: mean-pooling is permutation-invariant, so the shared row is exact and unlearnable",
            tied(arms) and result["init_gap"] < 1e-5 and result["init_gap"] / result["init_scale"] < 1e-4,
            f"confusion rows for {NAMES[0]} / {NAMES[1]} are element-wise identical: {by_row(arms)}. FramePool means the per-frame features "
            f"over time, so a clip and its reversal score within {column(arms, 'gap', '.1e')} of each other against logit scales of "
            f"{column(arms, 'scale', '.2f')} -- float summation noise, not a decision. It is architectural, not learned: one untrained "
            f"FramePool(pretrained=False) is already {result['init_gap']:.2e} apart at scale {result['init_scale']:.3f}, a relative "
            f"{result['init_gap'] / result['init_scale']:.1e}"),
        practice.Check(
            "CONTROL: the two classes are literally the same frames, so nothing appearance-based can separate them",
            result["multiset"] == 0.0,
            f"each {NAMES[1]} clip is generated as the bit-exact time-reversal of its {NAMES[0]} partner, so their sorted pixel multisets "
            f"differ by exactly {result['multiset']:.1f} over {PER_CLASS * T * 3 * SIZE * SIZE:,} values -- any leakage would have to come "
            f"from frame *order*, the one thing mean-pooling discards"),
        practice.Check(
            "CONTROL: the label is fully present in the data -- a zero-parameter motion read-out gets 1.000",
            result["centroid"] == 1.0,
            f"the sign of the intensity centroid's total displacement (dy < {-RISE / 2:.2f} => {NAMES[2]}, else dx > 0) labels "
            f"{result['centroid']:.3f} of the same {result['clips']} test clips with no parameters at all -- the task is not hard, it is "
            f"invisible to the architecture, which is what the exercise set out to show"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
