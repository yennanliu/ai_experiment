"""Exercise 1 — bce dice convergence speed.

    **(Easy)** Implement `bce_dice_loss` for a binary segmentation task
    (foreground vs background). Verify on a synthetic two-class dataset that the
    combined loss converges faster than BCE alone when the foreground is 5% of
    pixels.

Reading of the exercise: the lesson's own `synthetic_segmentation` is the
two-class dataset once the circle/square labels are merged, and 5% foreground is
not a knob on it -- it is a consequence of the image size, since the shapes keep
a fixed radius. `size=86` puts the foreground at 5.2% of pixels, so that is the
size used, and the number is measured rather than assumed. "Converges faster"
cannot be read off the loss curves: the two arms minimise different functionals
with different floors, so both are scored on the same held-out foreground IoU
instead. The real trap is that the exercise does not name an optimiser. Dice
exists to fix a gradient-*magnitude* imbalance, and Adam -- which the lesson's
own `main()` uses -- normalises gradient magnitude away, so the claim is true
under SGD and false under Adam. Both are run, on the same initialisation, the
same batch order and two seeds. Training is scaled down to a base-8 U-Net,
48 images and 7 epochs to keep the lesson under its runtime budget; the closed
form behind the result is computed exactly and needs no training at all. The
file stays 18 lines over D14's 120-line target, and what those lines buy is the
exercise's second and third deliverables. `arm` -- the training comparison it
asks for -- is 18 lines, and it has to run four times, because the Adam control
that is the point of this exercise is the same loop under a different optimiser.
The constant-logit sweep and the autograd slopes that explain the result are
another 10, across `score_at` and two lines of `solve`; without them the
"converges faster" claim would be asserted rather than measured. The six checks
are 44 lines, 19 of which are the measured numbers they quote.

How the pieces below fit together. `dice_term` is soft Dice on the single
foreground channel, scored once per image; `bce_dice_loss` is the loss the
exercise asks for, and taking `lam=0` gives the BCE-alone arm so both arms run
the same code path. `arm` is one training run, evaluated on held-out foreground
IoU after every epoch. `score_at` scores both terms at a single constant-logit
predictor -- the "predict background everywhere" shortcut -- and with `grad=True`
returns their derivative with respect to that logit instead. `solve` builds the
data by merging the lesson's circle and square labels into one foreground, runs
the four arms, sweeps the constant-logit predictors, and checks the Dice term
against the lesson's own `dice_loss` on a 16x16 patch: averaging the foreground
term with the same term on the complement is algebraically the lesson's
two-class Dice, so agreement there is a parity test rather than a restatement.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "07-semantic-segmentation-unet"

SIZE, SAMPLES, SPLIT = 86, 64, 48       # size 86 is what puts the foreground near 5%
BASE, EPOCHS, BATCH, SEEDS = 8, 7, 8, (0, 1)
ARMS = (("sgd", 0.05), ("adam", 1e-3))  # adam at 1e-3 is the lesson's own main()
LO, HI, GRID, TARGET = -6.0, 6.0, 241, 0.5   # the constant-logit sweep, and the IoU to reach

mean = lambda values: sum(values) / len(values)                                   # noqa: E731
series = lambda runs: " / ".join(" ".join(f"{v:.3f}" for v in r) for r in runs)    # noqa: E731
scores = lambda runs: {k: mean([mean(r) for r in v]) for k, v in runs.items()}     # noqa: E731
gains = lambda score: {n: score[n, 1.0] - score[n, 0.0] for n, _ in ARMS}          # noqa: E731
first = lambda run: next((i + 1 for i, v in enumerate(run) if v >= TARGET), 99)    # noqa: E731
epochs = lambda runs: {k: [first(r) for r in v] for k, v in runs.items()}          # noqa: E731


def dice_term(torch, logits, targets, eps=1e-6):
    probs, truth = torch.sigmoid(logits).flatten(1), targets.float().flatten(1)
    overlap = 2 * (probs * truth).sum(1) + eps
    return 1 - (overlap / (probs.sum(1) + truth.sum(1) + eps)).mean()


def bce_dice_loss(torch, functional, logits, targets, lam=1.0):
    return (functional.binary_cross_entropy_with_logits(logits, targets.float())
            + lam * dice_term(torch, logits, targets))


def arm(torch, functional, ref, data, optimiser, lam, seed):
    xtr, ytr, xva, yva = data
    torch.manual_seed(seed)
    net, (name, lr) = ref.UNet(3, 1, base=BASE), optimiser
    opt = (torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9) if name == "sgd"
           else torch.optim.Adam(net.parameters(), lr=lr))
    gen, curve, truth = torch.Generator().manual_seed(seed), [], yva.bool()
    for _ in range(EPOCHS):
        net.train()
        for start in range(0, SPLIT, BATCH):
            batch = torch.randperm(SPLIT, generator=gen)[start:start + BATCH]
            opt.zero_grad()
            bce_dice_loss(torch, functional, net(xtr[batch]), ytr[batch], lam).backward()
            opt.step()
        with torch.no_grad():
            hit = net.eval()(xva) > 0
        curve.append(float((hit & truth).sum() / max(int((hit | truth).sum()), 1)))
    return curve


def score_at(torch, functional, ytr, logit, grad=False):
    field = torch.full(ytr.shape, logit, requires_grad=grad)
    bce, dice = bce_dice_loss(torch, functional, field, ytr, lam=0.0), dice_term(torch, field, ytr)
    if grad:
        return [float(torch.autograd.grad(t, field, retain_graph=True)[0].sum()) for t in (bce, dice)]
    return 1 / (1 + math.exp(-logit)), float(bce), float(dice)


def solve():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    images, masks = ref.synthetic_segmentation(num_samples=SAMPLES, size=SIZE, seed=0)
    x, y = (torch.from_numpy(images).permute(0, 3, 1, 2),
            torch.from_numpy((masks > 0).astype("int64")).unsqueeze(1))
    data, ytr = (x[:SPLIT], y[:SPLIT], x[SPLIT:], y[SPLIT:]), y[:SPLIT]
    fraction, per_image = float(ytr.float().mean()), ytr.float().flatten(1).mean(1)
    runs = {(opt[0], lam): [arm(torch, functional, ref, data, opt, lam, s) for s in SEEDS]
            for opt in ARMS for lam in (0.0, 1.0)}
    sweep = [score_at(torch, functional, ytr, LO + (HI - LO) * i / (GRID - 1)) for i in range(GRID)]
    patch, mask = torch.randn(1, 1, 16, 16), ytr[:1, :, :16, :16]
    mine = 0.5 * (float(dice_term(torch, patch, mask)) + float(dice_term(torch, -patch, 1 - mask)))
    pair = torch.cat([torch.zeros_like(patch), patch], dim=1)   # the same image as 2-class logits
    return {"runs": runs, "fg": fraction, "shortcut": min(sweep, key=lambda row: row[1]),
            "half": min(sweep, key=lambda row: abs(row[0] - 0.5)),
            "slopes": score_at(torch, functional, ytr, math.log(fraction / (1 - fraction)), True),
            "spread": (float(per_image.min()), float(per_image.max())),
            "parity": (mine, float(ref.dice_loss(pair, mask[:, 0], 2)))}


def verify(result):
    runs, fg, (bce_slope, dice_slope) = result["runs"], result["fg"], result["slopes"]
    (p_star, bce_star, dice_star), (_, bce_half, dice_half) = result["shortcut"], result["half"]
    score, cross, (lo, hi) = scores(runs), epochs(runs), result["spread"]
    gain, (mine, lesson) = gains(score), result["parity"]
    return [
        practice.Check(
            "ANSWER: `bce_dice_loss` is BCE plus 1 - soft Dice, and its Dice term is the lesson's",
            abs(mine - lesson) < 1e-6,
            f"averaged with the same term on the complement it is the lesson's two-class `dice_loss`: "
            f"{mine:.9f} vs {lesson:.9f}, off by {abs(mine - lesson):.0e}"),
        practice.Check(
            "ANSWER: under SGD the combined loss does converge faster, at ~5% foreground",
            gain["sgd"] > 0.10 and mean(cross["sgd", 1.0]) < mean(cross["sgd", 0.0]),
            f"foreground {fg:.4f} of pixels at size {SIZE}. Held-out IoU per epoch under SGD -- BCE alone "
            f"{series(runs['sgd', 0.0])}; BCE+Dice {series(runs['sgd', 1.0])} -- mean over {EPOCHS} epochs, "
            f"{len(SEEDS)} seeds {score['sgd', 0.0]:.3f} -> {score['sgd', 1.0]:.3f}, epochs to IoU "
            f"{TARGET} {cross['sgd', 0.0]} -> {cross['sgd', 1.0]}"),
        practice.Check(
            "MECHANISM: the majority-class shortcut lowers BCE and *raises* Dice",
            bce_star < 0.35 * bce_half and dice_star > dice_half,
            f"over {GRID} constant-logit predictors on the real masks BCE bottoms out at p={p_star:.4f}, the "
            f"{fg:.4f} foreground rate, at {bce_star:.4f} nats against {bce_half:.4f} for p=0.5 -- "
            f"{1 - bce_star / bce_half:.0%} of the descent to zero. Dice there is {dice_star:.4f}, *worse* "
            f"than the {dice_half:.4f} it scores at p=0.5"),
        practice.Check(
            "MECHANISM: that shortcut is a stationary point of BCE and not of the combined loss",
            abs(bce_slope) < 0.01 * abs(dice_slope) and dice_slope < 0,
            f"autograd through a constant logit field at exactly p={fg:.4f}: d(BCE)/d(logit) {bce_slope:+.2e}, "
            f"d(Dice)/d(logit) {dice_slope:+.5f} -- BCE has nothing left to gain, Dice is still pushing"),
        practice.Check(
            "CONTROL: under Adam -- the lesson's own optimiser -- the advantage disappears",
            abs(gain["adam"]) < 0.4 * gain["sgd"] and cross["adam", 1.0] == cross["adam", 0.0],
            f"same data, initialisations and batch order, Adam at 1e-3: BCE alone {score['adam', 0.0]:.3f}, "
            f"BCE+Dice {score['adam', 1.0]:.3f}, a move of {gain['adam']:+.3f} against {gain['sgd']:+.3f} "
            f"under SGD, both arms reaching IoU {TARGET} in epochs {cross['adam', 0.0]}. Adam divides "
            "gradient magnitude out per parameter -- the one thing Dice was added to fix"),
        practice.Check(
            "CONTROL: '5% of pixels' is a fact about the set, not about any image in it",
            hi > 2.5 * lo,
            f"per-image foreground over the {SPLIT} training images runs {lo:.4f} to {hi:.4f}, a "
            f"{hi / lo:.1f}x range around the {fg:.4f} mean; Dice is per image, so the imbalance it sees is "
            "that whole range, not the one number the exercise quotes"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
