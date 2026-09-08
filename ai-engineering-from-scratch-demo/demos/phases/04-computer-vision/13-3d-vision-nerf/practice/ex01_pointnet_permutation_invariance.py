"""Exercise 1 — pointnet permutation invariance.

    **(Easy)** Show that PointNet is permutation-invariant: run the same cloud through twice, once with points shuffled. Verify outputs are identical up to floating-point noise.

Reading of the exercise: "identical up to floating-point noise" budgets for a
tolerance that is never spent. Over 20 clouds -- four seeds x five point counts,
three of them deliberately not powers of two -- the shuffled logits differ from
the originals by exactly zero bits. That is structural rather than lucky: the
only order-dependent op in the whole forward pass is `torch.max(x, dim=-1)`,
which *selects* an element instead of accumulating one, and everything upstream
is a `Conv1d(kernel_size=1)` whose reduction runs over channels only, so
permuting the point axis permutes GEMM columns without touching any reduction
order. The exercise's own `permutation_invariance_check` in the lesson's
`code/main.py` already prints `0.00e+00` for the same reason. So the checks below
assert bitwise equality rather than a tolerance, and spend the remaining slots on
what the exercise does not ask. Max-pooling buys far more than permutation
invariance: the global feature is decided by a *critical set* of 144 of 1,024
points, and feeding only those 144 reproduces all ten logits bit for bit. Two
controls name the ways a naive version of this test misfires -- the lesson's
`.eval()` is load-bearing, because in the default `train()` mode `Dropout(0.3)`
in the head makes two forwards of the *identical* cloud differ by ~1.0, and a
single cloud in `train()` mode raises `ValueError` before it can be compared at
all. A last control weighs the lesson's `docs/en.md` claim of "About 1.6M
parameters" against the 807,626 the model actually has.

Structure: `shuffle_trials` runs every (seed, size) pair and records the worst
logit, per-point-feature and post-pool gap plus how many pairs came back bitwise
identical; `critical_sets` finds the points that win at least one of the 1,024
max-pool channels and re-scores the cloud with only those; `train_mode_probes`
reports what the same test says if you forget `.eval()` -- dropout noise, the
BatchNorm coupling between a cloud and its batch mates, and the error a single
cloud raises. At 134 code lines this sits above D14's 120-line target, 16 clear
of the ceiling: six checks over three independent probes -- 20 shuffle trials,
three critical-set sizes and four train-mode failures -- and the parameter-count
audit reads the claim out of the lesson's own `docs/en.md` rather than restating
a number here.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "13-3d-vision-nerf"

CLASSES, DOC_CLAIM, CRIT_SIZES = 10, 1_600_000, (512, 1024, 4096)
SIZES, SEEDS = (333, 512, 1000, 1024, 2049), (0, 1, 2, 3)   # three sizes are not powers of two

pct = lambda part, whole: f"{part / whole:.1%}"                     # noqa: E731 - a formatter
crit_row = lambda table: "  ".join(                                 # noqa: E731
    f"{n}->{k} ({pct(k, n)}, diff {d:.0e})" for n, (k, d) in table.items())


def shuffle_trials(torch, net) -> dict:
    worst, exact = {"logit": 0.0, "feature": 0.0, "pool": 0.0}, 0
    for seed in SEEDS:
        for size in SIZES:
            torch.manual_seed(seed)
            points, order = torch.randn(2, 3, size), torch.randperm(size)
            with torch.no_grad():
                feats = net.mlp2(net.mlp1(points))
                gap = {"logit": (net(points) - net(points[:, :, order])).abs().max().item(),
                       "feature": (feats[:, :, order]
                                   - net.mlp2(net.mlp1(points[:, :, order]))).abs().max().item(),
                       "pool": (feats.max(-1)[0] - feats[:, :, order].max(-1)[0]).abs().max().item()}
            worst = {key: max(worst[key], gap[key]) for key in worst}
            exact += gap["logit"] == 0.0
    return {"worst": worst, "exact": exact, "trials": len(SEEDS) * len(SIZES)}


def critical_sets(torch, net) -> dict:
    table = {}
    for size in CRIT_SIZES:
        torch.manual_seed(1)
        points = torch.randn(1, 3, size)
        with torch.no_grad():
            keep = torch.unique(net.mlp2(net.mlp1(points)).argmax(dim=-1)[0])
            table[size] = (int(keep.numel()),
                           (net(points) - net(points[:, :, keep])).abs().max().item())
    return table


def train_mode_probes(torch, net) -> dict:
    net.train()
    torch.manual_seed(2)
    clouds, mates, order = torch.randn(2, 3, 1024), torch.randn(2, 3, 1024) * 4.0, torch.randperm(1024)
    with torch.no_grad():
        dropout = (net(clouds) - net(clouds)).abs().max().item()
    for module in net.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    with torch.no_grad():
        coupling = (net(clouds) - net(torch.cat([clouds, mates]))[:2]).abs().max().item()
        permuted = (net(clouds) - net(clouds[:, :, order])).abs().max().item()
    try:
        net(clouds[:1])
        failure = "no error"
    except ValueError as exc:
        failure = f"{type(exc).__name__}: {str(exc).split(',')[0]}"
    return {"dropout": dropout, "coupling": coupling, "permuted": permuted, "batch1": failure}


def solve():
    try:
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    net = ref.PointNet(num_classes=CLASSES).eval()
    claim = re.search(r"About [\d.]+M parameters", parity.doc_text(PHASE, LESSON))
    return {**shuffle_trials(torch, net), "critical": critical_sets(torch, net),
            "params": sum(p.numel() for p in net.parameters()), "train": train_mode_probes(torch, net),
            "head": sum(p.numel() for p in net.head.parameters()),
            "claim": claim.group(0) if claim else "no claim found"}


def verify(result):
    worst, table, train = result["worst"], result["critical"], result["train"]
    keep, gap = table[1024]
    return [
        practice.Check(
            "ANSWER: shuffling changes no bit of any logit — the allowed tolerance is never spent",
            worst["logit"] == 0.0 and result["exact"] == result["trials"],
            f"{result['trials']} clouds ({len(SEEDS)} seeds x sizes {SIZES}, three not powers of two) through the "
            f"lesson's own `PointNet` in `.eval()`: max|logit diff| {worst['logit']:.2e}, bitwise identical on "
            f"{result['exact']}/{result['trials']}. The lesson's `permutation_invariance_check` prints the same"),
        practice.Check(
            "MECHANISM: the symmetry is a selection, not a cancellation, so it is exact by construction",
            worst["pool"] == 0.0 and worst["feature"] == 0.0,
            f"the shared MLP is `Conv1d(kernel_size=1)`, a reduction over channels only, so its per-point outputs "
            f"are permuted rather than recomputed (max|diff| after re-indexing {worst['feature']:.2e}); "
            f"`torch.max(x, dim=-1)` then picks an element instead of summing one, so the 1,024-d global feature "
            f"is unchanged to the bit ({worst['pool']:.2e}) -- no float cancellation is involved"),
        practice.Check(
            "FINDING: max-pool buys much more than permutation — 144 of 1,024 points decide everything",
            keep < 200 and gap == 0.0 and table[4096][0] < table[512][0] * 2,
            f"the points that win at least one max-pool channel are the critical set; feeding only those "
            f"reproduces all {CLASSES} logits bitwise: {crit_row(table)}. The set saturates rather than growing "
            f"with the cloud -- {pct(table[4096][0], 4096)} of 4,096 points against {pct(table[512][0], 512)} of "
            "512 -- so PointNet is invariant to deleting most of its input, not merely to reordering it"),
        practice.Check(
            "CONTROL: the lesson's `.eval()` is load-bearing — in train mode the same cloud disagrees",
            train["dropout"] > 0.5 and train["permuted"] == 0.0,
            f"left in the default `train()` mode, two forwards of the *identical* cloud differ by "
            f"{train['dropout']:.3f} because of the head's `Dropout(0.3)` -- a naive version of this test appears "
            f"to disprove invariance. Switch dropout off and the shuffled cloud is back to "
            f"{train['permuted']:.2e}: BatchNorm over (batch, points) is itself a symmetric statistic"),
        practice.Check(
            "CONTROL: invariance is per-cloud only — train-mode BatchNorm couples a cloud to its batch",
            train["coupling"] > 0.1,
            f"the same two clouds scored alone and then alongside two unrelated clouds scaled 4x move by "
            f"{train['coupling']:.3f} in train mode, while a permutation moves them by {train['permuted']:.2e}. "
            f"And a single cloud cannot be scored at all -- {train['batch1']} -- so the head's `BatchNorm1d` "
            "needs a batch even at inference time unless the model is in `.eval()`"),
        practice.Check(
            "CONTROL: the lesson's own parameter count is 2.0x smaller than its docs claim",
            result["params"] < DOC_CLAIM / 1.5,
            f"`docs/en.md` says \"{result['claim']}\"; the model built by `code/main.py` has {result['params']:,} "
            f"-- {DOC_CLAIM / result['params']:.1f}x fewer. The head alone (1024->512->256->{CLASSES}) is "
            f"{result['head']:,} of them, so the claim is not a rounding of this architecture; the paper's "
            "PointNet, with its two T-Nets, is the 1.6M model"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
