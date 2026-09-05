"""Exercise 4 — a data pipeline validator, and what each of its four rules leaves out.

    **Create a data pipeline validator.** Write a function that checks for:
    duplicate samples across train/test splits, label distribution imbalance
    (>10:1 ratio), input normalization (mean near 0, std near 1), and NaN/Inf
    values in the data. Run it on a deliberately corrupted dataset.

Reading of the exercise: the validator is written to the four rules exactly as listed, and
check 1 runs it on a clean split and on one corrupted four ways. Each of the remaining checks
takes one rule and asks what it misses — three of the four are underspecified in a way that
decides whether they fire, and check 5 is the ordering constraint between them.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
except ImportError as exc:                       # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "03-deep-learning-core", "13-debugging-neural-networks"
N_TRAIN, N_TEST, FEATS, CLASSES = 600, 200, 10, 4
LEAK, RARE, RATIO, TOL, JITTER = 12, 6, 10.0, 0.1, 1e-5
OFFSET, SPREAD = 0.6, 0.8             # offset^2 + spread^2 = 1: pooled std stays at 1


def clean(seed=0) -> tuple:
    """A well-behaved split: standardised features, balanced labels, no overlap."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(N_TRAIN + N_TEST, FEATS, generator=gen)
    y = torch.arange(N_TRAIN + N_TEST) % CLASSES
    return x[:N_TRAIN], y[:N_TRAIN], x[N_TRAIN:], y[N_TRAIN:]


def validate(xs, ys, xt, _yt) -> dict:
    """The four rules, in the order the exercise lists them."""
    rows = {tuple(r.tolist()) for r in xs}
    counts = torch.bincount(ys, minlength=CLASSES).tolist()
    return {"dupes": sum(tuple(r.tolist()) in rows for r in xt),
            "imbalance": max(counts) / max(1, min(counts)), "counts": counts,
            "mean": float(xs.mean()), "std": float(xs.std()),
            "worst_mean": float(xs.mean(dim=0).abs().max()),
            "worst_std": float((xs.std(dim=0) - 1).abs().max()),
            "nonfinite": int((~torch.isfinite(xs)).sum())}


def corrupt(kind) -> tuple:
    """One deliberate fault at a time, on top of the clean split."""
    xs, ys, xt, yt = clean()
    if kind == "leak":
        xt[:LEAK] = xs[:LEAK]
    if kind == "jitter":
        xt[:LEAK] = xs[:LEAK] + JITTER
    if kind == "imbalance":
        ys = torch.cat([torch.zeros(N_TRAIN - RARE, dtype=torch.long),
                        torch.ones(RARE, dtype=torch.long)])
    if kind == "unnormalised":
        xs = xs * 4.0 + 3.0
    if kind == "shifted":
        xs = xs * SPREAD + torch.tensor([OFFSET, -OFFSET] * (FEATS // 2))
    if kind == "nan":
        xs[0, 0], xs[1, 1] = float("nan"), float("inf")
    return xs, ys, xt, yt


def solve():
    ref = parity.load_reference(PHASE, LESSON, "debug_neural_nets")
    kinds = ("clean", "leak", "jitter", "imbalance", "unnormalised", "shifted", "nan")
    return {"reports": {k: validate(*(clean() if k == "clean" else corrupt(k))) for k in kinds},
            "checks": [name for name in dir(ref.NetworkDebugger) if name.startswith("check_")]}


def flags(report) -> list:
    """Which of the four rules fired, as the validator would print them."""
    out = []
    if report["dupes"]:
        out.append(f"{report['dupes']} duplicate rows")
    if report["imbalance"] > RATIO:
        out.append(f"imbalance {report['imbalance']:.1f}:1")
    if abs(report["mean"]) > TOL or abs(report["std"] - 1) > TOL:
        out.append(f"unnormalised (mean {report['mean']:.2f}, std {report['std']:.2f})")
    if report["nonfinite"]:
        out.append(f"{report['nonfinite']} non-finite")
    return out


def listing(r) -> str:
    return "; ".join(f"{k}: {flags(r[k])[0]}" for k in
                     ("leak", "imbalance", "unnormalised", "nan"))


def verify(result):
    r = result["reports"]
    return [
        practice.Check("ANSWER: the validator is silent on the clean split and names each fault",
                       not flags(r["clean"]) and all(len(flags(r[k])) == 1 for k in
                                                     ("leak", "imbalance", "unnormalised", "nan")),
                       f"clean: {flags(r['clean']) or 'no findings'}; {listing(r)} — one rule "
                       f"each, on {N_TRAIN} train and {N_TEST} test rows of {FEATS} features. "
                       f"The lesson's own NetworkDebugger offers {result['checks']} and nothing "
                       f"that looks at the data"),
        practice.Check("FINDING: 'duplicate samples' needs an equality rule, and exact equality "
                       "is a lower bound",
                       r["leak"]["dupes"] == LEAK and r["jitter"]["dupes"] == 0,
                       f"copying {LEAK} training rows into the test set is caught "
                       f"({r['leak']['dupes']} found); adding {JITTER:g} to each of the same rows "
                       f"first hides all {LEAK} ({r['jitter']['dupes']} found). Any resize, "
                       f"re-encode or augmentation between the split and the check makes a leak "
                       f"invisible to an == test"),
        practice.Check("FINDING: '>10:1' does not say ratio of what",
                       r["imbalance"]["imbalance"] > RATIO
                       and max(r["imbalance"]["counts"]) / (N_TRAIN / CLASSES) < 4.0,
                       f"the imbalanced split is {r['imbalance']['counts']} — "
                       f"{r['imbalance']['imbalance']:.0f}:1 as max over min, which fires, but "
                       f"{max(r['imbalance']['counts']) / (N_TRAIN / CLASSES):.1f}:1 as max over "
                       f"the mean, which does not. The rule as written also cannot be evaluated "
                       f"at all when a class is absent, since min is 0"),
        practice.Check("FINDING: 'mean near 0, std near 1' passes a dataset with no feature "
                       "near either",
                       not flags(r["shifted"]) and r["shifted"]["worst_mean"] > 0.5,
                       f"offsetting alternate features by +/-{OFFSET} and scaling by {SPREAD} "
                       f"(offset^2 + spread^2 = 1) leaves the pooled mean at "
                       f"{r['shifted']['mean']:.3f} and the pooled std at "
                       f"{r['shifted']['std']:.3f}, so all four rules stay silent — while the "
                       f"worst *per-feature* mean is {r['shifted']['worst_mean']:.2f} and the "
                       f"worst per-feature std is off by {r['shifted']['worst_std']:.2f}. Two "
                       f"pooled scalars cannot see an offset that cancels across columns"),
        practice.Check("CONTROL: NaN is the only unambiguous rule, and it has to run first",
                       r["nan"]["nonfinite"] == 2 and r["nan"]["mean"] != r["nan"]["mean"],
                       f"two poisoned entries are counted exactly ({r['nan']['nonfinite']}), but "
                       f"the same tensor's mean and std come back NaN, so the normalisation rule "
                       f"reports `abs(nan) > {TOL}` = False and stays silent. A validator that "
                       f"runs the four rules in the order the exercise lists them reports the "
                       f"NaNs last and the normalisation not at all"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
