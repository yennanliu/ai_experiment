"""Exercise 3 — cross-entropy for 5 logits, target class 3, against PyTorch.

    Compute the cross-entropy loss for a 5-class classifier that outputs logits
    `[2.0, 0.5, -1.0, 3.0, 0.1]` when the correct class is index 3. Then verify
    your answer with PyTorch's `nn.CrossEntropyLoss`.

Reading of the exercise: the number itself is one line, so the checks go after
what makes this operator worth implementing carefully — the log-sum-exp shift —
and after the thing that actually trips people comparing against torch: check 2,
where torch's float32 default disagrees in the 8th decimal.
Check 4 feeds logits large enough to overflow `exp`, where the naive
softmax-then-log route returns `inf` or `nan` and the shifted one does not. The
lesson ships both `cross_entropy_loss` and `log_softmax`, so the comparison is
between its own two paths, not against a strawman.

Tier T1: needs torch, and skips with a remedy without it.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "06-probability-and-distributions"
LOGITS = [2.0, 0.5, -1.0, 3.0, 0.1]
TARGET = 3
TOL = 1e-9
HUGE = [1000.0, 999.0, 998.0, 1001.0, 997.0]


def naive_cross_entropy(logits, target):
    """softmax then log, with no shift — the version that overflows."""
    exp_values = [math.exp(v) for v in logits]
    total = sum(exp_values)
    return -math.log(exp_values[target] / total)


def solve():
    try:
        import torch
    except ImportError:
        raise practice.Skip("needs PyTorch — uv sync --extra llm") from None
    ref = parity.load_reference(PHASE, LESSON, "probability")
    mine = ref.cross_entropy_loss(LOGITS, TARGET)
    loss_fn = torch.nn.CrossEntropyLoss()
    target = torch.tensor([TARGET])
    # float64 to compare like with like: the lesson computes in Python floats
    theirs = loss_fn(torch.tensor([LOGITS], dtype=torch.float64), target).item()
    theirs32 = loss_fn(torch.tensor([LOGITS], dtype=torch.float32), target).item()
    probabilities = ref.softmax(LOGITS)
    try:
        naive = naive_cross_entropy(HUGE, TARGET)
    except OverflowError as exc:
        naive = f"OverflowError: {exc}"
    stable = ref.cross_entropy_loss(HUGE, TARGET)
    stable_torch = loss_fn(torch.tensor([HUGE], dtype=torch.float64), target).item()
    return {"mine": mine, "torch": theirs, "torch32": theirs32,
            "probabilities": probabilities, "naive_huge": naive, "stable_huge": stable,
            "torch_huge": stable_torch, "manual": -ref.log_softmax(LOGITS)[TARGET]}


def verify(result):
    probabilities = result["probabilities"]
    naive = result["naive_huge"]
    naive_broke = isinstance(naive, str) or not math.isfinite(naive)
    return [
        practice.Check("loss matches PyTorch's nn.CrossEntropyLoss in float64",
                       abs(result["mine"] - result["torch"]) <= TOL,
                       f"mine {result['mine']:.12f} vs torch {result['torch']:.12f}"),
        practice.Check("…and torch's float32 default differs, at float32's precision",
                       1e-9 < abs(result["mine"] - result["torch32"]) < 1e-6,
                       f"float32 gives {result['torch32']:.9f}, off by "
                       f"{abs(result['mine'] - result['torch32']):.2e} — torch.tensor() on "
                       f"Python floats silently narrows to float32, which is the usual "
                       f"reason a from-scratch loss 'disagrees' with torch"),
        practice.Check("…and equals −log_softmax(logits)[target], its definition",
                       abs(result["mine"] - result["manual"]) <= TOL,
                       f"−log_softmax[{TARGET}] = {result['manual']:.9f}"),
        practice.Check("softmax is a distribution, and class 3 is the argmax",
                       abs(sum(probabilities) - 1.0) <= TOL
                       and probabilities.index(max(probabilities)) == TARGET,
                       f"p = {[round(p, 4) for p in probabilities]}, "
                       f"p[{TARGET}] = {probabilities[TARGET]:.4f}"),
        practice.Check("the naive softmax-then-log route breaks on large logits",
                       naive_broke, f"logits ~1000 -> {naive}"),
        practice.Check("…while the lesson's shifted version stays exact, and matches torch",
                       math.isfinite(result["stable_huge"])
                       and abs(result["stable_huge"] - result["torch_huge"]) <= 1e-6,
                       f"{result['stable_huge']:.9f} vs torch {result['torch_huge']:.9f} — "
                       f"subtracting max(logits) changes nothing mathematically and "
                       f"everything numerically"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
