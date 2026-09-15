"""Exercise 1 — 33% more compute buys 1.2% less memory, at the calculator's own defaults.

    Modify the memory calculator to include activation checkpointing.
    With checkpointing, only store activations at every K-th layer (typical K=1,
    meaning recompute all). Show the memory-compute tradeoff: how much memory
    does checkpointing save, and how much does it slow down training (roughly
    33% more compute for full checkpointing)?

Reading of the exercise: the model is a 70B at the shape the lesson's own demo
uses -- `hidden_dim=8192`, `num_layers=80`, `sequence_length=2048` -- so the
calculator takes its real activation branch rather than its
`params * precision * 0.5` fallback. Checkpointing at every K-th layer keeps
`num_layers / K` of the activation term, so full recomputation keeps 1 of 80.
The 33% figure is taken as given and priced against what it buys.

**FINDING: the exercise's parenthetical contradicts its own definition.**
"Only store activations at every K-th layer" with K=1 stores *every* layer --
recomputing nothing. K=1 is the no-checkpointing case, not "recompute all",
which is K = num_layers.

**ANSWER: at the calculator's own defaults, almost nothing.** With
`batch_size_per_gpu=1`, activations are **10.7 GB of 850.7** -- 1.3% of the
total. Full checkpointing returns 10.6 GB, **1.2%** of per-GPU memory, for the
stated 33% more compute. The 70B model still does not fit on an 80 GB card, and
was never going to: weights, gradients and Adam states are 840 GB of the 850.7.

**FINDING: the answer is a statement about batch size, not about
checkpointing.** Activations are the only term that scales with the batch, so
the saving goes 1.2%, 2.5%, 4.8%, 9.2%, 16.8%, 28.7% as the batch goes 1, 2, 4,
8, 16, 32. Activations do not equal the fixed 840 GB until **batch size 78**.
The exercise asks "how much memory does checkpointing save" as though it had one
answer, and the calculator's default batch of 1 is the setting where the answer
is smallest.

**FINDING: the calculator's fallback overstates the term by 6.5x.** Call it
without `hidden_dim` and `num_layers` and activations become
`params * 2 * 0.5` = **70 GB** instead of 10.7. The default path inflates the
one quantity this exercise is about, which is the difference between "8% of
memory" and "1.3%".

Structure: `memory` calls the reference calculator at one batch size;
`checkpointed` applies the every-K-th-layer rule to its activation term.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "05-scaling-distributed"
PARAMS, LAYERS, COMPUTE_TAX = 70, 80, 0.33
SHAPE = dict(hidden_dim=8192, num_layers=LAYERS, sequence_length=2048)
BATCHES = (1, 2, 4, 8, 16, 32)


def memory(ref, batch, **overrides):
    return ref.memory_calculator(PARAMS, batch_size_per_gpu=batch, **dict(SHAPE, **overrides))


def checkpointed(report, every):
    """Per-GPU memory when only every `every`-th layer's activations are stored."""
    kept = report["activations_gb"] * max(1, LAYERS // every) / LAYERS
    return report["per_gpu_total_gb"] - report["activations_gb"] + kept


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reports = {batch: memory(ref, batch) for batch in BATCHES}
    base = reports[1]
    fixed = base["weights_gb"] + base["optimizer_gb"] + base["gradients_gb"]
    return {
        "saving": {batch: 1 - checkpointed(r, LAYERS) / r["per_gpu_total_gb"]
                   for batch, r in reports.items()},
        "activations": {batch: r["activations_gb"] for batch, r in reports.items()},
        "total": {batch: r["per_gpu_total_gb"] for batch, r in reports.items()},
        "fixed": fixed,
        "crossover": fixed / base["activations_gb"],
        "fits": base["fits_on_80gb"],
        "no_op": checkpointed(base, 1) == base["per_gpu_total_gb"],
        "fallback": ref.memory_calculator(PARAMS)["activations_gb"],
    }


def verify(result):
    saving, activations = result["saving"], result["activations"]
    total, fallback = result["total"], result["fallback"]
    return [
        practice.Check(
            "FINDING: K=1 stores every layer, so the parenthetical contradicts the definition",
            result["no_op"],
            "'only store activations at every K-th layer' with K=1 stores every layer and "
            "recomputes nothing -- it is the no-checkpointing case, and applying it to the "
            "calculator changes per-GPU memory by exactly 0 GB. 'Recompute all' is "
            f"K = num_layers = {LAYERS}, which is what the rest of this solution uses",
        ),
        practice.Check(
            "ANSWER: 1.2% of per-GPU memory, for 33% more compute, at the default batch of 1",
            saving[1] < 0.02 and not result["fits"],
            f"at batch 1 the activations are {activations[1]:.1f} GB of {total[1]:.1f} "
            f"({100 * activations[1] / total[1]:.1f}%), so recomputing all but one layer's worth "
            f"returns {activations[1] * (1 - 1 / LAYERS):.1f} GB -- {100 * saving[1]:.1f}% of "
            f"per-GPU memory -- for the stated {100 * COMPUTE_TAX:.0f}% more compute. The model "
            f"still does not fit on an 80 GB card and never could: weights, gradients and Adam "
            f"states are {result['fixed']:.0f} GB of the {total[1]:.1f}",
        ),
        practice.Check(
            "FINDING: the answer is a statement about batch size, not about checkpointing",
            saving[32] > 20 * saving[1] and result["crossover"] > 50,
            "activations are the only term that scales with the batch, so the saving goes "
            + ", ".join(f"{100 * saving[b]:.1f}%" for b in BATCHES)
            + f" as the batch goes {list(BATCHES)}. They do not equal the fixed "
            f"{result['fixed']:.0f} GB until batch size {result['crossover']:.0f}. The exercise "
            "asks how much checkpointing saves as though that had one answer, and the "
            "calculator's default batch of 1 is the setting where it is smallest",
        ),
        practice.Check(
            "FINDING: the calculator's own fallback overstates the term 6.5x",
            fallback > 6 * activations[1],
            f"called without hidden_dim and num_layers, memory_calculator falls back to "
            f"params * precision_bytes * 0.5 and reports {fallback:.0f} GB of activations "
            f"against the {activations[1]:.1f} its own formula gives for this shape -- "
            f"{fallback / activations[1]:.1f}x. That is the difference between activations "
            "being 8% of memory and 1.3% of it, on the one term this exercise is about",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
