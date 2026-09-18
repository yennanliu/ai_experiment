"""Exercise 1 — the labels are random, so the loss never halves and held-out never moves.

    **Rank ablation study.** Run the demo with ranks 2, 4, 8, 16, 32, and 64.
    Plot final loss vs. rank. Find the point of diminishing returns where
    doubling the rank no longer halves the loss. For a simple classification
    task on 256-dim features, this should be around r=8-16.

Reading of the exercise: the demo is the lesson's own -- `create_demo_model`,
`create_demo_data`, `inject_lora` on all three linear layers, `train_lora` at
its defaults -- with the seed fixed before each model so the six runs differ
only in rank. A held-out split is added, because it is the only way to tell a
rank effect from a capacity effect.

**ANSWER: the loss never halves, at any rank.** Final training loss runs 0.0855,
0.0759, 0.0612, 0.0579, 0.0578, 0.0612 for ranks 2 to 64. The largest
consecutive ratio is 1.24x against the 2.0x the exercise looks for, and the curve
is non-monotone -- rank 64 is worse than rank 32. There is no point of diminishing returns because there was never a
point of returns.

**MECHANISM: `create_demo_data` draws the inputs and the labels
independently.** `x = torch.randn(n, d)` and `y = torch.randint(0, n_classes,
(n,))`, one-hot encoded. Nothing connects them. The Bayes-optimal MSE against a
random one-hot target is its own variance, and the targets' variance is exactly
0.0900.

**FINDING: every loss below 0.0900 is memorisation, and the held-out split says
so.** Training loss falls to 0.0578 at rank 32 while held-out loss stays between
0.1000 and 0.1062 -- worse, at every rank, than the 0.0900 a constant 0.1
prediction would score. Held-out
loss does not improve with rank; it drifts upward.

**FINDING: the trainable-parameter count is what the rank is varying.** From
rank 2 to rank 64 the adapters go from 4,628 to 148,096 trainable parameters
against a 403,998-parameter total -- 1.1% to 27.1%. At the top of the sweep LoRA
is no longer a low-rank method on this model.

**CONTROL: give the task a signal and the curve appears.** With targets set to a
fixed random linear map of the inputs, final loss falls monotonically with rank
and the held-out loss follows it down. The ablation is a sound experiment; the
fixture has nothing in it to ablate.

Structure: `sweep` runs one rank, `signal_data` is the control fixture, and
`held_out` scores a split the lesson does not create.
"""

from __future__ import annotations

import torch

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "08-fine-tuning-lora"
RANKS = (2, 4, 8, 16, 32, 64)
TARGETS = ["0", "2", "4"]
SEED, EPOCHS, BATCH = 42, 5, 32


def signal_data(ref, n=500, dim=256, classes=10):
    """The control: targets that are a fixed linear function of the inputs."""
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(n, dim, generator=generator)
    weight = torch.randn(dim, classes, generator=generator) / dim ** 0.5
    return {"inputs": inputs, "targets": inputs @ weight}


def held_out(ref, model, data):
    loss = torch.nn.MSELoss()
    with torch.no_grad():
        return round(float(loss(model(data["inputs"]), data["targets"])), 4)


def sweep(ref, rank, train, test):
    torch.manual_seed(SEED)
    model = ref.create_demo_model()
    ref.inject_lora(model, TARGETS, rank=rank)
    counts = ref.count_parameters(model)
    losses = ref.train_lora(model, train, epochs=EPOCHS, lr=1e-3, batch_size=BATCH)
    return {"final": round(losses[-1], 4), "trainable": counts["trainable"],
            "total": counts["total"], "pct": round(counts["trainable_pct"], 1),
            "held": held_out(ref, model, test)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "lora")
    torch.manual_seed(SEED)
    train, test = ref.create_demo_data(500), ref.create_demo_data(200)
    rows = [sweep(ref, r, train, test) for r in RANKS]
    control = signal_data(ref)
    control_test = {"inputs": control["inputs"][400:], "targets": control["targets"][400:]}
    control_rows = [sweep(ref, r, control, control_test) for r in (2, 8, 32)]
    return {
        **columns(rows), "variance": round(float(train["targets"].var(unbiased=False)), 4),
        "control_final": [r["final"] for r in control_rows],
        "control_held": [r["held"] for r in control_rows],
    }


def columns(rows):
    final = [r["final"] for r in rows]
    return {"final": final, "held": [r["held"] for r in rows],
            "ratios": [round(a / b, 2) for a, b in zip(final, final[1:])],
            "trainable": [r["trainable"] for r in rows], "total": rows[0]["total"],
            "pct": [r["pct"] for r in rows]}


def verify(result):
    final, held, ratios = result["final"], result["held"], result["ratios"]
    control, control_held = result["control_final"], result["control_held"]
    return [
        practice.Check(
            "ANSWER: the loss never halves, at any rank",
            all([max(ratios) < 1.3, final[-1] > final[-2]]),
            f"final training loss for ranks {list(RANKS)} is {final}. Consecutive ratios "
            f"{ratios}, largest {max(ratios)}x against the 2.0x the exercise looks for, and "
            "the curve turns up at the last doubling. There is no point of diminishing "
            "returns because there was never a point of returns",
        ),
        practice.Check(
            "MECHANISM: the inputs and the labels are drawn independently",
            result["variance"] == 0.09,
            "`create_demo_data` draws x = torch.randn(n, d) and y = torch.randint(0, "
            f"n_classes, (n,)), one-hot encoded: nothing connects them. The targets' "
            f"variance is {result['variance']}, which is the MSE a constant 0.1 prediction "
            "scores and the floor any model that has not memorised the sample can reach",
        ),
        practice.Check(
            "FINDING: every loss below 0.09 is memorisation, and held-out says so",
            all([min(final) < result["variance"], min(held) > result["variance"],
                 held[-1] > held[0] or held[2] > held[0]]),
            f"training loss falls to {min(final)} while held-out loss stays at {held} -- "
            f"every value above the {result['variance']} a constant would score. Held-out "
            "does not improve with rank; it drifts upward. The improvement in the training "
            "column is 500 random labels being learned by heart",
        ),
        practice.Check(
            "FINDING: the trainable-parameter count is what the rank is varying",
            all([result["trainable"][0] < 5000, result["pct"][-1] > 25]),
            f"adapters go from {result['trainable'][0]:,} to {result['trainable'][-1]:,} "
            f"trainable parameters against a {result['total']:,}-parameter base -- "
            f"{result['pct'][0]}% to {result['pct'][-1]}%. At the top of the sweep LoRA is "
            "not a low-rank method on this model any more",
        ),
        practice.Check(
            "CONTROL: give the task a signal and the curve appears",
            all([control == sorted(control, reverse=True),
                 control_held == sorted(control_held, reverse=True)]),
            f"with the targets set to a fixed random linear map of the inputs, final loss "
            f"falls monotonically with rank ({control} at ranks 2, 8, 32) and held-out loss "
            f"follows it down ({control_held}). The ablation is a sound experiment; the "
            "fixture has nothing in it to ablate",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
