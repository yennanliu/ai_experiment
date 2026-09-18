"""Exercise 2 — the three layers differ in shape, not in role, so the comparison is capacity.

    **Target module comparison.** Modify inject_lora to target only layer "0",
    only layer "2", only layer "4", and all three. Train each variant for 20
    epochs. Compare convergence speed and final loss. This mirrors the real
    decision of targeting q_proj vs v_proj vs all linear layers.

Reading of the exercise: `inject_lora` already takes the target list, so no
modification is needed -- the four variants are four calls. The seed is fixed
before each model so the arms differ only in which layers carry an adapter, and
a held-out split is added because exercise 1 showed the training loss on this
fixture measures memorisation.

**ANSWER: all three wins, and it wins by having the most parameters.** Final
training loss after 20 epochs is 0.0536 for layer 0, 0.0609 for layer 2, 0.0862
for layer 4 and 0.0140 for all three -- and the trainable counts are 6,144,
8,192, 4,176 and 18,512. The winner is the variant with 2.3x the adapter of the
next best.

**FINDING: the ranking is not stable in the parameter count either.** Layer 0
has 6,144 trainable parameters and beats layer 2's 8,192. What separates them is
position: layer 0 sees the raw input, layer 2 sees a ReLU'd hidden state, and
layer 4 is the 512->10 output projection with only 4,176 adapter parameters
because its output dimension is 10.

**MECHANISM: the analogy to q_proj and v_proj does not hold.** In an attention
block those projections are the same shape and play different roles. Here the
three layers are 256->512, 512->512 and 512->10 -- different shapes, different
positions in the stack, and no two of them interchangeable. The comparison
measures where capacity was added, not which role matters.

**FINDING: "convergence speed" measured against each run's own endpoint inverts
the ranking.** Epochs to reach within 10% of the run's own final loss: 16, 14, 2
and 19. Layer 4 "converges" in 2 epochs because it never gets anywhere, and the
all-three arm takes 19 because it is still improving when the run ends. Against
a threshold every arm shares -- 0.09, the target variance -- the epochs are 3, 2,
15 and 2: layer 4 alone is slow and the other three are indistinguishable.

**FINDING: none of it survives a held-out split.** Held-out loss is 0.1013,
0.1032, 0.0967 and 0.1236 -- and the best training arm is the worst held-out
arm. On random labels the ranking the exercise asks for is inverted by the only
measurement that generalises.

Structure: `variant` trains one target list, `convergence` counts epochs to
within 10% of the final loss, and `held_out` scores the split.
"""

from __future__ import annotations

import torch

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "08-fine-tuning-lora"
VARIANTS = (["0"], ["2"], ["4"], ["0", "2", "4"])
SEED, RANK, EPOCHS, BATCH = 42, 8, 20, 32


def convergence(losses, tolerance=0.1):
    """The first epoch within `tolerance` of the run's own final loss."""
    target = losses[-1] * (1 + tolerance)
    return next(i + 1 for i, value in enumerate(losses) if value <= target)


def epochs_to(losses, threshold):
    """The first epoch under a threshold shared by every arm, or 0 if never."""
    return next((i + 1 for i, value in enumerate(losses) if value <= threshold), 0)


def held_out(model, data):
    loss = torch.nn.MSELoss()
    with torch.no_grad():
        return round(float(loss(model(data["inputs"]), data["targets"])), 4)


def variant(ref, targets, train, test):
    torch.manual_seed(SEED)
    model = ref.create_demo_model()
    layers = ref.inject_lora(model, targets, rank=RANK)
    counts = ref.count_parameters(model)
    losses = ref.train_lora(model, train, epochs=EPOCHS, lr=1e-3, batch_size=BATCH)
    return {"layers": sorted(layers), "trainable": counts["trainable"],
            "final": round(losses[-1], 4), "epochs": convergence(losses),
            "shared": epochs_to(losses, 0.09), "held": held_out(model, test)}


def shapes(ref):
    torch.manual_seed(SEED)
    model = ref.create_demo_model()
    return [(m.in_features, m.out_features) for m in model
            if isinstance(m, torch.nn.Linear)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "lora")
    torch.manual_seed(SEED)
    train, test = ref.create_demo_data(500), ref.create_demo_data(200)
    rows = [variant(ref, targets, train, test) for targets in VARIANTS]
    keys = ("trainable", "final", "epochs", "shared", "held")
    return {**{key: [r[key] for r in rows] for key in keys},
            "targets": [",".join(v) for v in VARIANTS],
            "layers": [len(r["layers"]) for r in rows], "shapes": shapes(ref)}


def verify(result):
    final, trainable, held = result["final"], result["trainable"], result["held"]
    best = final.index(min(final))
    return [
        practice.Check(
            "ANSWER: all three wins, and it wins by having the most parameters",
            all([best == 3, trainable[3] == max(trainable),
                 trainable[3] > 2 * sorted(trainable)[-2]]),
            f"final training loss after {EPOCHS} epochs: {final} for targets "
            f"{result['targets']}, against trainable counts {trainable}. The winner is the "
            f"variant with {trainable[3] / sorted(trainable)[-2]:.1f}x the adapter of the "
            "next best",
        ),
        practice.Check(
            "FINDING: the ranking is not stable in the parameter count either",
            all([trainable[0] < trainable[1], final[0] < final[1],
                 trainable[2] == min(trainable)]),
            f"layer 0 has {trainable[0]:,} trainable parameters and beats layer 2's "
            f"{trainable[1]:,} ({final[0]} against {final[1]}). Layer 4 has the fewest, "
            f"{trainable[2]:,}, because its output dimension is 10. Position and shape are "
            "both moving, and the exercise varies neither on purpose",
        ),
        practice.Check(
            "MECHANISM: the analogy to q_proj and v_proj does not hold",
            all([len(set(result["shapes"])) == 3, result["shapes"][2][1] == 10]),
            f"the three linear layers are {result['shapes']} -- different shapes, different "
            "positions in the stack, no two interchangeable. In an attention block q_proj "
            "and v_proj are the same shape and play different roles; here the shape is the "
            "difference, so the comparison measures where capacity was added",
        ),
        practice.Check(
            "FINDING: the convergence metric is self-referential and inverts the ranking",
            all([result["epochs"][2] == min(result["epochs"]),
                 result["epochs"][3] == max(result["epochs"]),
                 result["shared"][3] == min(result["shared"])]),
            f"epochs to reach within 10% of each run's own final loss: {result['epochs']} "
            f"out of {EPOCHS} -- layer 4 'converges' in {result['epochs'][2]} because it "
            f"never gets anywhere, and the all-three arm takes {result['epochs'][3]} "
            f"because it is still improving at the end. Against a threshold every arm "
            f"shares, 0.09, the epochs are {result['shared']}: layer 4 alone is slow and "
            "the other three land within one epoch of each other",
        ),
        practice.Check(
            "FINDING: none of it survives a held-out split",
            all([held[best] == max(held), min(held) > 0.09]),
            f"held-out loss is {held}, every value above the 0.09 a constant 0.1 prediction "
            f"scores -- and the best training arm, {result['targets'][best]!r}, is the worst "
            "held-out arm. On random labels the ranking the exercise asks for is inverted "
            "by the only measurement that generalises",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
