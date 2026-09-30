"""Exercise 2 -- without the division SGD's first step is 4.0x too big, but AdamW's is 1.000015x and hides it.

    Add a wrong scaling variant (no division) and show the parameter diff at step 1 against the reference.

Reading of the exercise: "the reference" is the full-batch step that the
lesson's `equivalence_check` builds: seed 7, a 16-sample batch, the
32-48-8 GELU net, SGD at lr 0.1. The wrong variant is the lesson's own
`train_one_optimizer_step` over four 4-sample chunks, with
`loss_scaled_for_accum` patched to return the raw loss (no `/ accum_steps`).
The correct variant is the same call unpatched. Both are compared with the
reference after one optimizer step. The same comparison is repeated with
AdamW (torch defaults, lr 0.1), because the lesson says the unscaled step is
"16 times too big" without naming an optimizer.

**ANSWER: the unscaled step moves the parameters 4.0x as far as the
reference.** After step 1 the largest parameter difference from the
reference is 0.0215. That is exactly (N - 1) * lr * max|g| = 3 * 0.1 *
0.0717, since every gradient entry is 4x its true value. The correctly
scaled variant differs from the reference by 1.5e-8, at float32 rounding.
The ratio of update norms, wrong over correct, is 4.0000002.

**FINDING: AdamW hides the missing division.** Adam divides the first moment
by the root of the second, so a gradient 4x too large gives almost the same
step. The update-norm ratio is 1.000015 instead of 4, and the parameter diff
at step 1 is 7.8e-4, down from 0.0215. It comes only from `eps` = 1e-8
mattering more for small gradients. The lesson's "the optimizer step is 16
times too big" holds for SGD, not for the Adam family.

**FINDING: the lesson's own log would still catch the bug.**
`train_one_optimizer_step` multiplies each scaled loss back by
`accum_steps` before logging it, so without the division the logged loss
reads 8.4088 instead of 2.1022 (4x), and the logged grad norm 1.2109
instead of 0.3027 (4x), under either optimizer.

Expected output: three PASS checks.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "46-gradient-accumulation"
SEED, BATCH, N, LR = 7, 16, 4, 0.1


def setup(ref, opt_cls):
    """The equivalence_check fixture: same batch, same initial weights."""
    gen = torch.Generator()
    gen.manual_seed(SEED)
    x, y = ref.synthetic_batch(BATCH, 32, 8, gen)
    ref.seed_everything(SEED)
    model = ref.make_model(32, 48, 8)
    return model, opt_cls(model.parameters(), lr=LR), x, y


def params(model):
    return [p.detach().clone() for p in model.parameters()]


def reference_step(ref, opt_cls):
    model, opt, x, y = setup(ref, opt_cls)
    start = params(model)
    nn.CrossEntropyLoss()(model(x), y).backward()
    grad_max = max(float(p.grad.abs().max()) for p in model.parameters())
    opt.step()
    return start, params(model), grad_max


def accum_step(ref, opt_cls, wrong):
    model, opt, x, y = setup(ref, opt_cls)
    micro = list(zip(torch.split(x, BATCH // N), torch.split(y, BATCH // N)))
    saved = ref.loss_scaled_for_accum
    if wrong:
        ref.loss_scaled_for_accum = lambda logits, target, accum_steps, loss_fn: loss_fn(logits, target)
    try:
        loss, norm = ref.train_one_optimizer_step(
            model, opt, micro, nn.CrossEntropyLoss(), no_sync_until_last=True, sync_counter=[0]
        )
    finally:
        ref.loss_scaled_for_accum = saved
    return params(model), round(loss, 4), round(norm, 4)


def compare(ref, opt_cls):
    start, after, grad_max = reference_step(ref, opt_cls)
    good, bad = accum_step(ref, opt_cls, False), accum_step(ref, opt_cls, True)

    def diff(ps):
        return max(float((a - b).abs().max()) for a, b in zip(ps, after))

    def move(ps):
        return sum(float((a - b).pow(2).sum()) for a, b in zip(ps, start)) ** 0.5

    return {"wrong_diff": diff(bad[0]), "right_diff": diff(good[0]), "ratio": move(bad[0]) / move(good[0]),
            "grad_max": grad_max, "logs": (good[1:], bad[1:])}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"sgd": compare(ref, torch.optim.SGD), "adamw": compare(ref, torch.optim.AdamW)}


def verify(result):
    s, a = result["sgd"], result["adamw"]
    predicted = (N - 1) * LR * s["grad_max"]
    return [
        practice.Check(
            "ANSWER: without the division SGD's step 1 lands 0.0215 from the reference, 4.0x the update",
            abs(s["wrong_diff"] - predicted) < 1e-4 * predicted and abs(s["wrong_diff"] - 0.0215) < 1e-4
            and s["right_diff"] < 1e-6 and abs(s["ratio"] - N) < 1e-5,
            f"wrong {s['wrong_diff']:.4g} vs 3*lr*max|g| {predicted:.4g}; scaled {s['right_diff']:.2g}; "
            f"update ratio {s['ratio']:.7f}",
        ),
        practice.Check(
            "FINDING: AdamW hides the missing division -- its step is 1.000015x, not 4x",
            abs(a["ratio"] - 1) < 1e-4 and a["wrong_diff"] < 0.05 * s["wrong_diff"] and a["right_diff"] < 1e-6,
            f"AdamW update ratio {a['ratio']:.6f}, step-1 diff {a['wrong_diff']:.2g} (SGD {s['wrong_diff']:.3g})",
        ),
        practice.Check(
            "FINDING: the lesson's own loss log reads 4x without the division",
            s["logs"] == a["logs"] == ((2.1022, 0.3027), (8.4088, 1.2109)),
            f"(loss, grad norm) scaled {s['logs'][0]}, unscaled {s['logs'][1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
