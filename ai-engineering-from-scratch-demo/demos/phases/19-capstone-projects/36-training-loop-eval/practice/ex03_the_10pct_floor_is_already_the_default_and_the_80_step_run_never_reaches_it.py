"""Exercise 3 -- the 10% floor is already the default, and the 80-step run never reaches it.

    Add a `min_lr` floor of 10 percent of `max_lr` to the cosine schedule and re-plot.

Reading of the exercise: `cosine_with_warmup` already takes `min_lr` and
decays to it, so "add a floor" means setting `min_lr = 0.1 * max_lr` and
comparing against a floorless schedule (`min_lr = 0`) over the demo's 80
steps (warmup 10, max_lr 3e-3). "Re-plot" is a text sparkline of both
curves, one character per two steps, plus the values at the turning points.
Both schedules then drive the lesson's own `train()` on the demo data to
see what the floor changes.

**ANSWER: with the floor, the LR ends at 3.01e-4 instead of 1.5e-6.**

    floor 10%  ▁▃▅▆██████████▇▇▇▇▆▆▆▅▅▅▄▄▄▃▃▃▃▂▂▂▂▂▂▁▁▁
    floor 0    ▁▃▅▆██████████▇▇▇▆▆▆▅▅▅▄▄▄▃▃▃▂▂▂▁▁▁▁▁▁▁▁

The floor lifts the mean LR over the run from 1.54e-3 to 1.67e-3 (+8.4%).
The final train loss moves from 1.130 to 1.101, a 0.029-nat difference on
a single batch, so over 80 steps the floor is close to cosmetic.

**FINDING: the exercise asks for what is already there.** `TrainConfig`
ships `max_lr = 3e-3` and `min_lr = 3e-4`, a ratio of exactly 0.1, and the
lesson's committed `outputs/losses.jsonl` carries that schedule: its `lr`
column matches `cosine_with_warmup` at 10% on all 80 rows, with a worst
difference of 0.0.

**FINDING: the run never touches the floor, and warmup does not start at
zero.** `train()` evaluates steps 0..79 against `total_steps = 80`, so the
last step sits at progress 69/70 and LR 3.0136e-4, 0.45% above the floor.
Only step 80, which never runs, returns 3e-4 exactly; the lesson's unit test
checks that step. The doc says warmup "ramps the learning rate from zero",
but step 0 gets max_lr x 1/10 = 3e-4, the floor value itself. Steps 9 and
10 both sit at the 3e-3 peak.

Structure: `curve()` evaluates the lesson's schedule; `spark()` draws it;
`final_loss()` runs the lesson's `train()` with one `min_lr`.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "36-training-loop-eval"
BARS = "▁▂▃▄▅▆▇█"


def curve(ref, cfg, min_lr):
    return [
        ref.cosine_with_warmup(s, cfg.warmup_steps, cfg.num_steps, cfg.max_lr, min_lr)
        for s in range(cfg.num_steps)
    ]


def spark(lrs, top):
    return "".join(BARS[min(int(v / top * len(BARS)), len(BARS) - 1)] for v in lrs[::2])


def final_loss(ref, cfg, min_lr):
    cfg = ref.TrainConfig(min_lr=min_lr)
    torch.manual_seed(0)
    model = ref.GPTModel(ref.ModelConfig(dropout=0.0))
    train = ref._synthetic_byte_tokens(4096, 256, 1)
    val = ref._synthetic_byte_tokens(1024, 256, 2)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        records = ref.train(model, train, val, cfg, torch.tensor([[7, 11, 13, 17]]),
                            log_path=pathlib.Path(tmp) / "l.jsonl")
    return round(records[-1]["train_loss"], 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = ref.TrainConfig()
    floor, zero = curve(ref, cfg, 0.1 * cfg.max_lr), curve(ref, cfg, 0.0)
    shipped = parity.lesson_dir(PHASE, LESSON) / "outputs" / "losses.jsonl"
    rows = [json.loads(line) for line in shipped.read_text().splitlines() if line.strip()]
    sched = lambda s: ref.cosine_with_warmup(s, 10, 80, 3e-3, 3e-4)  # noqa: E731
    return {
        "ratio": cfg.min_lr / cfg.max_lr,
        "spark": [spark(floor, cfg.max_lr), spark(zero, cfg.max_lr)],
        "last": [floor[-1], zero[-1]], "step80": sched(80), "step0": floor[0],
        "peaks": [s for s, v in enumerate(floor) if v == cfg.max_lr],
        "mean": [sum(floor) / len(floor), sum(zero) / len(zero)],
        "loss": [final_loss(ref, cfg, 0.1 * cfg.max_lr), final_loss(ref, cfg, 0.0)],
        "shipped_rows": len(rows),
        "shipped_worst": max(abs(r["lr"] - sched(r["step"])) for r in rows),
        "doc_zero": "ramps the learning rate from zero" in parity.doc_text(PHASE, LESSON),
        "unit_test_step": "cosine_with_warmup(100, warmup_steps=10, total_steps=100"
        in (parity.lesson_dir(PHASE, LESSON) / "code/tests/test_training.py").read_text(),
    }


def verify(result):
    r = result
    lift = r["mean"][0] / r["mean"][1] - 1
    last = (round(r["last"][0], 8), round(r["last"][1], 7))
    return [
        practice.Check(
            "ANSWER: with the floor, the LR ends at 3.01e-4 instead of 1.5e-6",
            (last, [round(m, 5) for m in r["mean"]], round(lift, 3), r["loss"])
            == ((3.0136e-4, 1.5e-6), [1.67e-3, 1.54e-3], 0.084, [1.101, 1.13]),
            f"floor {r['spark'][0]} | zero {r['spark'][1]}; last LR {r['last'][0]:.4e} vs "
            f"{r['last'][1]:.2e}; mean LR +{lift:.1%}; final train loss {r['loss']}",
        ),
        practice.Check(
            "FINDING: the exercise asks for what is already there",
            (round(r["ratio"], 12), r["shipped_rows"], r["shipped_worst"]) == (0.1, 80, 0.0),
            f"TrainConfig min_lr/max_lr = {r['ratio']:.3f}; shipped log {r['shipped_rows']} rows, "
            f"worst LR difference from the 10% schedule {r['shipped_worst']}",
        ),
        practice.Check(
            "FINDING: the run never touches the floor, and warmup does not start at zero",
            all([r["last"][0] > 3e-4, r["unit_test_step"], r["doc_zero"]])
            and (r["step80"], round(r["step0"], 12), r["peaks"]) == (3e-4, 3e-4, [9, 10]),
            f"step 79 LR {r['last'][0]:.4e} ({r['last'][0] / 3e-4 - 1:.2%} above), step 80 "
            f"{r['step80']}; step 0 {r['step0']:.1e}; peak at steps {r['peaks']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
