"""Exercise 5 -- held-out loss falls at all 5 checkpoints, but only BLEU-4 follows it; VQA ends below chance.

    Run the eval on the model at five checkpoints during training (step 0, 10, 20, 30, 40, 50) and plot the learning curve. Confirm the metric trajectories track the loss trajectory.

Reading of the exercise: the list names six steps, so all six are
evaluated (five intervals between them). The run is the lesson's own
`main()`, not a copy of its loop. Its `sample_batch` is wrapped so the
shipped `evaluate` runs from inside the loop before steps 0, 10, 20, 30 and
40, and its second `evaluate` call gives step 50. The wrapper restores train
mode after each evaluation, and the final metrics equal those of an unhooked
run. The loss trajectory is the training objective (contrastive + LM)
on the 50 eval pairs at the same six points; the printed training loss is
recorded beside it. The plot is one text sparkline per metric (low ' ' to
high '@'). A metric "tracks" the loss on an interval when it rises exactly
when the loss falls.

**ANSWER: the loss falls at every checkpoint, but only BLEU-4 tracks it on
all 5 intervals.** Held-out loss: 9.5825, 8.716, 8.4706, 8.0458, 7.6748, 7.2832.
BLEU-4 goes 0.122 -> 0.173 in step with it. R@10_i2t tracks on 3 of 5
intervals, R@5 and R@10_t2i on 2, and R@1 on 1. VQA exact match tracks on 0:
it sits at 0.02 for five checkpoints and drops to 0.0 at step 50.

**FINDING: the demo checks nothing, and two metrics end at or below the
lesson's own random baselines.** The lesson says the metrics are "expected
to be above the random baseline, which is what the demo checks". `main()`
prints the numbers and "done."; it has no assert, raise, exit or baseline. After 50 steps R@1_i2t is
0.0 against a baseline of 0.02, and VQA EM is 0.0 against 1/128.

**FINDING: R@1_i2t peaks mid-run and then collapses.** It reads 0.02, 0.02,
0.04, 0.0, 0.0, 0.0: the best retrieval checkpoint by R@1 is step 20, while
the loss keeps falling.

Structure: `run_lesson()` drives the shipped `main()` with the two wrappers;
`checkpoint()` evaluates and restores train mode; `spark()` plots;
`tracks()` counts agreeing intervals.
"""


from __future__ import annotations

import contextlib
import inspect
import io
import re

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "63-multimodal-eval"
MARKS = (0, 10, 20, 30, 40)
BARS = " .:-=+*#%@"


def heldout_loss(ref, model, suite):
    """The training objective (contrastive + LM) on the 50 eval pairs, no gradient."""
    with torch.no_grad():
        imgs = torch.cat([p.image for p in suite.retrieval])
        ids = torch.cat([p.caption_ids for p in suite.retrieval])
        contrast, lm, _ = model(imgs, ids)
    return round((contrast + lm).item(), 4)


def checkpoint(ref, saved_ev, model, suite, curve):
    metrics = saved_ev(model, suite)
    curve.append({**{k: round(v, 4) for k, v in metrics.items()},
                  "heldout": heldout_loss(ref, model, suite)})
    model.train()  # evaluate() leaves the model in eval mode; the loop expects train mode
    return metrics


def run_lesson(ref, hook):
    """The lesson's main(); with hook, evaluate at MARKS from inside its own loop."""
    state, saved_ev, saved_sb = {"model": None, "step": 0, "curve": []}, ref.evaluate, ref.sample_batch

    def evaluate(model, suite):
        state["model"], state["suite"] = model, suite
        if hook and state["step"]:  # the second call: step 50
            return checkpoint(ref, saved_ev, model, suite, state["curve"])
        return saved_ev(model, suite)

    def sample_batch(corpus, idx):
        if hook and state["step"] in MARKS:
            checkpoint(ref, saved_ev, state["model"], state["suite"], state["curve"])
        state["step"] += 1
        return saved_sb(corpus, idx)

    ref.evaluate, ref.sample_batch = evaluate, sample_batch
    try:
        with contextlib.redirect_stdout(io.StringIO()) as log:
            ref.main()
    finally:
        ref.evaluate, ref.sample_batch = saved_ev, saved_sb
    train = {int(s): float(v) for s, v in re.findall(r"step\s+(\d+)\s+total ([\d.]+)", log.getvalue())}
    after = re.findall(r"metrics AFTER training:\n((?:  .*\n)+)", log.getvalue())[0]
    return state["curve"], train, after


def spark(values):
    lo, hi = min(values), max(values)
    return "".join(BARS[round((v - lo) / (hi - lo) * 9) if hi > lo else 0] for v in values)


def tracks(curve, key):
    """Of the 5 intervals, how many see the metric rise as the held-out loss falls."""
    return sum((b["heldout"] < a["heldout"]) == (b[key] > a[key]) for a, b in zip(curve, curve[1:]))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    _, _, plain_after = run_lesson(ref, hook=False)
    curve, train, after = run_lesson(ref, hook=True)
    cols = {k: [c[k] for c in curve] for k in curve[0]}
    return {"cols": cols, "train": train, "same_run": after == plain_after,
            "plot": "; ".join(f"{k} [{spark(v)}]" for k, v in cols.items()),
            "tracks": {k: tracks(curve, k) for k in cols if k != "heldout"},
            "demo_claim": "which is what the demo checks" in parity.doc_text(PHASE, LESSON),
            "main_compares": [w for w in ("assert", "baseline", "raise", "exit")
                              if w in inspect.getsource(ref.main)]}


def verify(result):
    r, c = result, result["cols"]
    return [
        practice.Check(
            "ANSWER: the loss falls at every checkpoint, but only BLEU-4 tracks it on all 5 intervals",
            (r["same_run"], c["heldout"], c["bleu4"], r["tracks"])
            == (True, [9.5825, 8.716, 8.4706, 8.0458, 7.6748, 7.2832], [0.122, 0.1284, 0.1321, 0.137, 0.1519, 0.173],
                {"R@1_i2t": 1, "R@1_t2i": 1, "R@5_i2t": 2, "R@5_t2i": 2, "R@10_i2t": 3, "R@10_t2i": 2,
                 "vqa_em": 0, "bleu4": 5}),
            f"held-out loss {c['heldout']}; train loss {r['train']}; intervals tracked {r['tracks']}; {r['plot']}",
        ),
        practice.Check(
            "FINDING: the demo checks nothing, and two metrics end at or below the random baselines",
            (r["demo_claim"], r["main_compares"], c["R@1_i2t"][-1] < 1 / 50, c["vqa_em"][-1] < 1 / 128)
            == (True, [], True, True),
            f"comparison tokens in main(): {r['main_compares']}; after 50 steps R@1_i2t "
            f"{c['R@1_i2t'][-1]} (baseline 0.02), VQA EM {c['vqa_em'][-1]} (baseline 0.0078)",
        ),
        practice.Check(
            "FINDING: R@1_i2t peaks mid-run and then collapses",
            (c["R@1_i2t"], c["vqa_em"]) == ([0.02, 0.02, 0.04, 0.0, 0.0, 0.0], [0.02] * 5 + [0.0]),
            f"R@1_i2t {c['R@1_i2t']}; VQA EM {c['vqa_em']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
