"""Exercise 5 -- throughput holds a steady band on 76 of 80 steps, and every probe step drops it about 4x.

    Log per step throughput (tokens per second) next to the loss and confirm it stays in a steady band.

Reading of the exercise: throughput is measured on the lesson's own `train()`
running the demo config, without editing it. The module's `make_batches` is
wrapped for the training stream (seed 0) so each `next()` stamps a clock; the
gap between two stamps is one full step as `train()` runs it, probe work
included. Tokens per step is `inputs.numel()`, and `tokens_per_sec` is
written into each JSONL record next to `train_loss`. The clock is process CPU
time with torch pinned to one thread, so other load on the machine does not
move it. "Steady band" is read as: the 5th-95th percentile of non-probe steps
within 0.5x-2x of their median. The thresholds are loose on purpose, because
the absolute rate depends on the machine.

**ANSWER: yes, outside the probe steps.** Every step trains on 4 x 32 = 128
tokens. On the 76 non-probe steps the 5th-95th percentile throughput stays
within about 10% of the median (about 59,000 tokens/s on the authoring
machine; the check asserts only the 0.5x-2x band). The 80 records in the
rewritten JSONL all carry `tokens_per_sec` beside `train_loss`.

**FINDING: the probe steps are not training slowdowns.** Steps 19, 39, 59 and
79 each run 21 forward passes instead of 1: the training forward, 4
`evaluate_model` batches and 16 generation steps, counted with a forward
hook. Their throughput drops to about 1/4.4 of the median (asserted as more
than 2x slower). The eval and sample tokens do not count toward the 128, so
a per-step tokens/s that includes the probe reports 4 false dips per run.
To see a steady band, either time only the training part or leave the probe
steps out.

Structure: `timed()` wraps `make_batches` and a forward hook around one
`train()` call, then restores both; `log_rates()` turns the stamps into
rates and rewrites the JSONL; `solve()` summarises the band.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import statistics
import tempfile
import time

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "36-training-loop-eval"


def timed(ref, cfg, log_path):
    """Run the lesson's train(); return (records, [(cpu_time, tokens, forwards_so_far)])."""
    marks, forwards, original = [], [0], ref.make_batches

    def stamped(tokens, batch_size, context_length, seed=0):
        for batch in original(tokens, batch_size, context_length, seed):
            if seed == cfg.seed:
                marks.append((time.process_time(), batch[0].numel(), forwards[0]))
            yield batch

    torch.manual_seed(0)
    model = ref.GPTModel(ref.ModelConfig(dropout=0.0))
    hook = model.register_forward_hook(lambda *_: forwards.__setitem__(0, forwards[0] + 1))
    train = ref._synthetic_byte_tokens(4096, 256, 1)
    val = ref._synthetic_byte_tokens(1024, 256, 2)
    ref.make_batches = stamped
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            records = ref.train(model, train, val, cfg, torch.tensor([[7, 11, 13, 17]]),
                                log_path=log_path)
    finally:
        ref.make_batches = original
        hook.remove()
    marks.append((time.process_time(), 0, forwards[0]))
    return records, marks


def log_rates(records, marks, log):
    """Per-step tokens/s from consecutive stamps, written into each JSONL record."""
    steps = list(zip(marks, marks[1:]))
    rates = [a[1] / max(b[0] - a[0], 1e-9) for a, b in steps]
    for rec, rate in zip(records, rates):
        rec["tokens_per_sec"] = rate
    log.write_text("".join(json.dumps(r) + "\n" for r in records))
    return steps, rates, [json.loads(line) for line in log.read_text().splitlines()]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = ref.TrainConfig()
    with tempfile.TemporaryDirectory() as tmp:
        log = pathlib.Path(tmp) / "losses.jsonl"
        records, marks = timed(ref, cfg, log)
        steps, rates, rows = log_rates(records, marks, log)
    probe = [r["step"] for r in records if "val_loss" in r]
    rest = sorted(rate for s, rate in enumerate(rates) if s not in probe)
    median = statistics.median(rest)
    return {
        "tokens": sorted({m[1] for m in marks[:-1]}), "n": len(rates), "probe": probe,
        "band": [rest[int(0.05 * len(rest))] / median, rest[int(0.95 * len(rest))] / median],
        "median": median, "probe_ratio": [median / rates[s] for s in probe],
        "forwards": sorted({b[2] - a[2] for a, b in steps}),
        "probe_forwards": [steps[s][1][2] - steps[s][0][2] for s in probe],
        "logged": sum({"train_loss", "tokens_per_sec"} <= set(r) for r in rows),
    }


def verify(result):
    r = result
    lo, hi = r["band"]
    ratios = ", ".join(f"{x:.1f}x" for x in r["probe_ratio"])
    return [
        practice.Check(
            "ANSWER: yes, outside the probe steps",
            r["tokens"] == [128] and r["n"] == 80 and r["logged"] == 80 and 0.5 < lo <= hi < 2.0,
            f"{r['tokens'][0]} tokens/step; non-probe p5-p95 {lo:.3f}x-{hi:.3f}x of median "
            f"{r['median']:,.0f} tok/s (CPU); {r['logged']}/80 records log tokens_per_sec",
        ),
        practice.Check(
            "FINDING: the probe steps are not training slowdowns",
            r["probe"] == [19, 39, 59, 79] and r["forwards"] == [1, 21]
            and r["probe_forwards"] == [21] * 4 and min(r["probe_ratio"]) > 2.0,
            f"probe steps {r['probe']} run {r['probe_forwards']} forwards (others 1); "
            f"throughput drops {ratios}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
