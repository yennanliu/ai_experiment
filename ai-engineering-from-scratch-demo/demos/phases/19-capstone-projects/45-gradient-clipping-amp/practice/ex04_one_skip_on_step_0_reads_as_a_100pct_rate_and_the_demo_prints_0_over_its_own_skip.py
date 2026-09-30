"""Exercise 4 — the lesson's rolling rate reads one early skip as 100%, and its demo prints 0.0000 over its own skip.

    Add a rolling-window skip-rate computation and a CLI flag that fails the run if the rate exceeds a configured threshold for 100 consecutive steps.

Reading of the exercise: the rolling rate is the lesson's own
`rolling_skip_rate` over the lesson's `AmpTrainState.log`. It is not
rewritten. The CLI (`parse`) adds `--max-skip-rate`, and with it
`--window` and `--patience` (default 100 consecutive steps). `run` trains
the lesson's seed-7 toy model for `--steps` steps and returns exit status 1
at the first step whose last `patience` rates all exceed the threshold.
Skips are injected with the lesson's own `inject_inf_into_first_grad`, on
every `--skip-every`-th step or at `--skip-at` steps. Every scenario runs
300 steps with a 100-step window and a 5% threshold: the doc's alert level,
though the doc's window is 1,000 steps.

**ANSWER: the flag fails a steady 10% skip rate and passes a steady 2% one.**
A clean run and one skip in 50 both exit 0. One skip in 10 (on steps 9, 19,
...) keeps the rate above 5% from step 9 on, so the run exits 1 at step 108,
the 100th consecutive step over the threshold.

**FINDING: `rolling_skip_rate` divides by the steps seen so far, not by the
window, so an early skip reads as a storm.** One skip on step 0 reads 100%,
then 50%, and stays above 5% for 19 steps (0-18). An alert that fires on
the first step over the threshold would page on it. The 100-step patience
is what keeps it quiet: that run exits 0.

**FINDING: the lesson's demo prints `final_skip_rate=0.0000` right under its
own skip.** `run_demo` reports `skip_count=1` and then the rate over a
10-step window, which by step 19 no longer contains the skip on step 5.

Expected output: three PASS checks.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys

from harness import parity, practice

try:
    import torch  # noqa: F401  (the lesson's loop needs it)
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "45-gradient-clipping-amp"
BASE = ["--steps", "300", "--window", "100", "--max-skip-rate", "0.05"]


def parse(argv):
    cli = argparse.ArgumentParser(description="lesson 45 loop with a skip-rate kill switch")
    cli.add_argument("--steps", type=int, default=300)
    cli.add_argument("--window", type=int, default=1000)
    cli.add_argument("--max-skip-rate", type=float, default=None, help="fail when exceeded for --patience steps")
    cli.add_argument("--patience", type=int, default=100)
    cli.add_argument("--skip-every", type=int, default=0, help="inject an Inf grad on every Nth step")
    cli.add_argument("--skip-at", type=int, nargs="*", default=[])
    return cli.parse_args(argv)


def breach_step(rates, threshold, patience):
    """First step whose last `patience` rolling rates all exceed `threshold`, else None."""
    run_length = 0
    for step, rate in enumerate(rates):
        run_length = run_length + 1 if rate > threshold else 0
        if run_length >= patience:
            return step
    return None


def run(ref, argv):
    args = parse(argv)
    model, inputs, targets = ref.build_toy_model()
    state = ref.AmpTrainState(model=model, lr=1e-2, max_norm=1.0, device_type="cpu")
    for i in range(args.steps):
        bad = i in args.skip_at or (args.skip_every and i % args.skip_every == args.skip_every - 1)
        state.step(inputs, targets, gradient_corruptor=ref.inject_inf_into_first_grad if bad else None)
    rates = ref.rolling_skip_rate(state.log, window=args.window)
    step = None if args.max_skip_rate is None else breach_step(rates, args.max_skip_rate, args.patience)
    return {"exit": int(step is not None), "breach": step, "skips": state.skip_count, "rates": rates}


def main(argv):
    out = run(parity.load_reference(PHASE, LESSON, "main"), argv)
    print(f"skips={out['skips']} final_rate={out['rates'][-1]:.4f} breach_step={out['breach']}")
    return out["exit"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    scenarios = {"clean": [], "every_50": ["--skip-every", "50"], "every_10": ["--skip-every", "10"],
                 "one_at_0": ["--skip-at", "0"]}
    runs = {name: run(ref, BASE + flags) for name, flags in scenarios.items()}
    early = runs["one_at_0"]["rates"]
    with contextlib.redirect_stdout(io.StringIO()) as demo:
        ref.run_demo()
    return {
        "outcomes": {name: (r["exit"], r["breach"], r["skips"]) for name, r in runs.items()},
        "early_rates": [round(x, 4) for x in early[:3]], "early_steps_over": sum(x > 0.05 for x in early),
        "naive_breach": breach_step(early, 0.05, patience=1),
        "demo_tail": demo.getvalue().strip().splitlines()[-1],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: --max-skip-rate 0.05 fails a steady 10% run at step 108 and passes clean and 2% runs",
            {k: result["outcomes"][k] for k in ("clean", "every_50", "every_10")}
            == {"clean": (0, None, 0), "every_50": (0, None, 6), "every_10": (1, 108, 30)},
            f"(exit, breach step, skips): {result['outcomes']}",
        ),
        practice.Check(
            "FINDING: one skip on step 0 reads as a 100% rate and stays over 5% for 19 steps",
            result["early_rates"] == [1.0, 0.5, 0.3333] and result["early_steps_over"] == 19
            and result["naive_breach"] == 0 and result["outcomes"]["one_at_0"] == (0, None, 1),
            f"rates {result['early_rates']}..., {result['early_steps_over']} steps over 5%; patience 1 fails at "
            f"step {result['naive_breach']}, patience 100 gives {result['outcomes']['one_at_0']}",
        ),
        practice.Check(
            "FINDING: the lesson's demo prints final_skip_rate=0.0000 under skip_count=1",
            result["demo_tail"] == "skip_count=1 final_skip_rate=0.0000",
            f"run_demo's last line: {result['demo_tail']!r}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]) if sys.argv[1:] else practice.selfcheck(globals()))
