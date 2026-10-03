"""Exercise 1 -- one cold timed matmul each way, and a slower GPU prints "0x".

    Run the benchmark above and compare CPU vs GPU times

Reading of the exercise: this runner has no GPU and no torch, so "run the
benchmark" is read as running the lesson's own `gpu_check.check_gpu` against
a recording fake `torch` and a scripted clock that stands in for `time`. The
fake logs every tensor op, so the comparison is between what the protocol
measures and what it prints -- not between this host's chips. Host GPU state
never enters: `torch` is the fake, and `time` is the script.

**ANSWER: on a CPU-only machine the script prints no time at all.** With
`cuda.is_available()` False, `check_gpu` returns before the benchmark, so the
CPU half -- which needs no GPU -- never runs either. With a GPU, it times
exactly one 4000x4000 fp32 matmul on each device (128 GFLOP each), the GPU one
bracketed by `synchronize` with the 2 x 64 MB host-to-device copy left out,
and prints `cpu/gpu` -- 2.000 s against 0.020 s prints "Speedup: 100x".

**FINDING: both timings are cold.** The timed GPU matmul is the first CUDA
matmul in the process (0 before it) and nothing is repeated, so the number
includes cuBLAS start-up; the clock is `time.time`, wall time, read 4 times.

**FINDING: the ratio is formatted `:.0f`.** A GPU 2.5x slower than the CPU
prints "Speedup: 0x", and a GPU interval the clock cannot resolve (0 ticks)
raises ZeroDivisionError instead of printing anything.

**FINDING: the doc's benchmark is not the script's.** `docs/en.md` sets
`size = 5000` (250 GFLOP); `gpu_check.py` uses 4000 (128 GFLOP), so the two
"benchmarks" a reader runs differ by 1.95x in work.

Structure: `fake_torch` records ops; `run` drives `check_gpu` and returns its log.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import types
from unittest import mock

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "03-gpu-setup-and-cloud"


class Tensor:  # records every op; matmul advances the scripted clock
    def __init__(self, shape, device, env):
        self.shape, self.device, self.env = shape, device, env

    def to(self, device):
        self.env["log"].append(("to", device))
        return Tensor(self.shape, device, self.env)

    def __matmul__(self, other):
        self.env["log"].append(("matmul", self.device))
        self.env["now"] += self.env["cost"][self.device]
        return Tensor(self.shape, self.device, self.env)


def fake_torch(env, cuda):
    props = types.SimpleNamespace(total_memory=16e9, major=8, minor=0)
    return types.SimpleNamespace(
        __version__="fake", version=types.SimpleNamespace(cuda="12.4"),
        randn=lambda *shape: env["log"].append(("randn", shape)) or Tensor(shape, "cpu", env),
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda, get_device_name=lambda i: "FakeGPU",
            get_device_properties=lambda i: props,
            synchronize=lambda: env["log"].append(("sync", None))))


def run(cpu_s, gpu_s, cuda=True):
    """check_gpu's stdout, op log and clock reads, under a scripted clock."""
    env = {"log": [], "now": 0.0, "cost": {"cpu": cpu_s, "cuda": gpu_s}, "reads": 0}

    def clock():
        env["reads"] += 1
        env["log"].append(("time", env["now"]))
        return env["now"]
    ref = parity.load_reference(PHASE, LESSON, "gpu_check")
    ref.time = types.SimpleNamespace(time=clock)
    out = io.StringIO()
    with mock.patch.dict(sys.modules, {"torch": fake_torch(env, cuda)}):
        with contextlib.redirect_stdout(out):
            try:
                ref.check_gpu()
            except ZeroDivisionError as exc:
                env["error"] = type(exc).__name__
    return out.getvalue(), env


def speedup(text):
    return (re.findall(r"Speedup: (\S+)", text) or [None])[0]


def solve():
    no_gpu, _ = run(2.0, 0.02, cuda=False)
    text, env = run(2.0, 0.02)
    log, doc = env["log"], parity.doc_text(PHASE, LESSON)
    timed = log.index(("matmul", "cuda"))
    return {
        "cpu_only_prints_time": "matrix multiply" in no_gpu, "speedup": speedup(text),
        "shape": next(op[1] for op in log if op[0] == "randn"),
        "gpu_window": [op[0] for op in log[timed - 2:timed + 3]],
        "copies_untimed": max(i for i, op in enumerate(log) if op[0] == "to") < timed - 2,
        "cuda_matmuls_before": sum(op == ("matmul", "cuda") for op in log[:timed]),
        "matmuls": sum(op[0] == "matmul" for op in log), "clock_reads": env["reads"],
        "slow_gpu": speedup(run(2.0, 5.0)[0]),
        "zero_tick": run(2.0, 0.0)[1].get("error"),
        "doc_size": int(re.search(r"size = (\d+)", doc).group(1)),
    }


def verify(r):
    n, doc_n = r["shape"][0], r["doc_size"]
    return [
        practice.Check(
            "ANSWER: CPU-only prints no time; with a GPU, one 4000^2 matmul each way",
            all((not r["cpu_only_prints_time"], r["speedup"] == "100x", n == 4000,
                 r["copies_untimed"])),
            f"no GPU: no 'matrix multiply' line at all. GPU: shape {r['shape']}, "
            f"{2 * n**3 / 1e9:.0f} GFLOP per matmul, timed window {r['gpu_window']} with both "
            f".to('cuda') copies before it, "
            f"2.000 s / 0.020 s prints Speedup: {r['speedup']}",
        ),
        practice.Check(
            "FINDING: both timings are single cold runs on the wall clock",
            r["cuda_matmuls_before"] == 0 and r["matmuls"] == 2 and r["clock_reads"] == 4,
            f"{r['cuda_matmuls_before']} CUDA matmuls before the timed one, {r['matmuls']} "
            f"matmuls in total, time.time read {r['clock_reads']} times: no warm-up, no repeat",
        ),
        practice.Check(
            "FINDING: ':.0f' prints a slower GPU as 0x and a 0-tick GPU crashes",
            r["slow_gpu"] == "0x" and r["zero_tick"] == "ZeroDivisionError",
            f"GPU 2.5x slower prints Speedup: {r['slow_gpu']}; a GPU interval of 0 "
            f"ticks raises {r['zero_tick']}",
        ),
        practice.Check(
            "FINDING: the doc's benchmark is 5000^2, the script's 4000^2",
            doc_n == 5000 and n == 4000,
            f"docs/en.md size = {doc_n} ({2 * doc_n**3 / 1e9:.0f} GFLOP), gpu_check.py "
            f"{n} ({2 * n**3 / 1e9:.0f} GFLOP): {(doc_n / n) ** 3:.2f}x the work",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
