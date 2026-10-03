"""Exercise 2 -- on a Colab T4 the benchmark never touches the Tensor Cores.

    If you don't have a GPU, run it on Google Colab and compare

Reading of the exercise: Colab is out of reach here (no network, no GPU), so
the scaled-down runnable of DESIGN D11 ships instead: the lesson's own
`gpu_check.check_gpu` is run against a fake `torch` that reports what a Colab
T4 runtime reports (name "Tesla T4", 15102 MiB, compute capability 7.5) and a
scripted clock in which the GPU matmul takes exactly the T4's fp32 floor.
"Compare" is read as: what does the Colab run print, and what is it actually
measuring. The roofline numbers are NVIDIA's T4 datasheet peaks (8.1 TFLOPS
fp32, 65 TFLOPS fp16 on Tensor Cores), labelled below.

**ANSWER: on a T4 the script prints "Tesla T4", 15.8 GB, compute capability
7.5 and ~8B parameters.** Its 4000x4000 matmul is 128 GFLOP, so the GPU time it
can report is at least 15.8 ms; with a placeholder CPU second that prints
"Speedup: 63x".

**FINDING: the benchmark is fp32, so a T4 runs it without Tensor Cores.** Both
`randn` calls pass no dtype (torch's default, float32), and TF32 needs compute
capability 8.0 -- the T4 is 7.5. The Key Terms table credits Tensor Cores with
"4-8x"; on this card the fp16 floor is 1.97 ms, 8.0x below what the lesson
measures, so the Colab comparison understates the GPU by up to 8x.

**CONTROL: on Colab's default CPU runtime the script sends you to Colab.**
With `cuda.is_available()` False it prints "use Google Colab (free)" and runs
no benchmark -- step 2 of Option 2 (switch the runtime to T4) is not optional.

Structure: `fake_torch` reports a T4 and records randn kwargs; `run` drives it.
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
T4 = {"name": "Tesla T4", "mib": 15102, "cc": (7, 5)}   # as torch reports it on Colab
T4_FP32_TFLOPS, T4_FP16_TENSOR_TFLOPS = 8.1, 65.0      # NVIDIA T4 datasheet peaks
TF32_MIN_CC = (8, 0)                                    # Ampere introduced TF32
CPU_PLACEHOLDER_S = 1.0


class Tensor:
    def __init__(self, device, env):
        self.device, self.env = device, env

    def to(self, device):
        return Tensor(device, self.env)

    def __matmul__(self, other):
        self.env["now"] += self.env["cost"][self.device]
        return self


def fake_torch(env, cuda):
    props = types.SimpleNamespace(total_memory=T4["mib"] * 2**20, major=T4["cc"][0],
                                  minor=T4["cc"][1])

    def randn(*shape, **kwargs):
        env["randn"].append((shape, kwargs))
        return Tensor("cpu", env)

    return types.SimpleNamespace(
        __version__="fake", version=types.SimpleNamespace(cuda="12.4"), randn=randn,
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda, get_device_name=lambda i: T4["name"],
            get_device_properties=lambda i: props, synchronize=lambda: None))


def run(gpu_s, cuda=True):
    env = {"now": 0.0, "cost": {"cpu": CPU_PLACEHOLDER_S, "cuda": gpu_s}, "randn": []}
    ref = parity.load_reference(PHASE, LESSON, "gpu_check")
    ref.time = types.SimpleNamespace(time=lambda: env["now"])
    out = io.StringIO()
    with mock.patch.dict(sys.modules, {"torch": fake_torch(env, cuda)}):
        with contextlib.redirect_stdout(out):
            ref.check_gpu()
    return out.getvalue(), env["randn"]


def field(text, label):
    return re.search(rf"{label}: ?(.+)", text).group(1).strip()


def solve():
    flops = 2 * 4000**3
    fp32_floor = flops / (T4_FP32_TFLOPS * 1e12)
    fp16_floor = flops / (T4_FP16_TENSOR_TFLOPS * 1e12)
    text, calls = run(fp32_floor)
    cpu_runtime, _ = run(fp32_floor, cuda=False)
    return {
        "gpu": field(text, "GPU"), "memory": field(text, "Memory"),
        "cc": field(text, "Compute capability"),
        "estimate": field(text, "fp16\\)").split()[0],
        "speedup": field(text, "Speedup"),
        "dtypes": [kw.get("dtype") for _, kw in calls], "shapes": [s for s, _ in calls],
        "fp32_floor_ms": 1e3 * fp32_floor, "fp16_floor_ms": 1e3 * fp16_floor,
        "doc_tensor_core": "4-8x faster" in parity.doc_text(PHASE, LESSON),
        "cpu_runtime_advice": "Google Colab" in cpu_runtime,
        "cpu_runtime_benchmark": "Speedup" in cpu_runtime,
    }


def verify(r):
    cc = tuple(int(x) for x in r["cc"].split("."))
    gap = r["fp32_floor_ms"] / r["fp16_floor_ms"]
    return [
        practice.Check(
            "ANSWER: the T4 run prints Tesla T4, 15.8 GB, cc 7.5, ~8B",
            all((r["gpu"] == "Tesla T4", r["memory"] == "15.8 GB", r["estimate"] == "~8B",
                 r["speedup"] == "63x")),
            f"GPU {r['gpu']}, Memory {r['memory']}, cc {r['cc']}, estimate {r['estimate']}; "
            f"the fp32 floor for the 4000^2 matmul is {r['fp32_floor_ms']:.1f} ms, so a "
            f"placeholder CPU second prints Speedup: {r['speedup']}",
        ),
        practice.Check(
            "FINDING: fp32 on a cc-7.5 card, so no Tensor Cores and up to 8x unmeasured",
            all((r["dtypes"] == [None, None], cc < TF32_MIN_CC, r["doc_tensor_core"],
                 7.5 < gap < 8.5)),
            f"randn dtypes {r['dtypes']} (float32 default), cc {cc} below TF32's "
            f"{TF32_MIN_CC}; fp16 Tensor-Core floor {r['fp16_floor_ms']:.2f} ms against fp32 "
            f"{r['fp32_floor_ms']:.1f} ms, {gap:.1f}x -- the doc's Tensor Cores are '4-8x faster'",
        ),
        practice.Check(
            "CONTROL: on Colab's default CPU runtime the script points back to Colab",
            r["cpu_runtime_advice"] and not r["cpu_runtime_benchmark"],
            "cuda unavailable: prints 'use Google Colab (free)' and no Speedup line, so the "
            "runtime must be switched to T4 first",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
