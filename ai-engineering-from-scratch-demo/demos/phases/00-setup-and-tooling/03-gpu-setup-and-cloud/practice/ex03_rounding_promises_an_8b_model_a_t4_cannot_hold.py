"""Exercise 3 -- the estimate rounds up, promising an 8B model a T4 cannot hold.

    Check how much GPU memory you have and estimate the largest model you can
    fit (rule of thumb: 2 bytes per parameter for fp16)

Reading of the exercise: "how much GPU memory you have" is read on this runner
first -- the lesson's own `check_gpu` with `torch` made unimportable -- and
then, since the answer there is none, on four cards whose `total_memory` a fake
`torch` reports as the real drivers do (MiB, labelled in CARDS). The estimate
is the one the lesson's script prints, parsed from its stdout, and is checked
against real checkpoints' parameter counts (MODELS) at 2 bytes each.

**ANSWER: this runner has no torch and no GPU, so the script reports no
memory at all** ("PyTorch not installed"). On a T4 (15.8 GB) it prints ~8B, on
an RTX 4090 (25.8 GB) ~13B, on an A100 40GB (42.5 GB) ~21B, on an H100 80GB
(85.5 GB) ~43B -- `total_memory / 2e9`, rounded to a whole billion.

**FINDING: rounding to nearest overstates the fit on 3 of 4 cards.** The T4's
7.92B prints as ~8B, but Llama-3-8B (8.03B parameters) needs 16.06 GB of fp16
weights against 15.84 GB; the 4090's 12.88B prints as ~13B, and Llama-2-13B
needs 26.03 GB against 25.76 GB. A printed "~N B" is only safe when rounded
down -- and that is before the CUDA context, activations or KV cache.

**FINDING: the rule is inference weights only.** Mixed-precision Adam holds 16
bytes per parameter (ZeRO, Rajbhandari et al. 2019), so the same T4 trains at
most 0.99B -- 8x less than it prints.

**CONTROL: the printed figure is exactly `round(total_memory / 2e9)`** on all
four cards, so the overclaim is the format, not a different formula.

Structure: `printed` runs the lesson's script on one card and parses its estimate.
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
CARDS = {  # total_memory in MiB, as torch / nvidia-smi report it
    "Tesla T4": 15102, "RTX 4090": 24564, "A100 40GB": 40536, "H100 80GB": 81559}
MODELS = {"Llama-3-8B": 8_030_261_248, "Llama-2-13B": 13_015_864_320}  # parameter counts
FP16_BYTES, ADAM_MIXED_BYTES = 2, 16


class Tensor:
    """Just enough tensor for the benchmark to finish and reach the estimate."""

    def to(self, device):
        return self

    def __matmul__(self, other):
        return self


def fake_torch(name, total):
    props = types.SimpleNamespace(total_memory=total, major=8, minor=0)
    return types.SimpleNamespace(
        __version__="fake", version=types.SimpleNamespace(cuda="12.4"),
        randn=lambda *shape: Tensor(),
        cuda=types.SimpleNamespace(
            is_available=lambda: True, get_device_name=lambda i: name,
            get_device_properties=lambda i: props, synchronize=lambda: None))


def script_output(torch_module):
    ref = parity.load_reference(PHASE, LESSON, "gpu_check")
    ticks = iter(range(1, 100))  # a clock that always advances, so no 0-second interval
    ref.time = types.SimpleNamespace(time=lambda: next(ticks))
    out = io.StringIO()
    with mock.patch.dict(sys.modules, {"torch": torch_module}):
        with contextlib.redirect_stdout(out):
            ref.check_gpu()
    return out.getvalue()


def printed(name, total):
    """The script's own 'Memory' and 'Estimated max model size' for one card."""
    text = script_output(fake_torch(name, total))
    memory = re.search(r"Memory: ([\d.]+) GB", text).group(1)
    return float(memory), int(re.search(r"~(\d+)B parameters", text).group(1))


def solve():
    host = script_output(None)
    cards = {}
    for name, mib in CARDS.items():
        total = mib * 2**20
        gb, estimate = printed(name, total)
        cards[name] = {"total": total, "gb": gb, "estimate": estimate,
                       "exact_b": total / FP16_BYTES / 1e9,
                       "overclaims": estimate * 1e9 * FP16_BYTES > total}
    return {"host": host.strip(), "cards": cards}


def verify(r):
    c = r["cards"]
    t4, rtx = c["Tesla T4"], c["RTX 4090"]
    llama8, llama13 = (MODELS[m] * FP16_BYTES for m in ("Llama-3-8B", "Llama-2-13B"))
    trainable = t4["total"] / ADAM_MIXED_BYTES / 1e9
    return [
        practice.Check(
            "ANSWER: no GPU here; ~8B / ~13B / ~21B / ~43B on T4 / 4090 / A100 / H100",
            all(("not installed" in r["host"],
                 [v["estimate"] for v in c.values()] == [8, 13, 21, 43])),
            f"this runner prints {r['host']!r}; "
            + ", ".join(f"{k} {v['gb']} GB -> ~{v['estimate']}B" for k, v in c.items()),
        ),
        practice.Check(
            "FINDING: rounding to nearest promises a model that does not fit, 3 of 4 cards",
            all((sum(v["overclaims"] for v in c.values()) == 3, llama8 > t4["total"],
                 llama13 > rtx["total"])),
            f"T4 {t4['exact_b']:.2f}B prints ~8B; Llama-3-8B needs {llama8 / 1e9:.2f} GB "
            f"against {t4['total'] / 1e9:.2f}. 4090 {rtx['exact_b']:.2f}B prints ~13B; "
            f"Llama-2-13B needs {llama13 / 1e9:.2f} GB against {rtx['total'] / 1e9:.2f}",
        ),
        practice.Check(
            "FINDING: 2 bytes/param is inference weights; Adam training needs 16",
            all((trainable < 1.0, t4["estimate"] / trainable > 7.5)),
            f"at {ADAM_MIXED_BYTES} bytes/param the T4 trains at most {trainable:.2f}B, "
            f"{t4['estimate'] / trainable:.1f}x less than it prints",
        ),
        practice.Check(
            "CONTROL: the printed figure is round(total_memory / 2e9) on every card",
            all(v["estimate"] == round(v["exact_b"]) for v in c.values()),
            ", ".join(f"{k} {v['exact_b']:.2f} -> {v['estimate']}" for k, v in c.items()),
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
