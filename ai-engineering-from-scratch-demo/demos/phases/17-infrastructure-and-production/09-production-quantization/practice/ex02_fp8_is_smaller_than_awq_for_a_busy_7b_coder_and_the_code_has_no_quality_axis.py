"""Exercise 2 — FP8 is smaller than AWQ for a busy 7B coder, and the code has no quality axis.

    You have a 7B coding model. Pick a format and justify. If you were wrong
    about quality tolerance, what is the recovery path?

Reading of the exercise: the deployment is the lesson's own default shape, a
7B model on one H100 at 128 concurrent 2k-token sequences. "Justify" uses the
two things the lesson offers: its prose, which lists quality risk by workload,
and its code, which gives the HBM each format costs. "Wrong about quality
tolerance" means the quantized model fails the code eval after launch.
"Recovery path" is where it goes next, and what that costs.

**ANSWER: FP8, which here is both the safer choice and the smaller one.**
The lesson lists code-gen among the workloads where "quality is
non-negotiable" and FP8 is the safe default. Its reasoning section warns that
code-gen with long context "suffer[s] visibly" at INT4. It also says FP8's
"memory savings are half of INT4", but at 128 x 2k that is backwards: FP8
totals 18.22 GB and AWQ 25.58 GB. AWQ saves 3.5 GB of weights but keeps
21.73 GB of BF16 KV, while FP8 halves the KV. The recovery path is
FP8 -> BF16 (36.08 GB) on the same H100, a weight swap with no new hardware.
Keep the 14 GB BF16 checkpoint on disk and gate it on the HumanEval delta.
If the tolerance turns out looser than assumed, AWQ is also a same-GPU swap.

**FINDING: AWQ's memory win for a 7B only exists below 42 concurrent 2k
sequences.** Weight bytes (3.5 GB saved) and KV bytes (0.085 GB per sequence
saved by FP8 KV) cross between 41 and 42 sequences: AWQ is 10.81 against
10.83 GB at 41, and 10.98 against 10.92 at 42. At one sequence AWQ is 4.02
against 7.43. So "INT4 saves the most memory" holds only for a
lightly loaded 7B.

**FINDING: the code cannot express "wrong about quality tolerance".** `Format`
has five fields (name, weight_bits, kv_bits, engine, notes) and none is
accuracy, so no run of `main.py` distinguishes AWQ from GPTQ or FP8 from
BF16 on quality. The lesson's own benchmark source (Jarvis Labs, Qwen2.5-32B,
H200, fetched 2026-09-26) measures HumanEval Pass@1 of 56.1 at FP16, 51.8
for Marlin-AWQ and 45.7 for Marlin-GPTQ, so the gap the code cannot show
is 4.3 to 10.4 points. The recovery trigger has to be an eval, not this
calculator.

Structure: `solve()` runs the reference `memory_breakdown` for the three rungs
of the ladder and scans concurrency for the AWQ/FP8 crossover.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "09-production-quantization"
PARAMS, CONC, CTX = 7, 128, 2048
HUMANEVAL = {"FP16": 56.1, "Marlin-AWQ": 51.8, "Marlin-GPTQ": 45.7}  # Jarvis Labs, Qwen2.5-32B


def pick(ref, prefix):
    return next(f for f in ref.FORMATS if f.name.startswith(prefix))


def total(ref, fmt, conc=CONC):
    return round(ref.memory_breakdown(PARAMS, fmt, conc, CTX)["total"], 2)


def crossover(ref, awq, fp8):
    """Smallest concurrency at which AWQ's total exceeds FP8's."""
    return next(c for c in range(1, CONC + 1) if total(ref, awq, c) > total(ref, fp8, c))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    awq, fp8, bf16 = pick(ref, "AWQ"), pick(ref, "FP8"), pick(ref, "BF16")
    ladder = {f.name: total(ref, f) for f in (awq, fp8, bf16)}
    doc = parity.doc_text(PHASE, LESSON)
    cross = crossover(ref, awq, fp8)
    return {
        "ladder": ladder,
        "fits": {n: ref.gpu_check(t) for n, t in ladder.items()},
        "awq_kv": round(ref.memory_breakdown(PARAMS, awq, CONC, CTX)["kv"], 2),
        "bf16_disk": ref.memory_breakdown(PARAMS, bf16, 1, CTX)["weight"],
        "doc_says": all(s in doc for s in ("quality is non-negotiable (reasoning, medical, "
                                           "code-gen)", "Memory savings are half of INT4")),
        "cross": cross,
        "around": {c: (total(ref, awq, c), total(ref, fp8, c)) for c in (1, cross - 1, cross)},
        "fields": [f.name for f in dataclasses.fields(ref.Format)],
    }


def verify(result):
    lad, awq, fp8, bf16 = result["ladder"], *result["ladder"]
    drops = {k: round(HUMANEVAL["FP16"] - v, 1) for k, v in HUMANEVAL.items() if k != "FP16"}
    return [
        practice.Check(
            "ANSWER: FP8, which here is both the safer choice and the smaller one",
            all([result["doc_says"], lad[fp8] == 18.22, lad[awq] == 25.58, lad[bf16] == 36.08,
                 set(result["fits"].values()) == {"H100 80GB"}]),
            f"totals at {CONC} x {CTX}: {lad}; AWQ keeps {result['awq_kv']} GB of BF16 KV; "
            f"every rung fits {set(result['fits'].values())}, so FP8 -> BF16 is a weight "
            f"swap; BF16 checkpoint on disk {result['bf16_disk']} GB",
        ),
        practice.Check(
            "FINDING: AWQ's memory win for a 7B only exists below 42 concurrent 2k sequences",
            result["cross"] == 42 and result["around"][1] == (4.02, 7.43),
            f"AWQ vs FP8 totals by concurrency {result['around']}",
        ),
        practice.Check(
            "FINDING: the code cannot express 'wrong about quality tolerance'",
            result["fields"] == ["name", "weight_bits", "kv_bits", "engine", "notes"]
            and drops == {"Marlin-AWQ": 4.3, "Marlin-GPTQ": 10.4},
            f"Format fields {result['fields']}; the lesson's benchmark source measures "
            f"HumanEval drops {drops} points from FP16 that no field can hold",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
