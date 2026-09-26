"""Exercise 1 — there is no prompt-length crossover: disaggregation wins every row because the decode pool is given faster hardware.

    Run `code/main.py`. At what prompt length does disaggregation beat
    colocation?

Reading of the exercise: the crossover is searched for, not read off the
printed table -- the reference `ms_colocated` / `ms_disaggregated` are
evaluated at the table's rows and then solved for the prompt length where
the winner flips, over both transports, and the constants that decide it are
varied one at a time.

**ANSWER: at every prompt length -- there is no prompt-length crossover.**
Disaggregation wins all 8 rows (256..32768 tokens) over RDMA *and* TCP. The
code's winner flips on the prompt/output *ratio*: each output token saves
1/0.10 - 1/0.18 = 4.444 ms, each prompt token costs 0.00125 ms of RDMA
transfer (0.0125 ms TCP), so colocation wins only when prompt > 3555x output
over RDMA or > 355x output over TCP -- first at prompt 3556 / 356 for a
1-token output. With 0 output tokens colocation wins at any length.

**FINDING: the whole win is the decode pool's constant, not the split.**
`DECODE_TOK_PER_MS_DECODE_GPU` is 0.18 against 0.10 colocated -- an
"H200-like" GPU. Give the decode pool the same GPU and colocation wins every
row, by exactly the transfer time (0.32 ms at 256 tokens, 40.96 ms at 32768).
The simulator has no interference, batching or GPU count, so moving decode to
a separate pool of identical GPUs can only add the KV transfer.

**FINDING: the lesson's 512/200 threshold is not in the code.** "Prompts <
512 tokens and outputs < 200 tokens: transfer tax dominates gain", yet at
(256, 100) disaggregation wins by 444.1 ms against a 0.32 ms RDMA / 3.2 ms TCP
tax. The skill's "TCP raises the break-even to prompts >2K" is not there
either: TCP wins all 8 rows, and the Winner column compares RDMA only.

**FINDING: the constants exceed the lesson's own hardware numbers, and it
prints no cost.** 40 prefill tokens/ms at 2 x 70e9 FLOP/token is 5600 TFLOPS
against the text's "~2000 TFLOPS"; 100 decode tokens/s at batch 1 needs
7 TB/s of weight reads against "~3 TB/s", a bound of 42.9 tok/s. "Use It"
says the script "Reports throughput, cost per request"; it prints latency only.

Structure: `crossover()` scans prompt length for a fixed output; `patched()`
swaps one module constant and restores it.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "17-disaggregated-prefill-decode"
CASES = [(256, 100), (512, 200), (1024, 300), (2048, 400),
         (4096, 500), (8192, 800), (16384, 1200), (32768, 2000)]
PARAMS, DOC_TFLOPS, DOC_TB_S = 70e9, 2000, 3


def margins(ref, rdma=True):
    """colocated - disaggregated ms per row: positive means disaggregation wins."""
    return [round(ref.ms_colocated(p, o) - ref.ms_disaggregated(p, o, rdma), 2)
            for p, o in CASES]


def crossover(ref, output, rdma):
    prompt = 1
    while ref.ms_disaggregated(prompt, output, rdma) < ref.ms_colocated(prompt, output):
        prompt += 1
    return prompt


def patched(ref, name, value, fn):
    original = getattr(ref, name)
    setattr(ref, name, value)
    try:
        return fn()
    finally:
        setattr(ref, name, original)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.main()
    same_gpu = patched(ref, "DECODE_TOK_PER_MS_DECODE_GPU",
                       ref.DECODE_TOK_PER_MS_COLOCATED, lambda: margins(ref))
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "rdma": margins(ref), "tcp": margins(ref, False), "same_gpu": same_gpu,
        "cross": (crossover(ref, 1, True), crossover(ref, 1, False)),
        "zero_out": ref.ms_colocated(4096, 0) < ref.ms_disaggregated(4096, 0),
        "per_out": 1 / ref.DECODE_TOK_PER_MS_COLOCATED - 1 / ref.DECODE_TOK_PER_MS_DECODE_GPU,
        "tax": [round(ref.ms_disaggregated(p, 0, r) - ref.ms_colocated(p, 0), 2)
                for p, _ in CASES for r in (True, False)],
        "winners": [line.split()[-1] for line in out.getvalue().splitlines()
                    if line.strip()[:1].isdigit()],
        "tflops": ref.PREFILL_TOK_PER_MS * 1000 * 2 * PARAMS / 1e12,
        "bound": DOC_TB_S * 1e12 / PARAMS, "decode": ref.DECODE_TOK_PER_MS_COLOCATED * 1000,
        "doc": all(s in doc for s in ("< 512 tokens", "2000 TFLOPS", "3 TB/s", "cost per request")),
        "prints_cost": "cost" in out.getvalue().lower(),
    }


def verify(result):
    rdma, tcp, tax = result["rdma"], result["tcp"], result["tax"]
    return [
        practice.Check(
            "ANSWER: at every prompt length -- there is no prompt-length crossover",
            all([min(rdma) > 0, min(tcp) > 0, result["winners"] == ["disaggregated"] * 8,
                 result["cross"] == (3556, 356), result["zero_out"]]),
            f"disaggregation wins all 8 rows, RDMA margins {rdma}; each output token saves "
            f"{result['per_out']:.3f} ms, so colocation wins only past prompt "
            f"{result['cross'][0]} (RDMA) / {result['cross'][1]} (TCP) per output token",
        ),
        practice.Check(
            "FINDING: the whole win is the decode pool's constant, not the split",
            result["same_gpu"] == [-t for t in tax[::2]],
            f"with the decode pool on the colocated GPU every margin is minus the RDMA "
            f"transfer: {result['same_gpu']}",
        ),
        practice.Check(
            "FINDING: the lesson's 512/200 threshold is not in the code",
            result["doc"] and rdma[0] == 444.12 and tax[:2] == [0.32, 3.2],
            f"at (256, 100) disaggregation wins by {rdma[0]} ms against a {tax[0]} ms RDMA / "
            f"{tax[1]} ms TCP tax; TCP margins {tcp} are all positive",
        ),
        practice.Check(
            "FINDING: the constants exceed the lesson's own hardware numbers, and it prints no cost",
            all([result["tflops"] == 5600, result["decode"] > 2 * result["bound"],
                 not result["prints_cost"]]),
            f"prefill implies {result['tflops']:.0f} TFLOPS against {DOC_TFLOPS}; decode "
            f"{result['decode']:.0f} tok/s against a {result['bound']:.1f} tok/s bound at "
            f"{DOC_TB_S} TB/s; the output has no cost column",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
