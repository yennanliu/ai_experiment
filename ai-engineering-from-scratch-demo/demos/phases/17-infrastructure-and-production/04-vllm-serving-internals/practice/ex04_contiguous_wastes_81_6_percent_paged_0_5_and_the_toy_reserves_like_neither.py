"""Exercise 4 — contiguous wastes 81.6%, paged 0.5%, and the toy reserves like neither.

    Compute the KV cache fragmentation waste for a trace of 1,000 requests with
    mean 1,500 output tokens, std 600 tokens, under (a) contiguous per-request
    allocation at 8192 max, (b) PagedAttention with 16-token blocks.

Reading of the exercise: the trace is 1,000 seeded draws from N(1500, 600),
rounded and clipped to [1, 8192]. Two draws fall below 1, and the sample mean
is 1511. Prompts are left out, since the exercise gives only output lengths.
Waste is reserved-but-unused KV slots over reserved slots. It is reported at
completion, which is the usual figure, and averaged over every decode step of
every request, which is what the pool actually holds while requests run.

**ANSWER: (a) 81.6% wasted, (b) 0.50%.** Contiguous allocation reserves 8192
slots and a request uses 1511 on average. That is 6681 slots, or 2.19 GB of
BF16 KV for Llama 3.3 70B, idle per request. PagedAttention loses only the
tail of its last block, 7.6 tokens per request on average, or 0.50% of what it
holds. That is under the lesson's "<4%". The contiguous figure is just above
its "60-80%" range for classic allocation, and matches its own 82% example.

**FINDING: while requests run, contiguous waste is 89.4% and paged is 0.86%.**
A decoding request has used t of its tokens at step t. Averaged over the
decode, the contiguous reservation is emptier than the completion snapshot
shows. The paged pool still holds at most one partial block per request.

**FINDING: the toy reserves each request's final length at admission, so it
wastes 50.2%.** `simulate_continuous` adds `blocks_needed()`, which is ceil
of (prompt + output) / 16, the moment a request is admitted. Blocks are
16-token pages, but the full output is held from the first step, and the
scheduler knows that output length in advance, which a real one cannot.
On this trace, the toy's policy averages 50.2% waste while requests decode.
PagedAttention grows by one block at a time, and that growth is what the toy
does not model.

Structure: `trace()` draws the requests. `waste()` computes (reserved - used)
/ reserved at completion and step-averaged for one reservation policy.
"""

from __future__ import annotations

import inspect
import math
import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "04-vllm-serving-internals"
N, MEAN, STD, MAX_LEN, BLOCK = 1000, 1500, 600, 8192, 16
KV_BYTES_PER_TOKEN = 327_680  # Llama 3.3 70B, BF16: 80 layers x 2 x 8 heads x 128 x 2


def trace(seed=0):
    rng = random.Random(seed)
    raw = [round(rng.gauss(MEAN, STD)) for _ in range(N)]
    return [min(MAX_LEN, max(1, x)) for x in raw], sum(x < 1 for x in raw)


def waste(lengths, reserved):
    """reserved(length, step) -> slots held; returns (at completion, step-averaged)."""
    done = 1 - sum(lengths) / sum(reserved(n, n) for n in lengths)
    used = sum(n * (n + 1) / 2 for n in lengths)
    held = sum(sum(reserved(n, t) for t in range(1, n + 1)) for n in lengths)
    return round(done * 100, 2), round((1 - used / held) * 100, 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lengths, clipped = trace()

    def toy(n, t):  # the reference's own reservation: final length, rounded to blocks
        return ref.Request(0, 0, n, 0.0).blocks_needed() * ref.KV_BLOCK_SIZE

    mean = sum(lengths) / N
    return {
        "mean": round(mean), "clipped": clipped,
        "contiguous": waste(lengths, lambda n, t: MAX_LEN),
        "paged": waste(lengths, lambda n, t: BLOCK * math.ceil(t / BLOCK)),
        "toy": waste(lengths, toy),
        "tail": round(sum(BLOCK * math.ceil(n / BLOCK) - n for n in lengths) / N, 1),
        "idle_gb": round((MAX_LEN - mean) * KV_BYTES_PER_TOKEN / 1e9, 2),
        "admits_full": "blocks_used += r.blocks_needed()"
                       in inspect.getsource(ref.simulate_continuous),
        "block": ref.KV_BLOCK_SIZE,
    }


def verify(result):
    c, p, t = result["contiguous"], result["paged"], result["toy"]
    return [
        practice.Check(
            "ANSWER: (a) 81.6% wasted, (b) 0.50%",
            all([result["mean"] == 1511, result["clipped"] == 2, c[0] == 81.56, p[0] == 0.5,
                 result["tail"] == 7.6, result["idle_gb"] == 2.19, result["block"] == BLOCK]),
            f"mean {result['mean']} tokens ({result['clipped']} draws clipped at 1): contiguous "
            f"8192 wastes {c[0]}% ({result['idle_gb']} GB of BF16 KV per request), 16-token "
            f"blocks {p[0]}% ({result['tail']} tokens of tail per request)",
        ),
        practice.Check(
            "FINDING: while requests run, contiguous waste is 89.4% and paged is 0.86%",
            c[1] == 89.37 and p[1] == 0.86,
            f"step-averaged over each decode: contiguous {c[1]}%, paged {p[1]}%",
        ),
        practice.Check(
            "FINDING: the toy reserves each request's final length at admission, so it "
            "wastes 50.2%",
            result["admits_full"] and t == (0.5, 50.19),
            f"blocks_needed() is charged on admission: {t[0]}% at completion but {t[1]}% "
            "averaged over the decode, against paged growth's 0.86%",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
