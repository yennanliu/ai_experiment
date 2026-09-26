"""Exercise 2 — one FP8 H100 reaches 15,000 tok/s only if every sequence holds 160 tokens or less.

    A vendor quotes "15,000 tok/s on Llama 3.3 70B H100". Name three
    questions to ask before trusting it.

Reading of the exercise: each question is worth asking only if its answer
can move the number, so each one is checked against arithmetic: the
memory-bound decode floor for Llama 3.3 70B (80 layers, 8 KV heads, head dim
128, from the model's config.json; ~70.6B parameters) on one H100 SXM (80 GB,
3.35 TB/s, NVIDIA's spec page), and the lesson's own synthetic workload.

**ANSWER: (1) how many GPUs, at what precision? (2) input plus output tokens,
or output only? (3) at what latency -- what goodput under which SLO?**

(1) In BF16 the weights are 141 GB and do not fit one 80 GB H100. In FP8 they
are 70.6 GB, leaving 9.4 GB for KV cache at 160 KiB per token -- 57,373
tokens. Every decode step reads the weights plus the live KV, so with the
cache full a step takes 23.88 ms. 15,000 output tok/s then needs 358
sequences sharing 57,373 tokens, which is **160 tokens of context or less
each**. At the lesson workload's mean context of 2,394 tokens, one GPU tops
out at 963 tok/s. Across an 8-GPU node the same arithmetic gives 60,761 tok/s,
so "H100" almost certainly means a node, which is 1,875 tok/s per GPU.

(2) The lesson workload's mean prompt is 2,218 tokens against a mean output
of 176. Counting input tokens multiplies the same run's throughput by 13.6x.

(3) Throughput has no latency term. The lesson's own 2000 requests score
goodput 100%, 75.95% and 42.15% under its three SLO profiles, and the
throughput is the same in all three.

**FINDING: the lesson's code cannot compute throughput at all.** The lesson
defines `throughput = total_output_tokens / elapsed_time`, but `main.py` has
no throughput function, and `RequestTrace` has no arrival or completion
time, so the elapsed time of its workload is undefined.

Structure: `decode_ceiling()` is the roofline arithmetic; `solve()` reads
the reference workload for questions (2) and (3).
"""

from __future__ import annotations

import dataclasses
import statistics

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "08-inference-metrics-goodput"
CLAIM, PARAMS, GPU_BYTES, BANDWIDTH = 15_000, 70.6e9, 80e9, 3.35e12
KV_PER_TOKEN = 2 * 80 * 8 * 128  # K and V, 80 layers, 8 KV heads, head dim 128, 1 byte in FP8
PREFILL_MS_PER_TOKEN = 0.05  # synth_workload: prefill = prompt_len * 0.05


def decode_ceiling(gpus, context, bytes_per_param=1):
    """Max output tok/s with the KV cache full: B sequences, one token per step each."""
    weights = PARAMS * bytes_per_param
    free = gpus * GPU_BYTES - weights
    if free <= 0:
        return 0.0, 0
    kv_tokens = int(free / KV_PER_TOKEN)
    step_s = (weights / gpus + free / gpus) / BANDWIDTH  # tensor parallel: each GPU reads its share
    return (kv_tokens // context) / step_s, kv_tokens


def context_for_claim(kv_tokens, step_s):
    return kv_tokens / (CLAIM * step_s)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    traces = ref.synth_workload(n=2000)
    prompt = statistics.mean(t.prefill_ms / PREFILL_MS_PER_TOKEN for t in traces)
    output = statistics.mean(t.output_tokens for t in traces)
    context = round(prompt + output)
    _, kv_tokens = decode_ceiling(1, 1)
    step_s = GPU_BYTES / BANDWIDTH
    slos = [(800, 25, 3000), (500, 15, 2000), (300, 10, 1500)]
    return {
        "bf16": decode_ceiling(1, context, 2)[0], "kv_tokens": kv_tokens, "step_ms": step_s * 1e3,
        "sequences": CLAIM * step_s, "max_context": context_for_claim(kv_tokens, step_s),
        "one_gpu": decode_ceiling(1, context)[0], "node": decode_ceiling(8, context)[0],
        "context": context, "prompt": prompt, "output": output,
        "inflation": (prompt + output) / output,
        "goodputs": [round(ref.goodput(traces, *slo), 4) for slo in slos],
        "has_throughput": [n for n in dir(ref) if "throughput" in n.lower()],
        "trace_fields": [f.name for f in dataclasses.fields(ref.RequestTrace)],
    }


def verify(r):
    return [
        practice.Check(
            "ANSWER: how many GPUs at what precision, input or output tokens, at what goodput",
            all([r["bf16"] == 0, int(r["max_context"]) == 160, r["one_gpu"] < CLAIM < r["node"],
                 round(r["inflation"], 1) == 13.6, r["goodputs"] == [1.0, 0.7595, 0.4215]]),
            f"BF16 does not fit; FP8 leaves {r['kv_tokens']} KV tokens and a "
            f"{r['step_ms']:.2f} ms step, so {CLAIM} tok/s needs {r['sequences']:.0f} "
            f"sequences at {r['max_context']:.0f} tokens or less each; at {r['context']} tokens "
            f"one GPU gives {r['one_gpu']:.0f} tok/s, a node {r['node']:.0f}; counting input "
            f"tokens multiplies by {r['inflation']:.1f}x; the same traces score goodput "
            f"{r['goodputs']}",
        ),
        practice.Check(
            "FINDING: the lesson's code cannot compute throughput at all",
            r["has_throughput"] == [] and not any("time" in f for f in r["trace_fields"]),
            f"no throughput function in main.py; RequestTrace has only {r['trace_fields']}, "
            "no arrival or completion time",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
