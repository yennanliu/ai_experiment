"""Exercise 2 — the budget buys the toy no throughput, and one H100 has KV room for 13k tokens.

    Modify the toy scheduler to add `--max-num-batched-tokens`. What is the
    right value for an H100 running Llama 3.3 70B FP8? (Hint: it is a function
    of KV block size and number of free blocks, not raw HBM.)

Reading of the exercise: `step()` is the reference loop with one change. Each
iteration has a token budget: a decode costs 1 token, and a prefill takes
whatever budget is left, the way vLLM V1 chunks prefills. With no budget it
reproduces `simulate_continuous` exactly. The budget is swept on the
reference workload. The "right value" is the hint's bound, free blocks x
block size, computed from Llama 3.3 70B's real config on one H100 SXM5. That
is set against the default vLLM v0.18.0 actually picks.

**ANSWER: at most free_blocks x 16 = 13,104 tokens, and vLLM's default of
8192 fits under it.** Each token of KV takes 80 layers x 2 x 8 KV heads x 128
x 2 bytes = 327,680 bytes in BF16. An H100 has 81,559 MiB (nvidia-smi's
figure, assumed here). At `--gpu-memory-utilization 0.9` that is 76.97 GB,
and the FP8 checkpoint takes 72.67 GB of it. That leaves 4.30 GB, or 819
blocks of 16 tokens, before activations, so at most 13,104 tokens can be
scheduled in one step. With an FP8 KV cache it is 1,639 blocks. vLLM
v0.18.0's `get_batch_defaults` does not look at blocks at all. It picks 8192
for the API server, 16,384 for `LLM`, on any GPU of at least 70 GiB that is
not an A100.

**FINDING: in the toy the budget moves only latency, and 512-token chunks
bound no step.** Sweeping 128 to 28,800 tokens, throughput stays between 878
and 886 tok/s while P99 ITL rises from 10.8 ms to 262.2 ms. The toy's cost is
linear in tokens, so the fixed per-step cost that a bigger batch amortizes is
missing, and only the budget's latency side shows. Nor does the reference's
"chunked prefill" cap a step. Every prefilling request gets its own 512-token
chunk in the same iteration, so the largest step is 2694 tokens with chunking
and 18,563 without it. Only a budget caps a step.

**FINDING: the lesson's 1.25 GB is FP8 KV, and 128 concurrent leave ~100
tokens each.** 8192 tokens at 327,680 bytes is 2.68 GB in BF16; 1.25 GiB is
the FP8 KV cache. The lesson's headline setup is Llama 3.3 70B FP8 on one H100
at 128 concurrent. Its 13,104 tokens of KV room come to 102 tokens per
sequence, or 204 with FP8 KV.

Structure: `simulate(ref, reqs, max_num_batched_tokens, chunk)` is the modified
scheduler. `kv_budget()` is the hint's arithmetic.
"""

import collections

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "04-vllm-serving-internals"
KV_ELEMS_PER_TOKEN = 80 * 2 * 8 * 128  # layers x (K, V) x KV heads x head_dim, HF config.json
FP8_CHECKPOINT_BYTES = 72_669_954_704  # RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic
H100_BYTES, UTIL, CONCURRENT = 81_559 * 2**20, 0.9, 128
VLLM_H100_DEFAULTS = {"api_server": 8192, "llm_class": 16384}  # arg_utils.py @ v0.18.0
BUDGETS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 28800)


def schedule(ref, running, now, left, chunk):  # spend the budget in running order
    prefill, decoded = 0, []
    for r in running:
        if left <= 0:
            break
        take = min(r.prompt_len - r.prefilled, chunk or r.prompt_len, left) if r.in_prefill else 0
        r.prefilled, prefill, left = r.prefilled + take, prefill + take, left - (take or 1)
        if take and not r.in_prefill:
            r.ttft = now + prefill * ref.PREFILL_LATENCY_PER_TOKEN
        decoded += [r] * (not take)
    return prefill, decoded


def step(ref, running, now, budget, chunk, peak):  # -> (new time, largest step so far)
    prefill, decoded = schedule(ref, running, now, budget or float("inf"), chunk)
    now += (prefill * ref.PREFILL_LATENCY_PER_TOKEN
            + len(decoded) * ref.FORWARD_LATENCY_PER_TOKEN + ref.BATCH_OVERHEAD)
    for r in decoded:
        r.itl_samples.append(now - (r.last_token_at or r.ttft or now))
        r.generated, r.last_token_at = r.generated + 1, now
    return now, max(peak, prefill + len(decoded))


def admit(ref, waiting, running, used, now):  # the reference's rule: whole reservation fits
    while (waiting and waiting[0].arrived_at <= now
           and used + waiting[0].blocks_needed() <= ref.KV_BLOCKS_AVAILABLE):
        used += waiting[0].blocks_needed()
        running.append(waiting.popleft())
    return used


def simulate(ref, reqs, max_num_batched_tokens=None, chunk=None):  # -> (end, largest step)
    waiting, running, used, now, peak = collections.deque(reqs), [], 0, 0.0, 0
    while waiting or running:
        used = admit(ref, waiting, running, used, now)
        if not running:
            now = waiting[0].arrived_at
            continue
        now, peak = step(ref, running, now, max_num_batched_tokens, chunk, peak)
        used -= sum(r.blocks_needed() for r in running if r.done)
        running[:] = [r for r in running if not r.done]
    return now, peak


def measure(ref, **kw):
    reqs = ref.make_workload()  # fresh Requests on every call, already in arrival order
    end, peak = simulate(ref, reqs, **kw)
    itls = sorted(d for r in reqs for d in r.itl_samples)
    return {"tps": round(sum(r.generated for r in reqs) / end), "peak": peak,
            "p99": round(itls[int(0.99 * len(itls)) - 1] * 1e3, 1)}


def parity_with_reference(ref, chunked):
    mine, theirs = ref.make_workload(), ref.make_workload()
    simulate(ref, mine, chunk=ref.CHUNK_SIZE if chunked else None)
    ref.simulate_continuous(theirs, chunked=chunked)
    return all(a.itl_samples == b.itl_samples and a.ttft == b.ttft for a, b in zip(mine, theirs))


def kv_budget(kv_bytes, block):
    per_token, free = KV_ELEMS_PER_TOKEN * kv_bytes, H100_BYTES * UTIL - FP8_CHECKPOINT_BYTES
    tokens = int(free // (per_token * block)) * block
    return {"per_token": per_token, "free_gb": round(free / 1e9, 2), "tokens": tokens,
            "blocks": tokens // block, "seq_8192_gb": round(per_token * 8192 / 1e9, 2)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"parity": [parity_with_reference(ref, c) for c in (False, True)],
            "sweep": {b: measure(ref, max_num_batched_tokens=b) for b in BUDGETS},
            "peak": {c: measure(ref, chunk=c)["peak"] for c in (None, ref.CHUNK_SIZE)},
            "bf16": kv_budget(2, ref.KV_BLOCK_SIZE), "fp8": kv_budget(1, ref.KV_BLOCK_SIZE)}


def verify(result):
    sw, bf, fp = result["sweep"], result["bf16"], result["fp8"]
    tps, p99 = [sw[b]["tps"] for b in BUDGETS], [sw[b]["p99"] for b in BUDGETS]
    per_seq = [bf["tokens"] // CONCURRENT, fp["tokens"] // CONCURRENT]
    return [
        practice.Check(
            "ANSWER: at most free_blocks x 16 = 13,104 tokens, and vLLM's default of 8192 fits",
            all([result["parity"] == [True, True], bf["per_token"] == 327_680,
                 bf["free_gb"] == 4.30, bf["blocks"] == 819, bf["tokens"] == 13_104,
                 fp["blocks"] == 1639, VLLM_H100_DEFAULTS["api_server"] < bf["tokens"]]),
            f"loop == simulate_continuous; {bf['free_gb']} GB = {bf['blocks']} BF16 blocks "
            f"({bf['tokens']} tokens), {fp['blocks']} FP8; defaults {VLLM_H100_DEFAULTS}",
        ),
        practice.Check(
            "FINDING: in the toy the budget moves only latency, and 512-token chunks bound no step",
            all([min(tps) == 878, max(tps) == 886, p99[0] == 10.8, max(p99) == 262.2,
                 result["peak"] == {None: 18_563, 512: 2694}]),
            f"budgets {BUDGETS}: {tps} tok/s, P99 ITL {p99} ms; the reference's largest step "
            f"is {result['peak'][512]} tokens chunked, {result['peak'][None]} unchunked",
        ),
        practice.Check(
            "FINDING: the lesson's 1.25 GB is FP8 KV, and 128 concurrent leave ~100 tokens each",
            bf["seq_8192_gb"] == 2.68 and fp["seq_8192_gb"] == 1.34 and per_seq == [102, 204],
            f"8192 tokens is {bf['seq_8192_gb']} GB BF16, {fp['seq_8192_gb']} GB FP8; "
            f"{CONCURRENT} sequences get {per_seq} tokens each (BF16, FP8 KV)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
