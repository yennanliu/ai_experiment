"""Exercise 4 — TCP matters only for TTFT, where it adds 50%, and streaming KV per layer hides both behind prefill.

    Compute KV transfer cost: 4K prefill on 70B FP8 = ~500 MB KV. At RDMA 100
    GB/s, transfer = 5 ms. At TCP 10 GB/s = 50 ms. Which matters for your SLA?

Reading of the exercise: the transfer is computed with the reference
`ms_disaggregated` (4096 tokens, 0 output isolates it) and then placed in the
three SLA metrics it could touch -- TTFT, TPOT and end-to-end latency -- using
the lesson's own prefill rate and the table's 4096/500 row. The transfer is
then modelled two ways: sent after prefill ends (the code's model), and
streamed layer by layer as the 80 layers are produced.

**ANSWER: TCP matters, and only for TTFT.** The code's KV is 4096 x 125,000 B
= 512 MB: 5.12 ms over RDMA, 51.2 ms over TCP. The transfer happens once,
between prefill and the first decoded token, so TPOT is unchanged. On TTFT,
prefill is 102.4 ms: RDMA adds 5% (107.52 ms), TCP 50% (153.6 ms). Any TTFT
SLO between those fails on TCP only. End to end at 500 output tokens TCP adds
1.6% (2885.3 -> 2931.4 ms), which no latency SLA will notice.

**FINDING: streaming KV per layer hides both transports behind prefill.**
If each layer's KV is sent as soon as that layer is prefilled, only the last
layer's transfer is exposed: 0.064 ms over RDMA, 0.64 ms over TCP. The hiding
holds for any link faster than KV bytes per prefill ms = 125,000 x 40 = 5 GB/s.
Below that, the exposed tail grows fast: a 10 GbE link (1.25 GB/s) takes
409.6 ms and exposes 308.5 ms of it.

**FINDING: the lesson's own transfer figures disagree 4-16x.** The same page
says NIXL transfer is "typically 20-80 ms for KV cache of a 4K-token prompt on
70B FP8" and 5 ms over RDMA. And at Llama 3.3 70B's real FP8 GQA size
(163,840 B/token) the 4K KV is 671 MB, not ~500: 6.71 ms RDMA, 67.1 ms TCP.

Structure: `exposed()` runs the per-layer pipeline; `transfer()` asks the
reference for the transfer alone.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "17-disaggregated-prefill-decode"
PROMPT, OUTPUT, LAYERS = 4096, 500, 80
LLAMA_KV, TEN_GBE = 80 * 8 * 128 * 2, 1.25


def transfer(ref, rdma=True, prompt=PROMPT):
    return round(ref.ms_disaggregated(prompt, 0, rdma) - prompt / ref.PREFILL_TOK_PER_MS, 3)


def exposed(prefill_ms, transfer_ms, layers=LAYERS):
    """Transfer time left after prefill ends when each layer ships once it is computed."""
    done = 0.0
    for layer in range(1, layers + 1):
        done = max(layer * prefill_ms / layers, done) + transfer_ms / layers
    return round(done - prefill_ms, 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    prefill = PROMPT / ref.PREFILL_TOK_PER_MS
    rdma, tcp = transfer(ref), transfer(ref, False)
    kv_mb = PROMPT * ref.KV_BYTES_PER_TOKEN_70B_FP8 / 1e6
    slow = kv_mb / 1e3 / TEN_GBE * 1000
    llama_mb = PROMPT * LLAMA_KV / 1e6
    return {
        "kv_mb": kv_mb, "rdma": rdma, "tcp": tcp, "prefill": prefill,
        "ttft": (round(prefill + rdma, 2), round(prefill + tcp, 2)),
        "e2e": (round(ref.ms_disaggregated(PROMPT, OUTPUT), 1),
                round(ref.ms_disaggregated(PROMPT, OUTPUT, False), 1)),
        "hidden": (exposed(prefill, rdma), exposed(prefill, tcp)),
        "hide_gb_s": ref.KV_BYTES_PER_TOKEN_70B_FP8 * ref.PREFILL_TOK_PER_MS * 1000 / 1e9,
        "slow": (round(slow, 1), round(exposed(prefill, slow), 1)),
        "doc_range": "20-80 ms for KV cache of a 4K-token prompt" in parity.doc_text(PHASE, LESSON),
        "llama": (round(llama_mb), round(llama_mb / 1e3 / ref.NIXL_RDMA_GB_S * 1e3, 2),
                  round(llama_mb / 1e3 / ref.NIXL_TCP_GB_S * 1e3, 1)),
    }


def verify(result):
    ttft, e2e = result["ttft"], result["e2e"]
    return [
        practice.Check(
            "ANSWER: TCP matters, and only for TTFT",
            all([result["kv_mb"] == 512, (result["rdma"], result["tcp"]) == (5.12, 51.2),
                 ttft == (107.52, 153.6), round(e2e[1] / e2e[0] - 1, 3) == 0.016]),
            f"{result['kv_mb']:.0f} MB: RDMA {result['rdma']} ms, TCP {result['tcp']} ms; TTFT "
            f"{result['prefill']} -> {ttft[0]} / {ttft[1]} ms; TPOT unchanged; end to end "
            f"{e2e[0]} -> {e2e[1]} ms",
        ),
        practice.Check(
            "FINDING: streaming KV per layer hides both transports behind prefill",
            all([result["hidden"] == (0.064, 0.64), result["hide_gb_s"] == 5,
                 result["slow"] == (409.6, 308.5)]),
            f"exposed {result['hidden'][0]} ms RDMA / {result['hidden'][1]} ms TCP; hidden "
            f"above {result['hide_gb_s']:.0f} GB/s; 10 GbE takes {result['slow'][0]} ms and "
            f"exposes {result['slow'][1]} ms",
        ),
        practice.Check(
            "FINDING: the lesson's own transfer figures disagree 4-16x",
            result["doc_range"] and result["llama"] == (671, 6.71, 67.1),
            f"'20-80 ms' against {result['rdma']} ms RDMA; at {LLAMA_KV} B/token the KV is "
            f"{result['llama'][0]} MB: {result['llama'][1]} ms RDMA, {result['llama'][2]} ms TCP",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
