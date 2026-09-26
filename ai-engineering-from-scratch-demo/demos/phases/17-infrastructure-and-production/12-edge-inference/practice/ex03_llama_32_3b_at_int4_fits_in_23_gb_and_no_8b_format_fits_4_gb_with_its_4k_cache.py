"""Exercise 3 — Llama 3.2 3B at INT4 fits in 2.3 GB, and no 8B format fits 4 GB with its 4K cache.

    Your iOS app needs 4K-context streaming. Which model/format combination
    lets you stay under 4 GB active memory on an iPhone 16?

Reading of the exercise: active memory is weights plus a 4096-token KV cache,
counted from each model's published architecture (layers x KV heads x head
dim). Four models the lesson names or implies -- Llama 3.2 1B and 3B,
Phi-3.5-mini, Llama 3.1 8B -- cross four weight formats (FP16, INT8, INT4
with block-32 FP16 scales at 4.5 bits, the code's 3.6 GB Q3 for 8B) and two
KV formats (FP16, INT8). Runtime buffers are not modelled, so what is left of
4 GB is reported as headroom.

**ANSWER: Llama 3.2 3B, Core ML INT4 weights, FP16 KV -- 2.28 GB, 1.72 GB
headroom.** Weights are 1.81 GB and the 4K cache 0.47 GB (112 KiB a token).
WebLLM v0.2.84 lists the same model's q4f16 build at 2263.69 MB for a 4K
context, 0.5% below this estimate. 13 of the 26 combinations fit; this is
the largest model that leaves over 1 GB for the app with an FP16 cache.
Phi-3.5-mini at INT4 fits at 3.76 GB with 0.24 GB left, or 2.95 GB with an
INT8 cache; every Llama 3.1 8B combination but one misses.

**FINDING: no 8B format fits except Q3 weights with an INT8 cache, at 0.13 GB
headroom.** 8B INT4 weights alone are 4.52 GB. The code's 3.6 GB Q3 plus an
FP16 4K cache is 4.14 GB; with an INT8 cache it is 3.87.

**FINDING: Phi-3.5-mini's cache is 3.4x Llama 3.2 3B's at a similar size.**
It has 32 KV heads to Llama's 8 (no grouped-query attention): 384 KiB a
token, 1.61 GB at 4K, 43% of its footprint. WebLLM's 4K and 1K builds differ
by exactly 1152.00 MB -- 3072 tokens x 384 KiB.

**FINDING: the lesson's 32K trap undercounts the cache by half, and the code's
ceiling ignores it.** Llama 3.1 8B at 32K is 4.29 GB of FP16 KV, not "2 GB"
(2 GB is the INT8 cache). And `ceiling()` divides bandwidth by weights only:
at a full 4K context the 3B's per-token read grows 26%, cutting the A18's
ceiling from 33.2 to 26.4 tok/s.

Structure: `footprint()` is the arithmetic; the ceilings are the reference
`ceiling()` on the reference's own A18 target.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "12-edge-inference"
CTX, BUDGET = 4096, 4e9
ARCH = {  # name: (params, layers, kv heads, head dim) from each model's config.json
    "Llama 3.2 1B": (1.24e9, 16, 8, 64),
    "Llama 3.2 3B": (3.21e9, 28, 8, 128),
    "Phi-3.5-mini": (3.82e9, 32, 32, 96),
    "Llama 3.1 8B": (8.03e9, 32, 8, 128),
}
WEIGHT_BPW = {"FP16": 16, "INT8": 8, "INT4": 4.5}
KV_BYTES = {"FP16 KV": 2, "INT8 KV": 1}
WEBLLM_MB = {"3B q4f16 4K": 2263.69, "Phi 4K": 3672.07, "Phi 1K": 2520.07}  # src/config.ts


def kv_per_token(name, kv_bytes=2):
    _, layers, heads, dim = ARCH[name]
    return 2 * layers * heads * dim * kv_bytes


def footprint(name, bpw, kv_bytes, ctx=CTX, weights_gb=None):
    weights = weights_gb * 1e9 if weights_gb else ARCH[name][0] * bpw / 8
    return weights, kv_per_token(name, kv_bytes) * ctx


def grid():
    out = {}
    for name in ARCH:
        for fmt, bpw in WEIGHT_BPW.items():
            for kv, b in KV_BYTES.items():
                out[(name, fmt, kv)] = sum(footprint(name, bpw, b))
    for kv, b in KV_BYTES.items():
        out[("Llama 3.1 8B", "Q3 (code, 3.6 GB)", kv)] = sum(footprint("Llama 3.1 8B", 0, b,
                                                                        weights_gb=3.6))
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    a18 = next(t for t in ref.TARGETS if t.name.startswith("Apple A18"))
    w3, kv3 = footprint("Llama 3.2 3B", 4.5, 2)
    g = grid()
    fits = {k: BUDGET - v for k, v in g.items() if v < BUDGET}
    return {
        "grid": g, "w3": w3, "kv3": kv3, "fits": fits,
        "eight": [k for k in fits if k[0] == "Llama 3.1 8B"],
        "roomy": {k[0] for k, room in fits.items() if room > 1e9 and k[2] == "FP16 KV"},
        "ceiling_weights": ref.ceiling(a18, w3 / 1e9),
        "ceiling_with_kv": ref.ceiling(a18, (w3 + kv3) / 1e9),
        "kv_ratio": kv_per_token("Phi-3.5-mini") / kv_per_token("Llama 3.2 3B"),
        "phi_kv": footprint("Phi-3.5-mini", 4.5, 2),
        "phi_webllm_diff": WEBLLM_MB["Phi 4K"] - WEBLLM_MB["Phi 1K"],
        "kv_32k_8b": {kv: kv_per_token("Llama 3.1 8B", b) * 32768 for kv, b in KV_BYTES.items()},
    }


def verify(result):
    g = result["grid"]
    fits, eight = result["fits"], result["eight"]
    best = ("Llama 3.2 3B", "INT4", "FP16 KV")
    q3 = ("Llama 3.1 8B", "Q3 (code, 3.6 GB)")
    pw, pkv = result["phi_kv"]
    kv32 = result["kv_32k_8b"]
    drop = result["ceiling_with_kv"] / result["ceiling_weights"]
    return [
        practice.Check(
            "ANSWER: Llama 3.2 3B, Core ML INT4 weights, FP16 KV -- 2.28 GB, 1.72 GB headroom",
            all([round(g[best] / 1e9, 2) == 2.28, round(fits[best] / 1e9, 2) == 1.72,
                 abs(g[best] / 1e6 / WEBLLM_MB["3B q4f16 4K"] - 1) < 0.01,
                 ("Phi-3.5-mini", "INT4", "FP16 KV") in fits,
                 result["roomy"] == {"Llama 3.2 1B", best[0]}]),
            f"weights {result['w3'] / 1e9:.2f} GB + 4K cache {result['kv3'] / 1e9:.2f} GB; "
            f"WebLLM lists {WEBLLM_MB['3B q4f16 4K']} MB; {len(fits)} of {len(g)} "
            "combinations fit",
        ),
        practice.Check(
            "FINDING: no 8B format fits except Q3 weights with an INT8 cache, at 0.13 GB headroom",
            eight == [(*q3, "INT8 KV")] and round(fits[eight[0]] / 1e9, 2) == 0.13,
            f"8B INT4 weights {ARCH['Llama 3.1 8B'][0] * 4.5 / 8e9:.2f} GB; Q3 + FP16 KV "
            f"{g[(*q3, 'FP16 KV')] / 1e9:.2f} GB, + INT8 KV {g[(*q3, 'INT8 KV')] / 1e9:.2f} GB",
        ),
        practice.Check(
            "FINDING: Phi-3.5-mini's cache is 3.4x Llama 3.2 3B's at a similar size",
            round(result["kv_ratio"], 1) == 3.4
            and abs(result["phi_webllm_diff"] - 3072 * kv_per_token("Phi-3.5-mini") / 2**20) < 0.01,
            f"{kv_per_token('Phi-3.5-mini') // 1024} KiB/token, {pkv / 1e9:.2f} GB at 4K = "
            f"{pkv / (pw + pkv):.0%} of its footprint; WebLLM 4K - 1K = "
            f"{result['phi_webllm_diff']:.2f} MB",
        ),
        practice.Check(
            "FINDING: the lesson's 32K trap undercounts the cache by half, and the code's "
            "ceiling ignores it",
            round(kv32["FP16 KV"] / 1e9, 2) == 4.29 and round(kv32["INT8 KV"] / 1e9, 2) == 2.15
            and round(1 / drop - 1, 2) == 0.26,
            f"8B 32K cache FP16 {kv32['FP16 KV'] / 1e9:.2f} GB, INT8 {kv32['INT8 KV'] / 1e9:.2f}"
            f" GB; A18 ceiling {result['ceiling_weights']:.1f} -> "
            f"{result['ceiling_with_kv']:.1f} tok/s at a full 4K context",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
