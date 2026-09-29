"""Exercise 5 -- LoRA on a bf16 base cannot reach 10x lower memory; at 10x (4-bit base) it loses 0.198 agreement.

    Add LoRA SFT as an alternative to full fine-tune. Measure the quality gap at 10x lower memory.

Reading of the exercise: the lesson's SFT stage trains nothing, so both
methods are run on a toy layer. It is a 128x128 softmax map from a
"pretrained" W0 to a fine-tuned teacher W0 + delta, where delta is a rank-4
update plus a small full-rank part. Each method distils the teacher's soft
labels on 2,000 inputs with 300 Adam steps at lr 0.02. Quality is top-1
agreement with the teacher on 2,000 held-out inputs, plus the KL divergence.
Memory is weights plus gradients plus optimizer state, using the usual
mixed-precision Adam accounting. A trained parameter costs 16 bytes. A frozen
parameter costs 2 bytes in bf16, or 0.53 in 4-bit (block-64 absmax plus an
fp16 scale per block). Activations are left out. "10x lower memory" is the
LoRA configuration whose accounted memory is at most a tenth of full
fine-tuning: rank 4 on the 4-bit base. Quality is asserted to within 0.003.

**ANSWER: at 10x lower memory the quality gap is 0.198 agreement.** Full
fine-tuning reaches 0.998 agreement (KL 0.000). LoRA with rank 4 on a 4-bit
base (the QLoRA route) reaches 0.800 (KL 0.098), with memory 10.45x lower.
Most of the gap is the full-rank part of the update, which no low rank
captures: rank 8 on the same base scores 0.797. The 4-bit base costs a
further 0.036 (bf16 base, rank 4: 0.836).

**FINDING: plain LoRA cannot reach 10x.** With a bf16 base, memory is at
least 2 of full fine-tuning's 16 bytes per parameter, so the ceiling is 8x.
The toy's rank 4 gets 5.3x. For the Llama 3 8B architecture, LoRA at rank 16
on all linear layers trains 41.94M parameters, and memory falls from
128.5 GB to 16.7 GB, which is 7.68x. Only a quantised base crosses 10x: a
4-bit base comes to 4.9 GB, 26.0x lower.

**FINDING: the lesson's pipeline has no place for LoRA.** Setting
`cfg["lora"] = True` leaves every artifact hash unchanged. The SFT artifact
records no method, rank or memory, only a fixed 8 GPUs and 6.2 hours.

Structure: `world()` builds teacher and data; `train()` runs full or LoRA
Adam on a given base; `memory()` is the byte accounting; `solve()` also probes
the reference pipeline.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "07-end-to-end-fine-tuning-pipeline"
D, N, STEPS, LR, BLOCK = 128, 2000, 300, 0.02, 64
TOL = 0.003  # BLAS rounding can flip a near-tied argmax; 0.003 is 6 of 2,000 inputs
TRAINED, BF16, INT4 = 16, 2, 0.5 + 2 / BLOCK  # bytes per parameter
CFG = {"base_model": "llama-3.3-8b", "raw_examples": 300_000, "seed": 7, "dpo_beta": 0.08}
LAYERS, H, MLP, KV, PARAMS_8B = 32, 4096, 14336, 1024, 8.03e9  # Llama 3 8B: layers, hidden, MLP, K/V


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def world():
    r = np.random.default_rng(7)
    w0, delta = 3 * r.normal(size=(D, D)) / np.sqrt(D), 3 * (
        r.normal(size=(D, 4)) @ r.normal(size=(4, D)) * 0.5 + 0.15 * r.normal(size=(D, D))) / np.sqrt(D)
    x, xt = r.normal(size=(N, D)), r.normal(size=(N, D))
    return w0, softmax(x @ (w0 + delta).T), softmax(xt @ (w0 + delta).T), x, xt


def quant4(w):
    blocks = w.reshape(-1, BLOCK)
    scale = np.abs(blocks).max(1, keepdims=True) / 7
    return (np.clip(np.round(blocks / scale), -7, 7) * scale).reshape(w.shape)


def train(base, x, p, rank=None):
    params = {"W": base.copy()} if rank is None else {
        "A": np.random.default_rng(1).normal(size=(rank, D)) / np.sqrt(D), "B": np.zeros((D, rank))}
    state = {k: (0.0, 0.0) for k in params}
    for t in range(1, STEPS + 1):
        w = params["W"] if rank is None else base + params["B"] @ params["A"]
        g = (softmax(x @ w.T) - p).T @ x / N
        grads = {"W": g} if rank is None else {"B": g @ params["A"].T, "A": params["B"].T @ g}
        for k, gk in grads.items():
            m, v = state[k]
            m, v = 0.9 * m + 0.1 * gk, 0.999 * v + 0.001 * gk**2
            state[k] = (m, v)
            params[k] -= LR * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return params["W"] if rank is None else base + params["B"] @ params["A"]


def quality(w, xt, pt):
    q = softmax(xt @ w.T)
    kl = (pt * (np.log(pt + 1e-12) - np.log(q + 1e-12))).sum(1).mean()
    return round(float((q.argmax(1) == pt.argmax(1)).mean()), 4), round(float(kl), 4)


def memory(n, adapters, frozen):
    return round(n * TRAINED / (n * frozen + adapters * TRAINED), 2)


def llama_8b(rank=16):
    per_layer = 2 * (H + H) + 2 * (H + KV) + 3 * (H + MLP)  # q,o + k,v + gate,up,down
    adapters = rank * per_layer * LAYERS
    gb = [round(b / 1e9, 1) for b in (PARAMS_8B * TRAINED, PARAMS_8B * BF16 + adapters * TRAINED,
                                      PARAMS_8B * INT4 + adapters * TRAINED)]
    return {"adapters_m": round(adapters / 1e6, 2), "gb": gb,
            "x": [memory(PARAMS_8B, adapters, BF16), memory(PARAMS_8B, adapters, INT4)]}


def pipeline_probe(ref):
    runs = []
    for cfg in (CFG, {**CFG, "lora": True}):
        with parity.quiet():
            m = ref.run_pipeline(cfg)
        runs.append([a.content_hash() for a in m.artifacts.values()])
    return runs[0] == runs[1], m.get("sft_checkpoint").payload


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    w0, p, pt, x, xt = world()
    q4 = quant4(w0)
    rows = {"full": quality(train(w0, x, p), xt, pt), "lora4_bf16": quality(train(w0, x, p, 4), xt, pt),
            "qlora4": quality(train(q4, x, p, 4), xt, pt), "qlora8": quality(train(q4, x, p, 8), xt, pt)}
    same, sft = pipeline_probe(ref)
    return {"rows": rows, "mem": {"lora4_bf16": memory(D * D, 8 * D, BF16), "qlora4": memory(D * D, 8 * D, INT4),
                                  "ceiling_bf16": memory(D * D, 0, BF16)},
            "llama": llama_8b(), "lora_same_hashes": same, "sft_fields": sorted(sft)}


def verify(r):
    rows, mem, ll = r["rows"], r["mem"], r["llama"]
    gap = round(rows["full"][0] - rows["qlora4"][0], 3)
    return [
        practice.Check(
            "ANSWER: at 10x lower memory (rank-4 LoRA on a 4-bit base) the gap is 0.198 agreement",
            mem["qlora4"] == 10.45 and np.allclose(
                [gap, *rows["full"], *rows["qlora4"], *rows["qlora8"], *rows["lora4_bf16"]],
                [0.198, 0.9975, 0.0, 0.7995, 0.0976, 0.797, 0.0967, 0.8355, 0.0632], atol=TOL),
            f"{rows}; qlora4 memory {mem['qlora4']}x lower",
        ),
        practice.Check(
            "FINDING: plain LoRA cannot reach 10x",
            (mem["ceiling_bf16"], mem["lora4_bf16"], ll) == (8.0, 5.33, {
                "adapters_m": 41.94, "gb": [128.5, 16.7, 4.9], "x": [7.68, 26.02]}),
            f"bf16-base ceiling {mem['ceiling_bf16']}x, toy rank 4 {mem['lora4_bf16']}x; Llama 3 8B r=16: "
            f"{ll['adapters_m']}M adapters, {ll['gb']} GB full/LoRA/4-bit LoRA -> {ll['x']}x",
        ),
        practice.Check(
            "FINDING: the lesson's pipeline has no place for LoRA",
            r["lora_same_hashes"] and not {"method", "lora", "rank", "memory_gb"} & set(r["sft_fields"]),
            f"cfg lora=True leaves all hashes unchanged: {r['lora_same_hashes']}; sft fields {r['sft_fields']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
