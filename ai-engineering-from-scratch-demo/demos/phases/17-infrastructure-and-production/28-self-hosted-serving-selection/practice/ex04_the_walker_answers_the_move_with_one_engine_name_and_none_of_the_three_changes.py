"""Exercise 4 — the walker answers the move with one engine name and none of the three changes.

    Ollama dev to vLLM prod: what changes in quantization, configuration, and
    observability?

Reading of the exercise: the move is the lesson's own "pipeline pattern".
Nothing needs a GPU to answer it: the three changes are worked through on one
concrete model, Llama-3.1-8B (8.03B parameters, 32 layers, 8 KV heads of dim
128), dev on a laptop and prod on one 80 GB H100 at vLLM's default
`--gpu-memory-utilization 0.9`. The walker and the lesson text are then read
for what they say about the same move. Real command, for reference:
`vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192 --quantization fp8`.

**ANSWER: the weight file, the memory model and the metrics source all
change.** *Quantization*: Ollama serves a GGUF K-quant, Q4_K_M at about 4.85
bits/weight = 4.87 GB. vLLM serves HF safetensors -- FP16 at 16.06 GB or FP8
at 8.03 GB -- and its docs call GGUF "highly experimental and
under-optimized", so prod is a different file with different numerics, and
dev evals have to be rerun. *Configuration*: dev sizes one context window, and
one 8K context is 1.07 GB of FP16 KV (128 KiB/token). vLLM sizes a shared KV
pool, 72 GB minus weights = 427K tokens at FP16 weights, 52 concurrent 8K
sequences; FP8 weights make it 59. The knobs that matter become
`--max-model-len`, `--max-num-seqs` and `--gpu-memory-utilization`.
*Observability*: Ollama returns per-response timing fields and, as of an open
PR dated 2026-09-17, no server `/metrics`. vLLM serves Prometheus `/metrics`
and OTLP traces, which lesson 18's production-stack scrapes.

**FINDING: the walker's reasons say nothing about any of the three.** Across
the 100-cell grid, 0 outputs mention quantization, GGUF, safetensors,
metrics or tracing. Ollama -> vLLM is reachable by changing scale only on
Hopper. On CPU and Apple Silicon no scale reaches vLLM, and on AMD and
Blackwell no scale reaches Ollama. So a laptop-to-server move is also a
hardware change, and the walker has no single input for it.

**FINDING: the staging tier cannot "mirror production quantization".** The
lesson stages on llama.cpp, which runs GGUF, and serves prod from safetensors.
Its own text says both, so the file tested in staging is never the file that
ships.

Structure: `memory()` is the arithmetic; `solve()` greps the walker's 100
outputs and the lesson text.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "28-self-hosted-serving-selection"
PARAMS, LAYERS, KV_HEADS, HEAD_DIM = 8.03e9, 32, 8, 128
BITS = {"Q4_K_M": 4.85, "FP8": 8, "FP16": 16}
HBM_GB, UTIL, CONTEXT = 80, 0.9, 8192
TERMS = ("quant", "gguf", "safetensors", "metric", "prometheus", "trac", "observab")
HARDWARE = ("CPU", "Apple Silicon", "AMD", "NVIDIA Hopper", "NVIDIA Blackwell")
SCALES = ("single_user", "small_team", "production", "enterprise")
WORKLOADS = ("general chat", "agentic multi-turn", "RAG with heavy prefix reuse",
             "code generation", "long-context 128K")


def memory():
    kv_token = 2 * LAYERS * KV_HEADS * HEAD_DIM * 2  # K and V, fp16 bytes
    weights = {q: PARAMS * b / 8 / 1e9 for q, b in BITS.items()}
    pool = {q: (HBM_GB * UTIL - weights[q]) * 1e9 / kv_token for q in ("FP16", "FP8")}
    return {"kv_token": kv_token, "weights": weights, "ctx_gb": kv_token * CONTEXT / 1e9,
            "pool_tokens": pool, "seqs": {q: int(t // CONTEXT) for q, t in pool.items()}}


def reachable(ref):
    """Per hardware class, the set of engines some scale can reach."""
    return {hw: {ref.pick_engine(hw, sc, "general chat")["engine"] for sc in SCALES}
            for hw in HARDWARE}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    outputs = [ref.pick_engine(*c) for c in itertools.product(HARDWARE, SCALES, WORKLOADS)]
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "memory": memory(), "reach": reachable(ref),
        "mentions": sum(any(t in r.lower() for t in TERMS for r in o["reasons"]) for o in outputs),
        "outputs": len(outputs),
        "doc": {s: s in doc for s in ("staging mirrors production quantization",
                                      "GGUF for the llama.cpp family",
                                      "HF safetensors for the GPU engines",
                                      "Dev (Ollama) → staging (llama.cpp) → prod (vLLM)")},
    }


def verify(result):
    mem, reach = result["memory"], result["reach"]
    both = [hw for hw, engines in reach.items() if {"Ollama", "vLLM"} <= engines]
    return [
        practice.Check(
            "ANSWER: the weight file, the memory model and the metrics source all change",
            all([round(mem["weights"]["Q4_K_M"], 2) == 4.87,
                 round(mem["weights"]["FP16"], 2) == 16.06, mem["kv_token"] == 131072,
                 round(mem["ctx_gb"], 2) == 1.07, round(mem["pool_tokens"]["FP16"], -3) == 427000,
                 mem["seqs"] == {"FP16": 52, "FP8": 59}]),
            f"weights {({q: round(g, 2) for q, g in mem['weights'].items()})} GB; one 8K "
            f"context {mem['ctx_gb']:.2f} GB; the H100 pool holds "
            f"{mem['pool_tokens']['FP16']:.0f} tokens = {mem['seqs']} concurrent 8K sequences",
        ),
        practice.Check(
            "FINDING: the walker's reasons say nothing about any of the three",
            result["mentions"] == 0 and result["outputs"] == 100 and both == ["NVIDIA Hopper"]
            and "vLLM" not in reach["CPU"] | reach["Apple Silicon"]
            and "Ollama" not in reach["AMD"] | reach["NVIDIA Blackwell"],
            f"{result['mentions']} of {result['outputs']} outputs mention {TERMS}; engines "
            f"reachable by scale {reach}; Ollama and vLLM share only {both}",
        ),
        practice.Check(
            "FINDING: the staging tier cannot mirror production quantization",
            all(result["doc"].values()),
            f"the lesson says all of {list(result['doc'])}: staging runs GGUF, prod runs "
            "safetensors, so staging never tests the file that ships",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
