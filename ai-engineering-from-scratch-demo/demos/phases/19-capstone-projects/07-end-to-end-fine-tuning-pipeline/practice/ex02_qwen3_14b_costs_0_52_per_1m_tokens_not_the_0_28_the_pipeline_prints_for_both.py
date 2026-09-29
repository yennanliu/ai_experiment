"""Exercise 2 -- Qwen3 14B costs $0.52 per 1M tokens, not the $0.28 the pipeline prints for both models.

    Swap Llama 3.3 8B for Qwen3 14B. Measure the $/1M tokens at matched quality.

Reading of the exercise: the swap is made where the lesson's config makes it,
`cfg["base_model"]`, and the whole reference pipeline is rerun. Neither model
can be served here, so the price comes from the pipeline's own serving anchor
($0.28 per 1M tokens and 6,400 tokens/s at batch 32 for a 4.6 GB GPTQ-INT4
quant). Decode at batch 32 is bound by memory bandwidth, and every step reads
the weights once, so at a fixed quant $/1M scales with weight bytes. "Matched
quality" is read as the same quant recipe, the same bits per parameter, on
both bases. It is the only quality control this pipeline has: it prints the
same eval deltas for both. Parameter counts are 8.03B for the Llama 3 8B
architecture and 14.8B for Qwen3-14B (huggingface.co/Qwen/Qwen3-14B, read
2026-09-29).

**ANSWER: Qwen3 14B costs $0.52 per 1M tokens against Llama's $0.28, which is
1.84x.** The pipeline's 4.6 GB for 8.03B parameters works out to 4.58 bits per
parameter. At that rate Qwen3 14B needs 8.48 GB, and it serves 3,472 tokens/s
at batch 32 on the implied $6.45/hour of GPU.

**FINDING: the pipeline prices both models at $0.28 and gives them identical
quality.** After the swap, the endpoint hash is unchanged and the quant sizes
stay at 4.6/4.8/5.1 GB. The eval deltas stay at +3.2 MMLU-Pro, +0.41 MT-Bench-v2
and +0.08 RewardBench-2. Serving 14.8B parameters in 4.6 GB would mean 2.49
bits per parameter, which is below INT4. The model card still points at
`config/llama3.3-8b-domainX.yaml`.

**FINDING: the lesson's default base is not an open-weights release.** Meta's
Llama 3.3 was published in one size, 70B (the Llama-3.3-70B-Instruct model
card, read 2026-09-29). The 8B open weights are Llama 3.1.

Structure: `run()` reruns the reference pipeline per base; `price()` rescales
the pipeline's serving anchor by weight bytes.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "07-end-to-end-fine-tuning-pipeline"
CFG = {"base_model": "llama-3.3-8b", "raw_examples": 300_000, "seed": 7, "dpo_beta": 0.08}
PARAMS_B = {"llama-3.3-8b": 8.03, "qwen3-14b": 14.8}


def run(ref, base):
    with parity.quiet():
        m = ref.run_pipeline({**CFG, "base_model": base})
    return {a.name: (a.content_hash(), a.payload) for a in m.artifacts.values()}


def price(serve, quant_gb, params_b, anchor_params_b):
    bits = quant_gb * 8 / anchor_params_b
    gb = params_b * bits / 8
    tps = serve["tokens_per_sec_bs32"] * quant_gb / gb
    gpu_hour = serve["dollars_per_mtokens"] * serve["tokens_per_sec_bs32"] * 3600 / 1e6
    return {"bits": round(bits, 2), "gb": round(gb, 2), "tps": round(tps),
            "usd_per_m": round(gpu_hour / (tps * 3600) * 1e6, 3), "gpu_hour": round(gpu_hour, 2)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    llama, qwen = run(ref, "llama-3.3-8b"), run(ref, "qwen3-14b")
    serve, quant = llama["endpoint"][1], llama["quants"][1]
    evals = [{k: v for k, v in r["eval_report"][1].items() if k != "from"} for r in (llama, qwen)]
    q = price(serve, quant["gptq_int4_gb"], PARAMS_B["qwen3-14b"], PARAMS_B["llama-3.3-8b"])
    return {
        "llama_usd": serve["dollars_per_mtokens"], "qwen": q,
        "ratio": round(q["usd_per_m"] / serve["dollars_per_mtokens"], 2),
        "same_endpoint": llama["endpoint"] == qwen["endpoint"],
        "qwen_printed_usd": qwen["endpoint"][1]["dollars_per_mtokens"],
        "qwen_quants": [qwen["quants"][1][k] for k in ("gptq_int4_gb", "awq_int4_gb", "gguf_q4_km_gb")],
        "same_eval": evals[0] == evals[1], "eval": evals[1],
        "implied_qwen_bits": round(quant["gptq_int4_gb"] * 8 / PARAMS_B["qwen3-14b"], 2),
        "card_cmd": qwen["model_card"][1]["reproducibility_command"],
        "doc_names_33_8b": "Llama 3.3 8B" in parity.doc_text(PHASE, LESSON, "en"),
    }


def verify(result):
    r, q = result, result["qwen"]
    return [
        practice.Check(
            "ANSWER: Qwen3 14B costs $0.52 per 1M tokens at the same quant, 1.84x Llama's $0.28",
            (q["usd_per_m"], r["llama_usd"], r["ratio"], q["bits"], q["gb"], q["tps"], q["gpu_hour"])
            == (0.516, 0.28, 1.84, 4.58, 8.48, 3472, 6.45),
            f"{q['bits']} bits/param -> {q['gb']} GB, {q['tps']} tok/s at bs32 on ${q['gpu_hour']}/h "
            f"-> ${q['usd_per_m']}/1M vs ${r['llama_usd']}",
        ),
        practice.Check(
            "FINDING: the pipeline prices both models at $0.28 and gives them identical quality",
            (r["same_endpoint"], r["qwen_printed_usd"], r["qwen_quants"], r["same_eval"],
             r["implied_qwen_bits"], r["card_cmd"])
            == (True, 0.28, [4.6, 4.8, 5.1], True, 2.49, "./pipeline.sh config/llama3.3-8b-domainX.yaml"),
            f"endpoint unchanged: {r['same_endpoint']}; qwen quants {r['qwen_quants']} GB "
            f"({r['implied_qwen_bits']} bits/param); eval {r['eval']}; card runs {r['card_cmd']}",
        ),
        practice.Check(
            "FINDING: the lesson's default base is not an open-weights release",
            r["doc_names_33_8b"],
            "the lesson names 'Llama 3.3 8B'; Meta published Llama 3.3 at 70B only",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
