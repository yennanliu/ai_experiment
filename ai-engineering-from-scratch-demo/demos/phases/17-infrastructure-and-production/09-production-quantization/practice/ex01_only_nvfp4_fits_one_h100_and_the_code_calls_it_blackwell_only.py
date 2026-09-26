"""Exercise 1 — only NVFP4 fits one H100, and the code calls it Blackwell-only.

    Run `code/main.py`. For a 70B model at 128 concurrent with 2k context,
    compute the total HBM for each format. Which format lets you fit on one H100
    80GB?

Reading of the exercise: "total HBM" is the reference `memory_breakdown`
(weights + KV cache + activations) at params 70, concurrency 128, context
2048, and "fits" is the reference `gpu_check` (total <= 80 GB). The answer is
then checked against the hardware each format needs and against the lesson's
own prose budget for the same scenario.

**ANSWER: one format, NVFP4 + FP8 KV, at 72.86 GB.** Totals: BF16 212.22,
GGUF Q5_K_M 115.97, GGUF Q4_K_M / GPTQ / AWQ 107.22, FP8 107.86, NVFP4 + FP8
KV 72.86. Every other format lands on an H200 141GB; BF16 needs several GPUs.
The four 4-bit formats with BF16 KV are 107.22 GB each, and 68.72 GB of that
is KV cache: the weights are already down to 35 GB.

**FINDING: the one fit is a format the code itself calls Blackwell-only, and
the Hopper fit is a row the code lacks.** `main()` prints "NVFP4 + FP8 KV
stacks: shrink weights AND KV ; Blackwell-only", and the table tags it
TRT-LLM / "Blackwell aggressive", while `gpu_check` answers "H100 80GB". What
fits is 4-bit weights with 8-bit KV, and nothing about that needs FP4.
`Format("AWQ + FP8 KV", 4, 8, ...)`, i.e. AWQ weights with vLLM's
`kv_cache_dtype="fp8"`, is the same 72.86 GB, and FP8 KV on AWQ weights runs on
Hopper. `FORMATS` has no such row.

**FINDING: the lesson's "~60 GB, fits on H100" uses a KV cache 3.4x smaller
than its own code.** The KV cache trap section budgets AWQ-70B at 35 GB
weights + ~20 GB KV + ~5 GB activations. The code computes 68.72 GB of KV
(64 layers, 8 KV heads, head dim 128, BF16) and puts AWQ on an H200. With
Llama-3-70B's real 80 layers the KV is 85.9 GB. The ~20 GB would need 76 KB
per token, and 70B-class GQA at BF16 is 262-328 KB.

**FINDING: 72.86 GB is "fits" with nothing to spare.** vLLM reserves
`gpu_memory_utilization = 0.9` of the card by default, 72 GB of 80, so even
the one fit overshoots by 0.86 GB before CUDA context and fragmentation. At
full use, AWQ with BF16 KV fits 77 concurrent 2k sequences (0.537 GB each),
not 128.

Structure: `solve()` calls the reference functions unchanged. The extra row
is `dataclasses.replace` on the AWQ format, and the Llama-3 figure uses the
same formula with 80 layers.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "09-production-quantization"
PARAMS, CONC, CTX, H100 = 70, 128, 2048, 80
VLLM_UTIL = 0.9  # vLLM's default gpu_memory_utilization


def totals(ref, fmts):
    return {f.name: ref.memory_breakdown(PARAMS, f, CONC, CTX) for f in fmts}


def lacks_awq_fp8kv(ref):
    return not any(f.weight_bits == 4 and f.kv_bits == 8 and "NVFP4" not in f.name
                   for f in ref.FORMATS)


def doc_budget(ref):
    doc = parity.doc_text(PHASE, LESSON)
    return all(s in doc for s in ("~20 GB", "~60 GB", "fits on H100 80GB"))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    table = totals(ref, ref.FORMATS)
    awq = next(f for f in ref.FORMATS if f.name.startswith("AWQ"))
    awq_fp8kv = dataclasses.replace(awq, name="AWQ + FP8 KV", kv_bits=8)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        ref.main()
    per_seq = ref.memory_breakdown(PARAMS, awq, 1, CTX)["kv"]
    return {
        "totals": {n: round(m["total"], 2) for n, m in table.items()},
        "kv": {n: round(m["kv"], 2) for n, m in table.items()},
        "fits": [n for n, m in table.items() if ref.gpu_check(m["total"]) == "H100 80GB"],
        "awq_fp8kv": round(totals(ref, [awq_fp8kv])["AWQ + FP8 KV"]["total"], 2),
        "blackwell_only": "Blackwell-only" in out.getvalue(),
        "lacks_awq_fp8kv": lacks_awq_fp8kv(ref),
        "doc_budget": doc_budget(ref),
        "awq_gpu": ref.gpu_check(table[awq.name]["total"]),
        "llama3_kv": round(80 * 2 * 8 * 128 * CTX * 2 * CONC / 1e9, 1),
        "doc_kb_per_token": round(20e9 / (CONC * CTX) / 1e3, 1),
        "awq_max_conc": int((H100 - table[awq.name]["weight"] - table[awq.name]["act"]) / per_seq),
    }


def verify(result):
    t, kv = result["totals"], result["kv"]
    nvfp4 = "NVFP4 + FP8 KV (TRT-LLM)"
    awq = "AWQ-Int4 + Marlin (vLLM)"
    return [
        practice.Check(
            "ANSWER: one format, NVFP4 + FP8 KV, at 72.86 GB",
            all([result["fits"] == [nvfp4], t[nvfp4] == 72.86, t[awq] == 107.22,
                 kv[awq] == 68.72]),
            f"totals {t}; fits on one H100: {result['fits']}; the 4-bit BF16-KV formats "
            f"carry {kv[awq]} GB of KV each",
        ),
        practice.Check(
            "FINDING: the one fit is a format the code calls Blackwell-only",
            all([result["blackwell_only"], result["awq_fp8kv"] == t[nvfp4],
                 result["lacks_awq_fp8kv"]]),
            f"main() prints 'Blackwell-only' for NVFP4 + FP8 KV; AWQ weights with FP8 KV "
            f"total {result['awq_fp8kv']} GB, identical, and FORMATS has no such row",
        ),
        practice.Check(
            "FINDING: the lesson's '~60 GB, fits on H100' uses a KV cache 3.4x smaller than its code",
            all([result["doc_budget"], round(kv[awq] / 20, 1) == 3.4,
                 result["awq_gpu"] == "H200 141GB", result["llama3_kv"] == 85.9]),
            f"doc: ~20 GB KV, ~60 GB total on an H100; code: {kv[awq]} GB KV, AWQ on "
            f"{result['awq_gpu']}; 80-layer Llama-3-70B: {result['llama3_kv']} GB; ~20 GB "
            f"implies {result['doc_kb_per_token']} KB/token",
        ),
        practice.Check(
            "FINDING: 72.86 GB is 'fits' with nothing to spare",
            t[nvfp4] > H100 * VLLM_UTIL and result["awq_max_conc"] == 77,
            f"{t[nvfp4]} GB against vLLM's default {H100 * VLLM_UTIL:.0f} GB budget; AWQ with "
            f"BF16 KV fits {result['awq_max_conc']} concurrent 2k sequences, not {CONC}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
