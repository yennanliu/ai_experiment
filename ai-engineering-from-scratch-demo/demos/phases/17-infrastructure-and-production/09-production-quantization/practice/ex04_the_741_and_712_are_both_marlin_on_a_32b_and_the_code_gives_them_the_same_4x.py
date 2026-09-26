"""Exercise 4 — the 741 and 712 are both Marlin on a 32B, and the code gives them the same 4x.

    Read the Marlin-AWQ kernel paper or release notes. Explain in three
    sentences why AWQ hits 741 tok/s on 7B while raw GPTQ hits ~712.

Reading of the exercise: two sources were read, both fetched 2026-09-26. One
is the Marlin paper (Frantar et al., arXiv:2408.11743). The other is the
lesson's own benchmark source, Jarvis Labs' "vLLM Quantization Complete
Guide" (January 2026), where 741 and 712 come from. The three sentences are
the answer. The checks hold the lesson's text and its `relative_throughput`
against the measured table.

**ANSWER, in three sentences.** (1) The 712 is not raw GPTQ but GPTQ on
the same Marlin kernels (raw GPTQ runs 276.60 tok/s and raw AWQ 67.73), so
741 against 712 compares two INT4 checkpoints on one kernel family. (2)
Marlin is a mixed-precision kernel that reads 4-bit weights and computes
against FP16 activations, keeping "close to maximum (4x) quantization
speedup" up to batch 16-32 per its paper; that is why both land together at
1.55-1.61x the FP16 run's 461.04 tok/s, and why the kernel (10.9x for AWQ,
2.6x for GPTQ) matters far more than the algorithm. (3) The remaining 4% is
one 200-prompt run that neither source attributes to a mechanism; what AWQ
measurably buys is accuracy, HumanEval Pass@1 51.8 against Marlin-GPTQ's
45.7.

**FINDING: the lesson's code cannot produce the gap, or GGUF's penalty.**
`relative_throughput` is 16 / weight_bits, so AWQ, GPTQ and GGUF Q4_K_M all
come out 4.00x FP16. Measured, they are 1.607x, 1.545x and 0.202x. The code's
own KEY FINDINGS say GGUF's ~93 tok/s in vLLM "is the wrong engine", and its
throughput model gives GGUF the same 4x as AWQ. The fastest measured 4-bit
run is 2.5x short of the model's 4x.

**FINDING: none of these numbers are from a 7B.** The benchmark is
Qwen2.5-32B-Instruct on one H200, 200 ShareGPT prompts at max concurrency 10.
The lesson says "~741 tok/s on 7B", "~712 tok/s on 7B" and "~93 tok/s on 7B".
The source never mentions a 7B.

**FINDING: "best Pass@1 among INT4 formats" is a four-way tie.** Marlin-AWQ,
AWQ, GGUF Q4_K_M and bitsandbytes all score 0.5183. Only the GPTQ runs are
lower (0.4634, 0.4573).

Structure: `BENCH` holds the fetched table; `solve()` computes the ratios and
the reference's throughput for the matching formats.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "09-production-quantization"
BENCH = {  # jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks, fetched 2026-09-26
    "model": "Qwen2.5-32B-Instruct", "gpu": "H200", "prompts": 200, "concurrency": 10,
    "tok_s": {"FP16": 461.04, "AWQ": 67.73, "GPTQ": 276.60, "Marlin-GPTQ": 712.45,
              "Marlin-AWQ": 741.04, "GGUF Q4_K_M": 93.05, "bitsandbytes": 168.37},
    "pass1": {"FP16": 0.561, "AWQ": 0.5183, "GPTQ": 0.4634, "Marlin-GPTQ": 0.4573,
              "Marlin-AWQ": 0.5183, "GGUF Q4_K_M": 0.5183, "bitsandbytes": 0.5183},
}
MARLIN_IDEAL = 4.0  # arXiv:2408.11743: "close to maximum (4x) quantization speedup"
PAIRS = {"Marlin-AWQ": "AWQ-Int4", "Marlin-GPTQ": "GPTQ-Int4", "GGUF Q4_K_M": "GGUF Q4_K_M"}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tok = BENCH["tok_s"]
    code = {k: ref.relative_throughput(next(f for f in ref.FORMATS if f.name.startswith(v)))
            for k, v in PAIRS.items()}
    doc = parity.doc_text(PHASE, LESSON)
    int4 = {k: v for k, v in BENCH["pass1"].items() if k != "FP16"}
    return {
        "measured": {k: round(tok[k] / tok["FP16"], 3) for k in PAIRS},
        "code": code,
        "kernel_gain": {"AWQ": round(tok["Marlin-AWQ"] / tok["AWQ"], 1),
                        "GPTQ": round(tok["Marlin-GPTQ"] / tok["GPTQ"], 1)},
        "gap": round(tok["Marlin-AWQ"] / tok["Marlin-GPTQ"] - 1, 3),
        "doc_7b": all(f"{n} tok/s on 7B" in doc for n in ("~712", "~93", "741")),
        "doc_marlin_gptq": "Marlin kernels make it fast on GPU" in doc,
        "tie": sorted(k for k, v in int4.items() if v == max(int4.values())),
        "wrong_engine": "wrong engine" in inspect.getsource(ref.main),
    }


def verify(result):
    m, code, gain = result["measured"], result["code"], result["kernel_gain"]
    return [
        practice.Check(
            "ANSWER: 712 is Marlin-GPTQ, so 741 vs 712 is two checkpoints on one kernel family",
            all([result["doc_marlin_gptq"], BENCH["tok_s"]["GPTQ"] < 300,
                 gain == {"AWQ": 10.9, "GPTQ": 2.6}, 0.03 < result["gap"] < 0.05,
                 BENCH["pass1"]["Marlin-AWQ"] > BENCH["pass1"]["Marlin-GPTQ"]]),
            f"raw GPTQ {BENCH['tok_s']['GPTQ']}, raw AWQ {BENCH['tok_s']['AWQ']}; Marlin "
            f"gains {gain}; AWQ over GPTQ on Marlin +{result['gap']:.1%}; Pass@1 "
            f"{BENCH['pass1']['Marlin-AWQ']} vs {BENCH['pass1']['Marlin-GPTQ']}",
        ),
        practice.Check(
            "FINDING: the lesson's code cannot produce the gap, or GGUF's penalty",
            set(code.values()) == {MARLIN_IDEAL} and result["wrong_engine"]
            and m == {"Marlin-AWQ": 1.607, "Marlin-GPTQ": 1.545, "GGUF Q4_K_M": 0.202},
            f"relative_throughput {code} against measured {m} of FP16",
        ),
        practice.Check(
            "FINDING: none of these numbers are from a 7B",
            result["doc_7b"] and "7B" not in BENCH["model"],
            f"the lesson says '... tok/s on 7B' three times; the source ran "
            f"{BENCH['model']} on one {BENCH['gpu']}, {BENCH['prompts']} prompts at "
            f"concurrency {BENCH['concurrency']}",
        ),
        practice.Check(
            "FINDING: 'best Pass@1 among INT4 formats' is a four-way tie",
            result["tie"] == ["AWQ", "GGUF Q4_K_M", "Marlin-AWQ", "bitsandbytes"],
            f"tied at {max(BENCH['pass1'][k] for k in result['tie'])}: {result['tie']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
