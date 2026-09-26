"""Exercise 1 — the ceiling is 22 tok/s, so 6-8 is a third of it, and the code has one row above its own ceiling.

    Run `code/main.py`. For a 7B model in Q4 on a Snapdragon 8 Gen 3 (~77
    GB/s bandwidth), compute the decode ceiling. Compare to observed 6-8 tok/s
    -- is the runtime efficient?

Reading of the exercise: the ceiling is the reference's own `ceiling()` on
its own Snapdragon 8 Gen 3 target, fed the lesson's sizing for a 7B Q4 model
("3.5 GB", i.e. exactly 4 bits a weight). Because a real Q4 file carries
scales, the answer is also given at 4.5 bits a weight (Q4_0's layout) and at
the code's own 4.7 GB 8B file, so the verdict does not hang on one size.

**ANSWER: the ceiling is 22.0 tok/s, so 6-8 tok/s is 27-36% of it -- not
efficient.** 3.5 GB / 77 GB/s is 45 ms a token. At 4.5 bits a weight (3.94
GB) the ceiling is 19.6 and 6-8 is 31-41%; for the code's 8B file (4.7 GB) it
is 16.4 and 37-49%. Every sizing leaves over half the bandwidth unused.
Compute is not what is missing: 2 x 7e9 FLOPs a token at the 45 TOPS the
lesson quotes for Hexagon is a 3214 tok/s compute ceiling, 146x above the
bandwidth one. The gap is the runtime.

**FINDING: the code's own table has one row above its own ceiling.** Jetson
AGX Orin is listed at 45 tok/s on Llama 3.1 8B Q4 against a 43.6 tok/s
ceiling -- 103%, which a bandwidth-bound decode cannot do. `efficiency()`
prints it without comment. Every other row is 24-98%.

**FINDING: the lesson gives the iPhone two speeds for the same model.** The
Problem section says Llama 3.1 8B Q4 runs at 3 tok/s on an iPhone 16 Pro;
the code lists 8 tok/s on the A18. 3 tok/s would be 24% of the 12.8 tok/s
ceiling, 8 is 63%.

**FINDING: the quantization table scales from the Q4 size, so BF16 is 2.7 GB
too big.** It lists BF16 at 18.8 GB -- 4 x 4.7. Llama 3.1 8B has 8.03B
parameters, so BF16 is 16.06 GB and the iPhone BF16 ceiling is 3.7 tok/s,
not 3.2. The 4.7 GB Q4 file is 4.68 bits a weight.

Structure: every ceiling is the reference `ceiling()` on a reference
`Target`; only the model sizes are computed here.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "12-edge-inference"
PARAMS_7B, PARAMS_8B = 7e9, 8.03e9  # Llama 3.1 8B: 8.03B parameters
HEXAGON_TOPS = 45  # the lesson's Hexagon figure
OBSERVED = (6, 8)
SIZES_GB = {"lesson 3.5 GB (4.0 bpw)": 3.5, "Q4_0 layout (4.5 bpw)": PARAMS_7B * 4.5 / 8 / 1e9,
            "code's 8B file (4.7 GB)": 4.7}


def target(ref, prefix):
    return next(t for t in ref.TARGETS if t.name.startswith(prefix))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sd = target(ref, "Snapdragon 8 Gen 3")
    ceilings = {k: ref.ceiling(sd, gb) for k, gb in SIZES_GB.items()}
    rows = {t.name: t.observed_toks_per_s_llama8b_q4 / ref.ceiling(t, 4.7) for t in ref.TARGETS}
    a18 = ref.ceiling(target(ref, "Apple A18"), 4.7)
    doc = parity.doc_text(PHASE, LESSON, "en")
    over = [name for name, eff in rows.items() if eff > 1]
    return {
        "effs": {k: (OBSERVED[0] / v, OBSERVED[1] / v) for k, v in ceilings.items()},
        "over": over, "others": [eff for name, eff in rows.items() if name not in over],
        "bandwidth": sd.bandwidth_gb_s, "ceilings": ceilings,
        "compute_ceiling": HEXAGON_TOPS * 1e12 / (2 * PARAMS_7B),
        "rows": rows, "a18_ceiling": a18,
        "a18_observed": target(ref, "Apple A18").observed_toks_per_s_llama8b_q4,
        "doc_iphone_3": "iPhone 16 Pro, the same model runs at 3 tok/s" in doc,
        "bf16_real_gb": PARAMS_8B * 2 / 1e9, "q4_bpw": 4.7e9 * 8 / PARAMS_8B,
    }


def verify(result):
    c = result["ceilings"]
    lesson = c["lesson 3.5 GB (4.0 bpw)"]
    effs, rows, over, others = result["effs"], result["rows"], result["over"], result["others"]
    table = "; ".join(f"{k}: ceiling {c[k]:.1f}, observed {lo:.0%}-{hi:.0%}"
                      for k, (lo, hi) in effs.items())
    bf16 = 60 / result["bf16_real_gb"]
    return [
        practice.Check(
            "ANSWER: the ceiling is 22.0 tok/s, so 6-8 tok/s is 27-36% of it -- not efficient",
            all([result["bandwidth"] == 77, round(lesson, 1) == 22.0,
                 max(hi for _, hi in effs.values()) < 0.5,
                 result["compute_ceiling"] / lesson > 100]),
            table + f"; compute ceiling {result['compute_ceiling']:.0f} tok/s",
        ),
        practice.Check(
            "FINDING: the code's own table has one row above its own ceiling",
            over == ["Jetson AGX Orin"] and 0.23 < min(others) and max(others) < 0.99,
            f"AGX Orin at {rows['Jetson AGX Orin']:.0%} of its ceiling; the others span "
            f"{min(others):.0%}-{max(others):.0%}",
        ),
        practice.Check(
            "FINDING: the lesson gives the iPhone two speeds for the same model",
            result["doc_iphone_3"] and result["a18_observed"] == 8,
            f"docs say 3 tok/s ({3 / result['a18_ceiling']:.0%} of the "
            f"{result['a18_ceiling']:.1f} ceiling), code says 8 "
            f"({8 / result['a18_ceiling']:.0%})",
        ),
        practice.Check(
            "FINDING: the quantization table scales from the Q4 size, so BF16 is 2.7 GB too big",
            round(18.8 - result["bf16_real_gb"], 1) == 2.7 and round(bf16, 1) == 3.7,
            f"BF16 of 8.03B params is {result['bf16_real_gb']:.2f} GB, ceiling {bf16:.1f} "
            f"tok/s on 60 GB/s; the 4.7 GB Q4 file is {result['q4_bpw']:.2f} bits a weight",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
