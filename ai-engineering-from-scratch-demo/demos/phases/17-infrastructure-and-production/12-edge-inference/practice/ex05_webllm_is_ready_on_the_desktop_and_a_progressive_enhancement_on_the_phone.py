"""Exercise 5 — WebLLM is ready on the desktop and a progressive enhancement on the phone.

    Argue whether "WebLLM is production-ready in 2026." Cite the coverage,
    performance, and the Firefox Android gap.

Reading of the exercise: "production-ready" is split by platform, because
the lesson's own numbers split that way. Coverage is the lesson's figure,
parsed from its page; performance is the reference TARGETS table; memory is
WebLLM v0.2.84's `src/config.ts`. The Firefox Android status was checked on
2026-09-26 in the gpuweb wiki's Implementation Status page: WebGPU is
"behind a flag" there, and "Mozilla expects to do work on Android in 2026";
Safari iOS/iPadOS ships it in 26.

**ANSWER: production-ready on the desktop, and on the phone only as a
progressive enhancement over a server fallback.** On an M3 Max WebLLM decodes
Llama 3.1 8B Q4 at 41 tok/s, 74.5% of the 55 tok/s native runtime -- inside
the lesson's "70-80% of native". On mobile the lesson's ~70-75% coverage
leaves 25-30% of sessions with no WebGPU at all. Firefox Android is part of
that, and nothing in the lesson says how large a part. A launch that
depends on the local path fails those sessions; one that routes them to the
same API on a server (Exercise 2) does not.

**FINDING: on the phone the browser is not the bottleneck.** WebGPU on the
Pixel 9 decodes at 6 tok/s against the same 16.4 tok/s ceiling as the native
Snapdragon 8 Gen 3 row at 7 -- 86% of native, a smaller browser penalty than
the desktop's 74.5%. (The code gives the Pixel 9, a Tensor G4 phone, the
Snapdragon's 77 GB/s, so the two rows are one bandwidth.) What makes the phone slow is bandwidth and the runtime:
6 tok/s is 37% of its ceiling and a seventh of the desktop's 41. Llama
3.1 8B's q4f16 build needs 5001 MB in WebLLM, so a phone gets the 3B (2264 MB)
or 1B (879 MB), not the model benchmarked on the desktop.

**FINDING: the code's efficiency column is not the browser penalty.** The
note says "browser penalty ~25%", and 41 / 55 is 25% off native. The printed
efficiency for the same row is 48%, which is 41 over the bandwidth ceiling
and folds in native's own shortfall (55 / 85.1 = 65%).

Structure: every rate is a reference Target and every ceiling the reference
`ceiling()`; the coverage band is read from the lesson text.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "12-edge-inference"
WEBLLM_MB = {"Llama-3.1-8B q4f16": 5001.0, "Llama-3.2-3B q4f16": 2263.69,
             "Llama-3.2-1B q4f16": 879.04}  # src/config.ts, 4K context


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    t = {x.name: x for x in ref.TARGETS}
    doc = parity.doc_text(PHASE, LESSON, "en")
    lo, hi = map(int, re.search(r"~(\d+)-(\d+)% mobile coverage", doc).groups())
    obs = {k: v.observed_toks_per_s_llama8b_q4 for k, v in t.items()}
    ceil = {k: ref.ceiling(v, 4.7) for k, v in t.items()}
    return {
        "coverage": (lo, hi), "obs": obs, "ceil": ceil,
        "doc_native_band": "roughly 70-80% of native" in doc,
        "note": t["WebGPU on M3 Max"].notes,
        "printed_eff": ref.efficiency(obs["WebGPU on M3 Max"], ceil["WebGPU on M3 Max"]),
    }


def verify(result):
    obs, ceil = result["obs"], result["ceil"]
    desk = obs["WebGPU on M3 Max"] / obs["Apple M3 Max"]
    phone = obs["WebGPU on Pixel 9"] / obs["Snapdragon 8 Gen 3"]
    lo, hi = result["coverage"]
    phone_ceiling = obs["WebGPU on Pixel 9"] / ceil["WebGPU on Pixel 9"]
    return [
        practice.Check(
            "ANSWER: production-ready on the desktop, and on the phone only as a progressive "
            "enhancement over a server fallback",
            all([round(desk, 3) == 0.745, 0.70 <= desk <= 0.80, result["doc_native_band"],
                 (100 - hi, 100 - lo) == (25, 30)]),
            f"desktop WebGPU {obs['WebGPU on M3 Max']} tok/s = {desk:.1%} of native "
            f"{obs['Apple M3 Max']}; mobile coverage {lo}-{hi}% leaves "
            f"{100 - hi}-{100 - lo}% with no WebGPU",
        ),
        practice.Check(
            "FINDING: on the phone the browser is not the bottleneck",
            all([ceil["WebGPU on Pixel 9"] == ceil["Snapdragon 8 Gen 3"], phone > desk,
                 round(phone_ceiling, 2) == 0.37, WEBLLM_MB["Llama-3.1-8B q4f16"] > 5000]),
            f"Pixel 9 browser {obs['WebGPU on Pixel 9']} vs native {obs['Snapdragon 8 Gen 3']} "
            f"on one {ceil['WebGPU on Pixel 9']:.1f} ceiling = {phone:.0%} of native; "
            f"{phone_ceiling:.0%} of ceiling; WebLLM memory {WEBLLM_MB}",
        ),
        practice.Check(
            "FINDING: the code's efficiency column is not the browser penalty",
            "~25%" in result["note"] and result["printed_eff"].strip() == "48%"
            and round(1 - desk, 2) == 0.25,
            f"note {result['note']!r}, printed efficiency {result['printed_eff'].strip()}, "
            f"native M3 Max at {obs['Apple M3 Max'] / ceil['Apple M3 Max']:.0%} of its ceiling",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
