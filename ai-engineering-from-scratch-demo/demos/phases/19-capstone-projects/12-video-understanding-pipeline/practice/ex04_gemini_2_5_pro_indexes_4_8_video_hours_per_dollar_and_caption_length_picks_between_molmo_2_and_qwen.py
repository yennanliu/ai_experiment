"""Exercise 4 -- Gemini 2.5 Pro indexes 4.8 hours of video per dollar, the cheap VLMs 44-49, and caption length decides between those two.

    Benchmark ingest cost: hours-of-video-per-dollar across three VLM choices. Pick the sweet spot.

Reading of the exercise: the three VLMs are the ones the lesson names --
Gemini 2.5 Pro, Qwen3-VL and Molmo 2 -- priced from their public list prices
(read 2026-09-29, URLs in `PRICES`). The ingest call is the doc's step 4:
one 1280x720 keyframe per scene, a 50-token caption template, and a caption
of 30, 60 or 120 tokens. Image tokens follow each model's own rule: Gemini's
258 tokens per 768 px tile, Qwen3-VL's 32 px merged patches after
`smart_resize`, and Molmo 2's overlapping 378 px crops pooled 2x2 plus a
global view. Gemini 2.5 Pro cannot turn thinking off, and its smallest
budget is 128 tokens, billed as output. Scenes per hour come from the doc
(6k-8k scenes per 100 h) and from the lesson's own `SAMPLE` timings. ASR and
embedding are the same for every choice and are left out.

**ANSWER: the sweet spot is Molmo 2 8B or Qwen3-VL-Plus. They cost 9-10x
less than Gemini 2.5 Pro.** At 60-token captions and 80 scenes per hour,
Gemini 2.5 Pro indexes 4.8 hours of video per dollar, Qwen3-VL-Plus 44.3 and
Molmo 2 49.1. Molmo 2 wins at 60 and 120 tokens. Qwen3-VL-Plus wins at 30,
because Molmo's image costs more tokens and Qwen's output costs 8x more per
token. The crossover is a 40-token caption.

**FINDING: Gemini's bill is mostly tokens nobody reads.** With the minimum
thinking budget, output is 72.7% of Gemini 2.5 Pro's per-scene cost. Sending
the whole hour as native video instead of keyframes (263 tokens per second)
gives 0.7 hours per dollar, 7x worse than keyframes, even when the hour is
split to stay under the 200k-token price tier.

**FINDING: the lesson's own sample cuts scenes faster than the doc
budgets.** vid_001 has 5 scenes in 210 s (85.7 per hour) and vid_002 1 in
40 s (90 per hour). The doc budgets 60-80 per hour, so its cost figure
runs 7-50% low. There is no "Qwen3-VL-Max" on Alibaba's price list; the
nearest are qwen3-vl-plus and the older qwen-vl-max.

Structure: `tokens_*` are the three image-token rules; `cost_per_scene`
prices one call; `hours_per_dollar` scales it by scenes per hour.
"""

from __future__ import annotations

import math
import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "12-video-understanding-pipeline"
W, H, PROMPT, THINK_MIN = 1280, 720, 50, 128
PRICES = {  # USD per 1M tokens (input, output)
    "gemini-2.5-pro": (1.25, 10.00),  # https://ai.google.dev/gemini-api/docs/pricing (<= 200k prompt)
    "qwen3-vl-plus": (0.20, 1.60),  # https://www.alibabacloud.com/help/en/model-studio/model-pricing (0-32K, intl)
    "molmo-2-8b": (0.20, 0.20),  # https://pricepertoken.com/pricing-page/model/allenai-molmo-2-8b
}


def tokens_gemini(w, h):  # https://ai.google.dev/gemini-api/docs/tokens: 258 per 768x768 tile
    return 258 if max(w, h) <= 384 else 258 * math.ceil(w / 768) * math.ceil(h / 768)


def tokens_qwen(w, h, factor=32):  # huggingface.co/Qwen/Qwen3-VL-8B-Instruct: patch 16, merge 2
    return (round(w / factor) * factor // factor) * (round(h / factor) * factor // factor)


def tokens_molmo(w, h, crop=378, patch=14, margin=8, max_crops=8):
    """Port of select_tiling / build_overlapping_crops in huggingface.co/allenai/Molmo2-8B image_processing_molmo2.py."""
    window = (crop // patch - margin) * patch
    hh, ww = h - margin * patch, w - margin * patch
    tilings = sorted(((i, j) for i in range(1, max_crops + 1) for j in range(1, max_crops + 1) if i * j <= max_crops),
                     key=lambda t: (t[0] * t[1], t[0]))
    scales = [min(i * window / hh, j * window / ww) for i, j in tilings]
    if all(s < 1 for s in scales):
        ti, tj = tilings[scales.index(max(scales))]
    else:
        ti, tj = tilings[scales.index(min(s if s >= 1 else 1e10 for s in scales))]
    rows, cols = (ti * window + margin * patch) // patch, (tj * window + margin * patch) // patch
    glob = math.ceil(crop // patch / 2)
    return math.ceil(rows / 2) * math.ceil(cols / 2) + glob * glob


IMAGE = {"gemini-2.5-pro": tokens_gemini, "qwen3-vl-plus": tokens_qwen, "molmo-2-8b": tokens_molmo}


def cost_per_scene(model, caption):
    p_in, p_out = PRICES[model]
    out = caption + (THINK_MIN if model == "gemini-2.5-pro" else 0)
    return ((IMAGE[model](W, H) + PROMPT) * p_in + out * p_out) / 1e6, out * p_out / 1e6


def hours_per_dollar(model, caption, scenes_per_hour):
    return round(1 / (scenes_per_hour * cost_per_scene(model, caption)[0]), 1)


def lesson_scene_rates(ref):
    by_video = {}
    for s in ref.SAMPLE:
        by_video.setdefault(s.video_id, []).append(s)
    return {v: round(len(sc) / (max(s.end_ms for s in sc) / 3.6e6), 1) for v, sc in by_video.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lo, hi = map(int, re.search(r"~(\d+)k-(\d+)k scenes", parity.doc_text(PHASE, LESSON)).groups())
    doc_rates = (lo * 1000 / 100, hi * 1000 / 100)
    grid = {c: {m: hours_per_dollar(m, c, doc_rates[1]) for m in PRICES} for c in (30, 60, 120)}
    total, output = cost_per_scene("gemini-2.5-pro", 60)
    native_tokens = 263 * 3600
    native = 1 / (native_tokens * PRICES["gemini-2.5-pro"][0] / 1e6 + 80 * (60 + THINK_MIN) * 10 / 1e6)
    (q_in, q_out), (m_in, m_out) = PRICES["qwen3-vl-plus"], PRICES["molmo-2-8b"]
    extra_in = (tokens_molmo(W, H) + PROMPT) * m_in - (tokens_qwen(W, H) + PROMPT) * q_in
    crossover = extra_in / (q_out - m_out)
    return {
        "image_tokens": {m: f(W, H) for m, f in IMAGE.items()}, "grid": grid, "doc_rates": doc_rates,
        "winner": {c: max(row, key=row.get) for c, row in grid.items()}, "crossover": round(crossover, 1),
        "gemini_output_share": round(output / total, 3), "native": round(native, 1),
        "lesson_rates": lesson_scene_rates(ref),
    }


def verify(result):
    r, g = result, result["grid"]
    return [
        practice.Check(
            "ANSWER: Molmo 2 / Qwen3-VL-Plus index 9-10x more hours per dollar than Gemini 2.5 Pro",
            g[60] == {"gemini-2.5-pro": 4.8, "qwen3-vl-plus": 44.3, "molmo-2-8b": 49.1}
            and r["winner"] == {30: "qwen3-vl-plus", 60: "molmo-2-8b", 120: "molmo-2-8b"} and r["crossover"] == 40.3,
            f"image tokens {r['image_tokens']}; hours/$ at {r['doc_rates'][1]:.0f} scenes/h by caption length: {g}; "
            f"Qwen/Molmo crossover at a {r['crossover']}-token caption",
        ),
        practice.Check(
            "FINDING: output (mostly forced thinking) is most of Gemini's bill; native video is 7x worse",
            (r["gemini_output_share"], r["native"]) == (0.727, 0.7),
            f"output share of Gemini 2.5 Pro per-scene cost {r['gemini_output_share']:.1%}; native-video hours/$ "
            f"{r['native']} vs keyframes {g[60]['gemini-2.5-pro']}",
        ),
        practice.Check(
            "FINDING: the lesson's own sample cuts more scenes per hour than the doc budgets",
            r["lesson_rates"] == {"vid_001": 85.7, "vid_002": 90.0} and r["doc_rates"] == (60.0, 80.0),
            f"lesson scenes/h {r['lesson_rates']} vs doc {r['doc_rates']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
