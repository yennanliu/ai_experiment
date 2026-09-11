"""Exercise 2 — a hundred items cannot reach the published number.

    **Medium.** Score Qwen2.5-Omni-7B on 100 MMAU-Pro speech items. Compare to
    the paper's reported number.

Reading of the exercise: `transformers`, `torch`, `datasets`, `torchaudio` and
`accelerate` are all absent and MMAU-Pro is not here, so the scoring cannot run.
The comparison can, and it settles the exercise before any model is loaded:
**100 items cannot produce a number comparable to 57.4%.** This is the
`DESIGN D11` scaled-down run -- an 1800-item pool per model, each reproducing its
published rate exactly, sampled 4,000 times at n=100.

**57.4% is not on the grid.** A hundred items move accuracy in steps of exactly
**1.00 pp**, so the paper's figure is unreachable by construction; the nearest
scores a run can return are 57 and 58.

**The interval is wider than the whole table.** At `p = 0.574` the standard error
is 0.0494, so the 95% interval is **[47.7, 67.1]** -- **19.4 pp** wide, against a
`overall` column that spans **7.8 pp** from Gemini 2.5 Pro's ~60% to
Qwen2.5-Omni-7B's 52.2%. The confidence interval is **2.5x** the entire spread it
would have to resolve.

Sampling confirms it. Over 4,000 draws of 100 items:

| | |
|---|---|
| the five models come back in published order | **6.9%** of draws |
| the best model is picked | **57.6%** of draws |
| Qwen's overall, 95% of draws | **43.0 - 62.0%** (published 52.2) |

**What would be enough.** From `n = 2 z^2 p(1-p) / (p1-p2)^2`, separating
Qwen2.5-Omni-7B from Gemini 2.5 Pro takes **311** items -- three times the
exercise's hundred -- and separating it from GPT-4o Audio, 0.3 pp away, takes
**212,951**, which is **118x the whole 1800-item benchmark**.

The same arithmetic reaches the doc's own headline claim. It says multi-audio is
"barely above random" and quotes 26.5% against a 25% chance line; distinguishing
those needs **3,326** items, **1.8x** all of MMAU-Pro. The claim is almost
certainly true, and MMAU-Pro cannot be the thing that establishes it.

Structure: `pool` builds an 1800-item corpus at an exact rate; `draw` scores one
100-item subset; `sample` runs the 4,000 draws; `two_sample` and `one_sample` are
the size formulas.
"""

from __future__ import annotations

import importlib.util
import math
import random

from harness import practice

PHASE, LESSON = "06-speech-and-audio", "10-audio-language-models"
TABLE = (("Gemini 2.5 Pro", 60.0), ("Gemini 2.5 Flash", 57.0), ("Audio Flamingo 3", 54.0),
         ("GPT-4o Audio", 52.5), ("Qwen2.5-Omni-7B", 52.2))
SPEECH, POOL, SAMPLE, DRAWS, Z = 57.4, 1800, 100, 4000, 1.96
CHANCE, MULTI = 25.0, 26.5
ABSENT = ("transformers", "torch", "datasets", "torchaudio", "accelerate")


def pool(rate, rng, size=POOL):
    """An item pool whose accuracy is the published rate, to within 1/size."""
    hits = round(rate / 100 * size)
    items = [1] * hits + [0] * (size - hits)
    rng.shuffle(items)
    return items


def draw(pools, indices):
    return {name: sum(items[i] for i in indices) / len(indices) * 100
            for name, items in pools.items()}


def sample(pools, rng, watched, draws=DRAWS):
    """How often 100 items reproduce the published order, and where one model lands."""
    published = [name for name, _ in TABLE]
    exact, best, seen = 0, 0, []
    for _ in range(draws):
        scored = draw(pools, rng.sample(range(POOL), SAMPLE))
        order = sorted(scored, key=lambda name: -scored[name])
        exact += order == published
        best += order[0] == published[0]
        seen.append(scored[watched])
    seen.sort()
    return {"exact": exact / draws, "best": best / draws,
            "low": seen[int(0.025 * draws)], "high": seen[int(0.975 * draws)]}


def two_sample(first, second):
    """Items needed to separate two rates at 95%."""
    pooled = (first + second) / 2
    return 2 * Z**2 * pooled * (1 - pooled) / (first - second) ** 2


def one_sample(rate, target):
    """Items needed to separate one rate from a fixed line at 95%."""
    return Z**2 * rate * (1 - rate) / (rate - target) ** 2


def solve():
    rng = random.Random(0)
    pools = {name: pool(rate, rng) for name, rate in TABLE}
    error = math.sqrt(SPEECH / 100 * (1 - SPEECH / 100) / SAMPLE)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "rates": {name: sum(items) / POOL * 100 for name, items in pools.items()},
        "step": 100 / SAMPLE, "low": (SPEECH / 100 - Z * error) * 100,
        "high": (SPEECH / 100 + Z * error) * 100, "width": 2 * Z * error * 100,
        "span": TABLE[0][1] - TABLE[-1][1],
        "draws": sample(pools, random.Random(7), TABLE[-1][0]),
        "vs_best": two_sample(TABLE[-1][1] / 100, TABLE[0][1] / 100),
        "vs_near": two_sample(TABLE[-1][1] / 100, TABLE[-2][1] / 100),
        "vs_chance": one_sample(MULTI / 100, CHANCE / 100),
    }


def verify(result):
    draws = result["draws"]
    return [
        practice.Check(
            "CONTROL: nothing can score, and the pool reproduces every published rate",
            len(result["absent"]) == len(ABSENT)
            and max(abs(result["rates"][n] - r) for n, r in TABLE) < 0.03,
            f"find_spec is None for {result['absent']} and MMAU-Pro is not here, so an "
            f"{POOL}-item pool per model stands in: it reproduces every published rate to "
            f"{max(abs(result['rates'][n] - r) for n, r in TABLE):.3f} pp, which leaves sample "
            "size as the only thing the numbers below are about",
        ),
        practice.Check(
            "ANSWER: 57.4% is not a score 100 items can return",
            result["step"] == 1.0,
            f"{SAMPLE} items move accuracy in steps of exactly {result['step']:.2f} pp, so the "
            f"paper's {SPEECH}% is unreachable by construction and the nearest returnable "
            "scores are 57 and 58. Comparing to it means comparing to a number off the grid",
        ),
        practice.Check(
            "ANSWER: the 95% interval is 2.5x the whole table's spread",
            result["width"] > 2 * result["span"],
            f"at p = {SPEECH / 100:.3f} the 95% interval is [{result['low']:.1f}, "
            f"{result['high']:.1f}] -- {result['width']:.1f} pp wide -- against an overall "
            f"column spanning {result['span']:.1f} pp from {TABLE[0][0]} to {TABLE[-1][0]}. "
            f"One sample cannot resolve the table it would be read against",
        ),
        practice.Check(
            "ANSWER: at n=100 the published order comes back 7% of the time",
            draws["exact"] < 0.15 and draws["best"] < 0.7,
            f"over {DRAWS} draws the five models land in published order "
            f"{draws['exact'] * 100:.1f}% of the time and the best is picked "
            f"{draws['best'] * 100:.1f}%; {TABLE[-1][0]}'s own score spans "
            f"{draws['low']:.1f}-{draws['high']:.1f}% across 95% of them, against a published "
            f"{TABLE[-1][1]}",
        ),
        practice.Check(
            "FINDING: the doc's own multi-audio claim needs more items than MMAU-Pro has",
            result["vs_chance"] > POOL and result["vs_best"] > SAMPLE,
            f"separating {TABLE[-1][0]} from {TABLE[0][0]} needs {result['vs_best']:,.0f} items, "
            f"three times the exercise's {SAMPLE}, and from {TABLE[-2][0]} "
            f"{result['vs_near']:,.0f} -- {result['vs_near'] / POOL:.0f}x the benchmark. "
            f"Distinguishing the quoted {MULTI}% multi-audio score from the {CHANCE}% chance "
            f"line needs {result['vs_chance']:,.0f}, {result['vs_chance'] / POOL:.1f}x all of it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
