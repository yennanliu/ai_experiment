"""Exercise 5 — the premise is backwards on the lesson's own numbers.

    Emu3 beats SDXL on FID but not on VQAv2 vs specialized VLMs. Explain why the
    unified-loss approach shows different strengths vs specialists on different
    benchmarks.

Reading of the exercise: the three benchmark pairs the lesson reports are
transcribed and put on a common footing -- relative margin, since a 0.2 on FID
and a 0.01 on GenEval and a 2.7 on VQAv2 are three different scales -- and the
premise is checked before it is explained, because it turns out to be the wrong
way round.

**ANSWER: the premise is backwards.** On the lesson's own numbers Emu3 *beats*
LLaVA-1.6 on VQAv2, 75.1 against 72.4, and *loses* GenEval to SDXL, 0.54 against
0.55. The one benchmark where the unified model goes backwards is a generation
benchmark, not a perception one.

**FINDING: on a common footing the perception win is the largest of the three.**
Relative margins are **+3.6%** on FID, **-1.8%** on GenEval and **+3.7%** on
VQAv2. The three land in a **5.5-point** band centred near zero, and the widest
is the one the exercise says does not exist.

**FINDING: two margins of the same size are described differently.** GenEval's
0.01 on a 0-1 scale is 1.8% relative and is called a "statistical tie"; FID's
0.2 on 5.6 is 3.6% and is called a win. Neither has a variance estimate
attached, so the two labels differ by a factor of two in relative terms and by
nothing in evidence.

**ANSWER: which is the explanation.** A unified loss spends one capacity budget
over the union of the tasks, so it lands near the specialist everywhere and
past it nowhere by much -- a band, not a spike. A specialist trained on one
objective produces the opposite signature: one large margin and the remaining
benchmarks unreported, because they were never trained for. The lesson's own
three numbers are the band; the interesting question is not which model wins but
which shape the reported margins have.

Structure: `PAIRS` transcribes the lesson's three benchmark comparisons with the
direction each metric improves in, `relative` puts them on a common footing, and
`band` measures the spread they occupy.
"""

from __future__ import annotations

from harness import practice

# (Emu3, rival, lower_is_better) -- transcribed from the lesson's Benchmarks section
PAIRS = {
    "MJHQ-30K FID vs SDXL": (5.4, 5.6, True),
    "GenEval vs SDXL": (0.54, 0.55, False),
    "VQAv2 vs LLaVA-1.6": (75.1, 72.4, False),
}
TIE_LABEL = "GenEval vs SDXL"


def relative(ours, theirs, lower_is_better):
    """Emu3's margin as a percentage of the rival, signed so positive is better."""
    margin = (theirs - ours) if lower_is_better else (ours - theirs)
    return round(margin / theirs * 100, 1)


def solve():
    margins = {name: relative(*values) for name, values in PAIRS.items()}
    wins = [name for name, value in margins.items() if value > 0]
    return {
        "pairs": {name: (values[0], values[1]) for name, values in PAIRS.items()},
        "margins": margins,
        "wins": sorted(wins), "losses": sorted(set(margins) - set(wins)),
        "largest": max(margins, key=margins.get),
        "band": round(max(margins.values()) - min(margins.values()), 1),
        "centre": round(sum(margins.values()) / len(margins), 1),
        "tie_margin": abs(margins[TIE_LABEL]),
        "fid_margin": margins["MJHQ-30K FID vs SDXL"],
        "label_ratio": round(margins["MJHQ-30K FID vs SDXL"] / abs(margins[TIE_LABEL]), 1),
        "perception_beats_generation": (margins["VQAv2 vs LLaVA-1.6"]
                                        > margins["MJHQ-30K FID vs SDXL"]),
    }


def verify(result):
    margins, pairs = result["margins"], result["pairs"]
    return [
        practice.Check(
            "ANSWER: the premise is backwards -- Emu3 wins VQAv2 and loses GenEval",
            all([pairs["VQAv2 vs LLaVA-1.6"] == (75.1, 72.4),
                 pairs["GenEval vs SDXL"] == (0.54, 0.55),
                 result["losses"] == ["GenEval vs SDXL"],
                 len(result["wins"]) == 2]),
            f"the lesson reports {pairs}. Emu3 beats LLaVA-1.6 on VQAv2, 75.1 against 72.4, "
            f"and loses GenEval to SDXL, 0.54 against 0.55 -- so the one benchmark where the "
            f"unified model goes backwards is {result['losses'][0].split(' vs ')[0]}, a "
            "generation benchmark, not a perception one",
        ),
        practice.Check(
            "FINDING: on a common footing the perception win is the largest of the three",
            all([margins == {"MJHQ-30K FID vs SDXL": 3.6, "GenEval vs SDXL": -1.8,
                             "VQAv2 vs LLaVA-1.6": 3.7},
                 result["largest"] == "VQAv2 vs LLaVA-1.6",
                 result["perception_beats_generation"]]),
            f"as percentages of the rival's score the margins are {margins} -- and the widest "
            f"is {result['largest']}, the one the exercise says does not exist. A 0.2 on FID, "
            "a 0.01 on GenEval and a 2.7 on VQAv2 are three different scales until they are "
            "normalised",
        ),
        practice.Check(
            "FINDING: two margins of the same size are described differently",
            all([result["tie_margin"] == 1.8, result["fid_margin"] == 3.6,
                 result["label_ratio"] == 2.0]),
            f"GenEval's 0.01 on a 0-1 scale is {result['tie_margin']}% relative and is called "
            f"a statistical tie; FID's 0.2 on 5.6 is {result['fid_margin']}% and is called a "
            f"win. The two labels differ by {result['label_ratio']}x in relative terms and by "
            "nothing in evidence -- neither has a variance estimate attached",
        ),
        practice.Check(
            "ANSWER: the shape of the margins is the explanation",
            all([result["band"] == 5.5, abs(result["centre"]) <= 2.0,
                 len(PAIRS) == 3]),
            f"the three margins occupy a {result['band']}-point band centred at "
            f"{result['centre']}%. A unified loss spends one capacity budget over the union "
            "of the tasks, so it lands near the specialist everywhere and past it nowhere by "
            "much -- a band. A specialist produces the opposite signature: one large margin "
            "and the other benchmarks unreported, because it was never trained for them",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
