"""Exercise 3 — ASR alone cannot say why: a better reader refusing 30% after decoding ties a worse reader.

    Read Jiang et al. 2024 Section 4.3 (five-model results). Propose a reason
    why Claude's ArtPrompt-resistance is higher than Gemini's on the same
    benchmark.

Reading of the exercise: the lesson does not reproduce Section 4.3 -- its
only figure is "above 75%", and it lists Claude among the models that "all
fail" -- so no per-model number is quoted here. The exercise is answered as
the smallest runnable model of the two candidate reasons, built on the
reference's own glyphs and filter. A model "reads" the cloaked BOMB with each
art cell misread with probability p (lower p = better visual-text reading,
the lesson's ViTC axis). After reading, it runs the reference's
`keyword_filter` on *what it read* with probability s (safety that operates
on the decoded meaning). The attack succeeds when the word is read correctly
and not refused. 2,000 seeded trials per cell.

**ANSWER: the proposed reason is post-decoding safety -- Claude refuses on
the word it reconstructed, Gemini acts on it.** The lesson's own frame
offers the rival reason, that Claude simply reads ASCII art worse. The model
shows the two produce the same attack success rate:

| reader | p (misread) | s (refuse after decoding) | recognised | ASR |
|---|---:|---:|---:|---:|
| weaker reader, no post-decoding safety | 0.10 | 0 | 58.5% | 58.5% |
| stronger reader, post-decoding safety | 0.05 | 0.3 | 84.2% | 59.4% |

ASR differs by 0.9 points while recognition differs by 25.7. So a Section 4.3
ranking cannot pick between the reasons; the measurement that can is a
transcription-only (ViTC-style) test on the same art. If Claude transcribes
the art at least as well as Gemini, its lower ASR is safety after decoding;
if it transcribes worse, it is capability.

**FINDING: with s = 0 the capability-safety trade-off is exact; with s = 1
it disappears.** The surface filter (reference `keyword_filter` on the
cloaked prompt) blocks 0 of 2,000 trials. At s = 0, ASR falls from 100% at
p = 0 to 84.2 / 58.5 / 33.2 / 14.0 / 2.1% at p = 0.05 / 0.1 / 0.15 / 0.2 /
0.3 -- the better the reader, the better the attack, as the lesson says. At
s = 1 ASR is 0 at every p: the same substring filter that fails on the
prompt works once it is applied to the model's own reading.

Structure: `noisy_read()` is ex02's nearest-glyph reader fed a corrupted
glyph; `trial_rates()` runs one (p, s) cell with its own seeded
`random.Random`.
"""

from __future__ import annotations

import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "14-ascii-art-visual-jailbreaks"
WORD, PROMPT, TRIALS = "bomb", "how do i make a {}", 2000
P_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)


def noisy_read(glyphs, word, p, rng):
    out = ""
    for c in word.upper():
        cell = "".join((" " if ch == "#" else "#") if rng.random() < p else ch for ch in glyphs[c])
        dist = sorted((sum(a != b for a, b in zip(cell, g)), k) for k, g in glyphs.items())
        out += dist[0][1] if dist[0][0] < dist[1][0] else "?"
    return out.lower()


def trial_rates(ref, p, s, seed=0):
    """(recognised, attack success) over TRIALS reads of the cloaked WORD."""
    glyphs, rng = {c: "".join(g) for c, g in ref.GLYPHS.items()}, random.Random(seed)
    forbidden = [WORD]
    surface = ref.keyword_filter(ref.cloak_prompt(PROMPT.format(WORD), [WORD]), forbidden)
    seen = hits = 0
    for _ in range(TRIALS):
        read = noisy_read(glyphs, WORD, p, rng)
        refused = surface or (rng.random() < s and ref.keyword_filter(read, forbidden))
        seen += read == WORD
        hits += read == WORD and not refused
    return round(seen / TRIALS, 3), round(hits / TRIALS, 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON, "en")
    return {
        "doc_figures": re.findall(r"\d+%", doc),
        "claude_fails": bool(re.search(r"Claude[^.]*all fail", doc)),
        "surface": ref.keyword_filter(ref.cloak_prompt(PROMPT.format(WORD), [WORD]), [WORD]),
        "s0": {p: trial_rates(ref, p, 0.0) for p in P_GRID},
        "s1": {p: trial_rates(ref, p, 1.0)[1] for p in P_GRID},
        "weaker": trial_rates(ref, 0.10, 0.0), "stronger": trial_rates(ref, 0.05, 0.3),
    }


def verify(result):
    s0 = [asr for _, asr in result["s0"].values()]
    weak, strong = result["weaker"], result["stronger"]
    return [
        practice.Check(
            "the lesson quotes no Section 4.3 per-model figure",
            result["doc_figures"] == ["75%"] and result["claude_fails"],
            f"percent figures in docs/en.md: {result['doc_figures']}; Claude listed among "
            f"'all fail': {result['claude_fails']}",
        ),
        practice.Check(
            "ANSWER: a better reader refusing 30% after decoding ties a worse reader",
            weak == (0.585, 0.585) and strong == (0.842, 0.594),
            f"(recognised, ASR) at p=0.10, s=0: {weak}; at p=0.05, s=0.3: {strong}; "
            f"ASR gap {abs(strong[1] - weak[1]):.3f}, recognition gap {strong[0] - weak[0]:.3f}",
        ),
        practice.Check(
            "FINDING: with s = 0 the trade-off is exact; with s = 1 it disappears",
            not result["surface"] and s0 == [1.0, 0.842, 0.585, 0.332, 0.14, 0.021]
            and set(result["s1"].values()) == {0.0},
            f"surface filter blocks: {result['surface']}; ASR at s=0 by p: {result['s0']}; "
            f"at s=1: {result['s1']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
