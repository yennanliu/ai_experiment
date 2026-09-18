"""Exercise 3 — the seven residual levels share one codebook slot.

    Your vocabulary has 32k text + 4k image + 4k speech. Add 8k music and ~10
    separators. What is the embedding-matrix parameter cost at hidden dim 4096?

Reading of the exercise: the arithmetic is done as stated and then compared with
the lesson's own `build_vocab`, which allocates a different vocabulary -- two
speech slots rather than one and six separators rather than ten. The gap is
small in parameters and large in what it says about the residual-VQ design, so
both are reported.

**ANSWER: 48,394 entries at 4,096 wide is 198,221,824 parameters** -- 198.2M
tied, **396.4M** if the input and output embeddings are separate. That is
**2.83%** of a 7B model tied and 5.66% untied.

**FINDING: the lesson's own vocabulary is 52,486, not 48,394.** It allocates
`speech L0` *and* `speech L1..L7` at 4,096 each, and six separators rather than
ten -- **215.0M** parameters, **8.5%** above the exercise's figure.

**FINDING: and "speech L1..L7" is one 4,096-entry slot for seven codebooks.**
Residual-VQ levels each have their own codebook, so either the seven are tied to
one table or the slot is seven times too small. Allocating all eight separately
gives a vocabulary of **77,062** and **315.6M** parameters -- **46.8%** above
what the lesson prints, from a design detail its own table cannot express.

**FINDING: the music slot alone outweighs a whole projector.** 8,192 x 4,096 is
**33.6M** parameters -- **1.49x** LLaVA's entire two-layer projector from Lesson
12.05, and it is one row of a vocabulary table. Adding a modality to a shared
vocabulary is not a small change even when the modality is.

Structure: `embedding` prices a vocabulary at one hidden size, `STATED` is the
exercise's own allocation, `lesson_vocab` reads the shipped one, and `untied`
re-allocates the residual levels separately.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "16-mio-any-to-any-streaming"
HIDDEN = 4096
STATED = {"text": 32000, "image": 4096, "speech": 4096, "music": 8192, "separators": 10}
RESIDUAL_LEVELS = 8
LLAVA_PROJECTOR = 22_552_576          # Lesson 12.05
SEVEN_B = 7e9


def embedding(entries, hidden=HIDDEN):
    return entries * hidden


def lesson_vocab(ref):
    slots = ref.build_vocab()
    return slots[-1].end, {slot.name: slot.size for slot in slots}


def untied_vocab(ref):
    """The same layout with every residual level given its own codebook slot."""
    total, sizes = lesson_vocab(ref)
    speech = sizes["speech L0"]
    return total - sizes["speech L1..L7"] + (RESIDUAL_LEVELS - 1) * speech


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    stated = sum(STATED.values())
    shipped, sizes = lesson_vocab(ref)
    untied = untied_vocab(ref)
    return {
        "stated_entries": stated, "stated_params": embedding(stated),
        "stated_m": round(embedding(stated) / 1e6, 1),
        "untied_heads_m": round(2 * embedding(stated) / 1e6, 1),
        "share_of_7b": round(embedding(stated) / SEVEN_B * 100, 2),
        "share_untied": round(2 * embedding(stated) / SEVEN_B * 100, 2),
        "shipped_entries": shipped, "shipped_m": round(embedding(shipped) / 1e6, 1),
        "gap_entries": shipped - stated,
        "gap_pct": round((shipped / stated - 1) * 100, 1),
        "speech_slots": [name for name in sizes if name.startswith("speech")],
        "separators": sum(1 for size in sizes.values() if size == 1),
        "residual_slot": sizes["speech L1..L7"], "levels": RESIDUAL_LEVELS,
        "untied_entries": untied, "untied_m": round(embedding(untied) / 1e6, 1),
        "untied_increase_pct": round((untied / shipped - 1) * 100, 1),
        "music_params": embedding(STATED["music"]),
        "music_m": round(embedding(STATED["music"]) / 1e6, 1),
        "vs_projector": round(embedding(STATED["music"]) / LLAVA_PROJECTOR, 2),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 48,394 entries at 4,096 wide is 198,221,824 parameters",
            all([result["stated_entries"] == 48_394,
                 result["stated_params"] == 198_221_824,
                 result["stated_m"] == 198.2, result["untied_heads_m"] == 396.4,
                 result["share_of_7b"] == 2.83, result["share_untied"] == 5.66]),
            f"{STATED} sums to {result['stated_entries']:,} entries, which at {HIDDEN:,} "
            f"wide is {result['stated_params']:,} parameters -- {result['stated_m']}M tied "
            f"and {result['untied_heads_m']}M untied, or {result['share_of_7b']}% and "
            f"{result['share_untied']}% of a 7B model",
        ),
        practice.Check(
            "FINDING: the lesson's own vocabulary is 52,486, not 48,394",
            all([result["shipped_entries"] == 52_486, result["shipped_m"] == 215.0,
                 result["gap_entries"] == 4_092, result["gap_pct"] == 8.5,
                 len(result["speech_slots"]) == 2, result["separators"] == 6]),
            f"build_vocab allocates {result['speech_slots']} -- two speech slots -- and "
            f"{result['separators']} separators rather than 10, for "
            f"{result['shipped_entries']:,} entries and {result['shipped_m']}M parameters, "
            f"{result['gap_pct']}% above the exercise's figure",
        ),
        practice.Check(
            "FINDING: 'speech L1..L7' is one 4,096-entry slot for seven codebooks",
            all([result["residual_slot"] == 4096, result["untied_entries"] == 77_062,
                 result["untied_m"] == 315.6, result["untied_increase_pct"] == 46.8]),
            f"residual-VQ levels each have their own codebook, so one "
            f"{result['residual_slot']:,}-entry slot for seven of them means either they are "
            f"tied or it is seven times too small. Allocating all {result['levels']} "
            f"separately gives {result['untied_entries']:,} entries and "
            f"{result['untied_m']}M parameters -- {result['untied_increase_pct']}% above "
            "what the lesson prints, from a detail its own table cannot express",
        ),
        practice.Check(
            "FINDING: the music slot alone outweighs a whole projector",
            all([result["music_params"] == 33_554_432, result["music_m"] == 33.6,
                 result["vs_projector"] == 1.49]),
            f"{STATED['music']:,} x {HIDDEN:,} is {result['music_params']:,} parameters -- "
            f"{result['music_m']}M, {result['vs_projector']}x LLaVA's entire two-layer "
            "projector from Lesson 12.05 -- and it is one row of a vocabulary table. Adding "
            "a modality to a shared vocabulary is not a small change even when the modality "
            "is",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
