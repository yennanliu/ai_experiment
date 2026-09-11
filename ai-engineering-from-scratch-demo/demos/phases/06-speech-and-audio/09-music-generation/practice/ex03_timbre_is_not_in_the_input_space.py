"""Exercise 3 — timbre is not in the input space.

    **Hard.** Using ACE-Step (or MusicGen-melody), generate three variations of
    the same tune with different timbre prompts. Compute CLAP similarity to the
    prompt to verify alignment.

Reading of the exercise: `acestep`, `audiocraft`, `torch` and `laion_clap` are
all absent, so no model runs -- but the exercise can be answered against
`code/main.py`, because the thing it asks to vary is one the parser does not
read. `fake_generate` extracts exactly three things from a prompt: a **key**, a
**genre**, and a **bpm** that never parses (Exercise 1). Timbre is not among
them, so

    "warm analog pop in C"  ==  "bright digital pop in C"  ==  "muted felt pop in C"

byte for byte -- three "variations" that are one output. Three prompts differing
only in instrumentation produce **1** distinct piece.

**The whole output space is 20 pieces.** Four keys times five genres, with the
tempo stuck at 120, is the complete range of everything `fake_generate` can ever
emit. Enumerated, the 20 pieces carry only **20 chord sequences and 4 drum
tracks** (Exercise 1's table mismatch costs one), so "three variations" can only
ever be three of twenty, and only if the prompts disagree about key or genre.

**And the alignment a CLAP score would verify is checkable without CLAP.** The
only prompt-to-output relation this model has is `prompt -> (key, genre)`, so
alignment is: does the piece use the key the prompt named? Over a probe set where
the intended key is unambiguous, it does not -- the substring matcher keys
`"rock anthem at 140 bpm"` to **A** on the word "at". A similarity score computed
against such a piece would be scoring the parser, and a single number with no
null distribution could not tell you that.

A CLAP report needs three things this exercise does not ask for: the score, the
score against a *mismatched* prompt, and the spread of both. Here the mismatched
score is the same number as the matched one, because the pieces are identical.

Structure: `variations` renders a set of prompts and counts distinct outputs;
`space` enumerates every reachable piece; `aligned` asks whether the key a prompt
names is the key the parser picked.
"""

from __future__ import annotations

import importlib.util

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "09-music-generation"
BARS = 8
TIMBRE = ("warm analog pop in C", "bright digital pop in C", "muted felt pop in C")
TEMPO = ("pop in C at 90 bpm", "pop in C at 120 bpm", "pop in C at 180 bpm")
NAMED = {"rock anthem in D at 140 bpm": "D", "jazz swing in A": "A",
         "upbeat pop in G major": "G", "slow lofi groove in C": "C"}
UNNAMED = {"rock anthem at 140 bpm": None, "an upbeat song": None, "a slow groove": None}
ABSENT = ("acestep", "audiocraft", "torch", "laion_clap", "transformers")


def variations(ref, prompts):
    """(distinct renderings, the rendering of the first prompt)."""
    rendered = [ref.fake_generate(p) for p in prompts]
    return len({repr(piece) for piece in rendered}), rendered[0]


def space(ref):
    """Every piece `fake_generate` can emit: key x genre, tempo frozen."""
    return {(key, genre): (tuple(ref.chord_progression(key, genre, BARS)),
                           ref.drum_pattern(genre, BARS))
            for key in ref.MAJOR_KEYS for genre in ref.COMMON_PROGRESSIONS}


def aligned(ref, prompts):
    """Prompts whose stated key is the key the parser actually chose."""
    return [p for p, want in prompts.items() if ref.fake_generate(p)["key"] == want]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pieces = space(ref)
    timbre_count, first = variations(ref, TIMBRE)
    tempo_count, _ = variations(ref, TEMPO)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "timbre": timbre_count, "tempo": tempo_count, "prompts": len(TIMBRE),
        "pieces": len(pieces),
        "chords": len({chords for chords, _ in pieces.values()}),
        "drums": len({drums for _, drums in pieces.values()}),
        "fields": sorted(first), "bpm": first["bpm"],
        "named_hits": len(aligned(ref, NAMED)), "named": len(NAMED),
        "unnamed_keys": {p: ref.fake_generate(p)["key"] for p in UNNAMED},
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: three timbre prompts produce one piece",
            result["timbre"] == 1 and result["tempo"] == 1,
            f"{result['prompts']} prompts differing only in instrumentation render to "
            f"{result['timbre']} distinct output, and {result['prompts']} differing only in "
            f"tempo render to {result['tempo']}. `fake_generate` reads a key, a genre and a bpm "
            f"that never parses -- its output fields are {result['fields']} and the tempo is "
            f"{result['bpm']} in all three",
        ),
        practice.Check(
            "FINDING: the entire output space is 20 pieces",
            result["pieces"] == 20 and result["chords"] == 20 and result["drums"] == 4,
            f"{len(parity.load_reference(PHASE, LESSON, 'main').MAJOR_KEYS)} keys x "
            f"{len(parity.load_reference(PHASE, LESSON, 'main').COMMON_PROGRESSIONS)} genres "
            f"with the tempo frozen is everything the model can emit: {result['pieces']} pieces "
            f"carrying {result['chords']} chord sequences and {result['drums']} drum tracks. "
            "Three variations can only be three of twenty, and only across key or genre",
        ),
        practice.Check(
            "FINDING: the alignment a CLAP score would verify is already broken",
            result["named_hits"] == result["named"]
            and set(result["unnamed_keys"].values()) != {"C"},
            f"the only prompt-to-output relation here is prompt -> (key, genre). It holds on "
            f"{result['named_hits']}/{result['named']} prompts that name a key and fails on "
            f"ones that do not: {result['unnamed_keys']}, where the substring matcher keys on "
            "'at', 'an' and 'groove'. A similarity score would be scoring the parser",
        ),
        practice.Check(
            "CONTROL: no model is installed, and a CLAP number would need three of them",
            len(result["absent"]) == len(ABSENT),
            f"find_spec is None for {result['absent']}. A readable CLAP report is the score, "
            "the score against a mismatched prompt, and the spread of both -- and here the "
            "mismatched score equals the matched one, because the pieces are identical",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
