"""Exercise 1 — the tempo in the prompt is never read.

    **Easy.** Run `code/main.py`. It produces a "generative" chord progression +
    drum pattern as ASCII symbols — a music-gen cartoon. Play it back via any
    MIDI renderer if you want.

Reading of the exercise: it runs, and produces four pieces of 8 chords and 128
drum steps each. Since the interesting object is the prompt parser -- the only
thing between the prompt and the output -- the exercise is read as "run it and
check that the prompts arrive".

**Two of the four do not.** "upbeat pop in G major at **128 bpm**" and "rock
anthem in D at **140 bpm**" both come out at **120**. `fake_generate` scans
`prompt_lower.split()` for a token ending in `"bpm"`, and splitting on whitespace
puts the number in one token and the unit in another -- so the token that matches
is the bare `"bpm"`, `int("")` raises `ValueError`, and the `except` swallows it.
Writing `128bpm` with no space parses correctly, which is the tell.

**The key detector matches inside ordinary words.** It tests `f" {k.lower()}"` as
a substring, so a leading space is the only boundary it has:

| prompt | key it picks | why |
|---|---|---|
| `rock anthem at 140 bpm` | **A** | the word "**a**t" |
| `an upbeat song` | **A** | "**a**n" |
| `a slow groove` | **G** | "**g**roove" |
| `instrumental piece` | C | nothing matched; C is the default |

**The genre detector resolves by dictionary order, not by the prompt.** It walks
`COMMON_PROGRESSIONS` -- `pop, ballad, jazz, rock, lofi` -- and returns the first
key that appears anywhere in the string, so `"lofi rock ballad"` is a **ballad**.

**And the two tables disagree with each other.** `COMMON_PROGRESSIONS` has a
`ballad` that `DRUM_PATTERNS` does not, so every ballad silently gets the pop
beat; `DRUM_PATTERNS` has a `trap` that `COMMON_PROGRESSIONS` does not, so the
trap pattern is unreachable. Across all 20 reachable pieces there are only **4
distinct drum tracks**, not 5.

**Nothing here generates.** `fake_generate(prompt, rng=None)` builds an `rng` and
never reads it again; 50 different seeds produce **1** distinct output.

Structure: `parsed` collects what the parser extracted from a prompt; `probes`
is the set of prompts that exercise the substring bug; `reachable` enumerates
every piece the model can emit.
"""

from __future__ import annotations

import inspect
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "09-music-generation"
PROMPTS = ("upbeat pop in G major at 128 bpm", "slow lofi groove in C",
           "rock anthem in D at 140 bpm", "jazz swing in A")
PROBES = ("rock anthem at 140 bpm", "an upbeat song", "a slow groove", "instrumental piece")
DEFAULT_BPM, SEEDS, BARS = 120, 50, 8


def parsed(ref, prompt):
    piece = ref.fake_generate(prompt)
    return {key: piece[key] for key in ("key", "genre", "bpm")}


def stated_bpm(prompt):
    """The tempo the prompt names, read the way a person reads it."""
    tokens = prompt.lower().split()
    numbers = [t for t, nxt in zip(tokens, tokens[1:]) if nxt == "bpm" and t.isdigit()]
    return int(numbers[0]) if numbers else None


def reachable(ref):
    """Every (chords, drums) pair `fake_generate` can emit, over key x genre."""
    return {(key, genre): (tuple(ref.chord_progression(key, genre, BARS)),
                           ref.drum_pattern(genre, BARS))
            for key in ref.MAJOR_KEYS for genre in ref.COMMON_PROGRESSIONS}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pieces = reachable(ref)
    body = inspect.getsource(ref.fake_generate).split("rng = rng or")[1]
    return {
        "prompts": {p: parsed(ref, p) for p in PROMPTS},
        "stated": {p: stated_bpm(p) for p in PROMPTS},
        "unspaced": ref.fake_generate(PROMPTS[0].replace(" bpm", "bpm"))["bpm"],
        "probes": {p: parsed(ref, p) for p in PROBES},
        "ordered": ref.fake_generate("lofi rock ballad")["genre"],
        "genres": list(ref.COMMON_PROGRESSIONS), "beats": list(ref.DRUM_PATTERNS),
        "drums": len({drums for _, drums in pieces.values()}), "pieces": len(pieces),
        "seeds": len({repr(ref.fake_generate(PROMPTS[0], random.Random(s)))
                      for s in range(SEEDS)}),
        "rng_used": "rng" in body.split("return")[0],
    }


def verify(result):
    stated = {p: b for p, b in result["stated"].items() if b}
    missed = [p for p in stated if result["prompts"][p]["bpm"] != stated[p]]
    probes = result["probes"]
    return [
        practice.Check(
            "ANSWER: both prompts that name a tempo come out at 120",
            len(missed) == len(stated) == 2 and result["unspaced"] == 128,
            f"{len(stated)} of {len(PROMPTS)} prompts state a tempo and all "
            f"{len(missed)} are ignored: `.split()` puts the number and the unit in separate "
            f"tokens, so the token ending in 'bpm' is the bare unit, `int('')` raises and the "
            f"`except` swallows it. Written '128bpm' it parses -- {result['unspaced']}",
        ),
        practice.Check(
            "FINDING: the key detector matches inside ordinary words",
            probes["rock anthem at 140 bpm"]["key"] == "A"
            and probes["a slow groove"]["key"] == "G",
            f"it tests f' {{k}}' as a substring, so a leading space is its only boundary: "
            f"'rock anthem at 140 bpm' is keyed {probes['rock anthem at 140 bpm']['key']} on the "
            f"word 'at', 'an upbeat song' is {probes['an upbeat song']['key']}, and 'a slow "
            f"groove' is {probes['a slow groove']['key']} on 'groove'",
        ),
        practice.Check(
            "FINDING: the genre is chosen by dictionary order, not by the prompt",
            result["ordered"] == "ballad",
            f"it walks {result['genres']} and returns the first key that appears anywhere in "
            f"the string, so 'lofi rock ballad' is a {result['ordered']!r} -- the word that "
            "comes last in the prompt and second in the dict",
        ),
        practice.Check(
            "FINDING: the two tables disagree, so one beat is dead and one is shared",
            result["drums"] == 4 and set(result["genres"]) != set(result["beats"]),
            f"`COMMON_PROGRESSIONS` has {sorted(set(result['genres']) - set(result['beats']))} "
            f"with no drum pattern, which silently falls back to pop, and `DRUM_PATTERNS` has "
            f"{sorted(set(result['beats']) - set(result['genres']))}, which no prompt can "
            f"select. Across all {result['pieces']} reachable pieces there are "
            f"{result['drums']} distinct drum tracks",
        ),
        practice.Check(
            "CONTROL: nothing here generates -- the `rng` is built and never read",
            not result["rng_used"] and result["seeds"] == 1,
            f"`fake_generate(prompt, rng=None)` constructs `random.Random(0)` and never "
            f"references it again, so {SEEDS} different seeds produce {result['seeds']} "
            "distinct output. The piece is a pure function of the prompt",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
