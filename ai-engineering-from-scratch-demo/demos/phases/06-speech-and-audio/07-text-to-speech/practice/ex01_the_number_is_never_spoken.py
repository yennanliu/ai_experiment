"""Exercise 1 — the number is never spoken.

    **Easy.** Run `code/main.py`. Builds a phoneme dictionary from a toy vocab,
    estimates duration per phoneme, and prints a fake "mel" schedule.

Reading of the exercise: running it is one command, so the exercise is read as
"run it and check what came out". The pipeline works -- 42 phonemes, 210 mel
frames, **2.625 s** at a 12.5 ms hop -- and three things in that output are not
what the sentence above it says.

**The sentence it speaks is not the sentence it was given.** `phonemize` walks
the string trying 3-, 2- and 1-character keys and, on a miss, does `i += 1` with
no output. The only character of "Please remind me to water the plants at 6 pm."
that misses is **`6`**, so the demo silently synthesises "at pm". Text
normalisation -- digits, currency, dates, abbreviations -- is the first stage of
every real TTS front end, and here its absence is a `continue`.

**Six of the 42 duration entries can never be reached.** `DURATION_FRAMES` is
keyed by ARPAbet, but `G2P` emits only 36 distinct phonemes and never emits
`AA, AE, AY, OW, OY, ZH`. Two of those are unreachable because `G2P` maps their
own graphemes elsewhere: `"ay" -> EY` and `"ow" -> AW`. The table looks like a
phone inventory and is a superset of one.

**The jitter does not jitter.** `int(round(base * uniform(-0.1, 0.1)))` can only
be non-zero when `base * 0.1 >= 0.5`, so every phoneme with a base of 5 frames or
fewer -- **18 of the 42 entries**, including every stop and every glide -- is
exactly deterministic. Only **14 of this sentence's 42 phonemes** can move at all,
by at most one frame; at the shipped `seed=42` exactly **2** of them do, one up
and one down, and the total lands on 210, the same as with no jitter at all.
Across 200 seeds the total spans 205-215 with a standard deviation of **1.89
frames**, 0.9% of the sentence.

Structure: `unmatched` replays `phonemize`'s own longest-match loop to find the
characters it drops; `reachable` is the phoneme set `G2P` can emit; `deltas`
compares a jittered duration list against the table's base values.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "07-text-to-speech"
TEXT = "Please remind me to water the plants at 6 pm."
JITTER, SEED, HOP_MS, SEEDS = 0.1, 42, 12.5, 200


def unmatched(ref, text):
    """The characters `phonemize` skips -- its longest-match loop, kept honest."""
    lowered, missed, index = text.lower(), [], 0
    while index < len(lowered):
        hit = next((n for n in (3, 2, 1) if lowered[index:index + n] in ref.G2P), None)
        if hit is None:
            missed.append(lowered[index])
        index += hit or 1
    return missed


def reachable(ref):
    return {phone for phones in ref.G2P.values() for phone in phones}


def deltas(ref, phones, seed):
    """Per-phoneme jitter actually applied, against the table's base durations."""
    base = [ref.DURATION_FRAMES.get(p, 5) for p in phones]
    return [got - want for got, want in zip(ref.duration(phones, JITTER, seed), base)], sum(base)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    phones = ref.phonemize(TEXT)
    moved, base_total = deltas(ref, phones, SEED)
    totals = [sum(ref.duration(phones, JITTER, seed)) for seed in range(SEEDS)]
    _, span_ms = ref.mel_schedule(phones, ref.duration(phones, JITTER, SEED))
    frozen = {name for name, value in ref.DURATION_FRAMES.items() if value * JITTER <= 0.5}
    return {
        "phones": len(phones), "frames": sum(ref.duration(phones, JITTER, SEED)),
        "ms": span_ms, "missed": unmatched(ref, TEXT),
        "unreachable": sorted(set(ref.DURATION_FRAMES) - reachable(ref)),
        "entries": len(ref.DURATION_FRAMES), "frozen": len(frozen),
        "movable": sum(1 for p in phones if ref.DURATION_FRAMES.get(p, 5) * JITTER > 0.5),
        "moved": sum(1 for d in moved if d), "sum_moved": sum(moved), "base_total": base_total,
        "span": (min(totals), max(totals)), "sd": statistics.pstdev(totals),
    }


def verify(result):
    low, high = result["span"]
    return [
        practice.Check(
            "ANSWER: 42 phonemes, 210 mel frames, 2.625 s at a 12.5 ms hop",
            result["phones"] == 42 and result["frames"] == 210 and result["ms"] == 2625.0,
            f"{result['phones']} phonemes -> {result['frames']} frames -> {result['ms']:.1f} ms, "
            f"{result['frames'] * 300} samples at 24 kHz, a rate of "
            f"{result['phones'] / (result['ms'] / 1000):.1f} phonemes per second",
        ),
        practice.Check(
            "FINDING: the one character it cannot phonemize is the number",
            result["missed"] == ["6"],
            f"`phonemize` does `i += 1` with no output on a miss, and the only miss in the "
            f"sentence is {result['missed']} -- so the demo synthesises 'at pm'. Text "
            "normalisation is the first stage of every real TTS front end; here it is a skip",
        ),
        practice.Check(
            "FINDING: six duration entries can never be reached",
            result["unreachable"] == ["AA", "AE", "AY", "OW", "OY", "ZH"],
            f"`DURATION_FRAMES` has {result['entries']} entries and `G2P` emits "
            f"{len(result['unreachable'])} fewer: {result['unreachable']}. Two of them are "
            "unreachable because `G2P` sends their own graphemes elsewhere -- 'ay' to EY and "
            "'ow' to AW. The table reads as an inventory and is a superset of one",
        ),
        practice.Check(
            "FINDING: the jitter is exactly zero for 18 of the 42 duration entries",
            result["frozen"] == 18,
            f"`int(round(base * uniform(-{JITTER}, {JITTER})))` needs base * {JITTER} to exceed "
            f"0.5, since Python rounds a half to even, so every entry of 5 frames or fewer -- "
            f"{result['frozen']} of {result['entries']}, including every stop and glide -- is "
            f"deterministic. Only {result['movable']} of this sentence's {result['phones']} "
            "phonemes can move at all",
        ),
        practice.Check(
            "ANSWER: at the shipped seed two durations move and they cancel",
            result["moved"] == 2 and result["sum_moved"] == 0,
            f"seed {SEED} moves {result['moved']} of {result['phones']} phonemes, by "
            f"{result['sum_moved']:+d} frames in total, so the schedule lands on "
            f"{result['base_total']} -- the no-jitter number. Across {SEEDS} seeds the total "
            f"spans {low}-{high}, standard deviation {result['sd']:.2f} frames, "
            f"{result['sd'] / result['base_total'] * 100:.1f}% of the sentence",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
