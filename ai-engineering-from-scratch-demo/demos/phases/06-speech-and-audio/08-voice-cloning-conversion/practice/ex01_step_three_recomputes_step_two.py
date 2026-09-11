"""Exercise 1 — Step 3 recomputes Step 2, byte for byte.

    **Easy.** Run `code/main.py`. Demonstrates the speaker-embedding swap by
    computing the cosine between two "speakers" pre and post swap.

Reading of the exercise: "demonstrates the swap" is a claim about Step 3, so the
check is whether a swap happens. It does not. Step 3 calls
`extract_content(wav_bob_orig, contents)`, gets back the *exact dictionary key*
it started from, looks the original vector up by that key, and passes it to
`fake_tts(content, alice, 0.5)` -- which is the same call Step 2 already made.
`wav_converted == wav_cloned` is **True**, element for element. Nothing is
estimated, nothing is converted, and the number Step 3 prints is Step 2's number.

**The SECS it prints is a function of `mix` and nothing else.** `fake_tts` returns
`(1-mix)*content + mix*speaker`. The speaker vector is L2-normalised to 1.0 and
the content vector is a byte-hash scaled to `[-0.5, 0.5]`, norm **2.189**, so the
cosine to the speaker is fixed by the ratio of those two norms at the chosen mix:

| `mix` | SECS to the target | cosine to the content |
|---:|---:|---:|
| 0.10 | 0.1133 | 0.9987 |
| **0.50** (shipped) | **0.4620** | 0.9142 |
| 0.72 | **0.7800** | 0.7048 |
| 1.00 | 1.0000 | 0.0631 |

Two things follow. The demo prints `SECS = 0.462` and then says "production
ECAPA-TDNN on real clones lands SECS in 0.65 - 0.78" -- its own number is below
the worst row of its own leaderboard, and the row can be matched exactly by
setting `mix = 0.721`. And the SECS measures the mixing constant, not the clone.

**The speaker probe cannot say no.** `extract_speaker` sorts three cosines and
returns the largest, with no threshold and no reject option, so "should stay
alice" is guaranteed for any `mix > 0`: bob and carol are independent Gaussians
and score **-0.013** and near zero against the same wave.

**And the content channel is a hash.** `content_vector` is SHA-256 of the text,
so it has no metric structure at all: two different sentences score **-0.107**
and deleting one character scores **0.087**. `extract_content` is exact-match
retrieval by another name.

Structure: `build` assembles the demo's speakers, contents and waves; `sweep`
reports SECS and content cosine at a range of mixes; `mix_for` solves for the mix
that produces a target SECS.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "08-voice-cloning-conversion"
DIM, MIX = 64, 0.5
GREET, REMIND = "hello this is a test", "please remember to water plants"
MIXES = (0.1, 0.25, 0.5, 0.8, 1.0)
LEADERBOARD_TOP, LEADERBOARD_LOW = 0.78, 0.65


def build(ref):
    """`main()`'s own objects: three speakers, two contents, and the three waves."""
    speakers = {name: ref.speaker_vector(seed, DIM) for name, seed in
                (("alice", "alice_00001"), ("bob", "bob_00002"), ("carol", "carol_00003"))}
    contents = {text: ref.content_vector(text, DIM) for text in (GREET, REMIND)}
    cloned = ref.fake_tts(contents[REMIND], speakers["alice"], MIX)
    bob_original = ref.fake_tts(contents[REMIND], speakers["bob"], MIX)
    matched, _ = ref.extract_content(bob_original, contents)
    return speakers, contents, cloned, ref.fake_tts(contents[matched], speakers["alice"], MIX)


def sweep(ref, content, speaker):
    return {mix: (ref.cosine(speaker, ref.fake_tts(content, speaker, mix)),
                  ref.cosine(content, ref.fake_tts(content, speaker, mix))) for mix in MIXES}


def mix_for(ref, content, speaker, target, rounds=50):
    """The mixing constant that makes SECS equal `target` -- one bisection, no model."""
    low, high = 0.0, 1.0
    for _ in range(rounds):
        middle = (low + high) / 2
        if ref.cosine(speaker, ref.fake_tts(content, speaker, middle)) < target:
            low = middle
        else:
            high = middle
    return high


def norm(vector):
    return math.sqrt(sum(x * x for x in vector))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    speakers, contents, cloned, converted = build(ref)
    alice, content = speakers["alice"], contents[REMIND]
    return {
        "identical": cloned == converted,
        "secs": ref.cosine(alice, cloned), "rival": ref.cosine(speakers["bob"], cloned),
        "sweep": sweep(ref, content, alice),
        "target_mix": mix_for(ref, content, alice, LEADERBOARD_TOP),
        "content_norm": norm(content), "speaker_norm": norm(alice),
        "across_texts": ref.cosine(contents[GREET], content),
        "one_char": ref.cosine(contents[GREET], ref.content_vector(GREET[:-1], DIM)),
    }


def verify(result):
    sweep_rows = result["sweep"]
    return [
        practice.Check(
            "ANSWER: Step 3's 'converted' wave is Step 2's clone, element for element",
            result["identical"],
            "`extract_content` returns the exact dictionary key it was given, the lookup "
            "returns the original vector, and `fake_tts(content, alice, 0.5)` is the call Step 2 "
            "already made. `wav_converted == wav_cloned` is True -- nothing is estimated and "
            "nothing is converted, so the cosine Step 3 prints is Step 2's",
        ),
        practice.Check(
            "MECHANISM: the SECS is a function of `mix` and the two vector norms",
            abs(sweep_rows[MIX][0] - result["secs"]) < 1e-12 and result["content_norm"] > 2,
            f"`fake_tts` returns (1-mix)*content + mix*speaker with ||speaker|| = "
            f"{result['speaker_norm']:.3f} and ||content|| = {result['content_norm']:.3f}, so "
            f"SECS runs {sweep_rows[MIXES[0]][0]:.4f} -> {sweep_rows[MIXES[-1]][0]:.4f} across "
            f"mix {MIXES[0]} -> {MIXES[-1]} and the content cosine runs the other way, "
            f"{sweep_rows[MIXES[0]][1]:.4f} -> {sweep_rows[MIXES[-1]][1]:.4f}",
        ),
        practice.Check(
            "FINDING: the demo's own SECS is below the worst row of its own leaderboard",
            result["secs"] < LEADERBOARD_LOW and 0.7 < result["target_mix"] < 0.75,
            f"it prints {result['secs']:.3f} and then quotes 'production ECAPA-TDNN ... "
            f"{LEADERBOARD_LOW} - {LEADERBOARD_TOP}'. Matching the top row takes no model: "
            f"mix = {result['target_mix']:.3f} gives exactly {LEADERBOARD_TOP}. The number "
            "measures the mixing constant",
        ),
        practice.Check(
            "CONTROL: the speaker probe has no threshold and no reject option",
            result["rival"] < 0.05 < result["secs"],
            f"`extract_speaker` sorts three cosines and returns the largest, so 'should stay "
            f"alice' holds for any mix above zero: bob scores {result['rival']:.4f} against "
            f"alice's {result['secs']:.4f} because the speaker vectors are independent Gaussians",
        ),
        practice.Check(
            "CONTROL: the content channel is a hash, so it has no metric structure",
            abs(result["across_texts"]) < 0.2 and abs(result["one_char"]) < 0.2,
            f"`content_vector` is SHA-256 of the text: two different sentences score "
            f"{result['across_texts']:.4f} and deleting one character scores "
            f"{result['one_char']:.4f}. `extract_content` is exact-match retrieval under "
            "another name, which is why Step 3 could recover its input exactly",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
