"""Exercise 2 — one dial, and the two metrics pull against each other on it.

    **Medium.** Use OpenVoice v2 to clone your own voice. Measure SECS between
    reference and clone. Measure CER via Whisper.

Reading of the exercise: `openvoice`, `torch`, `whisper`, `transformers`,
`soundfile` and `sounddevice` are all absent and there is no microphone and no
audio file in the reference tree, so neither the clone nor either measurement can
be made against a real model. Against *this* model both can be made exactly, and
they turn out to be the same measurement read in two directions.

`fake_tts` has one parameter, `mix`. SECS is the cosine to the speaker half and
content fidelity is the cosine to the content half, so the two move in opposite
directions along a single dial and the exercise's two numbers are one number:

| `mix` | SECS | content cosine | retrieved correctly from 51 candidates |
|---:|---:|---:|---|
| 0.10 | 0.1133 | 0.9987 | yes |
| 0.50 (shipped) | 0.4620 | 0.9142 | yes |
| 0.70 | 0.7493 | 0.7082 | yes |
| **0.90** | **0.9726** | **0.2934** | **no** |
| 1.00 | 1.0000 | 0.0631 | no |

There is no trade-off curve to explore, because there is no second parameter:
every point on that table is `mix`, and a "better clone" is a larger number typed
into one call.

**CER cannot be defined here at all.** `content_vector` is SHA-256 of the text --
a one-way function -- so there is no pre-image to decode and no character
sequence to score. The closest measurable thing is retrieval: does
`extract_content` still return the right text out of a candidate set? That is a
**Bernoulli, not a rate**. Over 51 candidates it is right at `mix <= 0.7` and
wrong at `mix >= 0.9`, and it can never take a value between 0 and 1 for a single
utterance, because there are no characters to get partly right.

That is the substantive difference between a hash and a phonetic posteriorgram.
A real content channel degrades gracefully -- CER 2.1% means most characters
survived -- and the failure mode a CER is meant to catch, a clone that says
something slightly different, is exactly the one this representation cannot have.

Structure: `build` makes the speaker, the content and a candidate set;
`curve` sweeps `mix` reporting both cosines and whether retrieval still works;
`transition` finds the mix at which retrieval flips.
"""

from __future__ import annotations

import importlib.util
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "08-voice-cloning-conversion"
DIM, TEXT = 64, "please remember to water plants"
MIXES = (0.1, 0.5, 0.7, 0.9, 1.0)
DISTRACTORS = 50
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()
ABSENT = ("openvoice", "torch", "whisper", "transformers", "soundfile", "sounddevice")


def build(ref):
    """The target speaker, the content, and 50 distractor texts to retrieve against."""
    rnd = random.Random(1)
    speaker = ref.speaker_vector("alice_00001", DIM)
    candidates = {" ".join(rnd.choice(WORDS) for _ in range(5)): None
                  for _ in range(DISTRACTORS)}
    candidates[TEXT] = None
    return speaker, {text: ref.content_vector(text, DIM) for text in candidates}


def curve(ref, speaker, candidates):
    """SECS, content cosine and retrieval success at each mix."""
    content = candidates[TEXT]
    rows = {}
    for mix in MIXES:
        wave = ref.fake_tts(content, speaker, mix)
        top, _ = ref.extract_content(wave, candidates)
        rows[mix] = {"secs": ref.cosine(speaker, wave), "content": ref.cosine(content, wave),
                     "retrieved": top == TEXT}
    return rows


def transition(ref, speaker, candidates, steps=200):
    """The largest mix at which the right text is still retrieved."""
    content = candidates[TEXT]
    good = [i / steps for i in range(steps + 1)
            if ref.extract_content(ref.fake_tts(content, speaker, i / steps),
                                   candidates)[0] == TEXT]
    return max(good), len(good) / (steps + 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    speaker, candidates = build(ref)
    rows = curve(ref, speaker, candidates)
    edge, share = transition(ref, speaker, candidates)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "rows": rows, "edge": edge, "share": share, "candidates": len(candidates),
        "outcomes": {row["retrieved"] for row in rows.values()},
    }


def verify(result):
    rows = result["rows"]
    low, high = rows[MIXES[0]], rows[MIXES[-1]]
    return [
        practice.Check(
            "CONTROL: neither the clone nor either measurement can use a real model",
            len(result["absent"]) == len(ABSENT),
            f"find_spec is None for {result['absent']}, and the reference tree ships no audio "
            "file and no microphone. Against this module both numbers are computable exactly, "
            "which is what makes the next two checks possible",
        ),
        practice.Check(
            "ANSWER: SECS and content fidelity are one dial read in two directions",
            low["secs"] < high["secs"] and low["content"] > high["content"],
            f"`fake_tts` has one parameter. SECS runs {low['secs']:.4f} -> {high['secs']:.4f} "
            f"across mix {MIXES[0]} -> {MIXES[-1]} while the content cosine runs "
            f"{low['content']:.4f} -> {high['content']:.4f}. There is no trade-off curve to "
            "explore because there is no second parameter to trade against",
        ),
        practice.Check(
            "FINDING: a better SECS is a larger constant, not a better clone",
            rows[0.9]["secs"] > 0.95 and not rows[0.9]["retrieved"],
            f"at mix 0.9 the clone scores SECS {rows[0.9]['secs']:.4f} -- above every row of "
            f"the lesson's own leaderboard -- while its content cosine is "
            f"{rows[0.9]['content']:.4f} and the right text is no longer retrieved from "
            f"{result['candidates']} candidates. Nothing about the model changed",
        ),
        practice.Check(
            "FINDING: CER has no definition here -- retrieval is a Bernoulli, not a rate",
            result["outcomes"] == {True, False} and 0.6 < result["edge"] < 0.9,
            f"`content_vector` is SHA-256, a one-way function, so there is no pre-image to "
            f"decode and no characters to score. Retrieval is the nearest measurable thing and "
            f"it flips at mix {result['edge']:.3f} -- right below, wrong above, never partly "
            f"right ({result['share']:.0%} of the sweep succeeds). The failure a CER exists to "
            "catch, a clone saying something slightly different, this representation cannot have",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
