"""Exercise 3 — SECS is a property of the encoder, not of the clone.

    **Hard.** Record a 5-second reference clip of yourself. Use F5-TTS to clone
    it. Report SECS between reference and cloned output.

Reading of the exercise: `f5_tts`, `torch`, `soundfile` and `sounddevice` are all
absent, there is no microphone and no audio file anywhere in the reference tree,
so neither the recording nor the clone can happen. What can happen is the number
itself. SECS is speaker-encoder cosine similarity, and this phase already ships a
speaker encoder -- Lesson 06's `embed_mfcc_stats` and `cosine`. Running it over a
reference voice, four other voices, and a graded family of deliberately imperfect
clones is the `DESIGN D11` scaled-down version, and it shows why "report SECS"
is not a report.

**SECS has no zero.** On this encoder two utterances of the *same* voice average
**0.9958** and two utterances of *different* voices average **0.6468**. The whole
meaningful range is `[0.65, 1.00]`, so a bare "SECS = 0.78" is not 78% of
anything -- it is at the top of the different-speaker distribution.

**A high-looking SECS is a rejection.** The equal-error threshold on the same
encoder is **0.9909**. A clone whose formants are 5% off scores **0.9856** --
and is rejected as a different speaker by the very encoder that reported 98.6%
similarity. Without the threshold, and without the two distributions the
threshold comes from, the number cannot be read in either direction.

| clone error | SECS | accepted at the EER threshold |
|---:|---:|---|
| +2% | 0.9939 | yes |
| **+5%** | **0.9856** | **no** |
| +10% | 0.9412 | no |
| +25% | 0.7732 | no |
| +50% | 0.6238 | no -- the different-speaker mean |

**And the scale belongs to the encoder.** Dropping one coefficient -- MFCC `c0`,
the energy term -- moves the different-speaker floor from **0.6468 to 0.4854**
while the same-speaker top barely moves, 0.9958 to 0.9944. The same clone scores
0.9412 or 0.9123 depending on a choice inside the measuring instrument. Published
SECS numbers use Resemblyzer, WavLM or ECAPA embeddings, which are three different
scales, so they are not comparable to each other either.

Structure: `embed` builds n utterances of one voice through Lesson 06's encoder;
`secs` is the mean pairwise cosine between two sets; `strip` re-normalises an
embedding with coefficients removed; `clone` shifts a voice's formants by a
percentage.
"""

from __future__ import annotations

import importlib.util
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "07-text-to-speech"
ENCODER_LESSON = "06-speaker-recognition-verification"
SR, SECONDS, NOISE = 8000, 0.4, 0.04
REFERENCE = [200, 400, 600]
OTHERS = ([220, 330, 880], [300, 600, 1200], [180, 540, 1080], [260, 520, 780])
ERRORS = (2, 5, 10, 25, 50)
ENERGY = {0, 13}
ABSENT = ("f5_tts", "kokoro", "torch", "soundfile", "sounddevice")


def embed(enc, freqs, count):
    return [enc.embed_mfcc_stats(enc.tone_mix(freqs, SR, SECONDS, noise=NOISE), SR)
            for _ in range(count)]


def secs(enc, left, right, drop=frozenset()):
    """Mean pairwise cosine -- what every SECS report is, under some encoder."""
    return statistics.mean(enc.cosine(strip(enc, a, drop), strip(enc, b, drop))
                           for a in left for b in right)


def strip(enc, vector, drop):
    if not drop:
        return vector
    return enc.l2_normalize([x for i, x in enumerate(vector) if i not in drop])


def clone(percent):
    return [f * (1 + percent / 100) for f in REFERENCE]


def spread(enc, reference, drop=frozenset()):
    """(same-speaker scores, different-speaker scores) for one encoder variant."""
    same = [enc.cosine(strip(enc, a, drop), strip(enc, b, drop))
            for i, a in enumerate(reference) for b in reference[i + 1:]]
    others = [v for freqs in OTHERS for v in embed(enc, freqs, 2)]
    diff = [enc.cosine(strip(enc, a, drop), strip(enc, b, drop))
            for a in reference for b in others]
    return same, diff


def solve():
    enc = parity.load_reference(PHASE, ENCODER_LESSON, "main")
    random.seed(3)
    reference = embed(enc, REFERENCE, 6)
    clones = {error: embed(enc, clone(error), 3) for error in ERRORS}
    same, diff = spread(enc, reference)
    stripped_same, stripped_diff = spread(enc, reference, frozenset(ENERGY))
    rate, threshold = enc.eer(same, diff)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "same": statistics.mean(same), "diff": statistics.mean(diff),
        "eer": rate, "threshold": threshold,
        "clones": {error: secs(enc, reference, vectors) for error, vectors in clones.items()},
        "stripped_same": statistics.mean(stripped_same),
        "stripped_diff": statistics.mean(stripped_diff),
        "stripped_clone": secs(enc, reference, clones[10], frozenset(ENERGY)),
        "clone10": secs(enc, reference, clones[10]),
    }


def verify(result):
    clones, threshold = result["clones"], result["threshold"]
    accepted = [e for e, s in clones.items() if s >= threshold]
    return [
        practice.Check(
            "CONTROL: nothing here can record or clone, so the number is what is left",
            len(result["absent"]) == len(ABSENT),
            f"find_spec is None for {result['absent']} and the reference tree ships no audio "
            "file, so the SECS is computed with the encoder this phase already has -- Lesson "
            "06's `embed_mfcc_stats` and `cosine` -- over a reference voice, four others, and "
            "a graded family of imperfect clones",
        ),
        practice.Check(
            "ANSWER: SECS has no zero -- the whole range is 0.65 to 1.00",
            0.6 < result["diff"] < 0.7 < result["same"],
            f"two utterances of the same voice average {result['same']:.4f} and two of "
            f"different voices {result['diff']:.4f}, so a bare 'SECS = 0.78' is not 78% of "
            "anything: it sits at the top of the different-speaker distribution. The scale has "
            "to be reported with the number",
        ),
        practice.Check(
            "ANSWER: a clone at 0.9856 is rejected by the encoder that scored it",
            accepted == [2] and clones[5] < threshold,
            f"the equal-error threshold on this encoder is {threshold:.4f} (EER "
            f"{result['eer'] * 100:.2f}%), so of clones at {list(ERRORS)}% formant error only "
            f"{accepted}% is accepted: +5% scores {clones[5]:.4f} and fails, +10% "
            f"{clones[10]:.4f}, +50% {clones[50]:.4f} -- the different-speaker mean",
        ),
        practice.Check(
            "FINDING: one coefficient of the encoder moves the floor by 0.15",
            result["diff"] - result["stripped_diff"] > 0.1,
            f"dropping MFCC c0 moves the different-speaker mean from {result['diff']:.4f} to "
            f"{result['stripped_diff']:.4f} while the same-speaker mean barely moves, "
            f"{result['same']:.4f} to {result['stripped_same']:.4f}. The same +10% clone scores "
            f"{result['clone10']:.4f} or {result['stripped_clone']:.4f} depending on a choice "
            "inside the instrument -- and published SECS uses Resemblyzer, WavLM or ECAPA",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
